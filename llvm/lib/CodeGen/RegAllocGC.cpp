#include "RegAllocGC.h"
#include "AllocationOrder.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveIntervalCalc.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/InitializePasses.h"

#include <cassert>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc-gc"

INITIALIZE_PASS_BEGIN(RAGC, "graph-coloring",
                     "graph-coloring Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(SpillPlacement)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(RAGC, "graph-coloring",
                    "graph-coloring Allocator", false, false)

namespace llvm {

using RegisterSet = std::unordered_set<unsigned>;

const float Hysteresis = (2007 / 2048.0f); // 0.97998046875

static Register isFullCopyOf(const MachineInstr &MI, Register Reg) {
  if (!MI.isFullCopy())
    return Register();
  if (MI.getOperand(0).getReg() == Reg)
    return MI.getOperand(1).getReg();
  if (MI.getOperand(1).getReg() == Reg)
    return MI.getOperand(0).getReg();
  return Register();
}

class FastRemator {
public:
  enum RematResult : uint8_t {
    DontCare,
    Rematable,
    UnRematable
  };

  static bool isRealSpill(const MachineInstr &Def) {
    if (!Def.isImplicitDef())
      return true;
    assert(Def.getNumOperands() == 1 &&
          "Implicit def with more than one definition");
    return Def.getOperand(0).getSubReg();
  }

  static void getVDefInterval(const MachineInstr &MI, LiveIntervals &LIS) {
    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isDef() && Register::isVirtualRegister(MO.getReg()))
        LIS.getInterval(MO.getReg());
  }

  FastRemator(LiveRangeEdit &Edit, LiveIntervals &LIS,
              MachineRegisterInfo &MRI, RAGC &Allocator, VirtRegMap &VRM,
              RegisterSet &Visited)
              : Edit(Edit), LIS(LIS), MRI(MRI), Allocator(Allocator), VRM(VRM),
                Visited(Visited) {
    Original = VRM.getOriginal(Edit.getReg());
    StackSlot = VRM.getStackSlot(Original);
  }

  bool isRegToRemat(Register Reg) { return is_contained(RegsToRemat, Reg); }

  // A snippet is a tiny live range with only a single instruction using it
  // besides copies to/from Reg or spills/fills. We accept:
  //
  //   %snip = COPY %Reg / FILL fi#
  //   %snip = USE %snip
  //   %Reg = COPY %snip / SPILL %snip, fi#
  //
  bool isSnippet(const LiveInterval &SnipLI) {
    Register Reg = Edit.getReg();
    // The definition of snippet is the same as in InlineSpliier.
    if (SnipLI.getNumValNums() > 2 || !LIS.intervalIsInOneMBB(SnipLI))
      return false;

    MachineInstr *UseMI = nullptr;
    auto &TII = *Allocator.getMF().getSubtarget().getInstrInfo();
    for (MachineRegisterInfo::reg_instr_nodbg_iterator
            RI = MRI.reg_instr_nodbg_begin(SnipLI.reg()),
            E = MRI.reg_instr_nodbg_end();
        RI != E;) {
      MachineInstr &MI = *RI++;

      if (isFullCopyOf(MI, Reg))
        continue;

      int FI;
      if (SnipLI.reg() == TII.isLoadFromStackSlot(MI, FI) && FI == StackSlot)
        continue;

      if (SnipLI.reg() == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot)
        continue;

      if (UseMI && &MI != UseMI)
        return false;
      UseMI = &MI;
    }
    return true;
  }

  void collectRegsToRemat() {
    Register Reg = Edit.getReg();
    RegsToRemat.assign(1, Reg);

    for (MachineInstr &MI : llvm::make_early_inc_range(MRI.reg_instructions(Reg))) {
      Register SnipReg = isFullCopyOf(MI, Reg);
      if (!SnipReg.isVirtual() || VRM.getOriginal(SnipReg) != Original)
        continue;
      LiveInterval &SnipLI = LIS.getInterval(SnipReg);
      if (isSnippet(SnipLI)) {
        SnippetCopies.insert(&MI);
        if (!isRegToRemat(SnipReg) && !Visited.count(SnipReg))
          RegsToRemat.push_back(SnipReg);
      }
    }
  }

  void markValueUsed(const LiveInterval *LI, VNInfo *VNI) {
    SmallVector<std::pair<const LiveInterval*, VNInfo*>, 8> WorkList;
    WorkList.push_back(std::make_pair(LI, VNI));
    do {
      std::tie(LI, VNI) = WorkList.pop_back_val();
      if (!UsedValues.insert(VNI).second)
        continue;

      assert(!VNI->isPHIDef());

      // Follow snippet copies.
      MachineInstr *MI = LIS.getInstructionFromIndex(VNI->def);
      if (!SnippetCopies.count(MI))
        continue;
      LiveInterval &SnipLI = LIS.getInterval(MI->getOperand(1).getReg());
      assert(isRegToRemat(SnipLI.reg()) && "Unexpected register in copy");
      VNInfo *SnipVNI = SnipLI.getVNInfoAt(VNI->def.getRegSlot(true));
      assert(SnipVNI && "Snippet undefined before copy");
      WorkList.push_back(std::make_pair(&SnipLI, SnipVNI));
    } while (!WorkList.empty());
  }

  RematResult tryRematerializeFor(LiveInterval &VirtReg, MachineInstr &MI) {
    // Analyze instruction
    SmallVector<std::pair<MachineInstr *, unsigned>, 8> Ops;
    VirtRegInfo RI = AnalyzeVirtRegInBundle(MI, VirtReg.reg(), &Ops);

    if (!RI.Reads)
      return DontCare;

    SlotIndex UseIdx = LIS.getInstructionIndex(MI).getRegSlot(true);
    /*
     * TODO: We should think about the effect of expanding the live ranges of other operands.
     */
    VNInfo *ParentVNI = VirtReg.getVNInfoAt(UseIdx.getBaseIndex());

    if (!ParentVNI) {
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.isUse() && MO.getReg() == VirtReg.reg())
          MO.setIsUndef();
      return DontCare;
    }

    if (SnippetCopies.count(&MI))
      return UnRematable;

    LiveInterval &OrigLI = LIS.getInterval(Original);
    VNInfo *OrigVNI = OrigLI.getVNInfoAt(UseIdx);
    LiveRangeEdit::Remat RM(ParentVNI);
    RM.OrigMI = LIS.getInstructionFromIndex(OrigVNI->def);

    if (RI.Tied || !Edit.canRematerializeAt(RM, OrigVNI, UseIdx, false)) {
      markValueUsed(&VirtReg, ParentVNI);
      return UnRematable;
    }

    auto NewVReg = Edit.createFrom(Original);
    Allocator.setStage(NewVReg, RAGC::Reg_New);
    auto &TRI = *Allocator.getMF().getSubtarget().getRegisterInfo();
    SlotIndex DefIdx = Edit.rematerializeAt(*MI.getParent(), MI, NewVReg, RM, TRI);
    auto *NewMI = LIS.getInstructionFromIndex(DefIdx);
    NewMI->setDebugLoc(MI.getDebugLoc());
    for (const auto &OpPair : Ops) {
      MachineOperand &MO = OpPair.first->getOperand(OpPair.second);
      if (MO.isReg() && MO.isUse() && MO.getReg() == VirtReg.reg()) {
        MO.setReg(NewVReg);
        MO.setIsKill();
      }
    }
    return Rematable;
  }

  void tryRematerializeAll() {
    bool AnyRemat = false;
    for (Register &Reg : RegsToRemat) {
      LiveInterval &LI = LIS.getInterval(Reg);
      for (MachineInstr &MI : llvm::make_early_inc_range(MRI.reg_bundles(Reg))) {
        if (!MI.isDebugValue()) {
          assert(!MI.isDebugInstr() && "Did not expect to find a use in debug "
                "instruction that isn't a DBG_VALUE");
          AnyRemat |= tryRematerializeFor(LI, MI) == Rematable;
        }
      }
    }

    if (!AnyRemat)
      return;

    auto &TRI = *Allocator.getMF().getSubtarget().getRegisterInfo();
    for (Register Reg : RegsToRemat) {
      LiveInterval &LI = LIS.getInterval(Reg);
      for (VNInfo *VNI : LI.vnis()) {
        if (VNI->isUnused() || VNI->isPHIDef() || UsedValues.count(VNI))
          continue;
        auto *MI = LIS.getInstructionFromIndex(VNI->def);
        MI->addRegisterDead(Reg, &TRI);
        if (MI->allDefsAreDead())
          DeadDefs.push_back(MI);
      }
    }

    if (DeadDefs.empty())
      return;
    
    unsigned ResultPos = 0;
    Edit.eliminateDeadDefs(DeadDefs, RegsToRemat);
    for (Register Reg : RegsToRemat) {
      if (MRI.reg_nodbg_empty(Reg)) {
        Edit.eraseVirtReg(Reg);
        continue;
      }

      assert(LIS.hasInterval(Reg) &&
            (!LIS.getInterval(Reg).empty() || !MRI.reg_nodbg_empty(Reg)) &&
            "Empty and not used live-range?!");

      RegsToRemat[ResultPos++] = Reg;
    }
    RegsToRemat.erase(RegsToRemat.begin() + ResultPos, RegsToRemat.end());
  }

  bool coalesceStackAccess(MachineInstr *MI, Register Reg) {
    int FI = 0;
    auto &TII = *Allocator.getMF().getSubtarget().getInstrInfo();
    Register InstrReg = TII.isLoadFromStackSlot(*MI, FI);
    bool IsLoad = InstrReg;
    if (!IsLoad)
      InstrReg = TII.isStoreToStackSlot(*MI, FI);

    // not spill-like instruction.
    if (InstrReg != Reg || FI != StackSlot)
      return false;

    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
    return true;
  }

  void eliminateRedundantSpills(LiveInterval &SLI, VNInfo *VNI) {
    assert(VNI && "Missing value");
    SmallVector<std::pair<LiveInterval*, VNInfo*>, 8> WorkList;
    WorkList.push_back(std::make_pair(&SLI, VNI));
    assert(StackInt && "No stack slot assigned yet.");

    auto &TII = *Allocator.getMF().getSubtarget().getInstrInfo();
    do {
      LiveInterval *LI;
      std::tie(LI, VNI) = WorkList.pop_back_val();
      Register Reg = LI->reg();
      LLVM_DEBUG(dbgs() << "Checking redundant spills for " << VNI->id << '@'
                        << VNI->def << " in " << *LI << '\n');

      // Regs to spill are taken care of.
      if (isRegToRemat(Reg))
        continue;

      // Add all of VNI's live range to StackInt.
      StackInt->MergeValueInAsValue(*LI, VNI, StackInt->getValNumInfo(0));

      // Find all spills and copies of VNI.
      for (MachineInstr &MI :
          llvm::make_early_inc_range(MRI.use_nodbg_instructions(Reg))) {
        if (!MI.isCopy() && !MI.mayStore())
          continue;
        SlotIndex Idx = LIS.getInstructionIndex(MI);
        if (LI->getVNInfoAt(Idx) != VNI)
          continue;

        // Follow sibling copies down the dominator tree.
        if (Register DstReg = isFullCopyOf(MI, Reg)) {
          if (DstReg.isVirtual() && VRM.getOriginal(DstReg) == Original) {
            LiveInterval &DstLI = LIS.getInterval(DstReg);
            VNInfo *DstVNI = DstLI.getVNInfoAt(Idx.getRegSlot());
            assert(DstVNI && "Missing defined value");
            assert(DstVNI->def == Idx.getRegSlot() && "Wrong copy def slot");
            WorkList.push_back(std::make_pair(&DstLI, DstVNI));
          }
          continue;
        }

        // Erase spills.
        int FI;
        if (Reg == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot) {
          MI.setDesc(TII.get(TargetOpcode::KILL));
          DeadDefs.push_back(&MI);
        }
      }
    } while (!WorkList.empty());
  }

  void insertReload(Register NewVReg,
                    SlotIndex Idx,
                    MachineBasicBlock::iterator MI) {
    MachineBasicBlock &MBB = *MI->getParent();

    MachineInstrSpan MIS(MI, &MBB);
    auto &TII = *Allocator.getMF().getSubtarget().getInstrInfo();
    const auto *TRI = MRI.getTargetRegisterInfo();
    TII.loadRegFromStackSlot(MBB, MI, NewVReg, StackSlot,
                            MRI.getRegClass(NewVReg), TRI);

    LIS.InsertMachineInstrRangeInMaps(MIS.begin(), MI);
  }

  void insertSpill(Register NewVReg, bool isKill,
                   MachineBasicBlock::iterator MI) {
    assert(!MI->isTerminator() && "Inserting a spill after a terminator");
    MachineBasicBlock &MBB = *MI->getParent();

    MachineInstrSpan MIS(MI, &MBB);
    MachineBasicBlock::iterator SpillBefore = std::next(MI);
    bool IsRealSpill = isRealSpill(*MI);

    auto &TII = *Allocator.getMF().getSubtarget().getInstrInfo();
    const auto *TRI = MRI.getTargetRegisterInfo();
    if (IsRealSpill)
      TII.storeRegToStackSlot(MBB, SpillBefore, NewVReg, isKill, StackSlot,
                              MRI.getRegClass(NewVReg), TRI);
    else
      // Don't spill undef value.
      // Anything works for undef, in particular keeping the memory
      // uninitialized is a viable option and it saves code size and
      // run time.
      BuildMI(MBB, SpillBefore, MI->getDebugLoc(), TII.get(TargetOpcode::KILL))
          .addReg(NewVReg, getKillRegState(isKill));

    MachineBasicBlock::iterator Spill = std::next(MI);
    LIS.InsertMachineInstrRangeInMaps(Spill, MIS.end());
    for (const MachineInstr &MI : make_range(Spill, MIS.end()))
      getVDefInterval(MI, LIS);
  }

  void spillAroundUses(Register Reg) {
    LiveInterval &OldLI = LIS.getInterval(Reg);
    assert(OldLI.isSpillable());
    for (MachineInstr &MI :
          llvm::make_early_inc_range(MRI.reg_bundles(Reg))) {
      if (MI.isDebugValue()) {
        MachineBasicBlock *MBB = MI.getParent();
        buildDbgValueForSpill(*MBB, &MI, MI, StackSlot, Reg);
        continue;
      }

      assert(!MI.isDebugInstr() && "Did not expect to find a use in debug "
            "instruction that isn't a DBG_VALUE");

      if (SnippetCopies.count(&MI))
        continue;

      if (coalesceStackAccess(&MI, Reg))
        continue;

      SmallVector<std::pair<MachineInstr*, unsigned>, 8> Ops;
      VirtRegInfo RI = AnalyzeVirtRegInBundle(MI, Reg, &Ops);

      SlotIndex Idx = LIS.getInstructionIndex(MI).getRegSlot();
      if (VNInfo *VNI = OldLI.getVNInfoAt(Idx.getRegSlot(true)))
        if (SlotIndex::isSameInstr(Idx, VNI->def))
          Idx = VNI->def;

      // Check for a sibling copy.
      Register SibReg = isFullCopyOf(MI, Reg);
      if (SibReg && SibReg.isVirtual() && VRM.getOriginal(SibReg) == Original) {
        if (isRegToRemat(SibReg)) {
          SnippetCopies.insert(&MI);
          continue;
        }
        if (!RI.Writes) {
          LiveInterval &SibLI = LIS.getInterval(SibReg);
          eliminateRedundantSpills(SibLI, SibLI.getVNInfoAt(Idx));
        }
      }

      Register NewVReg = Edit.createFrom(Reg);

      if (RI.Reads)
        insertReload(NewVReg, Idx, &MI);

      bool HasLiveDef = false;
      for (const auto &OpPair : Ops) {
        MachineOperand &MO = OpPair.first->getOperand(OpPair.second);
        MO.setReg(NewVReg);
        if (MO.isUse()) {
          if (!OpPair.first->isRegTiedToDefOperand(OpPair.second))
            MO.setIsKill();
        } else {
          if (!MO.isDead())
            HasLiveDef = true;
        }
      }

      if (RI.Writes) {
        if (HasLiveDef)
          insertSpill(NewVReg, true, &MI);
      }
    }
  }

  void spillAll() {
    auto &LSS = Allocator.getAnalysis<LiveStacks>();
    if (StackSlot == VirtRegMap::NO_STACK_SLOT) {
      StackSlot = VRM.assignVirt2StackSlot(Original);
      StackInt = &LSS.getOrCreateInterval(StackSlot, MRI.getRegClass(Original));
      StackInt->getNextValue(SlotIndex(), LSS.getVNInfoAllocator());
    } else
      StackInt = &LSS.getInterval(StackSlot);

    if (Original != Edit.getReg())
      VRM.assignVirt2StackSlot(Edit.getReg(), StackSlot);

    assert(StackInt->getNumValNums() == 1 && "Bad stack interval values");
    for (Register Reg : RegsToRemat)
      StackInt->MergeSegmentsInAsValue(LIS.getInterval(Reg), StackInt->getValNumInfo(0));

    for (Register Reg : RegsToRemat)
      spillAroundUses(Reg);

    if (!DeadDefs.empty())
      Edit.eliminateDeadDefs(DeadDefs, RegsToRemat);

    for (Register Reg : RegsToRemat) {
      for (MachineInstr &MI :
          llvm::make_early_inc_range(MRI.reg_instructions(Reg))) {
        assert(SnippetCopies.count(&MI) && "Remaining use wasn't a snippet copy");
        LIS.RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
      }
    }

    for (Register Reg : RegsToRemat)
      Edit.eraseVirtReg(Reg);
  }

  void tryRemat() {
    collectRegsToRemat();
    tryRematerializeAll();
    if (!RegsToRemat.empty())
      spillAll();
  }

  int StackSlot;
  Register Original;
  LiveInterval *StackInt{nullptr};
  LiveRangeEdit &Edit;
  LiveIntervals &LIS;
  MachineRegisterInfo &MRI;
  RAGC &Allocator;
  VirtRegMap &VRM;
  RegisterSet &Visited;

  SmallVector<MachineInstr *, 8> DeadDefs;
  SmallVector<Register, 8> RegsToRemat;
  SmallPtrSet<MachineInstr*, 8> SnippetCopies;
  SmallPtrSet<VNInfo*, 8> UsedValues;
};

char RAGC::ID = 0;
char &RAGraphColoringID = RAGC::ID;
    
void RAGC::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addPreserved<MachineBlockFrequencyInfo>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<SpillPlacement>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RAGC::reset() {
  const auto &MRI = MF->getRegInfo();
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (getStage(Reg) == Reg_Done) {
      Matrix->unassign(LIS->getInterval(Reg));
      setStage(Reg, Reg_New);
    }
  }
  ItfGraph->reset();
}

void RAGC::constructItfGraph(bool FirstTime) {
  VRAI->calculateSpillWeightsAndHints();

  std::vector<LiveInterval *> LiveLI;
  const auto &MRI = MF->getRegInfo();
  const auto *TRI = MRI.getTargetRegisterInfo();
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    
    LiveInterval &VRegLI = LIS->getInterval(Reg);
    if (VRegLI.empty()) {
      LLVM_DEBUG(dbgs() << "LiveRange Empty! VREG: " << printReg(Reg, TRI));
      continue;
    }
    LiveLI.push_back(&VRegLI);
  }

  static auto PairHash =
      [] (const std::pair<int, int> &Pair) {
    return Pair.first ^ Pair.second;
  };

  auto &PQueue = ItfGraph->getPQueue();
  std::unordered_set<std::pair<int, int>, decltype(PairHash)> Visited(0, PairHash);
  for (int i = 0, e = LiveLI.size(); i < e; ++i) {
    auto *LI = LiveLI[i];
    auto Reg = LI->reg();
    const TargetRegisterClass *RC = MRI.getRegClass(Reg);

    auto &Node = ItfGraph->getOrCreateNode(LI);
    for (int j = 0, e = LiveLI.size(); j < e; ++j) {
      if (i != j) {
        auto *OtherLI = LiveLI[j];
        auto OtherReg = OtherLI->reg();
        auto *OtherRC = MRI.getRegClass(OtherReg);
        if (RC == OtherRC && !Visited.count({i, j})) {
          if (LI->overlaps(OtherLI)) {
            auto &OtherNode = ItfGraph->getOrCreateNode(OtherLI);
            Node.addAdjNode(&OtherNode);
            OtherNode.addAdjNode(&Node);
          }
          Visited.insert({i, j});
          Visited.insert({j, i});
        }
      }
    }
    PQueue.insert(&Node);
  }
}

void RAGC::simplify(StackTy &WorkStk) {
  auto &PQueue = ItfGraph->getPQueue();
  while (!PQueue.empty()) {
    auto *Node = PQueue.top();
    PQueue.pop();

    WorkStk.push(Node);
    for (auto *AdjNode : Node->getAdjNode()) {
      if (AdjNode->isAlive()) {
        auto OldWeight = ItfGraph->getWeight(*AdjNode);
        AdjNode->decDegree();
        PQueue.update(AdjNode, OldWeight);
      }
    }
    Node->dead();
  }
}

bool RAGC::select(StackTy &WorkStk, SmallVectorImpl<Register> &RegistersToSpill) {
  while (!WorkStk.empty()) {
    auto *Node = WorkStk.top();
    WorkStk.pop();

    RegisterSet AliveAdjRegs;
    for (auto *AdjNode : Node->getAdjNode()) {
      if (AdjNode->isAlive()) {
        auto *AliveLI = AdjNode->getLI();
        if (auto PhyReg = VRM->getPhys(AliveLI->reg()))
          AliveAdjRegs.insert(PhyReg.id());
      }
    }

    auto *LI = Node->getLI();
    assert(LI);

    MCRegister PhysReg;
    auto Order =
      AllocationOrder::create(LI->reg(), *VRM, RegClassInfo, Matrix);
    for (auto I = Order.begin(), E = Order.end(); I != E && !PhysReg; ++I) {
      assert(*I);
      auto Candidate = *I;
      if (AliveAdjRegs.count(Candidate) != 0) {
        if (!Matrix->checkRegUnitInterference(*LI, Candidate) &&
            !Matrix->checkRegMaskInterference(*LI, Candidate)) {
          if (I.isHint()) {
            PhysReg = Candidate;
            break;
          }
          PhysReg = Candidate;
        }
      }
    }

    if (!PhysReg.isValid()) {
      RegistersToSpill.push_back(LI->reg());
      continue;
    }

    Matrix->assign(*Node->getLI(), PhysReg);
    Node->reAlive();
  }
  return RegistersToSpill.empty();
}

void RAGC::doInstructionSplit(LiveRangeEdit &Edit) {
}

void RAGC::calcGapWeights(MCRegister PhysReg, SmallVectorImpl<float> &GapWeight) {
  assert(SA->getUseBlocks().size() == 1 && "Not a local interval");
  const SplitAnalysis::BlockInfo &BI = SA->getUseBlocks().front();
  ArrayRef<SlotIndex> Uses = SA->getUseSlots();
  const unsigned NumGaps = Uses.size()-1;

  // Start and end points for the interference check.
  SlotIndex StartIdx =
    BI.LiveIn ? BI.FirstInstr.getBaseIndex() : BI.FirstInstr;
  SlotIndex StopIdx =
    BI.LiveOut ? BI.LastInstr.getBoundaryIndex() : BI.LastInstr;

  GapWeight.assign(NumGaps, 0.0f);
  auto &TRI = VRM->getTargetRegInfo();
  for (MCRegUnitIterator Units(PhysReg, &TRI); Units.isValid(); ++Units) {
    if (!Matrix->query(const_cast<LiveInterval&>(SA->getParent()), *Units)
          .checkInterference())
      continue;

    LiveIntervalUnion::SegmentIter IntI =
      Matrix->getLiveUnions()[*Units] .find(StartIdx);
    for (unsigned Gap = 0; IntI.valid() && IntI.start() < StopIdx; ++IntI) {
      // Skip the gaps before IntI.
      while (Uses[Gap+1].getBoundaryIndex() < IntI.start())
        if (++Gap == NumGaps)
          break;
      if (Gap == NumGaps)
        break;

      // Update the gaps covered by IntI.
      const float Weight = IntI.value()->weight();
      for (; Gap != NumGaps; ++Gap) {
        GapWeight[Gap] = std::max(GapWeight[Gap], Weight);
        if (Uses[Gap+1].getBaseIndex() >= IntI.stop())
          break;
      }
      if (Gap == NumGaps)
        break;
    }
  }

  for (MCRegUnitIterator Units(PhysReg, &TRI); Units.isValid(); ++Units) {
    const LiveRange &LR = LIS->getRegUnit(*Units);
    LiveRange::const_iterator I = LR.find(StartIdx);
    LiveRange::const_iterator E = LR.end();

    // Same loop as above. Mark any overlapped gaps as HUGE_VALF.
    for (unsigned Gap = 0; I != E && I->start < StopIdx; ++I) {
      while (Uses[Gap+1].getBoundaryIndex() < I->start)
        if (++Gap == NumGaps)
          break;
      if (Gap == NumGaps)
        break;

      for (; Gap != NumGaps; ++Gap) {
        GapWeight[Gap] = huge_valf;
        if (Uses[Gap+1].getBaseIndex() >= I->end)
          break;
      }
      if (Gap == NumGaps)
        break;
    }
  }
}

void RAGC::doLocalSplit(LiveRangeEdit &Edit) {
  assert(SA->getUseBlocks().size() != 1);

  auto &LI = Edit.getParent();
  assert(!Matrix->checkRegMaskInterference(LI) && "Unsupported yet.");

  ArrayRef<SlotIndex> Uses = SA->getUseSlots();
  const unsigned NumGaps = Uses.size()-1;
  unsigned BestBefore = NumGaps;
  unsigned BestAfter = 0;
  float BestDiff = 0;

  const SplitAnalysis::BlockInfo &BI = SA->getUseBlocks().front();
  const float BlockFreq =
    SpillPlacer->getBlockFrequency(BI.MBB->getNumber()).getFrequency() *
    (1.0f / MBFI->getEntryFreq());
  SmallVector<float, 8> GapWeight;

  auto Order = AllocationOrder::create(LI.reg(), *VRM, RegClassInfo, Matrix);
  for (MCPhysReg PhysReg : Order) {
    assert(PhysReg);
    assert(!Matrix->checkRegMaskInterference(LI, PhysReg));

    unsigned SplitBefore = 0, SplitAfter = 1;
    calcGapWeights(PhysReg, GapWeight);
    float MaxGap = GapWeight[0];
    while (true) {
      const bool LiveBefore = SplitBefore != 0 || BI.LiveIn;
      const bool LiveAfter = SplitAfter != NumGaps || BI.LiveOut;

      if (!LiveBefore && !LiveAfter)
        break;

      bool Shrink = true;
      unsigned NewGaps = LiveBefore + SplitAfter - SplitBefore + LiveAfter;
      if (NewGaps < NumGaps && MaxGap < huge_valf) {
        const float EstWeight = normalizeSpillWeight(
            BlockFreq * (NewGaps + 1),
            Uses[SplitBefore].distance(Uses[SplitAfter]) +
                (LiveBefore + LiveAfter) * SlotIndex::InstrDist,
            1);
        if (EstWeight * Hysteresis >= MaxGap) {
          Shrink = false;
          float Diff = EstWeight - MaxGap;
          if (Diff > BestDiff) {
            LLVM_DEBUG(dbgs() << " (best)");
            BestDiff = Hysteresis * Diff;
            BestBefore = SplitBefore;
            BestAfter = SplitAfter;
          }
        }
      }

      if (Shrink) {
        if (++SplitBefore < SplitAfter) {
          LLVM_DEBUG(dbgs() << " shrink\n");
          if (GapWeight[SplitBefore - 1] >= MaxGap) {
            MaxGap = GapWeight[SplitBefore];
            for (unsigned I = SplitBefore + 1; I != SplitAfter; ++I)
              MaxGap = std::max(MaxGap, GapWeight[I]);
          }
          continue;
        }
        MaxGap = 0;
      }

      if (SplitAfter >= NumGaps)
        break;

      MaxGap = std::max(MaxGap, GapWeight[SplitAfter++]);
    }
  }
}

void RAGC::split(LiveRangeEdit &Edit) {
  auto Reg = Edit.getReg();
  auto &LI = LIS->getInterval(Reg);
  auto Stage = getStage(Reg);
  if (LIS->intervalIsInOneMBB(LI)) {
    assert(Stage == Reg_New || Stage == Reg_LocalSplit);
    if (Stage == Reg_New)
      doLocalSplit(Edit);
    else // Reg_LocalSplit
      doInstructionSplit(Edit);
  }
}

bool RAGC::isProfitableToSplit() const {
  auto &LI = SA->getParent();
  return !LI.isSpillable() || LI.getNumValNums() >= 2 || SA->getUseSlots().size() > 1;
}

void RAGC::trySplitOrSpill(SmallVectorImpl<Register> &RegistersToSpill) {
  RegisterSet Visited;
  SmallVector<Register> NewVRegs;
  for (auto Reg : RegistersToSpill) {
    assert(isRegAssignable(Reg));
    if (Visited.count(Reg))
      continue;

    auto &LI = LIS->getInterval(Reg);
    SmallPtrSet<MachineInstr *, 32> DeadRemats;
    LiveRangeEdit Edit(&LI, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    SE->reset(Edit, SplitEditor::SM_Speed);
    SA->analyze(&LI);
    if (!canFurtherSplit(Reg) || !isProfitableToSplit()) {
      FastRemator Remator(Edit, *LIS, *MRI, *this, *VRM, Visited);
      Remator.tryRemat();
    } else
      split(Edit);
  }
}

void RAGC::coalesce() {
}

void RAGC::allocatePhysRegs() {
  bool FirstTime = true;
  do {
    if (!FirstTime)
      reset();

    constructItfGraph(FirstTime);

    if (!FirstTime)
      coalesce();

    FirstTime = false;

    StackTy WorkStk;
    simplify(WorkStk);
    
    SmallVector<Register> RegistersToSpill;
    if (select(WorkStk, RegistersToSpill))
      break;
    
    trySplitOrSpill(RegistersToSpill);
  } while(true);
}

bool RAGC::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  DomTree = &getAnalysis<MachineDominatorTree>();
  Indexes = &getAnalysis<SlotIndexes>();
  LIS = &getAnalysis<LiveIntervals>();
  Loops = &getAnalysis<MachineLoopInfo>();
  Matrix = &getAnalysis<LiveRegMatrix>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  MRI = &getAnalysis<MachineRegisterInfo>();
  SpillPlacer = &getAnalysis<SpillPlacement>();
  VRAI = std::make_unique<VirtRegAuxInfo>(*MF, *LIS, *VRM, *Loops, *MBFI);
  VRM = &getAnalysis<VirtRegMap>();
  RegClassInfo.runOnMachineFunction(mf);

  LiveIntervalCalc LocalLICalc;
  LICalc = &LocalLICalc;

  regalloc_gc::DefaultInterferenceGraph ItfGraphObj;
  ItfGraph = &ItfGraphObj;

  SA.reset(new SplitAnalysis(*VRM, *LIS, *Loops));
  SE.reset(new SplitEditor(*SA, *LIS, *VRM, *DomTree, *MBFI, *VRAI));
  allocatePhysRegs();
  return true;
}

} // namespace llvm
