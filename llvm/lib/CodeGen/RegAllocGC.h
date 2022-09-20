#ifndef LLVM_CODEGEN_REGALLOCGC_H_
#define LLVM_CODEGEN_REGALLOCGC_H_

#include "SplitKit.h"
#include "SpillPlacement.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <stack>
#include <unordered_map>

namespace llvm {

namespace regalloc_gc {

template <typename T, class WeightTraits>
struct SmallHeap {
  using Less = typename WeightTraits::Less;
  using WeightTy = typename WeightTraits::WeightTy;

  void adjustUp(T *Node) {
    assert(Node && Node->PQIdx != -1);
    if (Container.size() == 1)
      return;

    int Idx = Node->PQIdx;
    for (; Idx && Less()(Node, Container[Idx >> 1]); Idx >>= 1) {
      Container[Idx >> 1]->PQIdx = Idx;
      Container[Idx] = Container[Idx >> 1];
    }
    Container[Idx] = Node;
    Node->PQIdx = Idx;
  }

  void adjustDown(int Idx) {
    assert(Idx != -1);
    if (Container.size() == 1)
      return;
      
    auto *Node = Container[Idx];
    assert(Node && Node->PQIdx);

    int Size = Container.size();
    while ((Idx << 1) < Size) {
      auto *Succ = Container[Idx << 1];
      if ((Idx << 1) + 1 < Size) {
        auto *Right = Container[(Idx << 1) + 1];
        if (Less()(Right, Succ))
          Succ = Right;
      }
        
      if (Less()(Node, Succ))
        break;
      swap(Idx, Succ->PQIdx);
      Idx <<= 1;
    }
  }

  void insert(T *Node) {
    Node->PQIdx = Container.size();
    Container.push_back(Node);
    adjustUp(Node);
  }

  void update(T *Node, WeightTy OldWeight) {
    assert(Node);
    auto Weight = WeightTraits::getWeight(Node);
    if (Less()(Weight, OldWeight))
      adjustUp(Node);
    else if (Less()(OldWeight, Weight))
      adjustDown(Node->PQIdx);
  }

  T *top() const {
    return Container.empty() ? nullptr : Container[0];
  }

  void pop() {
    if (Container.empty())
      return;

    auto *Node = Container[0];
    if (Container.size() != 1)
      swap(0, Container.size() - 1);
      
    Node->PQIdx = -1;
    Container.pop_back();
    adjustDown(0);
  }

  void swap(int Lhs, int Rhs) {
    auto &LhsNode = Container[Lhs], RhsNode = Container[Rhs];
    std::swap(LhsNode->PQIdx, RhsNode->PQIdx);
    std::swap(LhsNode, RhsNode);
  }

  bool empty() const { return Container.empty(); }

  SmallVector<T *> Container;
};

struct InterferenceGraphNode {
  using AdjNodeVecTy = SmallVector<InterferenceGraphNode *>;

  InterferenceGraphNode(LiveInterval *LI) : LI(LI) {}

  void addAdjNode(InterferenceGraphNode *AdjNode) {
    AdjNodes.push_back(AdjNode);
    ++Degree;
  }

  inline size_t getCurrDegree() const { return Degree; }
  inline size_t degree() const { return AdjNodes.size(); }
  inline void decDegree() { --Degree; }

  AdjNodeVecTy &getAdjNode() { return AdjNodes; }

  void dead() { Alive = false; }
  bool isAlive() const { return Alive; }

  LiveInterval *getLI() const { return LI; }

  void reAlive() {
    Alive = true;
    for (auto *Node : AdjNodes)
      ++Node->Degree;
  }

  bool Alive{true};
  int PQIdx{-1}, Degree{0};
  LiveInterval *LI{nullptr};
  AdjNodeVecTy AdjNodes;
};

template <typename WeightTraits>
struct InterferenceGraph {
  using Node = InterferenceGraphNode;
  using WeightTy = typename WeightTraits::WeightTy;
  using PQueueTy = SmallHeap<Node, WeightTraits>;

  Node &getOrCreateNode(LiveInterval *LI) {
    assert(LI && LI->reg().isValid());
    return LI2Nodes.insert({LI, Node(LI)}).first->second;
  }

  WeightTy getWeight(const InterferenceGraphNode &Node) const {
    return WeightTraits::getWeight(Node);
  }

  bool empty() const { return LI2Nodes.empty(); }

  size_t size() const { return LI2Nodes.size(); }

  PQueueTy &getPQueue() { return PQueue; }

  void reset() {
    assert(PQueue.empty());
    LI2Nodes.clear();
  }

  PQueueTy PQueue;
  std::unordered_map<const LiveInterval *, Node> LI2Nodes;
};

struct DefaultWeightTraits {
  // <degree, Split Cost, Size, ~Reg>
  using WeightTy = std::tuple<int32_t, float, int8_t>;

  static WeightTy getWeight(const InterferenceGraphNode &Node) {
    const auto *LI = Node.getLI();
    // try unspillable LI first.
    int8_t SpillWeight = LI->isSpillable() ? 1 : 0;
    return {Node.getCurrDegree(), -LI->weight(), SpillWeight};
  }

  struct Less {
    bool operator()(const InterferenceGraphNode *Lhs,
                    const InterferenceGraphNode *Rhs) {
      return getWeight(*Lhs) < getWeight(*Rhs);
    }
  };
};

struct DefaultInterferenceGraph : InterferenceGraph<DefaultWeightTraits> {};

} // namespace regalloc_gc


void initializeRAGCPass(PassRegistry&);

class LLVM_LIBRARY_VISIBILITY RAGC : public MachineFunctionPass,
                                     private LiveRangeEdit::Delegate {
public:
  using StackTy = std::stack<regalloc_gc::InterferenceGraphNode *>;

  // Reg_New => Reg_LocalSplit / Reg_RegionSplit / Reg_Done
  // Reg_LocalSplit => Reg_InstrSplit => Reg_Done
  // Reg_RegionSplit => Reg_BlockSplit => Reg_Done
  // Reg_Done == Assigned / Spill
  enum LiveRangeStage : uint8_t {
    Reg_New,
    Reg_LocalSplit,
    Reg_InstrSplit,
    Reg_RegionSplit,
    Reg_BlockSplit,
    Reg_Done
  };

  static char ID;
  
  RAGC() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage&) const override;

  bool runOnMachineFunction(MachineFunction&) final;

  MachineFunction &getMF() const { return *MF; }
  VirtRegAuxInfo &getVRAI() const { return *VRAI; }
  LiveIntervalCalc &getLICalc() const { return *LICalc; }

  LiveRangeStage getStage(Register Reg) const {
    return StageCache[Reg];
  }

  void setStage(Register Reg, LiveRangeStage Stage) {
    StageCache.grow(Reg.id());
    StageCache[Reg] = Stage;
  }

  void LRE_DidCloneVirtReg(Register New, Register Old) override {
    StageCache.grow(New.id());
    StageCache[New] = StageCache[Old] = Reg_New;
  }

  bool isRegAssignable(Register Reg) const {
    return getStage(Reg) != Reg_Done;
  }

  bool canFurtherSplit(Register Reg) const {
    auto Stage = getStage(Reg);
    return Stage != Reg_InstrSplit && Stage != Reg_BlockSplit;
  }

private:
  bool isProfitableToSplit() const;
  void calcGapWeights(MCRegister, SmallVectorImpl<float>&);
  void coalesce();
  void allocatePhysRegs();
  void constructItfGraph(bool);
  void reset();
  void simplify(StackTy&);
  void doInstructionSplit(LiveRangeEdit&);
  void doLocalSplit(LiveRangeEdit&);
  void split(LiveRangeEdit&);
  void spill(Register);
  bool select(StackTy&, SmallVectorImpl<Register>&);
  void trySplitOrSpill(SmallVectorImpl<Register>&);

  AliasAnalysis *AA{nullptr};
  LiveIntervals *LIS{nullptr};
  MachineLoopInfo *Loops{nullptr};
  LiveRegMatrix *Matrix{nullptr};
  MachineBlockFrequencyInfo *MBFI{nullptr};
  MachineDominatorTree *DomTree{nullptr};
  MachineFunction *MF{nullptr};
  MachineRegisterInfo *MRI{nullptr};
  SlotIndexes *Indexes{nullptr};
  VirtRegMap *VRM{nullptr};

  RegisterClassInfo RegClassInfo;
  regalloc_gc::DefaultInterferenceGraph *ItfGraph{nullptr};
  LiveIntervalCalc *LICalc{nullptr};
  SpillPlacement *SpillPlacer{nullptr};
  std::unique_ptr<SplitAnalysis> SA;
  std::unique_ptr<SplitEditor> SE;
  std::unique_ptr<VirtRegAuxInfo> VRAI;
  IndexedMap<LiveRangeStage, VirtReg2IndexFunctor> StageCache;
};

} // namespace llvm

#endif
