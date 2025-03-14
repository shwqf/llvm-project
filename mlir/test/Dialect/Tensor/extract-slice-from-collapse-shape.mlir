// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-rewrite-extract-slice-from-collapse-shape %s | FileCheck %s
// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns="test-rewrite-extract-slice-from-collapse-shape use-foreach" %s | FileCheck %s --check-prefix=FOREACH

func.func @extract_slice_static(%input: tensor<3x5x7x11xf32>) -> tensor<20x11xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x5x7x11xf32> into tensor<105x11xf32>
  %slice = tensor.extract_slice %collapsed [0, 0] [20, 11] [1, 1] : tensor<105x11xf32> to tensor<20x11xf32>
  return %slice : tensor<20x11xf32>
}

//     CHECK: func.func @extract_slice_static(%[[arg0:.+]]:
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c20:.+]] = arith.constant 20 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[c7:.+]] = arith.constant 7 : index
// CHECK-DAG: %[[init:.+]] = linalg.init_tensor [20, 11] :
// CHECK-DAG: %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[c20]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[iv]] into (%[[c3]], %[[c5]], %[[c7]]
//     CHECK:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 11] [1, 1, 1, 1] : 
//     CHECK:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} : 
//     CHECK:   %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 11] [1, 1] : 
//     CHECK:   scf.yield %[[update]] :
//     CHECK: return %[[tile]]

//     FOREACH: func.func @extract_slice_static(%[[arg0:.+]]:
// FOREACH-DAG: %[[c20:.+]] = arith.constant 20 : index
// FOREACH-DAG: %[[c3:.+]] = arith.constant 3 : index
// FOREACH-DAG: %[[c5:.+]] = arith.constant 5 : index
// FOREACH-DAG: %[[c7:.+]] = arith.constant 7 : index
// FOREACH-DAG: %[[init:.+]] = linalg.init_tensor [20, 11] :
//     FOREACH: %[[tile:.+]] = scf.foreach_thread (%[[iv:.+]]) in (%[[c20]]) shared_outs(%[[dest:.+]] = %[[init]])
//     FOREACH:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[iv]] into (%[[c3]], %[[c5]], %[[c7]]
//     FOREACH:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 11] [1, 1, 1, 1] : 
//     FOREACH:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} : 
//     FOREACH:   perform_concurrently
// FOREACH-NEXT:   tensor.parallel_insert_slice %[[sliceFlat]] into %[[dest]][%[[iv]], 0] [1, 11] [1, 1] :
//     FOREACH: return %[[tile]]

// -----


func.func @extract_slice_static_strided(%input: tensor<3x5x7x11xf32>) -> tensor<10x5xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x5x7x11xf32> into tensor<105x11xf32>
  %slice = tensor.extract_slice %collapsed [13, 0] [10, 5] [2, 2] : tensor<105x11xf32> to tensor<10x5xf32>
  return %slice : tensor<10x5xf32>
}

//     CHECK: #[[$map0:.+]] = affine_map<(d0) -> (d0 * 2 + 13)>
//     CHECK: func.func @extract_slice_static_strided(%[[arg0:.+]]:
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c10:.+]] = arith.constant 10 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[c7:.+]] = arith.constant 7 : index
//     CHECK: %[[init:.+]] = linalg.init_tensor [10, 5] :
//     CHECK: %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:   %[[inputIv:.+]] = affine.apply #[[$map0]](%[[iv]])
//     CHECK:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[inputIv]] into (%[[c3]], %[[c5]], %[[c7]]
//     CHECK:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 5] [1, 1, 1, 2] : 
//     CHECK:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} : 
//     CHECK:   %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 5] [1, 1] : 
//     CHECK:   scf.yield %[[update]] :
//     CHECK: return %[[tile]]


// -----


func.func @extract_slice_dynamic(%input: tensor<3x?x?x11xf32>, %offt: index, %size: index) -> tensor<?x5xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x?x?x11xf32> into tensor<?x11xf32>
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 5] [2, 2] : tensor<?x11xf32> to tensor<?x5xf32>
  return %slice : tensor<?x5xf32>
}

//     CHECK: #[[map0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//     CHECK: func.func @extract_slice_dynamic(%[[arg0:.+]]: tensor<{{.*}}>, %[[lb:.+]]: index, %[[sz:.+]]: index)
// CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[c3:.+]] = arith.constant 3 : index
//     CHECK:   %[[init:.+]] = linalg.init_tensor [%[[sz]], 5] : tensor<?x5xf32>
// CHECK-DAG:   %[[d1:.+]] = tensor.dim %arg0, %[[c1]] : tensor<3x?x?x11xf32>
// CHECK-DAG:   %[[d2:.+]] = tensor.dim %arg0, %[[c2]] : tensor<3x?x?x11xf32>
//     CHECK:   %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[sz]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:     %[[inputIv:.+]] = affine.apply #[[map0]](%[[iv]])[%[[lb]]]
//     CHECK:     %[[multiIndex:.+]]:3 = affine.delinearize_index %[[inputIv]] into (%[[c3]], %[[d1]], %[[d2]]) :
//     CHECK:     %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 5] [1, 1, 1, 2] :
//     CHECK:     %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} :
//     CHECK:     %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 5] [1, 1] :
//     CHECK:     scf.yield %[[update]] :
//     CHECK:   return %[[tile]] :

// -----


func.func @extract_slice_dynamic_multidim(%input: tensor<3x?x?x11x?xf32>, %offt0: index, %size0: index, %offt1: index, %size1: index) -> tensor<?x?xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3, 4]] : tensor<3x?x?x11x?xf32> into tensor<?x?xf32>
  %slice = tensor.extract_slice %collapsed [%offt0, %offt1] [%size0, %size1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %slice : tensor<?x?xf32>
}

//     CHECK: #[[map0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//     CHECK: func.func @extract_slice_dynamic_multidim(%[[arg0:.+]]: tensor<3x?x?x11x?xf32>, %[[lb1:.+]]: index, %[[sz1:.+]]: index, %[[lb2:.+]]: index, %[[sz2:.+]]: index)
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[c11:.+]] = arith.constant 11 : index
//     CHECK: %[[init:.+]] = linalg.init_tensor [%[[sz1]], %[[sz2]]] : tensor<?x?xf32>
// CHECK-DAG: %[[d1:.+]] = tensor.dim %[[arg0]], %[[c1]] : 
// CHECK-DAG: %[[d2:.+]] = tensor.dim %[[arg0]], %[[c2]] : 
// CHECK-DAG: %[[d4:.+]] = tensor.dim %[[arg0]], %[[c4]] :
//     CHECK: %[[tile1:.+]] = scf.for %[[iv1:.+]] = %[[c0]] to %[[sz1]] step %[[c1]] iter_args(%[[iterArg1:.+]] = %[[init]])
//     CHECK:   %[[tile2:.+]] = scf.for %[[iv2:.+]] = %[[c0]] to %[[sz2]] step %[[c1]] iter_args(%[[iterArg2:.+]] = %[[iterArg1]])
//     CHECK:       %[[inputIv1:.+]] = affine.apply #[[map0:.+]](%[[iv1]])[%[[lb1]]]
//     CHECK:       %[[multiIndex1:.+]]:3 = affine.delinearize_index %[[inputIv1]] into (%[[c3]], %[[d1]], %[[d2]]) :
//     CHECK:       %[[inputIv2:.+]] = affine.apply #[[map0:.+]](%[[iv2]])[%[[lb2]]]
//     CHECK:       %[[multiIndex2:.+]]:2 = affine.delinearize_index %[[inputIv2]] into (%[[c11]], %[[d4]]) :
//     CHECK:       %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex1]]#0, %[[multiIndex1]]#1, %[[multiIndex1]]#2, %[[multiIndex2]]#0, %[[multiIndex2]]#1] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : 
//     CHECK:       %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3, 4]{{\]}} : 
//     CHECK:       %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg2]][%[[iv1]], %[[iv2]]] [1, 1] [1, 1] : 
//     CHECK:       scf.yield %[[update]] :
//     CHECK:     scf.yield %[[tile2]] :
//     CHECK:   return %[[tile1]] : 

//     FOREACH: #[[map1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//     FOREACH: func.func @extract_slice_dynamic_multidim(%[[arg0:.+]]: tensor<3x?x?x11x?xf32>, %[[lb1:.+]]: index, %[[sz1:.+]]: index, %[[lb2:.+]]: index, %[[sz2:.+]]: index)
// FOREACH-DAG: %[[c1:.+]] = arith.constant 1 : index
// FOREACH-DAG: %[[c2:.+]] = arith.constant 2 : index
// FOREACH-DAG: %[[c3:.+]] = arith.constant 3 : index
// FOREACH-DAG: %[[c4:.+]] = arith.constant 4 : index
// FOREACH-DAG: %[[c11:.+]] = arith.constant 11 : index
//     FOREACH:     %[[init:.+]] = linalg.init_tensor [%[[sz1]], %[[sz2]]] : tensor<?x?xf32>
// FOREACH-DAG:     %[[d1:.+]] = tensor.dim %[[arg0]], %[[c1]] : 
// FOREACH-DAG:     %[[d2:.+]] = tensor.dim %[[arg0]], %[[c2]] : 
// FOREACH-DAG:     %[[d4:.+]] = tensor.dim %[[arg0]], %[[c4]] :
//     FOREACH:     %[[tile1:.+]] = scf.foreach_thread (%[[tid1:.+]], %[[tid2:.+]]) in (%[[sz1]], %[[sz2]]) shared_outs(%[[dest:.+]] = %[[init]])
// FOREACH-DAG:       %[[iv1:.+]] = affine.apply #[[map1]](%[[tid1]])[%[[lb1]]]
//     FOREACH:       %[[multiIndex1:.+]]:3 = affine.delinearize_index %[[iv1]] into (%[[c3]], %[[d1]], %[[d2]]) :
// FOREACH-DAG:       %[[iv2:.+]] = affine.apply #[[map1]](%[[tid2]])[%[[lb2]]]
//     FOREACH:       %[[multiIndex2:.+]]:2 = affine.delinearize_index %[[iv2]] into (%[[c11]], %[[d4]]) :
//     FOREACH:       %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex1]]#0, %[[multiIndex1]]#1, %[[multiIndex1]]#2, %[[multiIndex2]]#0, %[[multiIndex2]]#1] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : 
//     FOREACH:       %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3, 4]{{\]}} : 
//     FOREACH:       perform_concurrently
//FOREACH-NEXT:         tensor.parallel_insert_slice %[[sliceFlat]] into %[[dest]][%[[tid1]], %[[tid2]]] [1, 1] [1, 1] :

// -----

// Verifies that a linearized dimension that is not sliced does not generate a loop. Note that this
// only works for static shapes.

// CHECK: @extract_slice_non_sliced_linearized_dim(%[[arg0:.+]]: tensor<{{.*}}>,
func.func @extract_slice_non_sliced_linearized_dim(%input: tensor<3x?x?x11x2xf32>, %offt: index, %size: index) -> tensor<?x22xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3, 4]] : tensor<3x?x?x11x2xf32> into tensor<?x22xf32>  
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 22] [1, 1] : tensor<?x22xf32> to tensor<?x22xf32>
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // CHECK: %[[multiIndex:.+]]:3 = affine.delinearize_index
  // CHECK: tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0, 0] [1, 1, 1, 11, 2] [1, 1, 1, 1, 1]
  return %slice : tensor<?x22xf32>
}
