// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// Tests that MapReductionPartialsPass correctly lowers an all-outs linalg.generic
// with a FIFO-driven outer reduction loop (d1) and an output-only inner loop (d3).
//
// Input IR shape (as produced by ReductionLoopExposurePass):
//   - d1 outer loop: scf.for carrying acc0 (tensor<1x64xf16>) and acc1
//     (tensor<1x2x64xf16>) as iter_args, annotated {loop_type = reduction_loop}.
//   - d3 inner loop: scf.for carrying a slice of acc1, annotated
//     {ktdf.output_only_loop, loop_type = reduction_loop}.
//   - linalg.generic inside d3: map_data (d0,d1,d2), map_acc0 (d0,d2),
//     map_acc1 (d0,d3,d2); ops are arith.cmpf ogt + arith.select (max) and
//     arith.addf (sum).
//
// Expected transformations by MapReductionPartials:
//   1. The iter_arg tensors (acc0, acc1) are replaced by LRF allocations
//      initialised with splat-constant seeds via linalg.fill + memref.copy.
//   2. The splat-constant seed for acc0 (dense<0.0>) uses a scalar 0.0.
//      The splat-constant seed for acc1 (dense<0xFC00>) uses a scalar 0xFC00.
//   3. The spanning acc1 is initialised slot-by-slot (one linalg.fill + copy
//      per d3 slot, with memref.subview at offset [0, k, 0]).
//   4. The scf.for d1 loop loses its iter_args; d3 loop also loses its
//      iter_args (in-place buffer updates replace yield chains).
//   5. ktdf.read_from_fifo now produces a memref (not a tensor).
//   6. linalg.generic operates on memref subviews of the LRF allocations.

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0

// Compute sub-stage: LRF allocations for acc0 and the spanning acc1.
// CHECK:         ktdf.stage depends_in({{.*}}) depends_out({{.*}}) {

// LRF buffers: one for acc0 (1x64), two for acc1 slots (1x2x64 total).
// CHECK:           memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK:           memref.alloc() : memref<1x2x64xf16, "SFU_REG">

// acc0 seeded with 0.0 via linalg.fill.
// CHECK:           arith.constant {{.*}} : f16
// CHECK:           linalg.fill
// CHECK:           memref.copy

// acc1 seeded slot-by-slot with -inf (0xFC00) via linalg.fill + subview.
// CHECK:           arith.constant {{.*}} : f16
// CHECK:           linalg.fill
// CHECK:           memref.subview
// CHECK:           memref.copy
// CHECK:           memref.subview
// CHECK:           memref.copy

// The d1 reduction loop is now a plain scf.for (no iter_args).
// CHECK:           scf.for %[[D1:[^ ]*]] = {{.*}} to {{.*}} step {{.*}} {

// FIFO read produces a memref (not a tensor).
// CHECK:             = ktdf.read_from_fifo {{.*}} -> memref<

// The output-only d3 loop is also plain (no iter_args).
// CHECK:             scf.for %[[D3:[^ ]*]] = {{.*}} to {{.*}} step {{.*}} {

// Subviews of LRF buffers are passed as outs to the generic.
// CHECK:               memref.subview
// CHECK:               linalg.generic
// CHECK-SAME:              iterator_types = ["parallel", "reduction", "parallel", "reduction"]

// d3 loop closes with the output_only_loop attribute.
// CHECK:             } {ktdf.output_only_loop, loop_type = #ktdf.loop_type<reduction_loop>}

// On the last d1 iteration write the acc1 LRF buffer to the output FIFO.
// CHECK:             arith.cmpi eq, %[[D1]],
// CHECK:             scf.if
// CHECK:               ktdf.write_to_fifo
// CHECK:             }

// d1 loop closes with the reduction_loop attribute.
// CHECK:           } {loop_type = #ktdf.loop_type<reduction_loop>}

// ---------------------------------------------------------------------------
// Input IR (output of ReductionLoopExposurePass on the all-outs generic)
// ---------------------------------------------------------------------------
#map_data = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map_acc0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map_acc1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>

#set_in  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_out = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  module {
    func.func @output_only_reduction(%arg0: index, %arg1: index)
        attributes {grid = [1]} {
      call @local_schedule_0(%arg0, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index, %arg1: index)
        attributes {grid = [1]} {
      %c0  = arith.constant 0 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c1  = arith.constant 1 : index
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %c2  = arith.constant 2 : index
      %cst     = arith.constant dense<0.0>    : tensor<1x64xf16>
      %cst_neg = arith.constant dense<0xFC00> : tensor<1x2x64xf16>

      %in_view = ktdp.construct_memory_view %arg0, sizes: [2, 4, 64], strides: [256, 64, 1] {
        coordinate_set = #set_in,
        memory_space   = #ktdp.memory_space<global>
      } : memref<2x4x64xf16>

      %out_view = ktdp.construct_memory_view %arg1, sizes: [2, 2, 64], strides: [128, 64, 1] {
        coordinate_set = #set_out,
        memory_space   = #ktdp.memory_space<global>
      } : memref<2x2x64xf16>

      %in_cast  = memref.memory_space_cast %in_view  : memref<2x4x64xf16>  to memref<2x4x64xf16,  "DDR">
      %out_cast = memref.memory_space_cast %out_view : memref<2x2x64xf16>  to memref<2x2x64xf16,  "DDR">

      ktdf.pipeline {
        %priv:4 = ktdf.private ->
            (memref<2x1x4x64xf16, "L1">, memref<2x1x2x64xf16, "L1">,
             !ktdf.token, !ktdf.token) {
          %a  = memref.alloc() : memref<2x1x4x64xf16, "L1">
          %b  = memref.alloc() : memref<2x1x2x64xf16, "L1">
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %b, %t0, %t1
              : memref<2x1x4x64xf16, "L1">, memref<2x1x2x64xf16, "L1">,
                !ktdf.token, !ktdf.token
        }

        ktdf.stage depends_in(none) depends_out(%priv#2) {
          scf.for %i = %c0_2 to %c2 step %c1_3 {
            %s = arith.subi %i, %c0_2 : index
            %d = arith.divsi %s, %c1_3 : index
            ktdf.data_transfer from %in_cast[%i, 0, 0] size [1, 4, 64]
                                to %priv#0[%d, 0, 0, 0] size [1, 1, 4, 64]
                : memref<2x4x64xf16, "DDR">, memref<2x1x4x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU"]}

        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          scf.for %i = %c0_2 to %c2 step %c1_3 {
            %c0_6 = arith.constant 0 : index
            %c1_7 = arith.constant 1 : index
            %c3   = arith.constant 3 : index
            ktdf.pipeline {
              %slots:4 = ktdf.private ->
                  (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                   !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                   !ktdf.token, !ktdf.token) {
                %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %f0, %f1, %t2, %t3
                    : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                      !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                      !ktdf.token, !ktdf.token
              }

              // Load sub-stage: stream one row at a time into the FIFO.
              ktdf.stage depends_in(none) depends_out(%slots#2) {
                %s  = arith.subi %i, %c0_2 : index
                %d  = arith.divsi %s, %c1_3 : index
                %c4 = arith.constant 4 : index
                scf.for %r = %c0_6 to %c4 step %c1_7 {
                  ktdf.data_transfer from %priv#0[%d, 0, %r, 0] size [1, 1, 1, 64]
                                     to %slots#0 size [64]
                      : memref<2x1x4x64xf16, "L1">,
                        !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1LU"]}

              // Compute sub-stage: post-ReductionLoopExposure form.
              //   d1 outer loop (size 4): iter_args carry acc0 and acc1.
              //   d3 inner loop (size 2): output-only, iter_args carry acc1 slice.
              //   linalg.generic: arith.cmpf ogt + arith.select (running max into
              //   acc1) and arith.addf (running sum into acc0).
              ktdf.stage depends_in(%slots#2) depends_out(%slots#3) {
                %c4 = arith.constant 4 : index
                %4:2 = scf.for %d1 = %c0_6 to %c4 step %c1_7
                            iter_args(%acc0 = %cst, %acc1 = %cst_neg)
                            -> (tensor<1x64xf16>, tensor<1x2x64xf16>) {
                  %row = ktdf.read_from_fifo %slots#0
                      : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                  %c0_8 = arith.constant 0 : index
                  %c2_9 = arith.constant 2 : index
                  %c1_10 = arith.constant 1 : index
                  %5:3 = scf.for %d3 = %c0_8 to %c2_9 step %c1_10
                              iter_args(%arg_row = %row, %arg_a0 = %acc0, %arg_a1 = %acc1)
                              -> (tensor<1x1x64xf16>, tensor<1x64xf16>, tensor<1x2x64xf16>) {
                    %s0 = tensor.extract_slice %arg_a0[0, 0] [1, 64] [1, 1]
                        : tensor<1x64xf16> to tensor<1x64xf16>
                    %s1 = tensor.extract_slice %arg_a1[0, %d3, 0] [1, 1, 64] [1, 1, 1]
                        : tensor<1x2x64xf16> to tensor<1x1x64xf16>
                    %g:3 = linalg.generic {
                      indexing_maps = [#map_data, #map_acc0, #map_acc1],
                      iterator_types = ["parallel", "reduction", "parallel", "reduction"]
                    } outs(%row, %s0, %s1
                             : tensor<1x1x64xf16>, tensor<1x64xf16>, tensor<1x1x64xf16>) {
                    ^bb0(%v_data: f16, %v_a0: f16, %v_a1: f16):
                      %gt     = arith.cmpf ogt, %v_data, %v_a1 : f16
                      %new_a1 = arith.select %gt, %v_data, %v_a1 : f16
                      %new_a0 = arith.addf %v_a0, %v_data : f16
                      linalg.yield %v_data, %new_a0, %new_a1 : f16, f16, f16
                    } -> (tensor<1x1x64xf16>, tensor<1x64xf16>, tensor<1x1x64xf16>)
                    %ins0 = tensor.insert_slice %g#1 into %arg_a0[0, 0] [1, 64] [1, 1]
                        : tensor<1x64xf16> into tensor<1x64xf16>
                    %ins1 = tensor.insert_slice %g#2 into %arg_a1[0, %d3, 0] [1, 1, 64] [1, 1, 1]
                        : tensor<1x1x64xf16> into tensor<1x2x64xf16>
                    scf.yield %arg_row, %ins0, %ins1
                        : tensor<1x1x64xf16>, tensor<1x64xf16>, tensor<1x2x64xf16>
                  } {ktdf.output_only_loop, loop_type = #ktdf.loop_type<reduction_loop>}
                  %is_last = arith.cmpi eq, %d1, %c3 : index
                  scf.if %is_last {
                    ktdf.write_to_fifo %5#2, %slots#1
                        : tensor<1x2x64xf16>, <"SFU" -> "L1SU", 128xf16>
                  }
                  scf.yield %5#1, %5#2 : tensor<1x64xf16>, tensor<1x2x64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["SFU"]}

              ktdf.stage depends_in(%slots#3) depends_out(none) {
                %s = arith.subi %i, %c0_2 : index
                %d = arith.divsi %s, %c1_3 : index
                ktdf.data_transfer from %slots#1 size [128]
                                   to %priv#1[%d, 0, 0, 0] size [1, 1, 2, 64]
                    : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                      memref<2x1x2x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}

        ktdf.stage depends_in(%priv#3) depends_out(none) {
          scf.for %i = %c0_2 to %c2 step %c1_3 {
            %s = arith.subi %i, %c0_2 : index
            %d = arith.divsi %s, %c1_3 : index
            ktdf.data_transfer from %priv#1[%d, 0, 0, 0] size [1, 1, 2, 64]
                                to %out_cast[%i, 0, 0] size [1, 2, 64]
                : memref<2x1x2x64xf16, "L1">, memref<2x2x64xf16, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
