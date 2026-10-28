// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s

// Tests that ReductionLoopExposurePass correctly classifies and exposes an
// output-only reduction dimension (a dim present only in outs(), not in the
// FIFO data operand's indexing map).
//
// The generic has all-outs() operands (no ins()), modelling a kernel where
// the streaming data and the accumulators all live in outs():
//
//   outs[0] (data):  (d0,d1,d2,d3) -> (d0,d1,d2)  -- FIFO-fed, no d3
//   outs[1] (acc0):  (d0,d1,d2,d3) -> (d0,d2)      -- non-spanning acc
//   outs[2] (acc1):  (d0,d1,d2,d3) -> (d0,d3,d2)   -- spanning over d3
//
// Iterator types: [parallel(d0), reduction(d1), parallel(d2), reduction(d3)]
// Sizes:           d0=1,          d1=4,           d2=64,         d3=2
//
// The FIFO slot carries the entire data tensor (1x4x64 = 256 elements) in
// one shot, so the pass infers d1 loop size = 4 from outs[0]'s shape.
//
// Expected transformations by ReductionLoopExposure:
//   1. d1 (FIFO-driven reduction): becomes the outer scf.for loop with
//      {loop_type = reduction_loop}, carrying acc0 and acc1 as iter_args.
//   2. d3 (output-only reduction): becomes an inner scf.for loop with
//      {ktdf.output_only_loop, loop_type = reduction_loop} wrapping the
//      generic, inside the FIFO-driven loop body.
//   3. tensor.extract_slice / tensor.insert_slice pairs are emitted for
//      acc1 (the spanning outs operand) keyed by the d3 loop IV.
//   4. ktdf.read_from_fifo is emitted once per d1 iteration (inside the loop).
//   5. ktdf.write_to_fifo is guarded on the last d1 iteration.

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0

// The compute sub-stage contains the FIFO-driven d1 loop.
// CHECK:         ktdf.stage depends_in({{.*}}) depends_out({{.*}}) {
// CHECK:           scf.for {{.*}} {
// CHECK:             ktdf.pipeline {
// CHECK:               ktdf.stage depends_in({{.*}}) depends_out({{.*}}) {

// The FIFO-driven loop (d1): carries acc0 and acc1 as iter_args.
// CHECK:                 scf.for %[[D1:[^ ]*]] = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ACC0_ARG:[^ ]*]] = {{.*}}, %[[ACC1_ARG:[^ ]*]] = {{.*}})

// Inside d1: FIFO read produces a per-iteration row tensor.
// CHECK:                   = ktdf.read_from_fifo

// The output-only d3 loop is nested inside d1, carrying a slice of acc1.
// CHECK:                   scf.for %[[D3:[^ ]*]] = {{.*}} to {{.*}} step {{.*}}

// extract_slice for the spanning acc1 operand at offset [0, d3, 0].
// CHECK:                     tensor.extract_slice %{{.*}}[0, %[[D3]], 0]

// The linalg.generic is inside the output-only loop.
// CHECK:                     linalg.generic
// CHECK-SAME:                    iterator_types = ["parallel", "reduction", "parallel", "reduction"]

// insert_slice writes the updated acc1 slot back at [0, d3, 0].
// CHECK:                     tensor.insert_slice %{{.*}} into %{{.*}}[0, %[[D3]], 0]

// d3 loop closes with output_only_loop annotation.
// CHECK:                   } {ktdf.output_only_loop, loop_type = #ktdf.loop_type<reduction_loop>}

// On the last d1 iteration, write acc1 to the output FIFO.
// CHECK:                   arith.cmpi eq, %[[D1]],
// CHECK:                   scf.if
// CHECK:                     ktdf.write_to_fifo
// CHECK:                   }

// d1 loop closes with reduction_loop annotation.
// CHECK:                 } {loop_type = #ktdf.loop_type<reduction_loop>}

// ---------------------------------------------------------------------------
// Input IR (as produced by SplitReductionInnerOuterDimPass)
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
      %c1  = arith.constant 1 : index
      %c2  = arith.constant 2 : index

      // Initial accumulator tensors.
      // acc0 [1x64xf16]   – no d3 axis, simple running sum.
      // acc1 [1x2x64xf16] – spans d3 (size 2), running max per slot.
      %cst_zero    = arith.constant dense<0.0>    : tensor<1x64xf16>
      %cst_neg_inf = arith.constant dense<0xFC00> : tensor<1x2x64xf16>

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

        // Load stage: move one input tile (1x4x64) into local memory.
        ktdf.stage depends_in(none) depends_out(%priv#2) {
          scf.for %i = %c0 to %c2 step %c1 {
            %s = arith.subi %i, %c0 : index
            %d = arith.divsi %s, %c1 : index
            ktdf.data_transfer from %in_cast[%i, 0, 0] size [1, 4, 64]
                                to %priv#0[%d, 0, 0, 0] size [1, 1, 4, 64]
                : memref<2x4x64xf16, "DDR">, memref<2x1x4x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU"]}

        // Compute stage: FIFO-driven (d1) + output-only (d3) reduction.
        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          scf.for %i = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              // FIFO in:  256xf16 = 1×4×64 (full d1 slice per outer iteration).
              // FIFO out: 128xf16 = 1×2×64 (the acc1 result spanning d3).
              %slots:4 = ktdf.private ->
                  (!ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>,
                   !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                   !ktdf.token, !ktdf.token) {
                %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
                %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %f0, %f1, %t2, %t3
                    : !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>,
                      !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                      !ktdf.token, !ktdf.token
              }

              // Load sub-stage: stream the full 1x4x64 tile into the FIFO.
              ktdf.stage depends_in(none) depends_out(%slots#2) {
                %s  = arith.subi %i, %c0 : index
                %d  = arith.divsi %s, %c1 : index
                ktdf.data_transfer from %priv#0[%d, 0, 0, 0] size [1, 1, 4, 64]
                                   to %slots#0 size [256]
                    : memref<2x1x4x64xf16, "L1">,
                      !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
              } {applicable_units = ["L1LU"]}

              // Compute sub-stage: all-outs generic.
              //   outs[0] data  (d0,d1,d2) – FIFO-fed tensor<1x4x64xf16>
              //   outs[1] acc0  (d0,d2)    – running sum, tensor<1x64xf16>
              //   outs[2] acc1  (d0,d3,d2) – running max, tensor<1x2x64xf16>
              //
              // findFifoInOperandIdx identifies outs[0] as the FIFO operand
              // because it is produced by ktdf.read_from_fifo.
              // d1 (size 4, in data map) → FIFO-driven outer loop.
              // d3 (size 2, not in data map) → output-only inner loop.
              ktdf.stage depends_in(%slots#2) depends_out(%slots#3) {
                // Read the full 1x4x64 data tensor from the FIFO.
                %data_tile = ktdf.read_from_fifo %slots#0
                    : <"L1LU" -> "SFU", 256xf16> -> tensor<1x4x64xf16>

                %results:3 = linalg.generic {
                  indexing_maps = [#map_data, #map_acc0, #map_acc1],
                  iterator_types = ["parallel", "reduction", "parallel", "reduction"]
                } outs(%data_tile, %cst_zero, %cst_neg_inf
                         : tensor<1x4x64xf16>, tensor<1x64xf16>, tensor<1x2x64xf16>) {
                ^bb0(%v_data: f16, %v_acc0: f16, %v_acc1: f16):
                  %gt     = arith.cmpf ogt, %v_data, %v_acc1 : f16
                  %new_a1 = arith.select %gt, %v_data, %v_acc1 : f16
                  %new_a0 = arith.addf %v_acc0, %v_data : f16
                  linalg.yield %v_data, %new_a0, %new_a1 : f16, f16, f16
                } -> (tensor<1x4x64xf16>, tensor<1x64xf16>, tensor<1x2x64xf16>)

                // Only acc1 (spanning d3) is written to the output FIFO.
                ktdf.write_to_fifo %results#2, %slots#1
                    : tensor<1x2x64xf16>, <"SFU" -> "L1SU", 128xf16>
              } {applicable_units = ["SFU"]}

              // Store sub-stage: drain the output FIFO to local memory.
              ktdf.stage depends_in(%slots#3) depends_out(none) {
                %s = arith.subi %i, %c0 : index
                %d = arith.divsi %s, %c1 : index
                ktdf.data_transfer from %slots#1 size [128]
                                   to %priv#1[%d, 0, 0, 0] size [1, 1, 2, 64]
                    : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                      memref<2x1x2x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}

        // Store stage: write results to global memory.
        ktdf.stage depends_in(%priv#3) depends_out(none) {
          scf.for %i = %c0 to %c2 step %c1 {
            %s = arith.subi %i, %c0 : index
            %d = arith.divsi %s, %c1 : index
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
