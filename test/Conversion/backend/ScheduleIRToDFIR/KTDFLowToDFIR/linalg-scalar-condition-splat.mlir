// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that a select op with a scalar i1 condition (captured from outside the
// linalg.generic body, e.g. arith.cmpi on a loop IV) is lowered correctly by
// splatting the condition to a vector before emitting
// vectorchain.element_wise_selection.
//
// The existing linalg-cmpf-select-lowering.mlir covers the vector-condition
// path (arith.cmpf on vectors). This test covers the scalar i1 path where
// the condition is an external value.
//
// Expected lowering:
//   - vectorchain.constant_bitstream {value = [0x0]} : vector<1xi1>
//   - vectorchain.shuffle with pad(%i1_cond) {indices = [-1 : i32], repetition = 64}
//     producing vector<64xi1>
//   - vectorchain.element_wise_selection with the vector<64xi1> condition.

// CHECK-LABEL: func.func @scalar_condition_select
// CHECK: dataflow.program_unit
// CHECK:   scf.for %[[IV:[^ ]*]] =
// CHECK:     dataflow.receive {{.*}} : vector<64xf16>

// The scalar i1 (from arith.cmpi on loop IV) is splatted to vector<64xi1>.
// CHECK:     arith.cmpi eq, %[[IV]],
// CHECK:     vectorchain.constant_bitstream {value = [0x0]} : vector<1xi1>
// CHECK:     vectorchain.shuffle input(%{{.*}}), pad(%{{.*}}) {indices = [-1 : i32], repetition = 64 : i32} : vector<1xi1>, i1, vector<64xi1>
// CHECK:     vectorchain.element_wise_selection %{{.*}} ? %{{.*}} : %{{.*}} : vector<64xi1>

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @scalar_condition_select() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x4x64xf16, "L1">
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c4 step %c1 {
        ktdf.data_transfer from %alloc_l1[%c0, %i, %c0] size [1, 1, 64] to %fifo size [64]
            : memref<1x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    ktdf_lowering.execute_on %u_sfu {
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %neg_inf = arith.constant 0xFC00 : f16
      linalg.fill ins(%neg_inf : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c4 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        // The condition is a scalar i1 captured from outside the generic body.
        // This is the same pattern as the topk K-loop: arith.cmpi on the loop IV.
        %is_first = arith.cmpi eq, %i, %c0 : index
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%acc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          // arith.select with a scalar i1 condition — must be splatted.
          %sel = arith.select %is_first, %in, %out : f16
          linalg.yield %sel : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }
}
