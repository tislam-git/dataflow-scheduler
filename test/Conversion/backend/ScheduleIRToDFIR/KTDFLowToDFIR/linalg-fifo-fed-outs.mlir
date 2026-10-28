// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that a linalg.generic with a FIFO-fed outs() operand is lowered
// correctly:
//   - outs[0] is produced by ktdf.read_from_fifo (the data slice).
//   - outs[1] is an LRF accumulator memref.
//
// Expected lowering:
//   - dataflow.receive for outs[0] (no agen.vector_load on the FIFO result).
//   - agen.vector_load for outs[1] (the LRF accumulator).
//   - vectorchain.binary operating on the two vectors.
//   - agen.vector_store writing back to outs[1] only (no store to outs[0]).

// CHECK-LABEL: func.func @fifo_fed_outs
// CHECK: dataflow.program_unit
// CHECK:   dataflow.receive {{.*}} : vector<64xf16>
// CHECK-NOT: agen.vector_load {{.*}} vector<64xf16>
// CHECK:   agen.vector_load {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:   vectorchain.binary
// CHECK-SAME: binary_op = #vectorchain<binary_operator add>
// CHECK:   agen.vector_store {{.*}} : memref<1x64xf16, "SFU_REG">, vector<64xf16>
// CHECK-NOT: agen.vector_store {{.*}} vector<64xf16>{{.*}} vector<64xf16>{{.*}} vector<64xf16>

#map_data = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_acc  = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @fifo_fed_outs() attributes {grid = [2]} {
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

    // Load unit: stream rows into the FIFO.
    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c4 step %c1 {
        ktdf.data_transfer from %alloc_l1[%c0, %i, %c0] size [1, 1, 64] to %fifo size [64]
            : memref<1x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }

    // Compute unit: all-outs generic.
    //   outs[0] = ktdf.read_from_fifo  (FIFO data slice)
    //   outs[1] = LRF accumulator
    ktdf_lowering.execute_on %u_sfu {
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c4 step %c1 {
        // FIFO-fed operand in outs[0]: must not be loaded as a buffer.
        %data = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_data, #map_acc],
          iterator_types = ["parallel", "reduction", "parallel"]
        } outs(%data, %acc : memref<1x1x64xf16>, memref<1x64xf16, "SFU_REG">) {
        ^bb0(%v_data: f16, %v_acc: f16):
          %sum = arith.addf %v_data, %v_acc : f16
          linalg.yield %v_data, %sum : f16, f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }
}
