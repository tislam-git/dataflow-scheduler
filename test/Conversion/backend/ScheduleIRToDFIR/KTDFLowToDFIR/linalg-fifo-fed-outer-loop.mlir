// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that a linalg.generic whose FIFO-fed operand is defined at an outer
// loop level places the corresponding agen.vector_store at the same loop nest
// level as the dataflow.receive (outside the inner loop).

// CHECK-LABEL: func.func @fifo_fed_outer_loop
// CHECK: dataflow.program_unit
// CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c1 {
// CHECK:     [[RECV:%.+]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK:     agen.vector_store [[RECV]], %{{.*}}[%c0] {store_order = #map{{[0-9]+}}, store_set = #set{{[0-9]+}}} : memref<64xf16, "SFU_REG">, vector<64xf16>
// CHECK:     scf.for %{{.*}} = %c0 to %c4 step %c1 {
// CHECK-NOT:   agen.vector_store [[RECV]]
// CHECK:       agen.vector_load
// CHECK:       vectorchain.binary
// CHECK:     }
// CHECK:   }

#map_data = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_acc  = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @fifo_fed_outer_loop() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<64x4x64xf16, "L1">
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    // Load unit: stream rows into the FIFO.
    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c64 step %c1 {
        ktdf.data_transfer from %alloc_l1[%i, %c0, %c0] size [1, 1, 64] to %fifo size [64]
            : memref<64x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }

    // Compute unit: read from FIFO in outer loop, use in inner loop with store to register.
    ktdf_lowering.execute_on %u_sfu {
      %reg = memref.alloc() : memref<64xf16, "SFU_REG">
      %acc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c64 step %c1 {
        %data = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        scf.for %j = %c0 to %c4 step %c1 {
          linalg.generic {
            indexing_maps = [#map_data, #map_acc],
            iterator_types = ["parallel", "reduction", "parallel"]
          } outs(%data, %acc : memref<1x1x64xf16>, memref<1x64xf16, "SFU_REG">) {
          ^bb0(%v_data: f16, %v_acc: f16):
            memref.store %v_data, %reg[%c0] : memref<64xf16, "SFU_REG">
            %loaded = memref.load %reg[%c0] : memref<64xf16, "SFU_REG">
            %sum = arith.addf %loaded, %v_acc : f16
            linalg.yield %v_data, %sum : f16, f16
          }
        }
      }
    }
    return
  }
}
