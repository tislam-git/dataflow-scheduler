// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that a memref.subview of a memory view is folded into the agen
// accesses that use it: the subview's offsets are added to the access
// subscripts and no memref.subview survives.

// CHECK-LABEL: func.func @fold_subview
// CHECK:         dataflow.program_unit
// CHECK:           %[[BUF:.+]] = memref.alloc() : memref<1x4x64xf16, "SFU_REG">
// CHECK-NOT:       memref.subview
// CHECK:           scf.for %[[IV:.+]] =
// CHECK:             %[[RECV:.+]] = dataflow.receive
// CHECK-NEXT:        agen.vector_store %[[RECV]], %[[BUF]][%{{.+}}, %{{.+}} + %[[IV]], %{{.+}}]
// CHECK-SAME:          : memref<1x4x64xf16, "SFU_REG">, vector<64xf16>
// CHECK:           %[[RECV2:.+]] = dataflow.receive
// CHECK-NEXT:      agen.vector_store %[[RECV2]], %[[BUF]][%{{.+}}, %{{.+}} + 3, %{{.+}}]
// CHECK-NOT:       memref.subview

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @fold_subview() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      %src = memref.alloc() : memref<1x64xf16, "L1">
      scf.for %i = %c0 to %c4 step %c1 {
        ktdf.data_transfer from %src[%c0, %c0] size [1, 64] to %fifo size [64]
          : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
      ktdf.data_transfer from %src[%c0, %c0] size [1, 64] to %fifo size [64]
        : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }

    ktdf_lowering.execute_on %u_sfu {
      %buf = memref.alloc() : memref<1x4x64xf16, "SFU_REG">
      scf.for %i = %c0 to %c4 step %c1 {
        %sv = memref.subview %buf[0, %i, 0] [1, 1, 64] [1, 1, 1]
          : memref<1x4x64xf16, "SFU_REG"> to memref<1x1x64xf16, strided<[256, 64, 1], offset: ?>, "SFU_REG">
        %read = ktdf.read_from_fifo %fifo
          : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        memref.copy %read, %sv
          : memref<1x1x64xf16> to memref<1x1x64xf16, strided<[256, 64, 1], offset: ?>, "SFU_REG">
      }
      %sv3 = memref.subview %buf[0, 3, 0] [1, 1, 64] [1, 1, 1]
        : memref<1x4x64xf16, "SFU_REG"> to memref<1x1x64xf16, strided<[256, 64, 1], offset: 192>, "SFU_REG">
      %read3 = ktdf.read_from_fifo %fifo
        : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
      memref.copy %read3, %sv3
        : memref<1x1x64xf16> to memref<1x1x64xf16, strided<[256, 64, 1], offset: 192>, "SFU_REG">
    }

    return
  }
}
