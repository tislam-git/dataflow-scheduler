// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that a binary float op (arith.addf) with a scalar f16 constant operand
// is lowered correctly by splatting the constant to a vector before emitting
// vectorchain.binary.
//
// Also verifies the constant-on-the-left variant and that a non-constant scalar
// is rejected (xfail: see the rejected test at linalg-generic-reduction-rejected).
//
// Expected lowering:
//   - vectorchain.constant_bitstream holding the raw bits of the constant.
//   - vectorchain.shuffle with repetition = 64 to broadcast it.
//   - vectorchain.binary with two vector<64xf16> operands.

// CHECK-LABEL: func.func @scalar_constant_rhs
// CHECK: dataflow.program_unit
// CHECK:   dataflow.receive {{.*}} : vector<64xf16>
// The scalar 1.0 (0x3C00) is splatted before the binary.
// CHECK:   vectorchain.constant_bitstream {value = [0x3c00]} : vector<1xf16>
// CHECK:   vectorchain.shuffle input(%{{.*}}) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:   vectorchain.binary %{{.*}}, %{{.*}} {binary_op = #vectorchain<binary_operator add>
// CHECK-SAME: vector<64xf16>, vector<64xf16>, vector<64xf16>

// CHECK-LABEL: func.func @scalar_constant_lhs
// CHECK: dataflow.program_unit
// CHECK:   dataflow.receive {{.*}} : vector<64xf16>
// The scalar 2.0 (0x4000) is splatted when it is on the left side.
// CHECK:   vectorchain.constant_bitstream {value = [0x4000]} : vector<1xf16>
// CHECK:   vectorchain.shuffle input(%{{.*}}) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:   vectorchain.binary %{{.*}}, %{{.*}} {binary_op = #vectorchain<binary_operator mul>

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Scalar constant on the right: %v + 1.0
  func.func @scalar_constant_rhs() attributes {grid = [2]} {
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
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c4 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%acc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%v: f16, %out: f16):
          // scalar constant on the rhs — must be splatted to a vector
          %cst = arith.constant 1.0 : f16
          %result = arith.addf %v, %cst : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }

  // Scalar constant on the left: 2.0 * %v
  func.func @scalar_constant_lhs() attributes {grid = [2]} {
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
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%acc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c4 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%acc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%v: f16, %out: f16):
          // scalar constant on the lhs — must be splatted to a vector
          %cst = arith.constant 2.0 : f16
          %result = arith.mulf %cst, %v : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }
}
