// RUN: not dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s 2>&1 | FileCheck %s

// maxnumf on its own is not lowered, and neither is minnumf: they are the
// withdrawn 754-2008 operations, which leave NaN and signed-zero handling to the
// implementation, and what this unit does with either is not written down. Mapping
// them to a plain max or min would be a guess, so they are turned away and
// minimumf and maximumf -- which say what they mean -- are lowered instead.
//
// absf(x) maxnumf absf(y) still lowers, to abs_max; linalg-maxnumf-lowering covers
// it.

// The lowering refuses it before rewriting anything. The generic is left where it
// is once refused, so what cannot be lowered after it reports too. Only the first
// message is the one under test.
// CHECK: error: maxnumf outside the abs-max shape is not lowered

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // ── plain maxnumf: turned away ─────────────────────────────────────────────
  func.func @maxnumf_plain() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x256x64xf16, "L1">
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c256 step %c1 {
        ktdf.data_transfer from %alloc_l1[%c0, %i, %c0] size [1, 1, 64] to %fifo size [64] : memref<1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    ktdf_lowering.execute_on %u_sfu {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %neg_inf = arith.constant 0xFF80 : f16
      linalg.fill ins(%neg_inf : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c256 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%alloc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          %result = arith.maxnumf %in, %out : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }

}
