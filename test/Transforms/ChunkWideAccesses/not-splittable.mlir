// RUN: dataflow-scheduler-opt --chunk-wide-accesses --split-input-file --verify-diagnostics %s

ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

// A wide access that does not tile by the throttle is not legal, and not
// silently left wide.
func.func @does_not_tile(%buf: memref<1x2x80xf16, "SFU_REG">) {
  ktdf.pipeline {
    %p:1 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 160xf16>) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 160xf16>
      ktdf.private_yield %slot : !ktdf.fifo.slot<"SFU" -> "L1SU", 160xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{cannot tile an access of 160 elements by the throttle of 64}}
      ktdf.write_to_fifo %buf, %p#0 : memref<1x2x80xf16, "SFU_REG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 160xf16>
    } {applicable_units = ["SFU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

// A splat / pad load into a wide slot is not split.
func.func @splat_transfer(%src: memref<1x64xf16, "L1">) {
  ktdf.pipeline {
    %p:1 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
      ktdf.private_yield %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{cannot split a splat / pad transfer through a wide slot}}
      ktdf.data_transfer from %src[0, 0] size [1, 64] to %p#0 size [256] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
    } {applicable_units = ["L1LU"]}
  }
  return
}
