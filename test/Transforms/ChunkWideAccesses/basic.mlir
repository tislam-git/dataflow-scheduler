// RUN: dataflow-scheduler-opt --chunk-wide-accesses %s | FileCheck %s

// A FIFO slot wider than the throttle (64 f16 lanes on the sample device's
// SFU) is shrunk to the throttle, and each access to it is cut into
// throttle-sized transfers in a loop nest that covers the original extent.

ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

// The producer writes a 1x4x64 buffer, the consumer reads it as 1x1x4x64: both
// ends loop over the four rows, one throttle at a time.
// CHECK-LABEL: func.func @wide_slot
// CHECK:         ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:           ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:         scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}}[%{{.+}}, %{{.+}} + %[[I]], %{{.+}}] size [1, 1, 64] to %{{.+}} size [64]
// CHECK-NEXT:    }
// CHECK-NOT:     ktdf.write_to_fifo
// CHECK:         scf.for %[[J:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}} size [64] to %{{.+}}[0, 0, %[[J]], 0] size [1, 1, 1, 64]
// CHECK-NEXT:    }
// CHECK-NOT:     size [256]
func.func @wide_slot(%buf: memref<1x4x64xf16, "SFU_REG">, %out: memref<1x1x4x64xf16, "L1">) {
  ktdf.pipeline {
    %p:3 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %slot, %t0, %t1 : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#1) {
      ktdf.write_to_fifo %buf, %p#0 : memref<1x4x64xf16, "SFU_REG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(%p#1) depends_out(none) {
      ktdf.data_transfer from %p#0 size [256] to %out[0, 0, 0, 0] size [1, 1, 4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<1x1x4x64xf16, "L1">
    } {applicable_units = ["L1SU"]}
  }
  return
}

// The last dimension alone is wider than the throttle: it is stepped by the
// throttle.
// CHECK-LABEL: func.func @wide_innermost
// CHECK:         ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:         scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}}[0, %[[I]] * 64] size [1, 64] to %{{.+}} size [64]
func.func @wide_innermost(%buf: memref<1x128xf16, "SFU_REG">) {
  ktdf.pipeline {
    %p:1 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
      ktdf.private_yield %slot : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %buf[0, 0] size [1, 128] to %p#0 size [128] : memref<1x128xf16, "SFU_REG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
    } {applicable_units = ["SFU"]}
  }
  return
}

// A slot that is already a throttle is left alone.
// CHECK-LABEL: func.func @narrow_untouched
// CHECK-NOT:     scf.for
// CHECK:         ktdf.write_to_fifo
func.func @narrow_untouched(%buf: memref<1x1x64xf16, "SFU_REG">) {
  ktdf.pipeline {
    %p:1 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %slot : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %buf, %p#0 : memref<1x1x64xf16, "SFU_REG">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    } {applicable_units = ["SFU"]}
  }
  return
}
