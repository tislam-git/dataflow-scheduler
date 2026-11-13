// Checks that the `kEmitDFIR` pipeline registered by dataflow-scheduler-opt
// schedules every pass from `scheduler::buildKTDPToDFIRPipeline` (plus the
// leading `ensure-device-declaration` pass and the trailing `emit-split-dfir`
// pass, both added by the pipeline registration itself), in order.
//
// `--dump-pass-pipeline` prints the fully constructed pass manager, which lets
// us assert the passes are invoked without running the full pipeline on real
// KTIR input. The pipeline still starts to run afterwards and fails on the
// empty module (no device), so `-verify-diagnostics` consumes that error.
//
// RUN: dataflow-scheduler-opt -kEmitDFIR --dump-pass-pipeline %s \
// RUN:   -verify-diagnostics 2>&1 | FileCheck %s

// The passes must appear in the exact order established by
// buildKTDPToDFIRPipeline. CHECK directives match in order, and CHECK-NOT
// between the first and last entry would catch reordering.

// CHECK:        builtin.module(
// CHECK-NEXT:   ensure-device-declaration
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={pre_mapping}}
// CHECK-NEXT:   )
// CHECK-NEXT:   ktir-legality-check
// CHECK-NEXT:   compute-group-extraction
// CHECK-NEXT:   indirect-compute-group-split
// CHECK-NEXT:   indirect-addr-buf-legalization
// CHECK-NEXT:   indirect-access-loop-materialization
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   convert-elementwise-to-linalg,
// CHECK-NEXT:   linalg-morph-ops{category-to-generic=true generic-to-named=false named-to-category=false named-to-generic=true},
// CHECK-NEXT:   simplify-linalg,
// CHECK-NEXT:   fuse-linalg,
// CHECK-NEXT:   ktir-map-and-tile,
// CHECK-NEXT:   ktir-bufferize,
// CHECK-NEXT:   ktir-pipeline
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={pre_scheduling}}
// CHECK-NEXT:   hoist-invariants
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   materialize-registers
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   hoist-invariants
// CHECK-NEXT:   hoist-constant-storage
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   fold-fifo-relay-stages
// CHECK-NEXT:   path-expansion
// CHECK-NEXT:   indirect-addr-buf-fill-legalization
// CHECK-NEXT:   access-granularity-legalization
// CHECK-NEXT:   splat-legalization
// CHECK-NEXT:   normalize-scf-for-loops
// CHECK-NEXT:   canonicalize
// CHECK-NEXT:   tile-scf-for-loops
// CHECK-NEXT:   canonicalize
// CHECK-NEXT:   loop-invariant-code-motion
// CHECK-NEXT:   stage-coarsening
// CHECK-NEXT:   reduction-dim-chunking
// CHECK-NEXT:   split-reduction-inner-outer-dim
// CHECK-NEXT:   reduction-loop-exposure
// CHECK-NEXT:   map-reduction-partials
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   hoist-invariants
// CHECK-NEXT:   hoist-constant-storage
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   broadcast-promotion
// CHECK-NEXT:   double-buffering
// CHECK-NEXT:   tile-size-selection
// CHECK-NEXT:   canonicalize
// CHECK-NEXT:   parallelize-loops-across-instances
// CHECK-NEXT:   canonicalize
// CHECK-NEXT:   affine-min-canonicalization
// CHECK-NEXT:   subsume-linearize-index
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={post_scheduling}}
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   custom-linalg-bufferization
// CHECK-NEXT:   order-fifo-slots-by-use
// CHECK-NEXT:   chunk-wide-accesses
// CHECK-NEXT:   address-assignment
// CHECK-NEXT:   normalize-grid-to-1d
// CHECK-NEXT:   loop-iv-affine-mul-strength-reduction
// CHECK-NEXT:   ktdf-to-ktdflowering
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={post_lowering}}
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   ktdflowering-to-dfir
// CHECK-NEXT:   wrap-program-dfir
// CHECK-NEXT:   emit-split-dfir

// expected-error @below {{no 'ktdf_arch.device' is present in the module and no device file name is provided to ensure-device-declaration pass}}
module {
}
