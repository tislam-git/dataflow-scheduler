//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
/// KTDFLowToDFIR pass: creates one dataflow.program_unit per component type
/// from the KTDFLowering execute_on / signal IR.
///
//
//===----------------------------------------------------------------------===//

#include <mlir/Transforms/RegionUtils.h>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/FoldSubViews.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LogicalMemoryViewBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/PreludeWorkPartition.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ProgramUnitBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/QueryMapArithCollapse.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "ktdflowering-to-dfir"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFLOWTODFIRPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// Run the full canonicalization pattern set (all loaded dialects + registered
// ops) with region simplification (DCE) on `func`. Mirrors the upstream
// Canonicalizer pass; best-effort (non-convergence is not a failure).
static void canonicalizeFunc(mlir::func::FuncOp func) {
  mlir::MLIRContext* ctx = func.getContext();
  mlir::RewritePatternSet patterns(ctx);
  for (mlir::Dialect* dialect : ctx->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperations()) {
    op.getCanonicalizationPatterns(patterns, ctx);
  }
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  // Default GreedyRewriteConfig enables region simplification (DCE).
  (void)mlir::applyPatternsGreedily(func, frozen);
}

struct KTDFLowToDFIRPass
    : public impl::KTDFLowToDFIRPassBase<KTDFLowToDFIRPass> {
  KTDFLowToDFIRPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();

    // Obtain the default device, emitting a diagnostic on failure.
    const auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
    if (!default_device) {
      signalPassFailure();
      return;
    }

    // The lowerings ask the mapping which resource runs an op, so they can size
    // vectors and address memory the way that resource does.
    mlir::ktdf_arch::Mapping mapping(default_device.getRef());
    if (!mapping.byKind().getDefaultCompute()) {
      mapping.getDevice().getDeclaration().emitError(
          "no (unambiguous) default compute resource");
      signalPassFailure();
      return;
    }
    auto& memory_tree =
        default_device.getRef().getOrCreateView<arch_view::MemoryTree>();

    // The run's symbols, numbered once for the whole module: nothing sets a
    // range of ids aside, so two functions cannot each start from the top. This
    // also declares in the IR which symbol each of the run's inputs is, which
    // is what lets a consumer read the numbering rather than work it out again.
    SymbolAllocator symbols(module_op);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module_op.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      LDBG(1) << "Running " << PASS_NAME << " on " << func.getName();

      auto split = partitionPreludeAndWork(func);
      if (split.work_ops.empty()) {
        LDBG(1) << "  no work ops; skipping function";
        continue;
      }

      ResourceToUnits components;
      if (mlir::failed(discoverUnitTypes(split.work_ops, components))) {
        return signalPassFailure();
      }
      if (components.empty()) {
        LDBG(1) << "  no ktdf_lowering.execute_on ops; skipping function";
      } else {
        if (mlir::failed(buildProgramUnits(func, split.work_ops, components))) {
          return signalPassFailure();
        }
      }

      // Clean up side-effect-free ops left dead after extraction (e.g. ops that
      // were cloned into child modules but whose originals are now unused in
      // the top-level module). runRegionDCE recurses into nested regions, so
      // passing the module's regions covers child modules created above.
      mlir::IRRewriter rewriter(module_op.getContext());
      (void)mlir::runRegionDCE(rewriter, func.getBody());

      if (mlir::failed(buildLogicalMemoryViews(
              func, memory_tree, mapping.byKind(), scheduler_ctx_, symbols))) {
        return signalPassFailure();
      }
      // run arith folding into query maps after logical mem view is built to
      // cleanup any extraneous ops.
      if (mlir::failed(scheduler::collapseArithOverQueryMaps(func))) {
        func.emitError("failed to collapse arith ops over query maps");
        return signalPassFailure();
      }

      // Run operation lowerings after program units have been created
      if (mlir::failed(runOperationLowerings(func, schedulerExtContext(),
                                             components, mapping, symbols))) {
        func.emitError("failed to run operation lowerings for ")
            << func.getName();
        return signalPassFailure();
      }

      // The lowerings leave memref.subview between a memory view and the
      // agen accesses of it, which DFIR cannot express: fold them into the
      // accesses' subscripts.
      if (mlir::failed(foldSubViewsIntoAgenAccesses(func))) {
        return signalPassFailure();
      }

      // Final cleanup: canonicalize + DCE the fully-lowered function so the
      // emitted DFIR has no dead/duplicate/foldable leftovers (duplicate
      // folded constants from query-map collapse, addi(const,const), dead
      // NoMemoryEffect ops).
      canonicalizeFunc(func);
    }
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFLowToDFIRPass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass() {
  return std::make_unique<KTDFLowToDFIRPass>(
      SchedulerExtContext::dummyContext());
}
