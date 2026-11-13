//===-- Pipeline.cpp --------------------------------------------*- c++ -*-===//
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
// This file implements the scheduler pipeline registration.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Pipeline.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

using namespace scheduler;

void scheduler::buildKTIRFrontendPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  // Checkpoint 'pre_mapping':
  //   - The input is legal KTIR, but not necessarily scheduler-legal.
  //   - The scheduler has not made any mapping / scheduling decisions yet.
  //  -> Apply patterns that subtitute front-end constructs that might make
  //     scheduler-illegal programs legal.
  pm.addNestedPass<mlir::func::FuncOp>(
      createApplyDevicePatternsPass({"pre_mapping"}));

  pm.addPass(createKTIRLegalityCheckPass());
  pm.addPass(createComputeGroupExtractionPass());
  pm.addPass(createIndirectComputeGroupSplitPass());
  pm.addPass(createIndirectAddrBufLegalizationPass());
  pm.addPass(createIndirectAccessLoopMaterializationPass());
  {
    auto& nested = pm.nest<mlir::ModuleOp>().nest<mlir::func::FuncOp>();
    nested.addPass(mlir::createConvertElementwiseToLinalgPass());
    nested.addPass(mlir::createLinalgMorphOpsPass(
        {.categoryToGeneric = true, .namedToGeneric = true}));
    nested.addPass(createSimplifyLinalgPass());
    nested.addPass(createFuseLinalgPass());
    nested.addPass(createKTIRMapAndTilePass());
    nested.addPass(createKTIRBufferizePass());
    nested.addPass(createKTIRPipelinePass());
  }
}

void scheduler::buildSchedulerOptimizationPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  // Checkpoint 'pre_scheduling':
  //   - The input is 'ktdf' with nested modules (!), but not necessarily
  //     platform-legal.
  //   - The scheduler has placed some (but most likely not all) mapping
  //     constraints in the IR.
  //  -> Apply patterns that introduce or rewrite mapping constraints or
  //     interact with 'ktdf'.
  {
    auto& nested = pm.nest<mlir::ModuleOp>().nest<mlir::func::FuncOp>();
    nested.addPass(createApplyDevicePatternsPass({"pre_scheduling"}));
    nested.addPass(createHoistInvariantsPass());
  }
  // The patterns above rewrite inside a generic's body and can only insert
  // where they matched, so the registers they need land there.
  pm.addPass(createMaterializeRegistersPass());
  {
    auto& nested = pm.nest<mlir::ModuleOp>().nest<mlir::func::FuncOp>();
    nested.addPass(createHoistInvariantsPass());
    nested.addPass(createHoistConstantStoragePass());
  }
  // Path expansion can only split stages, so it expects transfers between
  // memories that are relayed through FIFOs in one stage.
  pm.addPass(createFoldFifoRelayStagesPass());
  pm.addPass(createPathExpansionPass(scheduler_ctx));
  // Path expansion materializes the stages and their units, which is what says
  // whether an indirect address buffer fill is on a memory its unit can read.
  pm.addPass(createIndirectAddrBufFillLegalizationPass());
  pm.addPass(createAccessGranularityLegalizationPass());
  pm.addPass(createSplatLegalizationPass());
  pm.addPass(createNormalizeSCFForLoopsPass());
  // Canonicalize to get rid of intervening code and single iteration loops
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createTileSCFForLoopsPass());
  // Canonicalize to simplify tiling arith ops
  // (probably not needed for custom tiling regime)
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::ktdf::createStageCoarseningPass());
  pm.addPass(mlir::ktdf::createReductionDimChunkingPass());
  pm.addPass(mlir::ktdf::createSplitReductionInnerOuterDimPass());
  pm.addPass(mlir::ktdf::createReductionLoopExposurePass());
  pm.addPass(mlir::ktdf::createMapReductionPartialsPass());
  // The pass above writes a reduction's identity into a buffer of its own and
  // copies the accumulator from it, and can only insert where the reduction is.
  // Hoisting lifts that write clear of the loop, which is what lets it become a
  // register's initial value rather than an immediate.
  {
    auto& nested = pm.nest<mlir::ModuleOp>().nest<mlir::func::FuncOp>();
    nested.addPass(createHoistInvariantsPass());
    nested.addPass(createHoistConstantStoragePass());
  }
  pm.addPass(mlir::ktdf::createBroadcastPromotionPass());
  pm.addPass(createDoubleBufferingPass(scheduler_ctx));
  // Tile sizes are selected before parallelizing, as the loop to parallelize is
  // chosen by its trip count, which is unknown until the tile sizes are. The
  // canonicalizer then folds the now-constant tile bounds and removes the
  // single-iteration loops, so they are not parallelized.
  pm.addPass(mlir::ktdf::createTileSizeSelectionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createAffineMinCanonicalizationPass());
  pm.addPass(mlir::ktdf::createSubsumeLinearizeIndexPass());

  // Checkpoint 'post_scheduling':
  //   - The input is 'ktdf' on memories and FIFOs, but memory is not yet
  //     allocated and parallelization has not happened.
  //   - The scheduler has decided on the final pipeline structure, tile sizes
  //     and materialized concrete trip counts.
  //  -> Apply patterns that add/remove allocations or coalesce FIFOs.
  pm.nest<mlir::ModuleOp>().addNestedPass<mlir::func::FuncOp>(
      createApplyDevicePatternsPass({"post_scheduling"}));
  // A pattern above can only hand a buffer it materialized to a tensor-typed
  // consumer as a tensor, so the consumer is rebuilt on buffers here, before
  // addresses are assigned to the buffers themselves.
  pm.addPass(createCustomLinalgBufferizationPass());
  // With the consumers in their final form, write each FIFO's slots in the
  // order the consumer uses them, as the FIFO delivers them in that order.
  pm.addPass(createOrderFifoSlotsByUsePass());
  // A FIFO slot wider than the throttle would be sent or received at once by
  // an access the hardware cannot issue; cut it, and its accesses, into
  // throttle-sized ones. After the reordering above, whose protocol looks for
  // the accesses of a slot in one block, and before addresses are assigned.
  pm.addPass(createChunkWideAccessesPass());

  pm.addPass(createAddressAssignmentPass(scheduler_ctx));
  // TODO: position of cross-instance parallelization is TBD
  // pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(createNormalizeGridTo1DPass());
}

void scheduler::buildDFIRBackendPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  pm.addPass(mlir::ktdf::createLoopIVAffineMulStrengthReductionPass());
  pm.addPass(createKTDFToKTDFLoweringPass(scheduler_ctx));

  // Checkpoint 'post_lowering':
  //   - The input is a mix of 'ktdf' and 'ktdf_lowering', but has materialized
  //     the execution unit program scopes.
  //   - The scheduler is down to using memory buffers, but still uses SSA.
  //  -> Apply patterns that produce custom DFIR.
  pm.nest<mlir::ModuleOp>().addNestedPass<mlir::func::FuncOp>(
      createApplyDevicePatternsPass({"post_lowering"}));

  pm.addPass(createKTDFLowToDFIRPass());
  // And each program given the two levels it is read at: what it is, and the
  // DataflowIR it is compiled from, which takes no arguments.
  pm.addPass(createWrapProgramDFIRPass());
}

void scheduler::buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm, const SchedulerExtContext& scheduler_ctx) {
  buildKTIRFrontendPipeline(pm, scheduler_ctx);
  buildSchedulerOptimizationPipeline(pm, scheduler_ctx);
  buildDFIRBackendPipeline(pm, scheduler_ctx);
}
