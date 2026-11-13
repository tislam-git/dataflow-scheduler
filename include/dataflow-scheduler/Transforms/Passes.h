//===-- Passes.h ------------------------------------------------*- c++ -*-===//
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
// This file declares all the scheduler passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_H_

#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace scheduler {
struct SchedulerExtContext;
}  // namespace scheduler

namespace scheduler {

std::unique_ptr<mlir::Pass> createNormalizeSCFForLoopsPass();
std::unique_ptr<mlir::Pass> createTileSCFForLoopsPass();
std::unique_ptr<mlir::Pass> createStripMineSCFForLoopsPass();
std::unique_ptr<mlir::Pass> createPathExpansionPass();
std::unique_ptr<mlir::Pass> createPathExpansionPass(
    const SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createNormalizeGridTo1DPass();

std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass();
std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass(
    const SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createDoubleBufferingPass();
std::unique_ptr<mlir::Pass> createDoubleBufferingPass(
    const SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createAffineMinCanonicalizationPass();
std::unique_ptr<mlir::Pass> createAddressAssignmentPass();
std::unique_ptr<mlir::Pass> createAddressAssignmentPass(
    const SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createAccessGranularityLegalizationPass();
std::unique_ptr<mlir::Pass> createIndirectAddrBufFillLegalizationPass();
std::unique_ptr<mlir::Pass> createChunkWideAccessesPass();
std::unique_ptr<mlir::Pass> createSplatLegalizationPass();
std::unique_ptr<mlir::Pass> createCustomLinalgBufferizationPass();
std::unique_ptr<mlir::Pass> createEnsureDeviceDeclarationPass();
std::unique_ptr<mlir::Pass> createIndirectAddrBufLegalizationPass();
std::unique_ptr<mlir::Pass> createIndirectAccessLoopMaterializationPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "dataflow-scheduler/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEnsureDeviceDeclarationPass(
    EnsureDeviceDeclarationPassOptions options);

[[nodiscard]]
std::unique_ptr<mlir::Pass> createApplyDevicePatternsPass(
    std::initializer_list<llvm::StringRef> enabled_groups);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORM_PASSES_H
