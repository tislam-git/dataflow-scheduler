//===------------------------------------------------------------*- c++ -*-===//
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
/// Fold memref.subview into the agen vector accesses that use it.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_FOLDSUBVIEWS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_FOLDSUBVIEWS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// DFIR addresses memory only through logical memory views, so no
/// memref.subview may survive lowering. Rewrite every
///   %sv = memref.subview %view[offsets] [sizes] [strides]
///   agen.vector_load/store ... %sv[map(operands)]
/// into an access of %view whose subscripts are
///   offset_i + stride_i * subscript_i,
/// then erase the subview once it has no users left.
///
/// Fails, with a diagnostic on the subview, if one is left over: because a user
/// is not an agen.vector_load/store, or the subview drops dimensions or has a
/// dynamic stride.
mlir::LogicalResult foldSubViewsIntoAgenAccesses(mlir::func::FuncOp func);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_FOLDSUBVIEWS_H_
