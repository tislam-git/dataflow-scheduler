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
/// Fold memref.subview into the agen vector accesses that use it.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/FoldSubViews.h"

#include <type_traits>

#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace scheduler;

namespace {

/// Folds the memref.subview feeding an agen.vector_load/store into its access.
///
/// The access subscripts are a map over SSA operands, so the subview's offsets
/// and strides are absorbed into that map: a constant becomes a constant term,
/// and a dynamic offset becomes a new map operand.
template <typename AccessOp>
struct FoldSubViewIntoAgenAccess : mlir::OpRewritePattern<AccessOp> {
  using mlir::OpRewritePattern<AccessOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      AccessOp op, mlir::PatternRewriter& rewriter) const override {
    auto subview =
        op.getMemRef().template getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview) {
      return rewriter.notifyMatchFailure(op, "memref is not a subview");
    }

    // A rank-reducing subview would also need the access sets rewritten.
    const int64_t rank = subview.getSourceType().getRank();
    if (subview.getType().getRank() != rank) {
      return rewriter.notifyMatchFailure(subview, "subview drops dimensions");
    }

    const mlir::AffineMap old_map = op.getAffineMap();
    if (old_map.getNumResults() != static_cast<unsigned>(rank)) {
      return rewriter.notifyMatchFailure(op, "subscripts do not cover memref");
    }

    mlir::MLIRContext* const ctx = op.getContext();
    const unsigned num_dims = old_map.getNumDims();
    const unsigned num_syms = old_map.getNumSymbols();

    // Dynamic offsets become extra map operands, numbered as dims after the
    // existing ones.
    llvm::SmallVector<mlir::Value> extra_operands;
    auto offset_expr = [&](mlir::OpFoldResult offset) -> mlir::AffineExpr {
      if (auto constant = mlir::getConstantIntValue(offset)) {
        return mlir::getAffineConstantExpr(*constant, ctx);
      }
      extra_operands.push_back(llvm::cast<mlir::Value>(offset));
      return mlir::getAffineDimExpr(num_dims + extra_operands.size() - 1, ctx);
    };

    llvm::SmallVector<mlir::AffineExpr> results;
    auto offsets = subview.getMixedOffsets();
    auto strides = subview.getMixedStrides();
    for (int64_t i = 0; i < rank; ++i) {
      auto stride = mlir::getConstantIntValue(strides[i]);
      if (!stride) {
        return rewriter.notifyMatchFailure(subview, "dynamic stride");
      }
      results.push_back(offset_expr(offsets[i]) +
                        *stride * old_map.getResult(i));
    }

    // Every operand is a dim: the old symbols move behind the extra dims.
    const unsigned new_num_dims =
        num_dims + static_cast<unsigned>(extra_operands.size());
    llvm::SmallVector<mlir::AffineExpr> dim_repl;
    for (unsigned i = 0; i < new_num_dims; ++i) {
      dim_repl.push_back(mlir::getAffineDimExpr(i, ctx));
    }
    llvm::SmallVector<mlir::AffineExpr> sym_repl;
    for (unsigned i = 0; i < num_syms; ++i) {
      sym_repl.push_back(mlir::getAffineDimExpr(new_num_dims + i, ctx));
    }
    mlir::AffineMap new_map = mlir::simplifyAffineMap(
        mlir::AffineMap::get(new_num_dims, num_syms, results, ctx)
            .replaceDimsAndSymbols(dim_repl, sym_repl, new_num_dims + num_syms,
                                   0));

    auto old_operands = op.getMapOperands();
    llvm::SmallVector<mlir::Value> operands(old_operands.begin(),
                                            old_operands.begin() + num_dims);
    operands.append(extra_operands);
    operands.append(old_operands.begin() + num_dims, old_operands.end());

    rewriter.setInsertionPoint(op);
    AccessOp folded = op.cloneWithNewAccessInfo(rewriter, subview.getSource(),
                                                new_map, operands);
    folded->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    if constexpr (std::is_same_v<AccessOp, mlir::agen::VectorLoadOp>) {
      rewriter.replaceOp(op, folded.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
};

}  // namespace

mlir::LogicalResult scheduler::foldSubViewsIntoAgenAccesses(
    mlir::func::FuncOp func) {
  mlir::RewritePatternSet patterns(func.getContext());
  patterns.add<FoldSubViewIntoAgenAccess<mlir::agen::VectorLoadOp>,
               FoldSubViewIntoAgenAccess<mlir::agen::VectorStoreOp>>(
      func.getContext());
  // The driver also erases each subview once its last user is rewritten.
  if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
    return mlir::failure();
  }

  mlir::LogicalResult result = mlir::success();
  func.walk([&](mlir::memref::SubViewOp subview) {
    subview.emitError(
        "memref.subview cannot be expressed in DFIR and could not be folded "
        "into its users");
    result = mlir::failure();
  });
  return result;
}
