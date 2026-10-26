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
// Lowering compute operations (linalg.generic, arith, math) into DFIR.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"

#include <llvm/ADT/APInt.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

// ---------------------------------------------------------------------------
// Returns true and sets `lhs_out`/`rhs_out` to the inputs of the abs ops when
// `maxnum_op` implements abs_max, i.e. both operands are produced by a
// math.absf.  The two abs ops are erased via `rewriter` after the caller
// emits its replacement.
// ---------------------------------------------------------------------------
static bool matchAbsMaxOperands(mlir::arith::MaxNumFOp maxnum_op,
                                mlir::Value& lhs_out, mlir::Value& rhs_out) {
  auto lhs_abs = maxnum_op.getLhs().getDefiningOp<mlir::math::AbsFOp>();
  auto rhs_abs = maxnum_op.getRhs().getDefiningOp<mlir::math::AbsFOp>();
  if (!lhs_abs || !rhs_abs) return false;
  lhs_out = lhs_abs.getOperand();
  rhs_out = rhs_abs.getOperand();
  return true;
}

/// Gets the compare operator standing for \p predicate, or nothing where the
/// unit has none. Only the ordered predicates map: an unordered one asks about
/// NaN, which the compare does not answer.
[[nodiscard]] auto compareOperatorFor(mlir::arith::CmpFPredicate predicate)
    -> std::optional<mlir::vectorchain::VectorChainElementWiseCompareOperator> {
  using Predicate = mlir::arith::CmpFPredicate;
  using Operator = mlir::vectorchain::VectorChainElementWiseCompareOperator;
  switch (predicate) {
    case Predicate::OEQ:
      return Operator::compare_eq;
    case Predicate::ONE:
      return Operator::compare_neq;
    case Predicate::OLT:
      return Operator::compare_lt;
    case Predicate::OLE:
      return Operator::compare_le;
    case Predicate::OGT:
      return Operator::compare_gt;
    case Predicate::OGE:
      return Operator::compare_ge;
    default:
      return std::nullopt;
  }
}

/// Pattern to lower linalg.generic compute operations
struct LowerLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    // Buffer-semantics path: init operand is a memref accumulator.
    if (generic_op.hasPureBufferSemantics())
      return lowerMemRefGenericOp(generic_op, rewriter);

    if (!generic_op.hasPureTensorSemantics() ||
        generic_op.getNumResults() != 1) {
      return mlir::failure();
    }

    // ReductionLoopExposure and MapReductionPartials run before this pass and
    // rewrite every tensor-semantics linalg.generic with reduction dims into
    // explicit scf.for loops, leaving only parallel iterators here. Nothing
    // below lowers a reduction, and the elementwise path would take one for an
    // elementwise op and give a wrong answer, so say so rather than assert it:
    // an assert is gone in a release build and the fall-through is silent.
    if (llvm::any_of(generic_op.getIteratorTypesArray(), [](auto iter_type) {
          return iter_type != mlir::utils::IteratorType::parallel;
        })) {
      return generic_op->emitError(
          "tensor-semantics linalg.generic with a reduction dimension reached "
          "the DataflowIR lowering; ReductionLoopExposure and "
          "MapReductionPartials run before it and should have rewritten it");
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Replace block arguments with generic inputs.  Any input that is a
    // constant tensor (e.g. a dense<0.0>) is converted to an equivalent
    // arith.constant with vector type first so that vectorchain.binary always
    // receives vector-typed operands.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      mlir::Value converted =
          convertConstTensorInputToVector(input, generic_op, rewriter);
      rewriter.replaceAllUsesWith(block_arg, converted);
    }

    // Identity affine map used as op_specific_map for binary ops.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // Collect compute operations to replace
    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator()) {
      ops_to_lower.push_back(&op);
    }

    // Process and lower compute operations via visitors
    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::arith::CmpFOp>([&](mlir::arith::CmpFOp op) {
                return lowerCompareFOp(op, rewriter);
              })
              .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp op) {
                return lowerSelectOp(op, rewriter);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                mlir::Value lhs, rhs;
                if (matchAbsMaxOperands(op, lhs, rhs)) {
                  // Capture abs ops before lowering replaces maxnumf (after
                  // which the operand values are no longer reachable via op).
                  mlir::Operation* lhs_abs = op.getLhs().getDefiningOp();
                  mlir::Operation* rhs_abs = op.getRhs().getDefiningOp();
                  mlir::LogicalResult res = lowerBinaryFOp(
                      op, lhs, rhs, rewriter, identity_map,
                      mlir::vectorchain::VectorChainBinaryOperator::abs_max);
                  // maxnumf is now replaced; absf results are unused — safe to
                  // erase.
                  if (mlir::succeeded(res)) {
                    rewriter.eraseOp(lhs_abs);
                    rewriter.eraseOp(rhs_abs);
                  }
                  return res;
                }
                // Only the abs-of-both shape lowers. maxnumf and minnumf
                // are the withdrawn 754-2008 operations, whose handling of NaN
                // and of signed zero is left to the implementation, and what
                // this unit does is not written down -- so mapping them to a
                // plain max or min would be a guess. minimumf and maximumf say
                // what they mean and are lowered instead.
                return rewriter.notifyMatchFailure(
                    op, "maxnumf outside the abs-max shape is not lowered");
              })
              .Case<mlir::math::AbsFOp>([&](mlir::math::AbsFOp op)
                                            -> mlir::LogicalResult {
                // Consumed and erased by the MaxNumFOp abs_max case above when
                // it is an operand of a maxnumf; nothing to emit here.
                if (op->hasOneUse() &&
                    mlir::isa<mlir::arith::MaxNumFOp>(*op->user_begin()))
                  return mlir::success();
                return op->emitError(
                    "unsupported standalone math.absf in linalg.generic body");
              })
              .Case<mlir::dataflow::OpaqueOp>([&](mlir::dataflow::OpaqueOp op) {
                // Already DFIR, and it reads and writes registers rather than
                // the lanes the body deals in, so it only has to leave the
                // body.
                rewriter.moveOpBefore(op, generic_op);
                return mlir::success();
              })
              .Case<mlir::memref::StoreOp>([&](mlir::memref::StoreOp op) {
                return lowerMemRefStore(op, rewriter);
              })
              .Case<mlir::memref::LoadOp>([&](mlir::memref::LoadOp op) {
                return lowerMemRefLoad(op, rewriter);
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });

      if (mlir::failed(result)) return mlir::failure();
    }

    // Replace the generic op with the yield operand
    rewriter.replaceOp(generic_op, yield_op.getOperand(0));
    return mlir::success();
  }

 private:
  /// Whether \p operand is a fifo read, or the vector that read was replaced
  /// by: a value rather than a buffer.
  static bool isFifoFed(mlir::Value operand) {
    return !mlir::isa<mlir::MemRefType>(operand.getType()) ||
           operand.getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
  }

  // Lowers a linalg.generic with buffer semantics (memref init operand).
  // The resulting accumulated vector is written back to the output buffer via
  // agen.vector_store.
  mlir::LogicalResult lowerMemRefGenericOp(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    mlir::Location loc = generic_op.getLoc();

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    const unsigned accumulators = generic_op.getNumDpsInits();
    if (!yield_op || yield_op.getNumOperands() != accumulators) {
      return mlir::failure();
    }

    // Replace input block arguments with their corresponding linalg ins
    // operands.  The body's ops become vectorchain ops, so each operand has to
    // reach them as a vector.
    //
    // A buffer is read with agen.vector_load, the same way an accumulator is
    // below.  An input arrives as a buffer when a device pattern materialized
    // one, which is also what wrote the values being read here.
    //
    // A buffer that a ktdf.read_from_fifo hands out is the exception: it holds
    // no address, and LowerReadFromFifoPattern replaces that read with the
    // vector a dataflow.receive yields.  Loading from it would only be undone
    // when that replacement rewrites the load's own operand.
    //
    // This pattern can fail part-way (a body store of a fifo read is not yet a
    // vector store) and be tried again once the read has been replaced. A block
    // argument with no uses was replaced by the first try, so it is skipped
    // rather than loaded a second time.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    rewriter.setInsertionPoint(generic_op);
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      if (block_arg.use_empty()) continue;
      mlir::Value operand = input;
      auto memref_type = mlir::dyn_cast<mlir::MemRefType>(input.getType());
      if (memref_type && !input.getDefiningOp<mlir::ktdf::ReadFromFifoOp>()) {
        mlir::VectorType vec_type = getFlattenedVectorType(memref_type);
        if (!vec_type) return mlir::failure();
        operand = scheduler::emitVectorLoad(rewriter, loc, vec_type, input);
      }
      rewriter.replaceAllUsesWith(block_arg, operand);
    }

    // Each output block argument is the accumulator value the matching memref
    // holds. Read each into a vector and let the body read that instead. There
    // is one per result: a compute that accumulates more than one thing
    // accumulates into two.
    //
    // An output that a ktdf.read_from_fifo hands out is the exception, for the
    // same reason as an input above: it holds no address, and its read becomes
    // the received vector, which is already the value the body wants. It is
    // also not written back below -- there is no buffer behind it. Once that
    // read has been replaced the output is the vector itself, which is the
    // same case.
    llvm::SmallVector<mlir::Value> out_memrefs;
    for (unsigned r = 0; r < accumulators; ++r) {
      mlir::Value out_memref = generic_op.getDpsInitOperand(r)->get();
      out_memrefs.push_back(out_memref);

      mlir::BlockArgument acc_arg = body.getArgument(num_inputs + r);
      if (acc_arg.use_empty()) continue;

      if (isFifoFed(out_memref)) {
        rewriter.replaceAllUsesWith(acc_arg, out_memref);
        continue;
      }

      auto out_memref_type = mlir::cast<mlir::MemRefType>(out_memref.getType());
      auto acc_vec_type = getFlattenedVectorType(out_memref_type);
      if (!acc_vec_type) return mlir::failure();

      rewriter.replaceAllUsesWith(
          acc_arg,
          scheduler::emitVectorLoad(rewriter, loc, acc_vec_type, out_memref));
    }

    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator())
      ops_to_lower.push_back(&op);

    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::arith::CmpFOp>([&](mlir::arith::CmpFOp op) {
                return lowerCompareFOp(op, rewriter);
              })
              .Case<mlir::arith::SelectOp>([&](mlir::arith::SelectOp op) {
                return lowerSelectOp(op, rewriter);
              })
              .Case<mlir::memref::StoreOp>([&](mlir::memref::StoreOp op) {
                return lowerMemRefStore(op, rewriter);
              })
              .Case<mlir::memref::LoadOp>([&](mlir::memref::LoadOp op) {
                return lowerMemRefLoad(op, rewriter);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                mlir::Value lhs, rhs;
                if (matchAbsMaxOperands(op, lhs, rhs)) {
                  mlir::Operation* lhs_abs = op.getLhs().getDefiningOp();
                  mlir::Operation* rhs_abs = op.getRhs().getDefiningOp();
                  mlir::LogicalResult res = lowerBinaryFOp(
                      op, lhs, rhs, rewriter, identity_map,
                      mlir::vectorchain::VectorChainBinaryOperator::abs_max);
                  if (mlir::succeeded(res)) {
                    rewriter.eraseOp(lhs_abs);
                    rewriter.eraseOp(rhs_abs);
                  }
                  return res;
                }
                // Only the abs-of-both shape lowers. maxnumf and minnumf
                // are the withdrawn 754-2008 operations, whose handling of NaN
                // and of signed zero is left to the implementation, and what
                // this unit does is not written down -- so mapping them to a
                // plain max or min would be a guess. minimumf and maximumf say
                // what they mean and are lowered instead.
                return rewriter.notifyMatchFailure(
                    op, "maxnumf outside the abs-max shape is not lowered");
              })
              .Case<mlir::math::AbsFOp>([&](mlir::math::AbsFOp op)
                                            -> mlir::LogicalResult {
                // Consumed and erased by the MaxNumFOp abs_max case above when
                // it is an operand of a maxnumf; nothing to emit here.
                if (op->hasOneUse() &&
                    mlir::isa<mlir::arith::MaxNumFOp>(*op->user_begin()))
                  return mlir::success();
                return op->emitError(
                    "unsupported standalone math.absf in linalg.generic body");
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });
      if (mlir::failed(result)) return mlir::failure();
    }

    // Write each accumulated vector back to the buffer it came from. A fifo
    // read has no buffer to write back to.
    for (unsigned r = 0; r < accumulators; ++r) {
      if (isFifoFed(out_memrefs[r])) continue;
      scheduler::emitVectorStore(rewriter, loc, yield_op.getOperand(r),
                                 out_memrefs[r]);
    }

    rewriter.eraseOp(generic_op);
    return mlir::success();
  }

  /// Converts a constant tensor input of a linalg.generic to a vector-typed
  /// arith.constant, preserving all element values.  This is needed because
  /// linalg.generic inputs can be constant tensors (e.g. a dense<0.0>), while
  /// vectorchain.binary requires vector operands.
  ///
  /// Only arith.constant ops whose value is a DenseElementsAttr are handled;
  /// any other input (non-constant tensors, vectors, scalars) is returned
  /// unchanged.
  mlir::Value convertConstTensorInputToVector(
      mlir::Value input, mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    // Only act on tensor-typed inputs — vectors and scalars pass through.
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensor_type) return input;

    // Must be a constant op with a dense attribute to convert.
    auto const_op =
        mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(input.getDefiningOp());
    if (!const_op) return input;
    auto dense_attr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!dense_attr) return input;

    // Determine the target vector type (same element type, flattened shape).
    auto vector_type = getFlattenedVectorType(tensor_type);
    if (!vector_type) return input;

    // Re-materialise the constant with the vector type, preserving all element
    // values by reinterpreting the same dense data into the flat vector shape.
    auto vec_attr = dense_attr.reshape(vector_type);
    rewriter.setInsertionPoint(generic_op);
    return mlir::arith::ConstantOp::create(rewriter, const_op.getLoc(),
                                           vector_type, vec_attr)
        .getResult();
  }

  /// Lowers a store into a register to the vector store that writes it.
  ///
  /// A register is written whole, so the value has to be a vector by the time
  /// this runs. Returns failure while it is still the body's scalar, so the
  /// driver comes back to it.
  mlir::LogicalResult lowerMemRefStore(mlir::memref::StoreOp op,
                                       mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<mlir::VectorType>(op.getValueToStore().getType())) {
      return mlir::failure();
    }

    const auto access = getRegisterAccess(op.getMemRef());
    if (!access) return mlir::failure();

    mlir::agen::VectorStoreOp::create(
        rewriter, op.getLoc(), op.getValueToStore(), op.getMemRef(),
        /*dbgName=*/nullptr, access->map, op.getIndices(), access->set,
        access->order);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  /// Lowers a load out of a register to the vector load that reads it.
  mlir::LogicalResult lowerMemRefLoad(mlir::memref::LoadOp op,
                                      mlir::PatternRewriter& rewriter) const {
    const auto vector_type = getFlattenedVectorType(
        llvm::cast<mlir::MemRefType>(op.getMemRef().getType()));
    if (!vector_type) return mlir::failure();

    const auto access = getRegisterAccess(op.getMemRef());
    if (!access) return mlir::failure();

    auto load = mlir::agen::VectorLoadOp::create(
        rewriter, op.getLoc(), vector_type, op.getMemRef(),
        /*dbgName=*/nullptr, access->map, op.getIndices(), access->set,
        access->order, /*multicast_info=*/nullptr);
    rewriter.replaceOp(op, load.getResult());
    return mlir::success();
  }

  /// Holds the maps that address a register: the whole of it, in lane order.
  struct RegisterAccess {
    mlir::AffineMap map;
    mlir::IntegerSet set;
    mlir::AffineMap order;
  };

  std::optional<RegisterAccess> getRegisterAccess(mlir::Value mem_ref) const {
    const auto type = mlir::dyn_cast<mlir::MemRefType>(mem_ref.getType());
    if (!type || !type.hasStaticShape()) return std::nullopt;

    auto* const ctx = mem_ref.getContext();
    const auto rank = static_cast<unsigned>(type.getRank());
    return RegisterAccess{mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
                          buildIntegerSetFromSizes(ctx, type.getShape()),
                          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx)};
  }

  // Unified helper: lowers any two-operand arith float op to
  // vectorchain.binary with the given binary_kind.
  mlir::LogicalResult lowerBinaryFOp(
      mlir::Operation* op, mlir::Value lhs, mlir::Value rhs,
      mlir::PatternRewriter& rewriter, mlir::AffineMap identity_map,
      mlir::vectorchain::VectorChainBinaryOperator binary_kind) const {
    const auto lhs_ty = llvm::dyn_cast<mlir::ShapedType>(lhs.getType());
    if (!lhs_ty) {
      return llvm::failure();
    }
    auto vector_type = getFlattenedVectorType(lhs_ty);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, op->getLoc(), vector_type, lhs, rhs,
        /*mask=*/nullptr, /*dbgName=*/nullptr, binary_kind, identity_map);

    rewriter.replaceOp(op, binary_op.getData());
    return mlir::success();
  }

  /// Lowers \p op to an element-wise compare.
  ///
  /// The result carries the operands' type rather than a boolean: it is what a
  /// selection takes as its condition, and the unit keeps it in a lane of the
  /// same width. The i1 form of this op is the separate mask operand.
  mlir::LogicalResult lowerCompareFOp(mlir::arith::CmpFOp op,
                                      mlir::PatternRewriter& rewriter) const {
    const auto compare_kind = compareOperatorFor(op.getPredicate());
    if (!compare_kind) return mlir::failure();

    const auto lhs_ty = llvm::dyn_cast<mlir::ShapedType>(op.getLhs().getType());
    if (!lhs_ty) return mlir::failure();
    auto operands = getFlattenedVectorType(lhs_ty);
    if (!operands) return mlir::failure();

    auto compare_op = mlir::vectorchain::ElementWiseCompareOp::create(
        rewriter, op->getLoc(), operands, op.getLhs(), op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr, *compare_kind);

    rewriter.replaceOp(op, compare_op.getData());
    return mlir::success();
  }

  /// Lowers \p op to an element-wise selection, taking a lane from one side or
  /// the other by the mask a compare left.
  mlir::LogicalResult lowerSelectOp(mlir::arith::SelectOp op,
                                    mlir::PatternRewriter& rewriter) const {
    const auto picked =
        llvm::dyn_cast<mlir::ShapedType>(op.getTrueValue().getType());
    if (!picked) return mlir::failure();
    auto result = getFlattenedVectorType(picked);
    if (!result) return mlir::failure();

    auto selection_op = mlir::vectorchain::ElementWiseSelectionOp::create(
        rewriter, op->getLoc(), result, op.getCondition(), op.getTrueValue(),
        op.getFalseValue(), /*mask=*/nullptr, /*dbgName=*/nullptr);

    rewriter.replaceOp(op, selection_op.getData());
    return mlir::success();
  }
};

/// Pattern to lower linalg.fill into:
///   vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
///   vectorchain.shuffle ... {indices = [0 : i32], repetition = N}
///       : vector<1xT>, vector<NxT>
///
/// Buffer semantics (memref output): the shuffle result is written to the
/// output memref via agen.vector_store and the fill is erased.
///
/// Tensor semantics (tensor output): the shuffle result directly replaces
/// the fill result (consumed by downstream vectorchain / FIFO ops).
///
/// N and T are derived from the output type shape and element type, and the
/// bitstream carries the fill value as the bits a lane holds.
struct LowerLinalgFillPattern
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  LowerLinalgFillPattern(mlir::MLIRContext* context,
                         scheduler::SymbolAllocator& symbols)
      : OpRewritePattern(context), symbols_(symbols) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::FillOp fill_op,
      mlir::PatternRewriter& rewriter) const override {
    // Match constant fill value input.
    if (fill_op.getInputs().size() != 1) {
      return rewriter.notifyMatchFailure(fill_op,
                                         "must have exactly one operand");
    }
    const mlir::Value filled = fill_op.getInputs()[0];
    auto* const fill_def = filled.getDefiningOp();
    mlir::Attribute value;
    const bool is_constant =
        fill_def && mlir::m_Constant(&value).match(fill_def);

    // A fill value the compiler does not know becomes a symbol: the bitstream
    // carries the id, and whatever resolves the symbols writes the value in.
    // What the symbol is is declared next to the program.
    int64_t symbol_id = 0;
    if (!is_constant) {
      auto program = fill_op->getParentOfType<mlir::func::FuncOp>();
      if (!program) {
        return rewriter.notifyMatchFailure(fill_op, "fill is in no function");
      }
      const auto declared =
          scheduler::declareScalarSymbol(filled, program, symbols_);
      if (mlir::failed(declared)) {
        return rewriter.notifyMatchFailure(
            fill_op, "fill value is neither a constant nor a symbol");
      }
      symbol_id = *declared;
    }

    llvm::APInt fill_bits;
    if (is_constant) {
      if (const auto attr = mlir::dyn_cast<mlir::FloatAttr>(value)) {
        fill_bits = attr.getValue().bitcastToAPInt();
      } else if (const auto attr = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
        fill_bits = attr.getValue();
      } else {
        return rewriter.notifyMatchFailure(fill_op,
                                           "fill value must be int or float");
      }
      if (fill_bits.getBitWidth() > 64) {
        return rewriter.notifyMatchFailure(
            fill_op, "fill value must not exceed 64 bits");
      }
      fill_bits = fill_bits.zext(64U);
    }

    // Derive output vector type from the output operand (memref or tensor).
    mlir::Value out_operand = fill_op.getOutputs()[0];
    mlir::VectorType out_vec_type = getFlattenedVectorType(
        llvm::cast<mlir::ShapedType>(out_operand.getType()));
    if (!out_vec_type) {
      return rewriter.notifyMatchFailure(
          fill_op, "output must convert to a flattened vector type");
    }

    mlir::Location loc = fill_op.getLoc();
    int64_t total_elements = out_vec_type.getNumElements();
    mlir::Type elem_type = out_vec_type.getElementType();

    rewriter.setInsertionPoint(fill_op);

    // Step 1: vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
    mlir::VectorType seed_type = mlir::VectorType::get({1}, elem_type);
    mlir::ArrayAttr value_attr = rewriter.getArrayAttr(
        {is_constant
             ? mlir::IntegerAttr::get(rewriter.getI64Type(), fill_bits)
             : mlir::IntegerAttr::get(rewriter.getI64Type(), symbol_id)});
    auto bitstream = mlir::vectorchain::ConstantBitstreamOp::create(
        rewriter, loc, seed_type, value_attr);
    if (!is_constant) {
      bitstream->setAttr("is_symbol", rewriter.getBoolAttr(true));
    }

    // Step 2: vectorchain.shuffle — splat to vector<NxT>
    mlir::ArrayAttr indices_attr = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getI32Type(), 0)});
    auto shuffle = mlir::vectorchain::ShuffleOp::create(
        rewriter, loc, out_vec_type, bitstream.getResult(),
        /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
        /*mask=*/nullptr, /*dbgName=*/nullptr, indices_attr,
        static_cast<uint32_t>(total_elements));

    // Step 3a (tensor): replace the fill result directly with the vector.
    if (!fill_op.getResultTensors().empty()) {
      rewriter.replaceOp(fill_op, shuffle.getOutput());
      return mlir::success();
    }

    // Step 3b (memref): write the filled vector into the output memref.
    scheduler::emitVectorStore(rewriter, loc, shuffle.getOutput(), out_operand);

    rewriter.eraseOp(fill_op);
    return mlir::success();
  }

 private:
  scheduler::SymbolAllocator& symbols_;
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns, SymbolAllocator& symbols) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext());
  patterns.add<LowerLinalgFillPattern>(patterns.getContext(), symbols);
}
