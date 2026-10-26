//===-- SplitReductionInnerOuterDim.cpp -------------------------*- c++ -*-===//
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
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "split-reduction-inner-outer-dim"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_SPLITREDUCTIONINNEROUTERDIMPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Split Reduction Inner/Outer Dim pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// A reduction dim expressed in loop-space together with its size in the input
// tensor type.
// ---------------------------------------------------------------------------
struct ReductionDimInfo {
  unsigned loop_dim;  // index into the iterator-types list
  int64_t size;       // static size of this dim in the input tensor
};

// ---------------------------------------------------------------------------
// Per-candidate analysis result.
// ---------------------------------------------------------------------------
struct CandidateInfo {
  linalg::GenericOp generic_op;
  // Inner dim: exactly one — the rightmost reduction dim in input order.
  SmallVector<ReductionDimInfo> inner_dim;
  // Outer dims: all remaining (leftward) reduction dims, same ordering.
  SmallVector<ReductionDimInfo> outer_dims;
};

// ---------------------------------------------------------------------------
// Collect all reduction dims of `generic_op` ordered by their position in the
// first input tensor (leftmost first), together with their static sizes.
// Returns failure (and emits a debug note) if any reduction dim has a dynamic
// size or a non-trivial (non-AffineDimExpr) map result.
// ---------------------------------------------------------------------------
static LogicalResult collectReductionDims(
    linalg::GenericOp generic_op, SmallVectorImpl<ReductionDimInfo>& dims) {
  auto iter_types = generic_op.getIteratorTypesArray();
  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  ArrayRef<int64_t> shape = input_type.getShape();

  // Iterate input dims left-to-right so the result list is in input order.
  for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(d));
    if (!dim_expr) continue;
    unsigned loop_dim = dim_expr.getPosition();
    if (iter_types[loop_dim] != utils::IteratorType::reduction) continue;

    int64_t sz = shape[d];
    if (sz == ShapedType::kDynamic) {
      LDBG(1) << PASS_NAME ": dynamic reduction dim size — skipping";
      return failure();
    }
    dims.push_back({loop_dim, sz});
  }
  return success();
}

// ---------------------------------------------------------------------------
// Return true when `generic_op` qualifies for the split:
//   1. It has at least two reduction iterator types (one inner + at least
//      one outer).
//   2. The rightmost input dimension maps to a reduction loop dim.
// ---------------------------------------------------------------------------
static bool isEligible(linalg::GenericOp generic_op) {
  auto iter_types = generic_op.getIteratorTypesArray();

  // Condition 1: count reduction dims.
  int num_reductions = 0;
  for (auto it : iter_types)
    if (it == utils::IteratorType::reduction) ++num_reductions;
  if (num_reductions < 2) return false;

  // Condition 2 guarded: skip output-only generics (empty ins()) which do
  // not have an input tensor to inspect; they are handled by
  // ReductionLoopExposure's output-only loop path.
  if (generic_op.getInputs().empty()) return false;

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  unsigned last = input_map.getNumResults() - 1;
  auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(last));
  if (!dim_expr) return false;

  // FIXME: Current implementation leads to invalid IR when processing
  //        contractions, so skip them for now to allow matmul to pass.
  if (linalg::isaContractionOpInterface(generic_op)) {
    LDBG() << "ignoring contraction";
    return false;
  }

  return iter_types[dim_expr.getPosition()] == utils::IteratorType::reduction;
}

// ---------------------------------------------------------------------------
// Partition the reduction dims of `generic_op` into:
//   inner dim  — exactly the rightmost reduction dim (in input order).
//   outer dims — all remaining (leftward) reduction dims.
// ---------------------------------------------------------------------------
static FailureOr<CandidateInfo> partitionReductionDims(
    linalg::GenericOp generic_op, int64_t vector_length) {
  CandidateInfo info;
  info.generic_op = generic_op;

  SmallVector<ReductionDimInfo> all_dims;
  if (failed(collectReductionDims(generic_op, all_dims))) return failure();

  // The rightmost reduction dim (last in all_dims) is the inner dim.
  assert(!all_dims.empty());
  ReductionDimInfo& inner_dim = all_dims.back();
  assert(inner_dim.size == vector_length &&
         "inner (rightmost) reduction dim size must equal vector length");

  // all_dims[0 .. N-2] are outer dims; all_dims[N-1] is the inner dim.
  for (int i = 0; i < static_cast<int>(all_dims.size()) - 1; ++i)
    info.outer_dims.push_back(all_dims[i]);
  info.inner_dim.push_back(inner_dim);

  LDBG(1) << PASS_NAME ": generic at " << generic_op.getLoc()
          << " vector_length=" << vector_length
          << " inner dim (loop_dim:size):";
  for (auto& d : info.inner_dim)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;
  LDBG(1) << " outer dims (loop_dim:size):";
  for (auto& d : info.outer_dims)
    LDBG(1) << "  [" << d.loop_dim << "]=" << d.size;

  return info;
}

// ---------------------------------------------------------------------------
// Clone the body region of `src` into a freshly-created (but not yet emitted)
// GenericOp `dst`.  The cloneInto helper prepends an empty placeholder block
// that must be removed after the copy.
// ---------------------------------------------------------------------------
static void cloneGenericBody(linalg::GenericOp src, linalg::GenericOp dst) {
  IRMapping mapping;
  src.getRegion().cloneInto(&dst.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it.
  Block& placeholder = dst.getRegion().front();
  if (&placeholder != &dst.getRegion().back()) placeholder.erase();
}

// ---------------------------------------------------------------------------
// Give `g2` the body that accumulates what `src` partially reduced.
//
// Not a copy of `src`'s body. That body works out each element's contribution
// as well as accumulating it -- the square, where the reduction is of squares
// -- and G1 has done that part already. What is left per result is the op that
// accumulated it, over the partial result and the running one.
//
// Fails where a result is not accumulated by one op over the value running
// through it, which is the only shape there is anything to say about.
// ---------------------------------------------------------------------------
static LogicalResult buildCombinerBody(linalg::GenericOp src,
                                       linalg::GenericOp g2) {
  Block& src_body = src.getRegion().front();
  auto src_yield = cast<linalg::YieldOp>(src_body.getTerminator());
  const unsigned results = static_cast<unsigned>(src.getNumDpsInits());
  const unsigned src_inputs = static_cast<unsigned>(src.getNumDpsInputs());

  OpBuilder builder(g2.getContext());
  Block* body = builder.createBlock(&g2.getRegion());

  // A partial result in, the running one out, per result.
  for (unsigned r = 0; r < results; ++r) {
    body->addArgument(src_body.getArgument(src_inputs + r).getType(),
                      g2.getLoc());
  }
  for (unsigned r = 0; r < results; ++r) {
    body->addArgument(src_body.getArgument(src_inputs + r).getType(),
                      g2.getLoc());
  }

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> accumulated;
  for (unsigned r = 0; r < results; ++r) {
    Operation* accumulates = src_yield.getOperand(r).getDefiningOp();
    BlockArgument running = src_body.getArgument(src_inputs + r);
    if (!accumulates || accumulates->getNumOperands() != 2) {
      return src.emitError("result ")
             << r << " is not accumulated by an op over two values";
    }

    // The running value can reach the accumulation through one op rather than
    // being an operand of it -- the abs of an absmax is over each side. So what
    // stands in for a side is that op rebuilt over the argument where there is
    // one, and the argument itself where there is not.
    auto reaches = [&](Value side) {
      if (side == running) return true;
      Operation* through = side.getDefiningOp();
      return through && through->getNumOperands() == 1 &&
             through->getOperand(0) == running;
    };
    auto standIn = [&](Value side, Value argument) -> Value {
      Operation* through = side.getDefiningOp();
      if (!through || through->getNumOperands() != 1) return argument;
      Operation* rebuilt = through->clone();
      rebuilt->setOperand(0, argument);
      builder.insert(rebuilt);
      return rebuilt->getResult(0);
    };

    Value first = accumulates->getOperand(0);
    Value second = accumulates->getOperand(1);
    const bool running_first = reaches(first);
    if (!running_first && !reaches(second)) {
      return src.emitError("result ")
             << r << " is not accumulated over the value running through it";
    }

    Value partial = body->getArgument(r);
    Value carried = body->getArgument(results + r);

    Operation* combines = accumulates->clone();
    combines->setOperand(
        0, running_first ? standIn(first, carried) : standIn(first, partial));
    combines->setOperand(
        1, running_first ? standIn(second, partial) : standIn(second, carried));
    builder.insert(combines);
    accumulated.push_back(combines->getResult(0));
  }

  linalg::YieldOp::create(builder, g2.getLoc(), accumulated);
  return success();
}

// ---------------------------------------------------------------------------
// Split a reduction linalg.generic into two:
//
//   Generic 1 (outer reduction): reduces the outer dims.
//     Input:  original input tensor
//     Output: intermediate tensor — input dims with outer reduction dims
//             removed; the inner dim is kept as parallel.
//     Iterator types: outer dims → reduction, inner dim → parallel,
//                     other input-map dims unchanged.
//
//   Generic 2 (inner reduction): reduces the inner dim.
//     Input:  intermediate tensor (result of Generic 1)
//     Output: same type as the original output
//     Iterator types: inner dim → reduction, all other dims → parallel.
//
// The original generic is replaced by the result of Generic 2 and erased.
// ---------------------------------------------------------------------------
static LogicalResult splitDim(CandidateInfo& info) {
  linalg::GenericOp generic_op = info.generic_op;
  LDBG(1) << PASS_NAME ": splitDim on generic at " << generic_op.getLoc();

  MLIRContext* ctx = generic_op.getContext();
  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ── Collect structural info ──────────────────────────────────────────────
  SmallVector<utils::IteratorType> orig_iter_types =
      generic_op.getIteratorTypesArray();
  unsigned N = static_cast<unsigned>(orig_iter_types.size());

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  AffineMap output_map = generic_op.getIndexingMapsArray().back();

  auto input_type =
      cast<RankedTensorType>(generic_op.getInputs().front().getType());
  auto output_type =
      cast<RankedTensorType>(generic_op.getOutputs().front().getType());
  ArrayRef<int64_t> input_shape = input_type.getShape();
  Type elem_type = output_type.getElementType();

  // Build sets of outer and inner loop dim indices for quick lookup.
  llvm::SmallDenseSet<unsigned> inner_loop_dims;
  llvm::SmallDenseSet<unsigned> outer_loop_dims;
  for (auto& d : info.inner_dim) inner_loop_dims.insert(d.loop_dim);
  for (auto& d : info.outer_dims) outer_loop_dims.insert(d.loop_dim);

  // ── Build G1 ─────────────────────────────────────────────────────────────
  // G1 loops over exactly the dims that appear in the input map, in order
  // (g1_num_dims = input map rank).  Any dims that appear only in the output
  // map are not part of G1's loop space — they belong entirely to G2.
  // orig_to_g1_dim: original loop dim index → G1 loop dim index.
  unsigned g1_num_dims = input_map.getNumResults();
  llvm::SmallDenseMap<unsigned, unsigned> orig_to_g1_dim;
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    orig_to_g1_dim[orig_ld] = r;
  }

  // G1 input map: identity over G1's loop dims.
  SmallVector<AffineExpr> g1_in_exprs;
  for (unsigned r = 0; r < g1_num_dims; ++r)
    g1_in_exprs.push_back(getAffineDimExpr(r, ctx));
  AffineMap g1_in_map = AffineMap::get(g1_num_dims, 0, g1_in_exprs, ctx);

  // G1 output map and intermediate shape: walk the input map left-to-right,
  // skipping outer reduction dims (which G1 reduces away).  Each surviving
  // dim contributes one result to G1's output map and one dim to the
  // intermediate tensor.
  SmallVector<int64_t> inter_shape;
  SmallVector<AffineExpr> g1_out_exprs;
  llvm::SmallDenseMap<unsigned, unsigned> orig_ld_to_inter_slot;
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    if (outer_loop_dims.count(orig_ld)) continue;
    orig_ld_to_inter_slot[orig_ld] = static_cast<unsigned>(inter_shape.size());
    inter_shape.push_back(input_shape[r]);
    g1_out_exprs.push_back(getAffineDimExpr(orig_to_g1_dim[orig_ld], ctx));
  }
  AffineMap g1_out_map = AffineMap::get(g1_num_dims, 0, g1_out_exprs, ctx);

  // G1 iterator types: outer dims → reduction, inner dim → parallel,
  // other input-map dims → same as original.
  SmallVector<utils::IteratorType> g1_iter_types(g1_num_dims);
  for (unsigned r = 0; r < g1_num_dims; ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(input_map.getResult(r)).getPosition();
    if (outer_loop_dims.count(orig_ld))
      g1_iter_types[r] = utils::IteratorType::reduction;
    else if (inner_loop_dims.count(orig_ld))
      g1_iter_types[r] = utils::IteratorType::parallel;
    else
      g1_iter_types[r] = orig_iter_types[orig_ld];
  }

  // ── Build G2 ─────────────────────────────────────────────────────────────
  // G2 loops over all original dims except the outer reduction dims, in
  // original order.  This includes the inner dim (now a reduction) and any
  // output-only dims.
  // orig_to_g2_dim: original loop dim index → G2 loop dim index (0..M-1).
  llvm::SmallDenseMap<unsigned, unsigned> orig_to_g2_dim;
  unsigned g2_idx = 0;
  for (unsigned i = 0; i < N; ++i) {
    if (!outer_loop_dims.count(i)) orig_to_g2_dim[i] = g2_idx++;
  }
  unsigned M = g2_idx;

  // G2 input map: maps G2's loop dims to the intermediate tensor slots.
  unsigned inter_rank = static_cast<unsigned>(inter_shape.size());
  SmallVector<AffineExpr> g2_in_exprs(inter_rank);
  for (auto& [orig_ld, slot] : orig_ld_to_inter_slot)
    g2_in_exprs[slot] = getAffineDimExpr(orig_to_g2_dim[orig_ld], ctx);
  AffineMap g2_in_map = AffineMap::get(M, 0, g2_in_exprs, ctx);

  // G2 output map: remap the original output map into G2's loop dim space.
  SmallVector<AffineExpr> g2_out_exprs;
  for (unsigned r = 0; r < output_map.getNumResults(); ++r) {
    unsigned orig_ld =
        cast<AffineDimExpr>(output_map.getResult(r)).getPosition();
    assert(orig_to_g2_dim.count(orig_ld) &&
           "output dim maps to an outer loop dim — unexpected");
    g2_out_exprs.push_back(getAffineDimExpr(orig_to_g2_dim[orig_ld], ctx));
  }
  AffineMap g2_out_map = AffineMap::get(M, 0, g2_out_exprs, ctx);

  // G2 iterator types: inner dim → reduction, all other non-outer dims →
  // parallel.
  SmallVector<utils::IteratorType> g2_iter_types(M);
  for (unsigned i = 0; i < N; ++i) {
    if (outer_loop_dims.count(i)) continue;
    g2_iter_types[orig_to_g2_dim[i]] = inner_loop_dims.count(i)
                                           ? utils::IteratorType::reduction
                                           : utils::IteratorType::parallel;
  }

  // ── Emit the initialisers and the two generics ───────────────────────────
  // Still two generics; it is the intermediates, the output initialisers and
  // the results of each that there is one of per result. A compute may
  // accumulate a pair -- more than one thing over the same input -- and
  // then both generics carry both halves.
  const unsigned results = static_cast<unsigned>(generic_op.getNumResults());
  auto inter_tensor_type = RankedTensorType::get(inter_shape, elem_type);

  SmallVector<Type> inter_types(results, inter_tensor_type);
  SmallVector<Value> inter_inits;
  for (unsigned r = 0; r < results; ++r) {
    inter_inits.push_back(
        tensor::EmptyOp::create(builder, loc, inter_shape, elem_type)
            .getResult());
  }

  SmallVector<AffineMap> g1_maps(generic_op.getNumDpsInputs(), g1_in_map);
  g1_maps.append(results, g1_out_map);
  auto g1 = linalg::GenericOp::create(builder, loc,
                                      /*resultTensorTypes=*/inter_types,
                                      /*inputs=*/generic_op.getInputs(),
                                      /*outputs=*/inter_inits, g1_maps,
                                      g1_iter_types);
  cloneGenericBody(generic_op, g1);

  // If any user of the original generic_op result lives inside a scf.if
  // (i.e. the combine-if from ReductionDimChunking), insert G2 after that
  // scf.if and feed it the scf.if result instead of G1's result.
  // Otherwise G2 is inserted immediately after G1 and fed G1's result.
  scf::IfOp combine_if;
  for (Operation* user : generic_op.getResult(0).getUsers()) {
    if (auto if_op = dyn_cast<scf::IfOp>(user->getParentOp())) {
      combine_if = if_op;
      break;
    }
  }

  SmallVector<Value> g2_inputs;
  if (combine_if) {
    builder.setInsertionPointAfter(combine_if);
    for (unsigned r = 0; r < results; ++r)
      g2_inputs.push_back(combine_if.getResult(r));
  } else {
    for (unsigned r = 0; r < results; ++r) g2_inputs.push_back(g1.getResult(r));
  }

  SmallVector<Type> out_types(results, output_type);
  SmallVector<Value> out_inits;
  for (unsigned r = 0; r < results; ++r) {
    out_inits.push_back(
        tensor::EmptyOp::create(builder, loc, output_type.getShape(), elem_type)
            .getResult());
  }

  SmallVector<AffineMap> g2_maps(results, g2_in_map);
  g2_maps.append(results, g2_out_map);
  auto g2 = linalg::GenericOp::create(builder, loc,
                                      /*resultTensorTypes=*/out_types,
                                      /*inputs=*/g2_inputs, out_inits, g2_maps,
                                      g2_iter_types);
  if (failed(buildCombinerBody(generic_op, g2))) return failure();

  // ── Replace and erase original generic and its now-dead output inits ─────
  // Transformation:
  //   before: generic_op  [combine_if]  write_to_fifo
  //   after:  g1          [combine_if]  g2  write_to_fifo
  //
  // With combine_if: replace generic_op results with g1 results, then replace
  // combine_if results (outside g2) with g2 results (so write_to_fifo sees g2).
  // Without combine_if: replace generic_op results with g2 results directly.
  for (unsigned r = 0; r < results; ++r) {
    if (combine_if) {
      generic_op.getResult(r).replaceAllUsesWith(g1.getResult(r));
      combine_if.getResult(r).replaceUsesWithIf(
          g2.getResult(r),
          [&](OpOperand& use) { return use.getOwner() != g2.getOperation(); });
    } else {
      generic_op.getResult(r).replaceAllUsesWith(g2.getResult(r));
    }
  }
  SmallVector<Value> orig_inits(generic_op.getOutputs());
  generic_op.erase();
  for (Value init : orig_inits) {
    if (!init.use_empty()) continue;
    if (auto* def = init.getDefiningOp()) def->erase();
  }
  return success();
}

struct SplitReductionInnerOuterDimPass
    : public mlir::ktdf::impl::SplitReductionInnerOuterDimPassBase<
          SplitReductionInnerOuterDimPass> {
  using SplitReductionInnerOuterDimPassBase<
      SplitReductionInnerOuterDimPass>::SplitReductionInnerOuterDimPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();

    // Obtain the default device, emitting a diagnostic on failure.
    const auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
    if (!default_device) {
      signalPassFailure();
      return;
    }
    mlir::ktdf_arch::Mapping mapping(default_device.getRef());
    if (!mapping.byKind().getDefaultCompute()) {
      mapping.getDevice().getDeclaration().emitError(
          "no (unambiguous) default compute resource");
      signalPassFailure();
      return;
    }

    // Collect eligible linalg.generic ops: at least two reduction iterator
    // types, with one reduction dim mapping to the rightmost non-1 input dim.
    SmallVector<linalg::GenericOp> candidates;
    module.walk([&](linalg::GenericOp generic_op) {
      if (isEligible(generic_op)) candidates.push_back(generic_op);
      return WalkResult::advance();
    });

    if (candidates.empty()) {
      LDBG(1) << PASS_NAME ": no eligible linalg.generic found — skipping";
      return;
    }

    for (linalg::GenericOp generic_op : candidates) {
      auto compute = mapping.getOrMap<mlir::ktdf_arch::ExecutionUnitOp>(
          generic_op, mapping.byKind().getDefaultCompute().getKind());
      if (!compute) {
        generic_op.emitError(
            "cannot determine the hardware vector width: no compute resource "
            "is mapped");
        signalPassFailure();
        return;
      }

      const auto simd_feature =
          compute.getFeature<mlir::ktdf_arch::feature::SIMD>();

      // Derive the element type from the first output (the accumulator).
      auto output_type =
          dyn_cast<ShapedType>(generic_op.getOutputs().front().getType());
      assert(output_type);
      Type elem_type = output_type.getElementType();

      const int64_t vector_length =
          std::max(simd_feature.getLanes(elem_type), int64_t(1));

      FailureOr<CandidateInfo> info =
          partitionReductionDims(generic_op, vector_length);
      if (failed(info)) continue;
      if (failed(splitDim(*info))) return signalPassFailure();
    }
  }
};

}  // namespace

auto mlir::ktdf::createSplitReductionInnerOuterDimPass()
    -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<SplitReductionInnerOuterDimPass>();
}
