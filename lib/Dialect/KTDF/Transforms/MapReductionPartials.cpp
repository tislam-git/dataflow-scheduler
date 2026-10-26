//===-- MapReductionPartials.cpp --------------------------------*- c++ -*-===//
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
// MapReductionPartials: lower reduction linalg.generic ops to buffer-semantics
// form backed by pre-allocated memrefs.  Two kinds of generics are handled:
//
// ── Outer-dim generic (loop-exposed by ReductionLoopExposure) ────────────────
//
// After ReductionLoopExposure the compute stage looks like:
//
//   %empty = tensor.empty() : tensor<CxDxf16>
//   %result = scf.for %r = 0 to R iter_args(%carry = %empty)
//                 {loop_type = reduction_loop}
//     %slice = ktdf.read_from_fifo %fifo -> tensor<CxDxf16>
//     %new_carry = linalg.generic ins(%slice) outs(%carry)
//     scf.if (%r == R-1): ktdf.write_to_fifo %new_carry, %fifo_out
//     scf.yield %new_carry
//
// This pass rewrites that to:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   linalg.fill(%zero, %alloc)
//   scf.for %r = 0 to R {
//     %slice = ktdf.read_from_fifo %fifo -> memref<CxDxf16>
//     linalg.generic ins(%slice) outs(%alloc)
//     scf.if (%r == R-1): ktdf.write_to_fifo %alloc, %fifo_out
//   }
//
// When the iter_arg initializer is a conditional (scf.if), the pass recurses
// into each branch and handles the yielded value:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   scf.if %cond {                     // no result; was -> (tensor<...>)
//     linalg.fill(%zero, %alloc)       // was: tensor.empty + yield
//   } else {
//     %r = ktdf.read_from_fifo ...     // memref form of read_from_fifo
//     memref.copy %r, %alloc
//   }
//   scf.for %r = 0 to R { ... }       // iter_arg stripped
//
// The iter_arg / loop result / scf.yield operand are stripped, and any
// write_to_fifo that consumed the loop result is patched to use %alloc.
//
// ── Inner-dim generic (produced by SplitReductionInnerOuterDim) ──────────────
//
// After the outer-dim rewrite above, %alloc_G1 (memref<D0x...xDNxf16, ms>)
// has been RAUW'd in place of the loop result, so the inner-dim generic sees:
//
//   %empty = tensor.empty() : tensor<D0x...xDN-1xf16>
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty: tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// A rank-reducing subview of %alloc_G1 (size=1 on reduction dims, original
// size on parallel dims) is used as the outs buffer; ins is the full %alloc_G1.
// After the reduction, the full %alloc_G1 — not the subview — is written to
// the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type is widened to match %alloc_G1's shape.
//
//   %sv = memref.subview %alloc_G1[0,...][D0,...,1,...][1,...]
//             : memref<D0x...xDNxf16, ms> to memref<D0x...xDN-1xf16, strided,
//             ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)  {original maps & iter_types}
//   ktdf.write_to_fifo %alloc_G1, %fifo_out  // full buffer
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "map-reduction-partials"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Map Reduction Partials pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_MAPREDUCTIONPARTIALSPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// ---------------------------------------------------------------------------
// Return true if `generic_op` has at least one reduction iterator.
// ---------------------------------------------------------------------------
static bool hasReductionIterator(linalg::GenericOp generic_op) {
  for (auto it : generic_op.getIteratorTypesArray())
    if (it == utils::IteratorType::reduction) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Return the typed neutral-element attribute for the reduction combiner of
// `generic_op`.
//
// The combiner is identified as the unique user of the output block argument
// (outs[0], the last block argument) — that is the accumulation step.  Any
// other ops in the body (e.g. element-wise pre-processing in a fused generic)
// are irrelevant, except for the abs-max pattern described below.
//
// Neutral elements per combiner:
//   addf / subf                          →  0.0
//   mulf                                 →  1.0
//   maximumf                             → -inf
//   minimumf                             → +inf
//   maxnumf(absf(%in), absf(%out))       →  0.0   (abs-max: result is >= 0)
//   addi / subi                          →  0
//   muli                                 →  1
//
// Returns failure() (with an emitted error) if the output block argument has
// zero or multiple users, the combiner is unrecognised, or the element type is
// unsupported.
// ---------------------------------------------------------------------------
static FailureOr<TypedAttr> getNeutralAttr(linalg::GenericOp generic_op,
                                           Type elem_type) {
  // The output block argument is the last argument of the body block.
  Value out_arg = generic_op.getRegion().front().getArguments().back();

  // The combiner is the unique op that uses out_arg as an operand.
  Operation* combiner =
      out_arg.hasOneUse() ? out_arg.getUses().begin()->getOwner() : nullptr;
  if (!combiner)
    return generic_op.emitError(
        "MapReductionPartials: output block argument of reduction "
        "linalg.generic must have exactly one user (the accumulation op)");

  // ── Abs-max pattern: maxnumf(absf(%in), absf(%out)) ───────────────────────
  // When the direct user of out_arg is math.absf (pre-processing the
  // accumulator before combining), check whether that absf feeds a
  // maxnumf whose other operand is also an absf.  Both inputs are
  // non-negative after the abs, so the correct neutral element is 0.0.
  bool is_absmax = false;
  if (isa<math::AbsFOp>(combiner) && combiner->hasOneUse()) {
    Operation* max_op = combiner->getUses().begin()->getOwner();
    if (isa<arith::MaxNumFOp>(max_op)) {
      Value abs_out_result = combiner->getResult(0);
      for (Value operand : max_op->getOperands()) {
        if (operand == abs_out_result) continue;
        if (operand.getDefiningOp<math::AbsFOp>()) {
          is_absmax = true;
          break;
        }
      }
    }
  }

  // ── Floating-point combiners ───────────────────────────────────────────────
  if (auto ftype = dyn_cast<FloatType>(elem_type)) {
    const llvm::fltSemantics& sem = ftype.getFloatSemantics();
    if (is_absmax)
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getZero(sem, /*negative=*/false)));
    if (isa<arith::AddFOp, arith::SubFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getZero(sem, /*negative=*/false)));
    if (isa<arith::MulFOp>(combiner))
      return cast<TypedAttr>(FloatAttr::get(elem_type, APFloat(sem, 1)));
    if (isa<arith::MaximumFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getInf(sem, /*negative=*/true)));
    if (isa<arith::MinimumFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getInf(sem, /*negative=*/false)));
    return combiner->emitError(
        "MapReductionPartials: unsupported floating-point reduction combiner");
  }

  // ── Integer combiners ──────────────────────────────────────────────────────
  if (auto itype = dyn_cast<IntegerType>(elem_type)) {
    unsigned width = itype.getWidth();
    if (isa<arith::AddIOp, arith::SubIOp>(combiner))
      return cast<TypedAttr>(IntegerAttr::get(elem_type, APInt(width, 0)));
    if (isa<arith::MulIOp>(combiner))
      return cast<TypedAttr>(IntegerAttr::get(elem_type, APInt(width, 1)));
    return combiner->emitError(
        "MapReductionPartials: unsupported integer reduction combiner");
  }

  return generic_op.emitError(
      "MapReductionPartials: unsupported accumulator element type — "
      "expected float or integer");
}

// ---------------------------------------------------------------------------
// Return the number of SIMD lanes the compute unit of `stage` has for
// `elem_type` -- the width of one register of the accumulator -- or 0 when the
// device does not say.
// ---------------------------------------------------------------------------
static int64_t getStageLanes(
    ktdf::StageOp stage, Type elem_type,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  const auto units = stage ? stage.getApplicableUnits() : std::nullopt;
  if (!units || units->size() != 1) return 0;
  auto kind = dyn_cast<mlir::ktdf_arch::KindAttr>(units->getValue().front());
  if (!kind) return 0;
  auto simd = resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(kind);
  return simd ? simd.getLanes(elem_type) : 0;
}

// ---------------------------------------------------------------------------
// Return the shape of a contiguous tile of `lanes` elements that `shape`
// divides into evenly.  The tile takes whole innermost dimensions until the
// next one only partly fits; that one is split, and the outer ones are 1:
//
//   shape = [1, 4, 64], lanes = 64  →  [1, 1, 64]
//   shape = [2, 8, 16], lanes = 64  →  [1, 4, 16]
//
// Fails when a dimension neither divides nor is divided by what remains of
// the tile, or when `shape` holds fewer than `lanes` elements.
// ---------------------------------------------------------------------------
static FailureOr<SmallVector<int64_t>> getLaneTileShape(
    ArrayRef<int64_t> shape, int64_t lanes) {
  SmallVector<int64_t> tile(shape.size(), 1);
  int64_t remaining = lanes;
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1;
       d >= 0 && remaining > 1; --d) {
    if (remaining % shape[d] == 0) {
      tile[d] = shape[d];
      remaining /= shape[d];
    } else if (shape[d] % remaining == 0) {
      tile[d] = remaining;
      remaining = 1;
    } else {
      return failure();
    }
  }
  if (remaining != 1) return failure();
  return tile;
}

// ---------------------------------------------------------------------------
// Give the accumulator `alloc_val` the value `init_val`.
//
// The value goes to a buffer of its own and the accumulator is copied from
// it; the accumulator is never filled directly.  A buffer written once and
// never read back is invariant, so what hoists such a write can lift it to
// where a register is initialised instead -- which carries the whole value.
//
// That buffer stays live for as long as the accumulator does, so it is only
// ever one register (`lanes` elements) wide.  When the device declares no lanes
// for the element type the register width is unknown, and the buffer is as
// wide as the accumulator:
//
//   accumulator of at most `lanes` elements → one memref.copy:
//     %init = memref.alloc() : memref<1x64xf16, ms>
//     linalg.fill ins(%cst) outs(%init)
//     memref.copy %init, %alloc
//
//   wider accumulator → the buffer is broadcast, one copy per lane-sized tile:
//     %init = memref.alloc() : memref<1x1x64xf16, ms>
//     linalg.fill ins(%cst) outs(%init)
//     %plain = memref.memory_space_cast %alloc : ... to memref<1x4x64xf16>
//     %sv_i = memref.subview %plain[0, i, 0] [1, 1, 64] [1, 1, 1]
//     memref.copy %init, %sv_i                     // for i in 0..3
// ---------------------------------------------------------------------------
static LogicalResult materializeInitialValue(OpBuilder& builder, Location loc,
                                             Value init_val, Value alloc_val,
                                             int64_t lanes) {
  auto alloc_type = cast<MemRefType>(alloc_val.getType());
  if (!alloc_type.hasStaticShape())
    return emitError(loc,
                     "MapReductionPartials: accumulator must have a static "
                     "shape to be initialised");
  const int64_t elements = alloc_type.getNumElements();

  // ── Copy: the accumulator fits in one register ────────────────────────────
  if (lanes <= 0 || elements <= lanes) {
    auto init = memref::AllocOp::create(builder, loc, alloc_type);
    linalg::FillOp::create(builder, loc, ValueRange{init_val},
                           ValueRange{init.getResult()});
    memref::CopyOp::create(builder, loc, init.getResult(), alloc_val);
    return success();
  }

  // ── Broadcast: one register copied into each lane-sized tile ──────────────
  ArrayRef<int64_t> shape = alloc_type.getShape();
  FailureOr<SmallVector<int64_t>> tile = getLaneTileShape(shape, lanes);
  if (failed(tile))
    return emitError(loc, "MapReductionPartials: accumulator ")
           << alloc_type << " does not divide into contiguous tiles of "
           << lanes << " lanes to broadcast its initial value over";

  auto init_type =
      MemRefType::get(*tile, alloc_type.getElementType(),
                      MemRefLayoutAttrInterface{}, alloc_type.getMemorySpace());
  auto init = memref::AllocOp::create(builder, loc, init_type);
  linalg::FillOp::create(builder, loc, ValueRange{init_val},
                         ValueRange{init.getResult()});

  // KTDFToKTDFLoweringPass replaces the alloc with a view that carries no
  // memory space, and memref.subview requires its source and result to agree
  // on one, so subview a cast of the alloc (as rewriteOutputOnlyGeneric does).
  Value plain = alloc_val;
  if (alloc_type.getMemorySpace())
    plain = memref::MemorySpaceCastOp::create(
                builder, loc,
                MemRefType::get(shape, alloc_type.getElementType(),
                                alloc_type.getLayout()),
                alloc_val)
                .getResult();

  const int64_t rank = alloc_type.getRank();
  SmallVector<int64_t> tiles_per_dim(rank);
  for (int64_t d = 0; d < rank; ++d) tiles_per_dim[d] = shape[d] / (*tile)[d];
  const SmallVector<int64_t> strides(rank, 1);

  for (int64_t t = 0, e = elements / lanes; t < e; ++t) {
    SmallVector<int64_t> offsets(rank);
    int64_t rest = t;
    for (int64_t d = rank - 1; d >= 0; --d) {
      offsets[d] = (rest % tiles_per_dim[d]) * (*tile)[d];
      rest /= tiles_per_dim[d];
    }
    auto subview =
        memref::SubViewOp::create(builder, loc, plain, offsets, *tile, strides);
    memref::CopyOp::create(builder, loc, init.getResult(),
                           subview.getResult());
  }
  return success();
}

// ---------------------------------------------------------------------------
// Transform the initializer `init_val` of a loop-carried accumulator in-place,
// emitting linalg.fill (or memref.copy) into `alloc_val` wherever the original
// tensor value was produced.  `alloc_val` is a memref.alloc already emitted at
// the top of the stage.
//
// `generic_op` is the reduction linalg.generic whose combiner determines the
// correct neutral element for the fill.
//
// Three leaf cases are handled; scf.if recurses into both branches:
//
//   tensor.empty()
//     → linalg.fill(%neutral, %alloc) inserted just before the tensor.empty,
//       then the tensor.empty is erased.
//
//   ktdf.read_from_fifo ... -> tensor<...>
//     → emit memref-typed read_from_fifo + memref.copy into %alloc just
//       before the original op, then erase it.
//
//   scf.if %cond -> (tensor<...>) { yield A } else { yield B }
//     → recurse on A (then-branch yield operand 0) and B (else-branch yield
//       operand 0), drop the yield operands so both yields become result-less,
//       then rebuild the scf.if with no result types.
// ---------------------------------------------------------------------------
static LogicalResult lowerIterArgInitializer(
    Value init_val, Value alloc_val, linalg::GenericOp generic_op,
    OpBuilder& builder,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  Operation* defining_op = init_val.getDefiningOp();
  assert(defining_op &&
         "lowerIterArgInitializer: init_val must be an op result");

  Type elem_type = cast<MemRefType>(alloc_val.getType()).getElementType();

  // The width of the register an initial value is held in.
  const int64_t lanes = getStageLanes(
      generic_op->getParentOfType<ktdf::StageOp>(), elem_type, resource_kinds);

  // ── Case: tensor.empty ────────────────────────────────────────────────────
  if (auto empty_op = dyn_cast<tensor::EmptyOp>(defining_op)) {
    auto neutral = getNeutralAttr(generic_op, elem_type);
    if (failed(neutral)) return failure();
    Location loc = empty_op.getLoc();
    Value neutral_val =
        arith::ConstantOp::create(builder, loc, neutral.value());

    if (failed(materializeInitialValue(builder, loc, neutral_val, alloc_val,
                                       lanes)))
      return failure();

    if (empty_op->use_empty()) empty_op->erase();
    return success();
  }

  // ── Case: arith.constant splat tensor ─────────────────────────────────────
  // The user-specified initial value is kept (not the combiner's neutral
  // element). Materialise it the same way as the tensor.empty case.
  if (auto const_op = dyn_cast<arith::ConstantOp>(defining_op)) {
    auto dense = dyn_cast<DenseElementsAttr>(const_op.getValue());
    if (!dense || !dense.isSplat() ||
        dense.getElementType() != elem_type)
      return const_op->emitError(
          "lowerIterArgInitializer: arith.constant initializer must be a "
          "splat tensor whose element type matches the accumulator");
    Location loc = const_op.getLoc();
    Value splat_val = arith::ConstantOp::create(
        builder, loc, cast<TypedAttr>(dense.getSplatValue<Attribute>()));
    if (failed(materializeInitialValue(builder, loc, splat_val, alloc_val,
                                       lanes)))
      return failure();

    // The constant may be shared; only erase it when nothing else uses it.
    if (const_op->use_empty()) const_op->erase();
    return success();
  }

  // ── Case: ktdf.read_from_fifo returning a tensor ──────────────────────────
  if (auto read_op = dyn_cast<ktdf::ReadFromFifoOp>(defining_op)) {
    OpBuilder read_builder(read_op);
    Value new_read = ktdf::convertFromTensorToMemref(read_builder, read_op);
    memref::CopyOp::create(read_builder, read_op.getLoc(), new_read, alloc_val);
    read_op->erase();
    return success();
  }

  // ── Case: scf.if returning a tensor — recurse into both branches ──────────
  if (auto if_op = dyn_cast<scf::IfOp>(defining_op)) {
    // Drop the yield operand *before* recursing so that when the recursive
    // call erases the defining op (e.g. tensor.empty) no uses remain.
    {
      auto then_yield =
          cast<scf::YieldOp>(if_op.getThenRegion().front().getTerminator());
      Value then_init = then_yield.getOperand(0);
      then_yield->setOperands({});
      OpBuilder then_builder(then_yield);
      if (failed(lowerIterArgInitializer(then_init, alloc_val, generic_op,
                                         then_builder, resource_kinds)))
        return failure();
    }

    {
      auto else_yield =
          cast<scf::YieldOp>(if_op.getElseRegion().front().getTerminator());
      Value else_init = else_yield.getOperand(0);
      else_yield->setOperands({});
      OpBuilder else_builder(else_yield);
      if (failed(lowerIterArgInitializer(else_init, alloc_val, generic_op,
                                         else_builder, resource_kinds)))
        return failure();
    }

    // Rebuild the scf.if without result types and transfer the rewritten
    // regions.
    OpBuilder builder(if_op);
    auto new_if =
        scf::IfOp::create(builder, if_op.getLoc(),
                          /*resultTypes=*/TypeRange{}, if_op.getCondition(),
                          /*withElseRegion=*/true);
    new_if.getThenRegion().takeBody(if_op.getThenRegion());
    new_if.getElseRegion().takeBody(if_op.getElseRegion());
    if_op.erase();
    return success();
  }

  return defining_op->emitError(
      "lowerIterArgInitializer: unrecognised initializer form — expected "
      "tensor.empty, ktdf.read_from_fifo, or scf.if");
}

// ---------------------------------------------------------------------------
// Walk up the iter_arg chain rooted at `acc_iter_arg` (outs[0] of the
// original linalg.generic, captured before erasure) and rebuild each
// enclosing scf.for without that iter_arg slot, replacing every use of
// the dropped iter_arg and its loop result with `alloc_val`.
//
// After the innermost loop is rebuilt the old loop result may itself be an
// iter_arg of an outer scf.for — we keep climbing until the chain exits a
// scf.for.
//
// The initializer at the top of the chain has already been handled by
// lowerIterArgInitializer before this function is called, so it is NOT
// erased here.
//
// Cannot remove an iter_arg in-place: getInitArgsMutable().erase() only drops
// the operand but leaves the region block argument and loop result intact,
// producing an inconsistent op that fails the verifier.  We clone instead.
// ---------------------------------------------------------------------------
static void removeUnusedIterArgChain(ArrayRef<BlockArgument> acc_iter_args,
                                     ArrayRef<Value> alloc_vals) {
  // `current` starts as the innermost iter_args and advances to their inits
  // at each level.
  SmallVector<Value> current(acc_iter_args.begin(), acc_iter_args.end());
  SmallVector<Value> current_allocs(alloc_vals.begin(), alloc_vals.end());

  while (true) {
    // Filter to only those elements that are BlockArguments of an scf.for loop.
    SmallVector<BlockArgument> block_args;
    SmallVector<Value> active_allocs;
    for (auto [val, alloc] : llvm::zip(current, current_allocs)) {
      if (auto ba = dyn_cast<BlockArgument>(val)) {
        if (isa<scf::ForOp>(ba.getOwner()->getParentOp())) {
          block_args.push_back(ba);
          active_allocs.push_back(alloc);
        }
      }
    }
    if (block_args.empty()) break;

    auto for_op =
        cast<scf::ForOp>(block_args.front().getOwner()->getParentOp());

    llvm::SmallDenseSet<unsigned> dropped;
    SmallVector<Value> next;
    SmallVector<Value> next_allocs;

    for (auto [k, iter_arg] : llvm::enumerate(block_args)) {
      if (iter_arg.getOwner()->getParentOp() != for_op.getOperation())
        continue;

      // Block arg 0 of an scf.for is the induction variable; iter_args start
      // at index 1, so subtract 1 to get the iter_arg slot index.
      const unsigned iter_idx = iter_arg.getArgNumber() - 1;
      dropped.insert(iter_idx);

      // The init is what this accumulator advances to next.
      next.push_back(for_op.getInits()[iter_idx]);
      next_allocs.push_back(active_allocs[k]);

      // Replace all uses of the dropped iter_arg and its loop result.
      for_op.getResult(iter_idx).replaceAllUsesWith(active_allocs[k]);
      iter_arg.replaceAllUsesWith(active_allocs[k]);
    }

    // Rebuild the loop without the dropped iter_args.
    OpBuilder builder(for_op);
    Location loc = for_op.getLoc();

    SmallVector<Value> new_inits;
    for (auto [i, init] : llvm::enumerate(for_op.getInits()))
      if (!dropped.contains(i)) new_inits.push_back(init);

    auto new_for =
        scf::ForOp::create(builder, loc, for_op.getLowerBound(),
                           for_op.getUpperBound(), for_op.getStep(), new_inits);

    for (auto attr : for_op->getAttrs())
      if (attr.getName() != "operandSegmentSizes")
        new_for->setAttr(attr.getName(), attr.getValue());

    // Map old to new iter args, skipping the dropped ones.
    IRMapping body_map;
    body_map.map(for_op.getInductionVar(), new_for.getInductionVar());
    unsigned new_arg_idx = 0;
    for (auto [i, arg] : llvm::enumerate(for_op.getRegionIterArgs()))
      if (!dropped.contains(i))
        body_map.map(arg, new_for.getRegionIterArgs()[new_arg_idx++]);

    // Ensure the new_for body is empty (clearing any default yield added by
    // scf::ForOp::build when iterArgs is empty) before cloning into it.
    auto* old_yield = for_op.getBody()->getTerminator();
    new_for.getBody()->clear();

    OpBuilder body_builder(new_for.getBody(), new_for.getBody()->end());
    for (auto& op : for_op.getBody()->without_terminator())
      body_builder.clone(op, body_map);

    SmallVector<Value> new_yield_operands;
    for (auto [i, operand] : llvm::enumerate(old_yield->getOperands()))
      if (!dropped.contains(i))
        new_yield_operands.push_back(body_map.lookupOrDefault(operand));
    scf::YieldOp::create(body_builder, loc, new_yield_operands);

    unsigned new_res_idx = 0;
    for (auto [i, res] : llvm::enumerate(for_op.getResults()))
      if (!dropped.contains(i))
        res.replaceAllUsesWith(new_for.getResult(new_res_idx++));

    // Advance to the next parent loop in the iter arg chain.
    current = next;
    current_allocs = next_allocs;
    for_op.erase();
  }
}

// ---------------------------------------------------------------------------
// Lower a linalg.generic that was cloned by ReductionLoopExposure inside an
// output-only loop nest.
//
// The generic lives inside an scf.for tagged {ktdf.output_only_loop} and has
// all-outs() operands (no ins()):
//
//   outs[0]         = ktdf.read_from_fifo result (FIFO data slice)
//   outs[1..ns-1]   = tensor.extract_slice of FIFO-loop carry (non-spanning)
//   outs[ns..N-1]   = tensor.extract_slice of output-only iter_arg (spanning)
//
// After this rewrite:
//   - Memref allocs are placed at stage top for every non-FIFO outs.
//   - Non-spanning allocs are initialised from the FIFO-loop outermost seed.
//   - Spanning allocs are initialised from their outermost seed.
//   - The extract_slice / insert_slice pairs are replaced with
//     memref.subview reads / in-place writes.
//   - The generic is rebuilt as buffer-semantics (0 results).
//   - Unused iter_args are stripped from both the output-only and FIFO loops
//     via removeUnusedIterArgChain.
// ---------------------------------------------------------------------------
static LogicalResult rewriteOutputOnlyGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space) return failure();

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // --------------------------------------------------------------------------
  // Step 1: Identify the output-only loop this generic lives in.
  // --------------------------------------------------------------------------
  auto out_only_loop = generic_op->getParentOfType<scf::ForOp>();
  assert(out_only_loop && out_only_loop->hasAttr("ktdf.output_only_loop") &&
         "rewriteOutputOnlyGeneric: generic must be inside output-only loop");

  // --------------------------------------------------------------------------
  // Step 2: Locate the FIFO data outs operand.
  // --------------------------------------------------------------------------
  std::optional<unsigned> fifo_idx = ktdf::findFifoInOperandIdx(generic_op);
  if (!fifo_idx)
    return generic_op.emitError(
        "rewriteOutputOnlyGeneric: no FIFO data operand found");
  Value orig_fifo_operand = ktdf::getFifoInOperand(generic_op, *fifo_idx);
  if (auto ext = orig_fifo_operand.getDefiningOp<tensor::ExtractSliceOp>())
    orig_fifo_operand = ext.getSource();
  auto orig_read = orig_fifo_operand.getDefiningOp<ktdf::ReadFromFifoOp>();
  assert(orig_read && "FIFO operand must come from a read_from_fifo");

  // Convert the tensor read to a memref read at the location of orig_read
  // (outside the output-only loop), so the FIFO slice is read once per outer
  // iteration and shared across all inner k-iterations.
  OpBuilder read_builder(orig_read);
  Value new_fifo_read = ktdf::convertFromTensorToMemref(read_builder, orig_read);

  // --------------------------------------------------------------------------
  // Step 3: For each non-FIFO outs slot, detect spanning vs non-spanning.
  //
  //   Spanning: the out value is a tensor.extract_slice whose source is a
  //             BlockArgument of the output-only loop (iterating over k).
  //   Non-spanning: the out value is either a FIFO-loop BlockArgument directly
  //             (after canonicalization folds away identity extract_slices) or
  //             an identity tensor.extract_slice of a FIFO-loop BA.
  //
  // In both cases we record:
  //   - the out_idx in the generic
  //   - the source tensor value (before any extract_slice, i.e. the BA itself)
  //   - the extract_slice op (nullptr for the direct-BA non-spanning case)
  //   - whether it is spanning
  // --------------------------------------------------------------------------
  const unsigned num_outs = generic_op.getNumDpsInits();

  struct OutSlot {
    unsigned out_idx;
    Value source_val;              // BA feeding the outs (spanning: output-only
                                   // BA; non-spanning: FIFO BA)
    tensor::ExtractSliceOp ext_op; // the extract_slice, if any (may be null)
    bool is_spanning;
    Value alloc;                   // filled in step 4
  };

  SmallVector<OutSlot> out_slots;
  for (unsigned r = 0; r < num_outs; ++r) {
    if (r == *fifo_idx) continue;
    Value out_val = generic_op.getDpsInitOperand(r)->get();

    // Case A: extract_slice (spanning or pre-canonicalization non-spanning).
    if (auto ext = out_val.getDefiningOp<tensor::ExtractSliceOp>()) {
      Value src = ext.getSource();
      bool is_spanning = false;
      if (auto ba = dyn_cast<BlockArgument>(src))
        if (isa<scf::ForOp>(ba.getOwner()->getParentOp()) &&
            ba.getOwner()->getParentOp() == out_only_loop.getOperation())
          is_spanning = true;
      out_slots.push_back({r, src, ext, is_spanning, {}});
      continue;
    }

    // Case B: direct BlockArgument (non-spanning after canonicalization).
    if (auto ba = dyn_cast<BlockArgument>(out_val)) {
      out_slots.push_back({r, out_val, {}, /*is_spanning=*/false, {}});
      continue;
    }

    return generic_op.emitError(
        "rewriteOutputOnlyGeneric: non-FIFO outs must be extract_slice or "
        "BlockArgument");
  }

  // --------------------------------------------------------------------------
  // Step 4: Allocate memrefs at stage top for every non-FIFO slot.
  //
  //   Spanning:     alloc matches the extract_slice *source* tensor shape
  //                 (the full k-dim accumulator, e.g. tensor<1x4x64xf16>).
  //   Non-spanning: alloc matches the out value's tensor shape
  //                 (e.g. tensor<1x64xf16>).
  // --------------------------------------------------------------------------
  builder.setInsertionPointToStart(stage.getBody());
  for (auto& slot : out_slots) {
    RankedTensorType src_tensor_type;
    if (slot.is_spanning)
      src_tensor_type = cast<RankedTensorType>(slot.source_val.getType());
    else
      src_tensor_type = cast<RankedTensorType>(
          generic_op.getDpsInitOperand(slot.out_idx)->get().getType());
    auto alloc_type =
        MemRefType::get(src_tensor_type.getShape(),
                        src_tensor_type.getElementType(),
                        MemRefLayoutAttrInterface{}, mem_space);
    slot.alloc = memref::AllocOp::create(builder, loc, alloc_type).getResult();
  }

  // --------------------------------------------------------------------------
  // Step 5: Initialise allocs from their outermost seeds (walk the BA chain).
  // --------------------------------------------------------------------------
  for (auto& slot : out_slots) {
    // Walk up through scf.for iter_arg chain to the non-BA seed.
    Value cursor = slot.source_val;
    Value outermost_init;
    while (auto ba = dyn_cast<BlockArgument>(cursor)) {
      auto for_op = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp());
      if (!for_op) break;
      outermost_init = for_op.getInits()[ba.getArgNumber() - 1];
      cursor = outermost_init;
    }
    Value seed = outermost_init ? outermost_init : cursor;
    OpBuilder init_builder(
        seed.getDefiningOp()
            ? (stage->isProperAncestor(seed.getDefiningOp())
                   ? OpBuilder(seed.getDefiningOp())
                   : builder)
            : builder);
    if (failed(lowerIterArgInitializer(seed, slot.alloc, generic_op,
                                       init_builder, resource_kinds)))
      return failure();
  }

  // --------------------------------------------------------------------------
  // Step 6: Build replacement memref operands for the buffer generic.
  //         Spanning → memref.subview at the k-slot offset from ext_op.
  //         Non-spanning → the full alloc.
  // --------------------------------------------------------------------------
  builder.setInsertionPoint(generic_op);
  SmallVector<Value> new_outs(num_outs);
  new_outs[*fifo_idx] = new_fifo_read;
  for (auto& slot : out_slots) {
    if (slot.is_spanning) {
      auto ext = slot.ext_op;
      auto src_memref_type = cast<MemRefType>(slot.alloc.getType());
      // KTDFToKTDFLoweringPass replaces the alloc with a
      // dataflow.get_logical_memory_view that returns a memref WITHOUT a memory
      // space annotation (see LogicalMemoryViewBuilder.cpp plain_type).  The
      // memref.subview verifier requires source and result to carry the same
      // memory space, so cast the alloc to a no-space type first; the subview
      // result type is then also built without a memory space.
      auto src_no_space_type =
          MemRefType::get(src_memref_type.getShape(),
                          src_memref_type.getElementType(),
                          src_memref_type.getLayout());
      Value src_no_space_val =
          src_memref_type.getMemorySpace()
              ? memref::MemorySpaceCastOp::create(builder, loc,
                                                  src_no_space_type,
                                                  slot.alloc)
                    .getResult()
              : slot.alloc;
      auto sv_result_type =
          cast<MemRefType>(memref::SubViewOp::inferResultType(
              src_no_space_type, ext.getMixedOffsets(), ext.getMixedSizes(),
              ext.getMixedStrides()));
      auto subview = memref::SubViewOp::create(
          builder, loc, sv_result_type, src_no_space_val,
          ext.getMixedOffsets(), ext.getMixedSizes(), ext.getMixedStrides());
      new_outs[slot.out_idx] = subview.getResult();
    } else {
      new_outs[slot.out_idx] = slot.alloc;
    }
  }

  // --------------------------------------------------------------------------
  // Step 7: Clone the generic in buffer mode (0 results).
  // --------------------------------------------------------------------------
  ktdf::cloneLinalgGenericAsBufferOp(builder, generic_op, /*inputs=*/{},
                                     new_outs);

  // --------------------------------------------------------------------------
  // Step 8: Replace generic tensor results and patch downstream users.
  //
  //   Spanning results:
  //     - tensor.insert_slice users → replaced by the dest (the output-only
  //       loop BA), so the loop's scf.yield still sees a tensor value; once
  //       removeUnusedIterArgChain strips that iter_arg the yield slot also
  //       disappears.
  //     - write_to_fifo users → retargeted to the spanning alloc (memref).
  //     - Remaining uses of the result → replaced by the spanning alloc.
  //   Non-spanning results:
  //     - All uses replaced by the non-spanning alloc (memref).
  //   FIFO data result → replaced by new_fifo_read (memref).
  // --------------------------------------------------------------------------

  // Spanning: collect insert_slices before erasing them.
  SmallVector<tensor::InsertSliceOp> inserts_to_erase;
  for (auto& slot : out_slots) {
    if (!slot.is_spanning) continue;
    Value span_result = generic_op.getResult(slot.out_idx);
    for (Operation* user : llvm::make_early_inc_range(span_result.getUsers()))
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user))
        inserts_to_erase.push_back(ins);
  }
  // Replace insert_slice results with their dest (the output-only BA), so the
  // output-only yield slot still type-checks until the iter_arg is stripped.
  for (auto ins : inserts_to_erase) {
    ins.getResult().replaceAllUsesWith(ins.getDest());
    ins.erase();
  }

  // Non-spanning: replace generic results with their allocs.
  for (auto& slot : out_slots) {
    if (slot.is_spanning) continue;
    generic_op.getResult(slot.out_idx).replaceAllUsesWith(slot.alloc);
  }

  // FIFO data result.
  const unsigned num_ins = generic_op.getNumDpsInputs();
  if (*fifo_idx >= num_ins) {
    Value fifo_result = generic_op.getResult(*fifo_idx - num_ins);
    for (Operation* user : llvm::make_early_inc_range(fifo_result.getUsers())) {
      if (auto ins = dyn_cast<tensor::InsertSliceOp>(user)) {
        ins.getResult().replaceAllUsesWith(ins.getDest());
        ins.erase();
      }
    }
    fifo_result.replaceAllUsesWith(new_fifo_read);
  }

  // Spanning: patch write_to_fifo and clear remaining uses.
  for (auto& slot : out_slots) {
    if (!slot.is_spanning) continue;
    Value span_result = generic_op.getResult(slot.out_idx);
    for (Operation* user : llvm::make_early_inc_range(span_result.getUsers()))
      if (auto w = dyn_cast<ktdf::WriteToFifoOp>(user))
        w.getDataMutable().assign(slot.alloc);
    span_result.replaceAllUsesWith(slot.alloc);
  }

  // Erase the tensor generic (all results replaced).
  generic_op.erase();

  // Retarget any remaining uses of the old tensor read (e.g. in out_only_loop inits)
  // to new_fifo_read and erase orig_read before loop rebuilding.
  if (orig_read) {
    for (OpOperand& use :
         llvm::make_early_inc_range(orig_read.getResult().getUses()))
      use.set(new_fifo_read);
    orig_read->erase();
  }

  // --------------------------------------------------------------------------
  // Step 9: Strip unused iter_args from both the output-only loop and the
  //         enclosing FIFO loop via removeUnusedIterArgChain.
  //
  //   All iter_args of the output-only loop (including any FIFO iter_args created
  //   by tileLinalgOp) are passed in ONE call so that the loop is rebuilt
  //   exactly once without iter_args. Walking the chain further (from each
  //   init) rebuilds the FIFO loop with all dropped slots.
  // --------------------------------------------------------------------------

  // Erase spanning extract_slice ops (must be done before RAUW of their source
  // BAs so they don't see alloc operands during the rebuild).
  for (auto& slot : out_slots) {
    if (slot.ext_op && slot.ext_op->use_empty())
      slot.ext_op.erase();
  }

  // Collect ALL output-only loop iter_args in iter_arg order.
  SmallVector<BlockArgument> all_out_only_bas;
  SmallVector<Value> all_out_only_allocs;

  for (auto [ba, init] : llvm::zip(out_only_loop.getRegionIterArgs(),
                                   out_only_loop.getInits())) {
    Value alloc_val = nullptr;
    for (auto& slot : out_slots) {
      if (slot.is_spanning && slot.source_val == Value(ba)) {
        alloc_val = slot.alloc;
        break;
      }
      if (!slot.is_spanning && (init == slot.source_val)) {
        alloc_val = slot.alloc;
        break;
      }
    }
    // If not matching a non-FIFO out_slot (e.g. the FIFO data iter_arg created by
    // tileLinalgOp), use new_fifo_read as the replacement value.
    if (!alloc_val)
      alloc_val = new_fifo_read;

    all_out_only_bas.push_back(ba);
    all_out_only_allocs.push_back(alloc_val);
  }

  // Single call strips all output-only iter_arg slots and walks up to strip
  // the corresponding FIFO iter_arg slots.
  if (!all_out_only_bas.empty())
    removeUnusedIterArgChain(all_out_only_bas, all_out_only_allocs);

  return success();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Lower a single reduction linalg.generic to buffer semantics.
// `generic_op` must have at least one reduction iterator and its input must
// be a ktdf.read_from_fifo.
// ---------------------------------------------------------------------------
static LogicalResult rewriteGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  // Step 1: decide which memory kind backs the accumulator.
  Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space) return failure();

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // One accumulator per loop-carried output: a compute that accumulates more
  // than one thing has one of each, and each is buffered on its own.  The
  // outputs that are not iter_args (e.g. the FIFO data operand of a generic with
  // no ins()) are not accumulators and are left alone.
  const SmallVector<unsigned> acc_out_idx =
      ktdf::getIterArgOutputIndices(generic_op);
  const unsigned accumulators = static_cast<unsigned>(acc_out_idx.size());

  // Step 2: allocate an accumulator memref per result at the top of the stage.
  builder.setInsertionPointToStart(stage.getBody());
  SmallVector<Value> allocs;
  for (unsigned r = 0; r < accumulators; ++r) {
    auto out_tensor_type = cast<RankedTensorType>(
        generic_op.getDpsInitOperand(acc_out_idx[r])->get().getType());
    auto alloc_type = MemRefType::get(out_tensor_type.getShape(),
                                      out_tensor_type.getElementType(),
                                      MemRefLayoutAttrInterface{}, mem_space);
    allocs.push_back(
        memref::AllocOp::create(builder, loc, alloc_type).getResult());
  }

  // Step 3: capture the iter_args and walk up to find the outermost initializer
  // of each, before anything is erased.
  SmallVector<BlockArgument> acc_iter_args;
  SmallVector<Value> outermost_inits;
  Operation* outermost_for = nullptr;
  for (unsigned r = 0; r < accumulators; ++r) {
    auto acc_iter_arg =
        cast<BlockArgument>(generic_op.getOutputs()[acc_out_idx[r]]);
    acc_iter_args.push_back(acc_iter_arg);

    Value outermost_init;
    Value cursor = acc_iter_arg;
    while (auto ba = dyn_cast<BlockArgument>(cursor)) {
      auto for_op = cast<scf::ForOp>(ba.getOwner()->getParentOp());
      assert(for_op);
      outermost_init = for_op.getInits()[ba.getArgNumber() - 1];
      outermost_for = for_op;
      cursor = outermost_init;
    }
    assert(outermost_init && "could not find iter_arg initializer");
    outermost_inits.push_back(outermost_init);
  }

  // Step 4: replace all uses of each outermost initializer (e.g. the scf.for
  // init operand) with its alloc *before* lowering it, so that when
  // lowerIterArgInitializer erases the defining op no uses remain.
  for (unsigned r = 0; r < accumulators; ++r) {
    outermost_inits[r].replaceAllUsesWith(allocs[r]);
  }

  // Step 4b: lower each initializer — emit linalg.fill (and/or memref.copy) in
  // the right place and clean up tensor ops.
  for (unsigned r = 0; r < accumulators; ++r) {
    OpBuilder init_builder =
        (outermost_inits[r].getDefiningOp() &&
         stage->isProperAncestor(outermost_inits[r].getDefiningOp()))
            ? OpBuilder(outermost_inits[r].getDefiningOp())
            : (outermost_for ? OpBuilder(outermost_for) : OpBuilder(generic_op));
    if (failed(lowerIterArgInitializer(outermost_inits[r], allocs[r],
                                       generic_op, init_builder,
                                       resource_kinds)))
      return failure();
  }

  // Step 5: emit a new ktdf.read_from_fifo with a memref result type so the
  // buffer-semantics linalg.generic below has a pure-buffer FIFO operand.  The
  // FIFO operand is ins[0], or — for a generic with no ins() — the outs()
  // operand produced by the read.
  builder.setInsertionPoint(generic_op);
  std::optional<unsigned> fifo_idx = ktdf::findFifoInOperandIdx(generic_op);
  if (!fifo_idx)
    return generic_op.emitError(
        "rewriteGeneric: no operand is fed by a ktdf.read_from_fifo");
  Value orig_input = ktdf::getFifoInOperand(generic_op, *fifo_idx);
  auto orig_read = orig_input.getDefiningOp<ktdf::ReadFromFifoOp>();
  assert(orig_read &&
         "rewriteGeneric: the FIFO operand must be a ktdf.read_from_fifo");
  Value new_read = ktdf::convertFromTensorToMemref(builder, orig_read);

  // Step 6: pure-buffer linalg.generic — memref operands, no result.  The new
  // read keeps the position of the operand it replaces: ins[0] when there are
  // ins, otherwise its slot among the outs, with the accumulator allocs in the
  // iter_arg slots.
  const unsigned num_ins = generic_op.getInputs().size();
  if (num_ins > 0) {
    ktdf::cloneLinalgGenericAsBufferOp(
        builder, generic_op, /*inputs=*/ValueRange{new_read},
        /*outputs=*/allocs);
  } else {
    SmallVector<Value> new_outs(generic_op.getOutputs());
    new_outs[*fifo_idx - num_ins] = new_read;
    for (auto [r, out_idx] : llvm::enumerate(acc_out_idx))
      new_outs[out_idx] = allocs[r];
    ktdf::cloneLinalgGenericAsBufferOp(builder, generic_op, /*inputs=*/{},
                                       /*outputs=*/new_outs);
  }

  // Step 7: replace generic result with alloc, patch write_to_fifo users,
  // then erase the generic and its now-dead tensor read_from_fifo input.
  for (unsigned r = 0; r < accumulators; ++r) {
    Value generic_result = generic_op.getResult(acc_out_idx[r]);
    for (Operation* user : generic_result.getUsers())
      if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user))
        write_op.getDataMutable().assign(allocs[r]);
    generic_result.replaceAllUsesWith(allocs[r]);
  }
  // The buffer generic writes the FIFO operand in place, so its result (if
  // anything reads it) is now the new read.
  if (*fifo_idx >= num_ins)
    generic_op.getResult(*fifo_idx - num_ins).replaceAllUsesWith(new_read);

  generic_op.erase();
  if (orig_input.use_empty())
    if (auto* orig_input_op = orig_input.getDefiningOp())
      orig_input_op->erase();

  // Step 8: walk up the iter_arg chain and rebuild each scf.for without it.
  // The initializer has already been transformed; no erasure is needed here.
  removeUnusedIterArgChain(acc_iter_args, allocs);
  return success();
}

// ---------------------------------------------------------------------------
// Find the "combine-with-partial" scf.if that ReductionDimChunking wrapped
// around `value`.  On the first chunk the IfOp passes `value` straight through,
// so `value` is an operand of the then-branch yield:
//
//   %combined:N = scf.if %is_first -> (tensor<...>, ...) {
//     scf.yield %value, ...        // first chunk: nothing to combine with
//   } else {
//     %p0, ... = ktdf.read_from_fifo ...
//     %g:N = linalg.generic(parallel) ins(%p0, ...) outs(%value, ...)
//     scf.yield %g#0, ...
//   }
//
// The IfOp is therefore never a direct user of `value` — the then-branch yield
// is.  Returns a null IfOp when `value` has no such user.
// ---------------------------------------------------------------------------
static scf::IfOp findCombineIf(Value value) {
  for (Operation* user : value.getUsers()) {
    auto yield = dyn_cast<scf::YieldOp>(user);
    if (!yield) continue;
    auto if_op = dyn_cast<scf::IfOp>(yield->getParentOp());
    if (!if_op) continue;
    if (yield->getParentRegion() != &if_op.getThenRegion()) continue;
    return if_op;
  }
  return {};
}

// ---------------------------------------------------------------------------
// The outermost scf.for enclosing `op` without leaving its ktdf.stage — the
// head of the reduction loop nest ReductionLoopExposure built, whose results
// carry the accumulators out of the nest.  Null when `op` is not in a loop.
// ---------------------------------------------------------------------------
static scf::ForOp outermostForInStage(Operation* op) {
  auto stage = op->getParentOfType<ktdf::StageOp>();
  scf::ForOp outermost;
  for (auto for_op = op->getParentOfType<scf::ForOp>(); for_op;
       for_op = for_op->getParentOfType<scf::ForOp>()) {
    if (for_op->getParentOfType<ktdf::StageOp>() != stage) break;
    outermost = for_op;
  }
  return outermost;
}

// ---------------------------------------------------------------------------
// Lower a "combine-with-partial" scf.if to a result-less scf.if that combines
// each previous chunk's partial straight into its accumulator destination.
//
// By the time this runs, the bypass rewrite has RAUW'd each result of the if
// to the then-branch yield operand at the same index, and the reduction rewrite
// has turned those yield operands into the memref(s) backing the
// accumulator(s). Each else-branch yield operand is a linalg.generic whose outs
// already point at the same memref.  We iterate over all results:
//
//   scf.if %is_first {
//     // nothing — each dest already holds this chunk's result
//   } else {
//     %p0 = ktdf.read_from_fifo %partial_slot_0 -> memref<...>
//     linalg.generic(parallel) ins(%p0) outs(%dest0)
//     // ... one per result ...
//   }
//
// The original scf.if is left in place; the caller erases it once nothing uses
// its results.
// ---------------------------------------------------------------------------
static LogicalResult lowerCombineIfToMemref(scf::IfOp if_op) {
  // ── Emit the lowered result-less scf.if ──────────────────────────────────
  OpBuilder builder(if_op);
  Location loc = if_op.getLoc();
  auto new_if =
      scf::IfOp::create(builder, loc, /*resultTypes=*/TypeRange{},
                        if_op.getCondition(), /*withElseRegion=*/true);

  // then-branch: empty — dest already holds the correct data.
  // (The default result-less scf.yield terminator is already present.)

  // else-branch: one buffer-semantics generic mirroring the combine generic.
  // The combine generic has N inputs (one partial read per accumulator) and N
  // outputs (one per result); we emit one memref read per input and collect all
  // dests from the then-branch yield at the matching indices.
  {
    Block& new_else = new_if.getElseRegion().front();
    OpBuilder else_b = OpBuilder::atBlockBegin(&new_else);

    Operation* then_yield = if_op.getThenRegion().front().getTerminator();
    auto else_yield =
        cast<scf::YieldOp>(if_op.getElseRegion().front().getTerminator());

    // All else-yield operands are results of the same combine generic.
    auto combine_generic =
        else_yield.getOperand(0).getDefiningOp<linalg::GenericOp>();
    if (!combine_generic)
      return if_op.emitError(
          "lowerCombineIfToMemref: else-branch yield is not a linalg.generic "
          "result");

    // Collect memref-typed partials (one per input of the combine generic).
    SmallVector<Value> partials;
    for (Value tensor_read : combine_generic.getInputs()) {
      auto read_op = tensor_read.getDefiningOp<ktdf::ReadFromFifoOp>();
      if (!read_op)
        return if_op.emitError(
            "lowerCombineIfToMemref: combine generic input is not a "
            "ktdf.read_from_fifo");
      auto tensor_type = cast<RankedTensorType>(tensor_read.getType());
      auto memref_type =
          MemRefType::get(tensor_type.getShape(), tensor_type.getElementType());
      partials.push_back(ktdf::ReadFromFifoOp::create(else_b, loc, memref_type,
                                                      read_op.getFifoSlot())
                             .getResult());
    }

    // Collect dests from the then-branch yield (one per result).
    SmallVector<Value> dests;
    for (Value then_operand : then_yield->getOperands()) {
      Value dest = then_operand;
      if (auto subview = dest.getDefiningOp<memref::SubViewOp>())
        dest = subview.getSource();
      if (!isa<MemRefType>(dest.getType()))
        return if_op.emitError(
            "lowerCombineIfToMemref: the combine destination is not a memref");
      dests.push_back(dest);
    }

    // Buffer-semantics generic: ins = partials, outs = dests, no result.
    auto buf_generic = linalg::GenericOp::create(
        else_b, loc, /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/partials, /*outputs=*/dests,
        combine_generic.getIndexingMapsAttr(),
        combine_generic.getIteratorTypesAttr(),
        /*doc=*/StringAttr{}, /*library_call=*/StringAttr{});
    IRMapping mapping;
    combine_generic.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
    Block& placeholder = buf_generic.getRegion().front();
    if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();
    // Terminator (result-less scf.yield) is already present in new_else.
  }

  return success();
}

// ---------------------------------------------------------------------------
// Lower a combine previously bypassed (results RAUW'd to then-branch yield
// operands), and erase the original.  No-op when there is no combine.
//
// The bypass rewrite turned each then-branch yield operand into the memref
// backing its accumulator, so lowerCombineIfToMemref reads destinations from
// there.  The combine is lowered in place, keeping its order relative to the
// rest of the stage.
//
// When an accumulator memref is a subview of a larger buffer — the subview an
// inner-dim reduction reduces into — the combine still runs at the buffer's
// full width, because the chunk pipeline sends and receives the buffer entire.
// ---------------------------------------------------------------------------
static LogicalResult lowerAndEraseCombineIf(scf::IfOp if_op) {
  if (!if_op) return success();
  if (failed(lowerCombineIfToMemref(if_op))) return failure();
  if_op.erase();
  return success();
}

// ---------------------------------------------------------------------------
// Set `new_type` on a PrivateOp result and its corresponding inner value
// (the private_yield operand at the same index).
//
// Always called with a direct PrivateOp result; the inner value is derived
// by indexing into private_yield.
//
// Example:
//   %2#2 = ktdf.private -> (!ktdf.fifo.slot<"A"->"B", 4xf16>, ...)
//     → widenPrivateResult(%2#2, !ktdf.fifo.slot<"A"->"B", 32xf16>)
//   %2#3 = ktdf.private -> (memref<4xf16, ms>, ...)
//     → widenPrivateResult(%2#3, memref<2x16xf16, ms>)
// ---------------------------------------------------------------------------
static void widenPrivateResult(Value priv_res, Type new_type) {
  auto priv_op = cast<ktdf::PrivateOp>(cast<OpResult>(priv_res).getOwner());
  auto result_idx = cast<OpResult>(priv_res).getResultNumber();
  Value inner = priv_op.getYieldOp().getOperand(result_idx);
  inner.setType(new_type);
  priv_res.setType(new_type);
}

// ---------------------------------------------------------------------------
// Iterate the direct uses of `fifo_slot` and for each data_transfer that
// references it update the FIFO-side static size field to reflect `new_shape`.
// ---------------------------------------------------------------------------
static void widenFifoUses(Value fifo_slot, ArrayRef<int64_t> new_shape,
                          MLIRContext* ctx) {
  auto new_sizes_attr = DenseI64ArrayAttr::get(ctx, new_shape);

  for (Operation* user : fifo_slot.getUsers()) {
    auto xfer = dyn_cast<ktdf::DataTransferOp>(user);
    if (!xfer) continue;

    // Only update the FIFO-side size on this transfer.  The alloc side (its
    // size, map, and type) is left entirely unchanged — the alloc's shape is
    // determined by its own allocation context and is shared with other
    // transfers (e.g. the store-out to global memory) that must not be
    // disturbed.
    if (xfer.isSourceFifo())
      xfer.setStaticSourceSizesAttr(new_sizes_attr);
    else if (xfer.isDestFifo())
      xfer.setStaticDestSizesAttr(new_sizes_attr);
  }
}

// ---------------------------------------------------------------------------
// Lower an inner-dim reduction linalg.generic to buffer semantics.
//
// Two sub-cases are handled:
//
// ── Case A: alloc_in is a local accumulator from rewriteGeneric ──────────
//
// At this point (after rewriteGeneric has run on G1) the stage contains:
//
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty:   tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// %alloc_G1 is already a memref (RAUW'd by rewriteGeneric).  We reuse it as
// both input and output buffer:
//
//   ins  = %alloc_G1 (full memref<D0x...xDNxf16>)
//   outs = a rank-reducing subview of %alloc_G1 with size=1 on all reduction
//          dims (rank-reduced away) and original size on parallel dims.  This
//          gives a strided view into alloc_G1 where reduced results are
//          written back in-place.
//
// After the inner reduction, the full %alloc_G1 buffer — not the subview —
// is sent to the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type and the paired memref.alloc in the ktdf.private block are
// widened to match %alloc_G1's shape.
//
// Example (memref<2x64xf16, ms>, reduction over dim 1):
//
//   %sv = memref.subview %alloc_G1[0, 0][2, 1][1, 1]
//             : memref<2x64xf16, ms> to memref<2xf16, strided<[64]>, ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)
//   ktdf.write_to_fifo %alloc_G1, %fifo_out    // full buffer, not subview
//
//   // ktdf.private widened (flat count 2 → 128, shape 2xf16 → 2x64xf16):
//   %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"A" -> "B", 128xf16>
//   %acc  = memref.alloc() : memref<2x64xf16, ct_local>
//
// ── Case B: alloc_in is a ktdf.read_from_fifo (no outer-dim loop) ────────
//
// When there is no outer-dim reduction loop, rewriteGeneric is never called.
// alloc_in is the memref result of a ktdf.read_from_fifo.
//
// We allocate ONE local buffer (local_reg), copy the FIFO data into it, and
// use it as BOTH ins and the subview source:
//
//   %local_reg = memref.alloc() : memref<D0x...xDNxf16, "REG">
//   memref.copy %fifo_read_memref, %local_reg
//   %sv = memref.subview %local_reg[0,...][..., 1, ...][...]
//   linalg.generic ins(%local_reg) outs(%sv)   // both refer to local_reg
//   ktdf.write_to_fifo %local_reg, %fifo_out
// ---------------------------------------------------------------------------
static LogicalResult rewriteInnerDimGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ins[0] is a memref — either the outer-dim alloc from rewriteGeneric, or a
  // memref-typed read_from_fifo emitted by the caller when no outer-dim loop
  // exists. One input per accumulator: a compute reducing more than one thing
  // has an intermediate of each, and each half reduces into its own buffer.
  // They are the same type, so one set of offsets and sizes describes every
  // subview.
  SmallVector<Value> allocs_in(generic_op.getInputs());
  Value alloc_in = allocs_in.front();
  auto in_memref_type = cast<MemRefType>(alloc_in.getType());
  unsigned in_rank = in_memref_type.getRank();
  if (llvm::any_of(allocs_in,
                   [&](Value in) { return in.getType() != in_memref_type; })) {
    return generic_op.emitError(
        "rewriteInnerDimGeneric: the inputs are not all the same memref type");
  }
  if (generic_op.getOutputs().size() != allocs_in.size()) {
    return generic_op.emitError(
        "rewriteInnerDimGeneric: expected an output per input");
  }

  // Inspect indexing maps and iterator types to classify each dim.
  auto iter_types = generic_op.getIteratorTypesArray();
  auto indexing_maps = generic_op.getIndexingMapsArray();
  AffineMap in_map = indexing_maps.front();
  AffineMap out_map = indexing_maps.back();
  ArrayRef<int64_t> in_shape = in_memref_type.getShape();

  // Build a loop-dim → input-dimension-size lookup from the input map so we
  // can resolve sizes for dims that appear in the output map but not in a
  // direct positional correspondence to in_shape.
  SmallVector<int64_t> loop_dim_to_size(iter_types.size(), 1);
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    assert(dim_expr && "inner-dim input map must be identity-like");
    loop_dim_to_size[dim_expr.getPosition()] = in_shape[d];
  }

  // SubViewOp requires offsets/sizes/strides arrays of length == source rank
  // (in_rank).  The rank reduction is expressed by placing
  // size=1 on reduction dimensions; inferRankReducedResultType then produces
  // the correctly strided result type with the reduced rank.
  unsigned out_rank = out_map.getNumResults();

  // Build source-rank-sized arrays by iterating over the input map: each
  // source dim d maps to loop dim loop_dim; if that loop dim is a reduction
  // iterator set size=1 (to be dropped), otherwise keep the full size.
  SmallVector<OpFoldResult> sv_offsets(in_rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sv_sizes_ofr(in_rank);
  SmallVector<OpFoldResult> sv_strides(in_rank, builder.getIndexAttr(1));
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    assert(dim_expr && "inner-dim input map must be identity-like");
    unsigned loop_dim = dim_expr.getPosition();
    int64_t sz = (iter_types[loop_dim] == utils::IteratorType::reduction)
                     ? 1
                     : loop_dim_to_size[loop_dim];
    sv_sizes_ofr[d] = builder.getIndexAttr(sz);
  }

  // Rank-reduced result shape: parallel dims only, in out_map order, so the
  // subview result rank matches the output indexing map.
  SmallVector<int64_t> sv_result_shape;
  for (unsigned d = 0; d < out_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(out_map.getResult(d));
    assert(dim_expr && "inner-dim output map must be identity-like");
    unsigned loop_dim = dim_expr.getPosition();
    if (iter_types[loop_dim] != utils::IteratorType::reduction)
      sv_result_shape.push_back(loop_dim_to_size[loop_dim]);
  }

  // Case B: alloc_in is a FIFO-read buffer (no outer-dim loop).
  //
  // ktdf.opaque requires a constant-address operand (read_from_fifo fails
  // that check).  Allocate a local register per accumulator, copy the FIFO data
  // into each, and use them as both ins and subview sources.
  //
  // One each rather than one shared: two accumulations cannot be reduced
  // straight out of the fifo, since each element's own contribution still has
  // to be worked out, so the second would overwrite the first.
  SmallVector<Value> effective(allocs_in);
  if (allocs_in.front().getDefiningOp<ktdf::ReadFromFifoOp>()) {
    Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
    if (!mem_space) return failure();

    auto alloc_type = MemRefType::get(in_shape, in_memref_type.getElementType(),
                                      MemRefLayoutAttrInterface{}, mem_space);

    for (auto [r, in] : llvm::enumerate(allocs_in)) {
      // Each alloc serves as both ins and subview source for its accumulator.
      builder.setInsertionPointToStart(stage.getBody());
      Value local_reg =
          memref::AllocOp::create(builder, loc, alloc_type).getResult();

      // Copy its FIFO data into it immediately before the generic.
      builder.setInsertionPoint(generic_op);
      memref::CopyOp::create(builder, loc, in, local_reg);
      effective[r] = local_reg;
    }
  }

  // Let SubViewOp infer the correct strided layout from the subview source.
  auto sv_result_type =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          sv_result_shape, cast<MemRefType>(effective.front().getType()),
          sv_offsets, sv_sizes_ofr, sv_strides));

  builder.setInsertionPoint(generic_op);
  SmallVector<Value> subviews;
  for (Value in : effective) {
    subviews.push_back(memref::SubViewOp::create(builder, loc, sv_result_type,
                                                 in, sv_offsets, sv_sizes_ofr,
                                                 sv_strides)
                           .getResult());
  }

  // Buffer linalg.generic: ins = the full buffers, outs = a rank-reduced
  // subview of each. In Case A those are the outer-dim accumulators; in Case B
  // they are the local registers copied from the fifo. Original maps and
  // iterator_types are preserved.
  ktdf::cloneLinalgGenericAsBufferOp(builder, generic_op, /*inputs=*/effective,
                                     /*outputs=*/subviews);

  // Patch each write_to_fifo to send the whole buffer its result reduced into
  // rather than the subview, and capture the slot it wrote to, whose type is
  // widened below.
  SmallVector<Value> old_fifo_slots;
  for (auto [r, in] : llvm::enumerate(effective)) {
    Value generic_result = generic_op.getResult(r);
    for (Operation* user :
         llvm::make_early_inc_range(generic_result.getUsers())) {
      if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user)) {
        old_fifo_slots.push_back(write_op.getFifoSlot());
        write_op.getDataMutable().assign(in);
      }
    }
    generic_result.replaceAllUsesWith(subviews[r]);
  }

  // Widen each FIFO slot type and all downstream allocs/transfers that use
  // it. The new element count is the product of all input dimensions.
  for (Value old_fifo_slot : old_fifo_slots) {
    auto old_slot_type = cast<ktdf::FifoSlotType>(old_fifo_slot.getType());
    Type elem_type = old_slot_type.getElementType();

    int64_t new_num_elems = 1;
    for (int64_t sz : in_shape) new_num_elems *= sz;

    auto new_slot_type = ktdf::FifoSlotType::get(
        generic_op.getContext(), old_slot_type.getSrc(),
        old_slot_type.getDest(), new_num_elems, elem_type);
    // Widen the PrivateOp result for the fifo slot (and its inner allocate).
    widenPrivateResult(old_fifo_slot, new_slot_type);

    // Update the FIFO-side size on all data_transfer uses of the fifo slot.
    // The paired alloc in ktdf.private is intentionally left unchanged.
    widenFifoUses(old_fifo_slot, in_shape, generic_op.getContext());
  }

  // Erase the original tensor.empty output initializers and the tensor
  // generic.
  SmallVector<Value> orig_out_inits(generic_op.getOutputs());
  generic_op.erase();
  for (Value init : orig_out_inits) {
    if (!init.use_empty()) continue;
    if (auto* def = init.getDefiningOp()) def->erase();
  }

  return success();
}

struct MapReductionPartialsPass
    : public ktdf::impl::MapReductionPartialsPassBase<
          MapReductionPartialsPass> {
  using MapReductionPartialsPassBase<
      MapReductionPartialsPass>::MapReductionPartialsPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError("Unable to import the device specification.");
      signalPassFailure();
      return;
    }
    auto& group_local_mem =
        device_manager.getOrCreateView<scheduler::arch_view::GroupLocalMemory>(
            *device);
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    // Collect generics in three buckets.
    //
    //   output_only  — cloned generics inside a {ktdf.output_only_loop} nest
    //                  (produced by ReductionLoopExposure for output-only dims).
    //                  Processed first: they may be nested inside the same
    //                  FIFO-driven scf.for that carries loop_exposed generics,
    //                  so their iter_args must be stripped before the outer
    //                  loop_exposed pass touches the loop.
    //   loop_exposed — outer-dim generics (outs are FIFO-loop iter_args).
    //                  Their loop results are RAUW'd to alloc memrefs before
    //                  inner_dim generics are processed.
    //   inner_dim    — inner-dim generics (produced by SplitReductionInnerOuter).
    SmallVector<linalg::GenericOp> output_only, loop_exposed, inner_dim;
    module.walk([&](linalg::GenericOp generic_op) {
      // FIXME: Current implementation leads to invalid IR when processing
      //        contractions, so skip them for now to allow matmul to pass.
      if (linalg::isaContractionOpInterface(generic_op)) {
        return;
      }
      if (!hasReductionIterator(generic_op)) return;
      if (generic_op.getOutputs().empty()) return;
      // Detect generics inside an output-only loop nest.
      bool inside_output_only = false;
      for (auto* parent = generic_op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (parent->hasAttr("ktdf.output_only_loop")) {
          inside_output_only = true;
          break;
        }
        if (isa<ktdf::StageOp>(parent)) break;
      }
      if (inside_output_only) {
        output_only.push_back(generic_op);
        return;
      }
      // Any output may be the loop-carried one: a generic with no ins() carries
      // its FIFO data in outs[0] and the accumulators after it.
      if (!ktdf::getIterArgOutputIndices(generic_op).empty())
        loop_exposed.push_back(generic_op);
      else
        inner_dim.push_back(generic_op);
    });

    // Process output-only generics first.
    for (auto generic_op : output_only) {
      if (failed(rewriteOutputOnlyGeneric(generic_op, group_local_mem,
                                    resource_kinds))) {
        signalPassFailure();
        return;
      }
    }

    for (auto generic_op : loop_exposed) {
      // Find the combine before rewriteGeneric rebuilds the loop nest.  It
      // hangs off the nest's result, not the generic's — the generic only
      // feeds the innermost yield.  This is where the chunk pipeline puts it:
      // after the outer-dim reduction and ahead of the inner-dim one, so an
      // inner-dim generic in the same stage consumes the combined accumulator.
      scf::IfOp combine_if;
      if (auto outermost_for = outermostForInStage(generic_op))
        combine_if = findCombineIf(outermost_for.getResult(0));
      if (combine_if) {
        Operation* then_yield =
            combine_if.getThenRegion().front().getTerminator();
        for (auto [res, operand] :
             llvm::zip(combine_if.getResults(), then_yield->getOperands()))
          res.replaceAllUsesWith(operand);
      }

      if (failed(rewriteGeneric(generic_op, group_local_mem,
                                    resource_kinds))) {
        signalPassFailure();
        return;
      }
      // By now the then-branch yields the accumulator memref (RAUW'd by
      // rewriteGeneric).
      if (failed(lowerAndEraseCombineIf(combine_if))) {
        signalPassFailure();
        return;
      }
    }
    for (auto generic_op : inner_dim) {
      // ins[0] is a tensor-typed ktdf.read_from_fifo when there was no
      // outer-dim loop feeding this generic (i.e. rewriteGeneric did not run
      // on its pipeline).  Emit a memref-typed read in its place so that
      // rewriteInnerDimGeneric can assume ins[0] is already a memref.
      Operation* stale_tensor_read = nullptr;
      std::optional<unsigned> fifo_idx = ktdf::findFifoInOperandIdx(generic_op);
      if (!fifo_idx || generic_op.getInputs().empty()) {
        // No ins() to reduce from: the inner-dim rewrite cannot handle it.
        generic_op.emitError(
            "MapReductionPartials: an inner-dim reduction generic needs an "
            "ins() operand");
        signalPassFailure();
        return;
      }
      if (auto read_op = ktdf::getFifoInOperand(generic_op, *fifo_idx)
                             .getDefiningOp<ktdf::ReadFromFifoOp>()) {
        OpBuilder builder(generic_op);
        stale_tensor_read = read_op;
        Value new_read = ktdf::convertFromTensorToMemref(builder, read_op);
        generic_op.getInputsMutable().assign(new_read);
      }
      // Find the combine before rewriteInnerDimGeneric erases generic_op.
      // With an outer-dim reduction in the stage the combine sits ahead of this
      // generic and the loop above already lowered it; one here belongs to a
      // stage whose only reduction is this inner-dim one, so there was no
      // accumulator to combine against any earlier.
      scf::IfOp combine_if = findCombineIf(generic_op.getResult(0));

      // Take it out of the reduction's dataflow up front, so
      // rewriteInnerDimGeneric sees the write_to_fifo downstream of it as a
      // user of the generic and retargets it onto the whole accumulator (and
      // widens its FIFO slot to match).
      if (combine_if) {
        Operation* then_yield =
            combine_if.getThenRegion().front().getTerminator();
        for (auto [res, operand] :
             llvm::zip(combine_if.getResults(), then_yield->getOperands()))
          res.replaceAllUsesWith(operand);
      }

      // Transform inner-dim generic into memref-typed generic.
      if (failed(rewriteInnerDimGeneric(generic_op, group_local_mem))) {
        signalPassFailure();
        return;
      }
      if (stale_tensor_read && stale_tensor_read->use_empty())
        stale_tensor_read->erase();
      // By now the then-branch yields the subview (RAUW'd by
      // rewriteInnerDimGeneric), which lowerAndEraseCombineIf resolves back to
      // the buffer it slices.
      if (failed(lowerAndEraseCombineIf(combine_if))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

auto mlir::ktdf::createMapReductionPartialsPass() -> std::unique_ptr<Pass> {
  return std::make_unique<MapReductionPartialsPass>();
}
