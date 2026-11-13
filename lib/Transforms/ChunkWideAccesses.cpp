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
// ChunkWideAccesses: split every FIFO access that moves more elements
// than the throttle into throttle-sized accesses enclosed in a loop nest that
// covers the original extent.
//
// The throttle is the number of SIMD lanes the default compute unit provides
// for the access's element type; it is the value KTIRMapAndTile records as
// `dataflow_scheduler.throttle` on the compute ops.
//
// Runs on ktdf, after the scheduler has decided on the FIFO slot sizes and
// before addresses are assigned. A FIFO slot wider than the throttle is
// lowered to one wide agen.vector_load / agen.vector_store plus a wide
// dataflow.send / dataflow.receive, which the hardware cannot issue. The
// slot's three kinds of access are handled here:
//
//   ktdf.write_to_fifo %buf, %slot                   (memref data)
//   ktdf.read_from_fifo %slot -> memref<...>
//   ktdf.data_transfer between %slot and a memref    (either direction)
//
// For a 1x4x64 access and a throttle of 64, the write
//
//   ktdf.write_to_fifo %buf, %slot : memref<1x4x64xf16>, <... 256xf16>
//
// becomes
//
//   scf.for %i = 0 to 4 {
//     %v = memref.subview %buf[0, %i, 0] [1, 1, 64] [1, 1, 1]
//     ktdf.write_to_fifo %v, %slot : memref<1x1x64xf16, ...>, <... 64xf16>
//   }
//
// and the slot itself is shrunk to the throttle. A FIFO delivers elements in
// the order they were written, and the pieces of an access, taken in loop
// order, are consecutive in row-major order of the access's dimensions. The
// sender and the receiver derive their chunking from that order alone, so both
// ends of a slot stay in step even when they are shaped differently.
//
// All the accesses to a slot are split together, as the slot's size has to
// match all of them. A slot that cannot be split this way (a user other than
// the three above, a tensor operand, a splat / pad transfer, dynamic sizes, an
// extent that does not tile by the throttle, or a slot that is not a result of
// `ktdf.private`) is reported as an error, as such an access is not legal on
// the hardware.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <memory>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "chunk-wide-accesses"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable the Throttle Wide Agen Accesses pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_CHUNKWIDEACCESSESPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// How a wide access is cut into throttle-sized pieces: the extent of one
/// piece in each dimension, and the loops (outermost first) that walk the
/// pieces.
struct Chunking {
  struct Loop {
    unsigned dim;    // memref dimension the loop advances
    int64_t extent;  // trip count
    int64_t stride;  // indices of `dim` advanced per iteration
  };
  llvm::SmallVector<int64_t> chunk_sizes;
  llvm::SmallVector<Loop> loops;
};

/// Cut an access of extents @p sizes into pieces of exactly @p throttle
/// elements that, in row-major order, are consecutive lanes of the access:
///   - the trailing dimensions whose extents multiply to @p throttle form one
///     piece, and every leading dimension is looped over; or
///   - the innermost non-unit dimension alone is a multiple of @p throttle,
///     and is looped over in steps of @p throttle.
/// Returns nullopt when neither applies.
std::optional<Chunking> planChunking(llvm::ArrayRef<int64_t> sizes,
                                     int64_t throttle) {
  Chunking plan;
  plan.chunk_sizes.assign(sizes.size(), 1);

  int64_t covered = 1;
  size_t first = sizes.size();
  while (first > 0 && covered < throttle) covered *= sizes[--first];

  auto loopLeading = [&](size_t end) {
    for (size_t i = 0; i < end; ++i)
      if (sizes[i] != 1) plan.loops.push_back({(unsigned)i, sizes[i], 1});
  };

  if (covered == throttle) {
    for (size_t i = first; i < sizes.size(); ++i)
      plan.chunk_sizes[i] = sizes[i];
    loopLeading(first);
    return plan;
  }

  // Overshot: the last dimension taken is the only non-unit one at or below it
  // and must itself hold a whole number of pieces.
  if (first < sizes.size() && covered / sizes[first] == 1 &&
      sizes[first] % throttle == 0) {
    plan.chunk_sizes[first] = throttle;
    loopLeading(first);
    plan.loops.push_back(
        {(unsigned)first, sizes[first] / throttle, /*stride=*/throttle});
    return plan;
  }
  return std::nullopt;
}

/// Emit the loop nest of @p plan at the rewriter's insertion point and leave
/// the insertion point in the innermost body. Returns the induction variables,
/// in the order of plan.loops.
llvm::SmallVector<mlir::Value> emitLoopNest(mlir::RewriterBase& rewriter,
                                            mlir::Location loc,
                                            const Chunking& plan) {
  llvm::SmallVector<mlir::Value> ivs;
  for (const Chunking::Loop& loop : plan.loops) {
    auto lower = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto upper =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, loop.extent);
    auto step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto for_op = mlir::scf::ForOp::create(rewriter, loc, lower, upper, step);
    ivs.push_back(for_op.getInductionVar());
    rewriter.setInsertionPointToStart(for_op.getBody());
  }
  return ivs;
}

/// @p map with one new trailing dimension per loop of @p plan, each advancing
/// the memref dimension that loop walks.
mlir::AffineMap extendSubscripts(mlir::AffineMap map, const Chunking& plan) {
  mlir::MLIRContext* context = map.getContext();
  llvm::SmallVector<mlir::AffineExpr> results(map.getResults());
  const unsigned base = map.getNumDims();
  for (auto [i, loop] : llvm::enumerate(plan.loops)) {
    results[loop.dim] =
        results[loop.dim] + mlir::getAffineConstantExpr(loop.stride, context) *
                                mlir::getAffineDimExpr(base + i, context);
  }
  return mlir::AffineMap::get(base + plan.loops.size(), 0, results, context);
}

/// Plan the split of @p extents, the shape moved through a slot wider than
/// @p throttle, reporting on @p op when it cannot be tiled.
std::optional<Chunking> planExtents(mlir::Operation* op,
                                    llvm::ArrayRef<int64_t> extents,
                                    int64_t throttle) {
  auto plan = planChunking(extents, throttle);
  if (!plan) {
    int64_t total = 1;
    for (int64_t extent : extents) total *= extent;
    op->emitError(PASS_NAME) << ": cannot tile an access of " << total
                             << " elements by the throttle of " << throttle;
  }
  return plan;
}

/// Plan the split of one access to a slot wider than @p throttle. Reports on
/// the access, and returns nullopt, when it cannot be split.
std::optional<Chunking> planAccess(mlir::Operation* user, int64_t throttle) {
  if (auto write = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(user)) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(write.getData().getType());
    if (!type || !type.hasStaticShape()) {
      write.emitError(PASS_NAME)
          << ": cannot split a write of a value that is not a static memref";
      return std::nullopt;
    }
    return planExtents(write, type.getShape(), throttle);
  }

  if (auto read = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(user)) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(read.getType());
    if (!type || !type.hasStaticShape() || read.getSplat()) {
      read.emitError(PASS_NAME)
          << ": cannot split a read that is a splat or not a static memref";
      return std::nullopt;
    }
    // The read is a transient buffer; it can be cut only where it is copied
    // straight on, as then the copy can be cut with it.
    auto copy = read.getResult().hasOneUse()
                    ? mlir::dyn_cast<mlir::memref::CopyOp>(
                          *read.getResult().getUsers().begin())
                    : nullptr;
    if (!copy || copy.getSource() != read.getResult()) {
      read.emitError(PASS_NAME)
          << ": cannot split a read whose only use is not a memref.copy";
      return std::nullopt;
    }
    return planExtents(read, type.getShape(), throttle);
  }

  auto transfer = mlir::cast<mlir::ktdf::DataTransferOp>(user);
  if (transfer.getTransferMode() != mlir::ktdf::TransferMode::Copy) {
    transfer.emitError(PASS_NAME)
        << ": cannot split a splat / pad transfer through a wide slot";
    return std::nullopt;
  }
  auto sizes = transfer.isSourceFifo() ? transfer.getStaticDestSizesArray()
                                       : transfer.getStaticSourceSizesArray();
  if (!sizes) {
    transfer.emitError(PASS_NAME)
        << ": cannot split a transfer whose memref sizes are dynamic";
    return std::nullopt;
  }
  return planExtents(transfer, *sizes, throttle);
}

/// Emit, in the loop nest of @p plan at the rewriter's insertion point, one
/// throttle-sized `ktdf.data_transfer` between @p slot and the piece of
/// @p memref that the iteration covers. @p to_slot gives the direction.
mlir::ktdf::DataTransferOp emitPieceTransfer(
    mlir::RewriterBase& rewriter, mlir::Location loc, mlir::Value slot,
    mlir::Value memref, mlir::AffineMap memref_map,
    mlir::ValueRange memref_indices, bool to_slot, const Chunking& plan,
    int64_t throttle) {
  auto ivs = emitLoopNest(rewriter, loc, plan);
  llvm::SmallVector<mlir::Value> indices(memref_indices);
  indices.append(ivs.begin(), ivs.end());
  const mlir::AffineMap map = extendSubscripts(memref_map, plan);
  const llvm::ArrayRef<int64_t> slot_sizes(&throttle, 1);
  const llvm::ArrayRef<int64_t> piece_sizes(plan.chunk_sizes);
  const auto copy_mode = mlir::ktdf::TransferMode::Copy;
  if (to_slot) {
    return mlir::ktdf::DataTransferOp::create(
        rewriter, loc, memref, map, indices, piece_sizes, slot,
        mlir::AffineMap(), mlir::ValueRange{}, slot_sizes, copy_mode);
  }
  return mlir::ktdf::DataTransferOp::create(
      rewriter, loc, slot, mlir::AffineMap(), mlir::ValueRange{}, slot_sizes,
      memref, map, indices, piece_sizes, copy_mode);
}

/// Zero indices and the identity map for a whole memref of @p memref's rank.
std::pair<mlir::AffineMap, llvm::SmallVector<mlir::Value>> wholeMemref(
    mlir::RewriterBase& rewriter, mlir::Location loc, mlir::Value memref) {
  const unsigned rank =
      mlir::cast<mlir::MemRefType>(memref.getType()).getRank();
  llvm::SmallVector<mlir::Value> zeros(
      rank, mlir::arith::ConstantIndexOp::create(rewriter, loc, 0));
  return {mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
          zeros};
}

void splitWrite(mlir::RewriterBase& rewriter, mlir::ktdf::WriteToFifoOp write,
                const Chunking& plan, int64_t throttle) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(write);
  auto [map, zeros] = wholeMemref(rewriter, write.getLoc(), write.getData());
  emitPieceTransfer(rewriter, write.getLoc(), write.getFifoSlot(),
                    write.getData(), map, zeros, /*to_slot=*/true, plan,
                    throttle);
  rewriter.eraseOp(write);
}

void splitRead(mlir::RewriterBase& rewriter, mlir::ktdf::ReadFromFifoOp read,
               const Chunking& plan, int64_t throttle) {
  auto copy =
      mlir::cast<mlir::memref::CopyOp>(*read.getResult().getUsers().begin());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(copy);
  auto [map, zeros] = wholeMemref(rewriter, copy.getLoc(), copy.getTarget());
  emitPieceTransfer(rewriter, copy.getLoc(), read.getFifoSlot(),
                    copy.getTarget(), map, zeros, /*to_slot=*/false, plan,
                    throttle);
  rewriter.eraseOp(copy);
  rewriter.eraseOp(read);
}

void splitTransfer(mlir::RewriterBase& rewriter,
                   mlir::ktdf::DataTransferOp transfer, const Chunking& plan,
                   int64_t throttle) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(transfer);
  const bool to_slot = transfer.isDestFifo();
  auto piece = emitPieceTransfer(
      rewriter, transfer.getLoc(),
      to_slot ? transfer.getDestination() : transfer.getSource(),
      to_slot ? transfer.getSource() : transfer.getDestination(),
      to_slot ? *transfer.getSourceMap() : *transfer.getDestMap(),
      to_slot ? transfer.getSourceIndices() : transfer.getDestIndices(),
      to_slot, plan, throttle);
  piece->setDiscardableAttrs(transfer->getDiscardableAttrDictionary());
  rewriter.eraseOp(transfer);
}

/// The number of elements one access of @p elem_type may move, or 0 when the
/// compute unit declares no vector width for it (such a slot is not throttled).
int64_t throttleFor(mlir::Type elem_type,
                    mlir::ktdf_arch::ExecutionUnitOp compute) {
  if (!elem_type.isIntOrFloat()) return 0;
  return compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
      elem_type);
}

/// Split every access to @p slot, then shrink the slot to @p throttle.
mlir::LogicalResult splitSlot(mlir::RewriterBase& rewriter, mlir::Value slot,
                              int64_t throttle) {
  // Collect and check first: nothing is rewritten unless all accesses split.
  llvm::SmallVector<std::pair<mlir::Operation*, Chunking>> accesses;
  for (mlir::Operation* user : slot.getUsers()) {
    if (mlir::isa<mlir::ktdf::PrivateYieldOp>(user)) continue;
    if (!mlir::isa<mlir::ktdf::WriteToFifoOp, mlir::ktdf::ReadFromFifoOp,
                   mlir::ktdf::DataTransferOp>(user)) {
      return user->emitError(PASS_NAME)
             << ": cannot split a slot that is used by this op";
    }
    auto plan = planAccess(user, throttle);
    if (!plan) return mlir::failure();
    accesses.emplace_back(user, std::move(*plan));
  }

  if (!mlir::ktdf::resizePrivateFifoSlot(rewriter, slot, throttle)) {
    return mlir::emitError(slot.getLoc(), PASS_NAME)
           << ": cannot shrink a slot that is not a ktdf.private result "
              "yielded from a ktdf.fifo.allocate";
  }
  for (auto& [op, plan] : accesses) {
    LDBG(1) << "splitting wide access to " << throttle << " elements: " << *op;
    if (auto write = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op)) {
      splitWrite(rewriter, write, plan, throttle);
    } else if (auto read = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op)) {
      splitRead(rewriter, read, plan, throttle);
    } else {
      splitTransfer(rewriter, mlir::cast<mlir::ktdf::DataTransferOp>(op), plan,
                    throttle);
    }
  }
  return mlir::success();
}

struct ChunkWideAccessesPass
    : public scheduler::impl::ChunkWideAccessesPassBase<
          ChunkWideAccessesPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);
    mlir::ktdf_arch::ExecutionUnitOp compute =
        resource_kinds.getDefaultCompute();
    if (!compute) {
      module_op->emitError(PASS_NAME)
          << ": no (unambiguous) default compute resource to take the "
             "throttle from";
      signalPassFailure();
      return;
    }

    // Collect first: splitting a slot replaces its allocation.
    llvm::SmallVector<std::pair<mlir::Value, int64_t>> wide;
    module_op.walk([&](mlir::ktdf::PrivateOp private_op) {
      for (mlir::Value result : private_op.getResults()) {
        auto slot = mlir::dyn_cast<mlir::ktdf::FifoSlotType>(result.getType());
        if (!slot) continue;
        const int64_t throttle = throttleFor(slot.getElementType(), compute);
        if (throttle > 0 && slot.getNumElements() > throttle)
          wide.emplace_back(result, throttle);
      }
    });

    mlir::IRRewriter rewriter(&getContext());
    for (auto [slot, throttle] : wide) {
      if (mlir::failed(splitSlot(rewriter, slot, throttle))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createChunkWideAccessesPass() {
  return std::make_unique<ChunkWideAccessesPass>();
}
