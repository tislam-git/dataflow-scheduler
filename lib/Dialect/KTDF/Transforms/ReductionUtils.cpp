//===-- ReductionUtils.cpp --------------------------------------*- c++ -*-===//
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
// Definitions for StageFactory methods declared in ReductionUtils.h.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::ktdf {

// ---------------------------------------------------------------------------
// findInnerDimLoopDim
// ---------------------------------------------------------------------------
std::optional<unsigned> findInnerDimLoopDim(linalg::GenericOp generic_op) {
  if (generic_op.getInputs().empty())
    return std::nullopt;

  auto input_type =
      dyn_cast<RankedTensorType>(generic_op.getInputs().front().getType());
  if (!input_type) return std::nullopt;

  int64_t last = input_type.getRank() - 1;

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(last));
  if (!dim_expr) return std::nullopt;

  unsigned loop_dim = dim_expr.getPosition();
  auto iter_types = generic_op.getIteratorTypesArray();
  if (iter_types[loop_dim] == utils::IteratorType::reduction) return loop_dim;
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// findFifoInOperandIdx / getFifoInOperand
// ---------------------------------------------------------------------------
std::optional<unsigned> findFifoInOperandIdx(linalg::GenericOp generic_op) {
  unsigned num_inputs = generic_op.getInputs().size();
  if (num_inputs > 0) return 0;
  for (auto [i, out] : llvm::enumerate(generic_op.getOutputs())) {
    Value base = out;
    if (auto slice = base.getDefiningOp<tensor::ExtractSliceOp>())
      base = slice.getSource();
    if (base.getDefiningOp<ktdf::ReadFromFifoOp>()) return num_inputs + i;
  }
  return std::nullopt;
}

Value getFifoInOperand(linalg::GenericOp generic_op, unsigned idx) {
  unsigned num_inputs = generic_op.getInputs().size();
  return idx < num_inputs ? generic_op.getInputs()[idx]
                          : generic_op.getOutputs()[idx - num_inputs];
}

// ---------------------------------------------------------------------------
// getIterArgOutputIndices
// ---------------------------------------------------------------------------
SmallVector<unsigned> getIterArgOutputIndices(linalg::GenericOp generic_op) {
  SmallVector<unsigned> indices;
  for (auto [i, out] : llvm::enumerate(generic_op.getOutputs())) {
    auto arg = dyn_cast<BlockArgument>(out);
    if (arg && isa<scf::ForOp>(arg.getOwner()->getParentOp()))
      indices.push_back(i);
  }
  return indices;
}

// ---------------------------------------------------------------------------
// StageFactory method definitions
// ---------------------------------------------------------------------------

FifoPrivateResult StageFactory::buildFifoPrivate(
    const ChunkPipelineConfig& cfg) {
  auto tok_type = pipe_bldr_.getType<ktdf::TokenType>();

  SmallVector<Type> result_types;
  for (auto slot_type : cfg.in_slot_types) {
    result_types.push_back(slot_type);
  }
  for (auto slot_type : cfg.partial_slot_types) {
    result_types.push_back(slot_type);
  }
  for (auto slot_type : cfg.out_slot_types) {
    result_types.push_back(slot_type);
  }
  for (unsigned i = 0; i < cfg.n_tokens; ++i) {
    result_types.push_back(tok_type);
  }

  auto priv_op = ktdf::PrivateOp::create(pipe_bldr_, loc_, result_types);
  OpBuilder body_bldr(priv_op.getRegion());

  SmallVector<Value> yield_vals;
  auto alloc = [&](ktdf::FifoSlotType slot_type) {
    return ktdf::FifoAllocateOp::create(body_bldr, loc_, TypeRange{slot_type},
                                        ValueRange{})
        .getResult(0);
  };
  for (auto slot_type : cfg.in_slot_types) {
    yield_vals.push_back(alloc(slot_type));
  }
  for (auto slot_type : cfg.partial_slot_types) {
    yield_vals.push_back(alloc(slot_type));
  }
  for (auto slot_type : cfg.out_slot_types) {
    yield_vals.push_back(alloc(slot_type));
  }
  for (unsigned i = 0; i < cfg.n_tokens; ++i) {
    yield_vals.push_back(
        ktdf::CreateTokenOp::create(body_bldr, loc_, tok_type).getResult());
  }

  ktdf::PrivateYieldOp::create(body_bldr, loc_, yield_vals);

  auto n_in = static_cast<unsigned>(cfg.in_slot_types.size());
  auto n_partial = static_cast<unsigned>(cfg.partial_slot_types.size());
  auto n_out = static_cast<unsigned>(cfg.out_slot_types.size());
  FifoPrivateResult result;
  result.priv_op = priv_op;
  result.in_begin = 0;
  result.partial_begin = n_in;
  result.out_begin = n_in + n_partial;
  result.tok_begin = n_in + n_partial + n_out;
  return result;
}

void StageFactory::buildLoadStage(
    Value condition, Value input_memref, MemRefType input_memref_type,
    Value partial_memref, MemRefType output_memref_type,
    ArrayRef<unsigned> reduction_dims, ArrayRef<int64_t> chunk_sizes,
    ArrayRef<Value> dim_ivs, const ChunkPipelineConfig& cfg) {
  // token 0: Load → Compute synchronisation (ChunkPipelineConfig convention).
  auto stage =
      ktdf::StageOp::create(pipe_bldr_, loc_,
                            /*depends_in=*/ValueRange{},
                            /*depends_out=*/ValueRange{slots_.token(0)});
  if (load_units_) {
    stage.setApplicableUnitsAttr(load_units_);
  }

  OpBuilder b(stage.getBody(), stage.getBody()->end());
  Value c0 = arith::ConstantIndexOp::create(b, loc_, 0);
  Value local_slot_idx = emitLocalSlotIdx(b, c0);

  // Build the shared source index vector and AffineMap for all in-slots.
  //
  // dim-0 of the memref is the batch/local slot (index = local_slot_idx, size =
  // 1). dims 1..rank-1 correspond to linalg.generic dims 0..rank-2:
  //   - reduction dim j → index operand = dim_ivs[j],
  //                        map result    = d_j * chunk_size[j]  (affine),
  //                        static size   = chunk_size[j]
  //   - parallel dim    → index operand = c0,
  //                        map result    = d_k  (identity),
  //                        static size   = full dim size
  //
  // Encoding iv * chunk_size inside the AffineMap avoids emitting an
  // arith.muli SSA value as an index operand, which the backend does not
  // support.
  int64_t input_rank = input_memref_type.getRank();
  SmallVector<Value> src_indices;
  SmallVector<int64_t> src_static_sizes;
  SmallVector<AffineExpr> map_results;
  src_indices.reserve(static_cast<size_t>(input_rank));
  src_static_sizes.reserve(static_cast<size_t>(input_rank));
  map_results.reserve(static_cast<size_t>(input_rank));

  // dim-0: batch/local slot.
  src_indices.push_back(local_slot_idx);
  src_static_sizes.push_back(1);
  map_results.push_back(getAffineDimExpr(0, b.getContext()));

  // dims 1..rank-1.
  for (int64_t i = 1; i < input_rank; ++i) {
    int64_t generic_dim = i - 1;
    auto it = llvm::find(reduction_dims, generic_dim);
    unsigned dim_idx = static_cast<unsigned>(i);
    if (it != reduction_dims.end()) {
      size_t j = static_cast<size_t>(it - reduction_dims.begin());
      src_indices.push_back(dim_ivs[j]);
      src_static_sizes.push_back(chunk_sizes[j]);
      // iv_j * chunk_size[j]: constant multiplication stays in the map.
      map_results.push_back(getAffineDimExpr(dim_idx, b.getContext()) *
                            chunk_sizes[j]);
    } else {
      src_indices.push_back(c0);
      src_static_sizes.push_back(input_memref_type.getDimSize(i));
      map_results.push_back(getAffineDimExpr(dim_idx, b.getContext()));
    }
  }

  AffineMap input_map =
      AffineMap::get(static_cast<unsigned>(input_rank),
                     /*symbolCount=*/0, map_results, b.getContext());

  // Transfer the input chunk into each in-slot.
  for (unsigned i = 0; i < cfg.in_slot_types.size(); ++i) {
    Value fifo_in_slot = slots_.inSlot(i);
    auto fifo_type = cast<ktdf::FifoSlotType>(fifo_in_slot.getType());
    ktdf::DataTransferOp::create(
        b, loc_, input_memref, input_map, src_indices, src_static_sizes,
        fifo_in_slot, AffineMap{}, ValueRange{},
        ArrayRef<int64_t>{static_cast<int64_t>(fifo_type.getNumElements())});
  }

  // On non-first iterations, feed each partial accumulator into its partial-
  // slot so the Compute stage can read back the running result.
  if (!cfg.partial_slot_types.empty()) {
    auto if_op =
        scf::IfOp::create(b, loc_, /*resultTypes=*/TypeRange{}, condition,
                          /*withElseRegion=*/true);
    OpBuilder else_bldr =
        OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
    int64_t partial_rank = output_memref_type.getRank();
    AffineMap partial_id =
        AffineMap::getMultiDimIdentityMap(partial_rank, else_bldr.getContext());
    SmallVector<Value> partial_src_indices;
    SmallVector<int64_t> partial_src_sizes;
    partial_src_indices.reserve(static_cast<size_t>(partial_rank));
    partial_src_sizes.reserve(static_cast<size_t>(partial_rank));
    partial_src_indices.push_back(local_slot_idx);
    partial_src_sizes.push_back(1);
    for (int64_t i = 1; i < partial_rank; ++i) {
      partial_src_indices.push_back(c0);
      partial_src_sizes.push_back(output_memref_type.getDimSize(i));
    }
    for (unsigned i = 0; i < cfg.partial_slot_types.size(); ++i) {
      Value fifo_partial_slot = slots_.partialSlot(i);
      auto fifo_type = cast<ktdf::FifoSlotType>(fifo_partial_slot.getType());
      ktdf::DataTransferOp::create(
          else_bldr, loc_, partial_memref, partial_id, partial_src_indices,
          partial_src_sizes, fifo_partial_slot, AffineMap{}, ValueRange{},
          ArrayRef<int64_t>{static_cast<int64_t>(fifo_type.getNumElements())});
    }
  }
}

void StageFactory::buildComputeStage(Value condition,
                                     RankedTensorType chunk_input_tensor_type,
                                     RankedTensorType output_tensor_type,
                                     linalg::GenericOp generic_op,
                                     const ChunkPipelineConfig& cfg) {
  // token 0:          Load → Compute (depends_in).
  // token n_tokens-1: Compute → Store (depends_out).
  auto stage = ktdf::StageOp::create(
      pipe_bldr_, loc_,
      /*depends_in=*/ValueRange{slots_.token(0)},
      /*depends_out=*/ValueRange{slots_.token(cfg.n_tokens - 1)});
  if (compute_units_) {
    stage.setApplicableUnitsAttr(compute_units_);
  }

  OpBuilder b(stage.getBody(), stage.getBody()->end());

  // Read the input chunk unconditionally.
  auto in_tensor = ktdf::ReadFromFifoOp::create(
      b, loc_, chunk_input_tensor_type, slots_.inSlot(0));

  // First linalg.generic: reduce the input chunk into a fresh tensor.empty.
  // This is always computed; the conditional below determines whether to
  // accumulate it against a prior partial result.
  Value empty_tensor =
      tensor::EmptyOp::create(b, loc_, output_tensor_type.getShape(),
                              output_tensor_type.getElementType());
  {
    IRMapping mapping;
    mapping.map(generic_op.getInputs().front(), in_tensor.getResult());
    mapping.map(generic_op.getOutputs().front(), empty_tensor);
    empty_tensor =
        cast<linalg::GenericOp>(b.clone(*generic_op.getOperation(), mapping))
            .getResult(0);
  }

  // Guard: on the first iteration (condition == true) the first generic result
  // is the final result.  On subsequent iterations it is combined with the
  // running partial accumulator read back from fifo_partial[0].
  //
  //   then: yield first_generic_result                   (no prior partial)
  //   else: partial ← read_from_fifo(fifo_partial[0])
  //         second_generic ← linalg.generic(partial, first_generic_result)
  //         yield second_generic_result
  auto if_op = scf::IfOp::create(b, loc_, TypeRange{output_tensor_type},
                                 condition, /*withElseRegion=*/true);
  {
    OpBuilder then_bldr =
        OpBuilder::atBlockBegin(&if_op.getThenRegion().front());
    scf::YieldOp::create(then_bldr, loc_, empty_tensor);
  }
  {
    OpBuilder else_bldr =
        OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
    Value partial_tensor = ktdf::ReadFromFifoOp::create(
        else_bldr, loc_, output_tensor_type, slots_.partialSlot(0));

    // The second generic combines partial_tensor (ins) with the first generic
    // result (outs).  Both are output_tensor_type, so we need identity maps
    // for both operands and all-parallel iterators — the original generic_op
    // has a higher-rank input map and cannot be cloned directly here.
    int64_t out_rank = output_tensor_type.getRank();
    AffineMap id_map =
        AffineMap::getMultiDimIdentityMap(out_rank, else_bldr.getContext());
    SmallVector<utils::IteratorType> parallel_iters(
        static_cast<size_t>(out_rank), utils::IteratorType::parallel);
    SmallVector<AffineMap> indexing_maps = {id_map, id_map};
    auto combine_generic = linalg::GenericOp::create(
        else_bldr, loc_,
        /*resultTensorTypes=*/TypeRange{output_tensor_type},
        /*inputs=*/ValueRange{partial_tensor},
        /*outputs=*/ValueRange{empty_tensor},
        /*indexingMaps=*/indexing_maps,
        /*iteratorTypes=*/parallel_iters);
    Type elem_type = output_tensor_type.getElementType();
    Block* new_body = &combine_generic.getRegion().emplaceBlock();
    new_body->addArgument(elem_type, loc_);  // ins[0] element
    new_body->addArgument(elem_type, loc_);  // outs[0] element

    // Clone the scalar body ops from the original generic, remapping its block
    // args to the new block args.
    IRMapping body_map;
    Block* orig_body = generic_op.getBody();
    body_map.map(orig_body->getArgument(0), new_body->getArgument(0));
    body_map.map(orig_body->getArgument(1), new_body->getArgument(1));
    OpBuilder body_bldr = OpBuilder::atBlockBegin(new_body);
    for (auto& op : *orig_body) body_bldr.clone(op, body_map);

    scf::YieldOp::create(else_bldr, loc_, combine_generic.getResult(0));
  }

  ktdf::WriteToFifoOp::create(b, loc_, if_op.getResult(0), slots_.outSlot(0));
}

void StageFactory::buildStoreStage(Value partial_memref,
                                   const ChunkPipelineConfig& cfg) {
  // token n_tokens-1: Compute → Store (ChunkPipelineConfig convention).
  auto stage = ktdf::StageOp::create(
      pipe_bldr_, loc_,
      /*depends_in=*/ValueRange{slots_.token(cfg.n_tokens - 1)},
      /*depends_out=*/ValueRange{});
  if (store_units_) {
    stage.setApplicableUnitsAttr(store_units_);
  }

  OpBuilder b(stage.getBody(), stage.getBody()->end());
  Value c0 = arith::ConstantIndexOp::create(b, loc_, 0);
  Value local_slot_idx = emitLocalSlotIdx(b, c0);

  auto dest_memref_type = cast<MemRefType>(partial_memref.getType());
  int64_t dest_rank = dest_memref_type.getRank();
  AffineMap dest_id =
      AffineMap::getMultiDimIdentityMap(dest_rank, b.getContext());
  SmallVector<Value> dest_indices;
  SmallVector<int64_t> dest_sizes;
  dest_indices.reserve(static_cast<size_t>(dest_rank));
  dest_sizes.reserve(static_cast<size_t>(dest_rank));
  dest_indices.push_back(local_slot_idx);
  dest_sizes.push_back(1);
  for (int64_t i = 1; i < dest_rank; ++i) {
    dest_indices.push_back(c0);
    dest_sizes.push_back(dest_memref_type.getDimSize(i));
  }

  for (unsigned i = 0; i < cfg.out_slot_types.size(); ++i) {
    Value fifo_out_slot = slots_.outSlot(i);
    auto fifo_type = cast<ktdf::FifoSlotType>(fifo_out_slot.getType());
    ktdf::DataTransferOp::create(
        b, loc_, fifo_out_slot, AffineMap{}, ValueRange{},
        ArrayRef<int64_t>{static_cast<int64_t>(fifo_type.getNumElements())},
        partial_memref, dest_id, dest_indices, dest_sizes);
  }
}

Value StageFactory::emitLocalSlotIdx(OpBuilder& bldr, Value c0) {
  Value c1 = arith::ConstantIndexOp::create(bldr, loc_, 1);
  Value sub = arith::SubIOp::create(bldr, loc_, batch_iv_, c0);
  return arith::DivSIOp::create(bldr, loc_, sub, c1);
}

auto StageFactory::findLoadStage(ktdf::PipelineOp pipeline,
                                 ktdf::StageOp compute_stage) -> StageOp {
  for (Value tok : compute_stage.getDependsIn()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value out_tok : sibling.getDependsOut()) {
        if (out_tok == tok) return sibling;
      }
    }
  }
  return {};
}

auto StageFactory::findStoreStage(ktdf::PipelineOp pipeline,
                                  ktdf::StageOp compute_stage) -> StageOp {
  for (Value tok : compute_stage.getDependsOut()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value in_tok : sibling.getDependsIn()) {
        if (in_tok == tok) return sibling;
      }
    }
  }
  return {};
}

auto StageFactory::findLoadTransfer(ktdf::StageOp load_stage,
                                    linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp> {
  auto read_from_fifo =
      generic_op.getInputs().front().getDefiningOp<ktdf::ReadFromFifoOp>();
  if (!read_from_fifo) return {nullptr, nullptr};

  Value fifo_in = read_from_fifo.getFifoSlot();
  ktdf::DataTransferOp load_transfer;
  load_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceMemRef() && dt.isDestFifo() &&
        dt.getDestination() == fifo_in) {
      load_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {load_transfer, read_from_fifo};
}

auto StageFactory::findStoreTransfer(ktdf::StageOp compute_stage,
                                     ktdf::StageOp store_stage,
                                     linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp> {
  ktdf::WriteToFifoOp write_to_fifo;
  compute_stage.getBody()->walk([&](ktdf::WriteToFifoOp write) {
    if (write.getData() == generic_op.getResult(0)) {
      write_to_fifo = write;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!write_to_fifo) return {nullptr, nullptr};

  Value fifo_out = write_to_fifo.getFifoSlot();
  ktdf::DataTransferOp store_transfer;
  store_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceFifo() && dt.isDestMemRef() && dt.getSource() == fifo_out) {
      store_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {store_transfer, write_to_fifo};
}

}  // namespace mlir::ktdf
