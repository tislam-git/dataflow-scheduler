//===-- ReductionUtils.h ----------------------------------------*- c++ -*-===//
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
// Utilities for reduction-related transforms that work with a Load / Compute /
// Store three-stage pipeline pattern:
//
//   FifoPrivateResult    — named-accessor wrapper for a ktdf.private result
//                          list produced by StageFactory::buildFifoPrivate.
//   ChunkPipelineConfig  — describes the FIFO slot layout and token count for
//                          one chunk-pipeline iteration.
//   StageFactory         — builds Load / Compute / Store stages inside a
//                          ktdf.pipeline body driven by a ChunkPipelineConfig.
//                          Also exposes static pipeline-shape query methods:
//                            findLoadStage    — upstream sibling via token
//                            chain. findStoreStage   — downstream sibling via
//                            token chain. findLoadTransfer — trace the
//                            memref→fifo_in path. findStoreTransfer— trace the
//                            fifo_out→memref path.
//
// These helpers assume the Load / Compute / Store pipeline shape produced by
// StageCoarsening and are intentionally kept in the transform layer rather
// than the dialect utilities, because they encode pipeline-shape assumptions
// that do not hold in general.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_

#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir::ktdf {

// ---------------------------------------------------------------------------
// Return the loop dimension index that corresponds to the innermost reduction:
// the rightmost input dimension (input_rank - 1), if it is a reduction
// iterator. Returns std::nullopt if the rightmost input dimension is not a
// reduction.
// ---------------------------------------------------------------------------
std::optional<unsigned> findInnerDimLoopDim(linalg::GenericOp generic_op);

// ---------------------------------------------------------------------------
// Index (into [ins..., outs...]) of the operand fed from the input FIFO.  With
// a non-empty ins() this is the first input (the pre-existing convention; it is
// not checked against a read_from_fifo).  A generic with an empty ins() (e.g.
// a top-k whose data tensor is also updated in place) carries it in outs()
// instead, as the operand produced by a ktdf.read_from_fifo.  Returns
// std::nullopt when no such operand exists.
// ---------------------------------------------------------------------------
std::optional<unsigned> findFifoInOperandIdx(linalg::GenericOp generic_op);

// The operand at `idx` in [ins..., outs...] (see findFifoInOperandIdx).
Value getFifoInOperand(linalg::GenericOp generic_op, unsigned idx);

// ---------------------------------------------------------------------------
// Indices (into outs()) of the outputs that are loop-carried accumulators,
// i.e. block arguments (iter_args) of an enclosing scf.for.  The remaining
// outputs, if any, are not accumulators (e.g. the FIFO data operand of a
// generic that has no ins()).
// ---------------------------------------------------------------------------
SmallVector<unsigned> getIterArgOutputIndices(linalg::GenericOp generic_op);

// ---------------------------------------------------------------------------
// Result of StageFactory::buildFifoPrivate.
//
// Wraps the ktdf.private op and records the begin-offset of each group of
// results so callers can retrieve any slot or token by logical index without
// knowing the raw result-number layout.
//
// Yield order inside the ktdf.private body (and thus result order):
//   [in_slots... | partial_slots... | out_slots... | tokens...]
// ---------------------------------------------------------------------------
struct FifoPrivateResult {
  ktdf::PrivateOp priv_op;
  // Begin offset of each group in the PrivateOp result list.
  unsigned in_begin;
  unsigned partial_begin;
  unsigned out_begin;
  unsigned tok_begin;

  Value inSlot(unsigned i) { return priv_op.getResult(in_begin + i); }
  Value partialSlot(unsigned i) { return priv_op.getResult(partial_begin + i); }
  Value outSlot(unsigned i) { return priv_op.getResult(out_begin + i); }
  Value token(unsigned i) { return priv_op.getResult(tok_begin + i); }
};

// ---------------------------------------------------------------------------
// Describes the FIFO slot layout and token count for one chunk pipeline.
//
// Passed to StageFactory::buildFifoPrivate, buildLoadStage, buildComputeStage,
// and buildStoreStage so that explicit per-slot index arguments are no longer
// needed at every call site.
//
// Slot groups (parallel vectors; index i selects one logical slot within that
// group):
//   in_slot_types     — FifoSlotType for each input-chunk FIFO (Load → Compute)
//   partial_slot_types— FifoSlotType for each partial-accum FIFO (Load →
//                       Compute, non-first iterations only); may be empty.
//   out_slot_types    — FifoSlotType for each result FIFO (Compute → Store)
//
// Token assignment (fixed convention used by all three stage builders):
//   token 0 … n_tokens-2 : Load → Compute synchronisation tokens
//   token n_tokens-1      : Compute → Store synchronisation token
//
// Currently the convention is n_tokens == 2: token 0 for Load→Compute and
// token 1 for Compute→Store.
// ---------------------------------------------------------------------------
struct ChunkPipelineConfig {
  SmallVector<ktdf::FifoSlotType> in_slot_types;
  SmallVector<ktdf::FifoSlotType> partial_slot_types;
  SmallVector<ktdf::FifoSlotType> out_slot_types;
  unsigned n_tokens = 0;
};

// ---------------------------------------------------------------------------
// StageFactory: builds the Load / Compute / Store stages of one chunk-pipeline
// iteration inside a ktdf.pipeline body.
//
// Usage sequence:
//   1. Construct with the pipeline body builder, location, batch IV, and
//      per-stage applicable-unit attributes.
//   2. Call buildFifoPrivate(cfg) to allocate FIFO slots and tokens described
//      by ChunkPipelineConfig cfg; it returns a FifoPrivateResult with named
//      accessors.
//   3. Call setSlots() with that result so the stage builders can address
//      any slot or token by index.
//   4. Call buildLoadStage(…, cfg), buildComputeStage(…, cfg),
//      buildStoreStage(…, cfg) in order.
//
// The ChunkPipelineConfig carries all slot-type lists and the token count, so
// no per-slot index arguments need to be passed to the stage builders.
//
// The local slot index arithmetic (batch_iv → local slot index) is encapsulated
// in emitLocalSlotIdx() and never needs to be passed as a parameter.
//
// This class can be reused whenever a 3-stage Load / Compute / Store pipeline
// is required.
// ---------------------------------------------------------------------------
class StageFactory {
 public:
  // -------------------------------------------------------------------------
  // Records the pipeline body builder, location, batch induction variable,
  // and per-stage applicable-unit attributes.  Does not emit any IR.
  // -------------------------------------------------------------------------
  StageFactory(OpBuilder& pipe_bldr, Location loc, Value batch_iv,
               ArrayAttr load_units, ArrayAttr compute_units,
               ArrayAttr store_units)
      : pipe_bldr_(pipe_bldr),
        loc_(loc),
        batch_iv_(batch_iv),
        load_units_(load_units),
        compute_units_(compute_units),
        store_units_(store_units) {}

  // -------------------------------------------------------------------------
  // Emit a ktdf.private op that yields FIFO slots and tokens in the order:
  //   [in_slots... | partial_slots... | out_slots... | tokens...]
  //
  // The layout is driven entirely by cfg:
  //   cfg.in_slot_types.size()      in-slots
  //   cfg.partial_slot_types.size() partial-slots (skipped if empty)
  //   cfg.out_slot_types.size()     out-slots
  //   cfg.n_tokens                  ktdf.token values
  //
  // Returns a FifoPrivateResult whose named accessors (inSlot, partialSlot,
  // outSlot, token) index into the corresponding group.  Call setSlots() with
  // the result before invoking any stage builder.
  // -------------------------------------------------------------------------
  FifoPrivateResult buildFifoPrivate(const ChunkPipelineConfig& cfg);

  // Bind the FifoPrivateResult produced by buildFifoPrivate() into the
  // factory.  Must be called before any of the stage builders.
  void setSlots(FifoPrivateResult slots) { slots_ = slots; }

  // -------------------------------------------------------------------------
  // Emit the Load stage inside a chunk pipeline.
  //
  // For each slot in cfg.in_slot_types: transfers one input chunk slice from
  // input_memref into the corresponding in-slot (input_memref → fifo_in[i]).
  //
  // For each slot in cfg.partial_slot_types: on non-first iterations
  // (!condition) transfers the running partial accumulator from partial_memref
  // into the corresponding partial-slot (partial_memref → fifo_partial[i]).
  // If cfg.partial_slot_types is empty the scf.if guard is not emitted.
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token 0 : Load → Compute (depends_out).
  //
  // Parameters:
  //   condition          — i1 value; true on the first chunk iteration.
  //   input_memref       — source buffer for the input chunk.
  //   input_memref_type  — type of input_memref (cached to avoid re-cast).
  //   partial_memref     — local memory buffer reused as the partial
  //   accumulator. output_memref_type — type of partial_memref (cached to avoid
  //   re-cast). reduction_dims     — linalg.generic dims that are being
  //   chunked. chunk_sizes        — size of one chunk along each reduction dim.
  //   dim_ivs            — loop IVs (or c0) for each reduction dim.
  //   cfg                — slot layout and token count for this pipeline.
  //
  // Input memref layout: dim-0 is the batch/local slot index; dims 1..rank-1
  // correspond to linalg.generic dims 0..rank-2.  For reduction dims the
  // source offset is encoded as iv_j * chunk_size[j] in the AffineMap (to
  // avoid emitting arith.muli, which the backend does not accept as an index).
  // -------------------------------------------------------------------------
  void buildLoadStage(Value condition, Value input_memref,
                      MemRefType input_memref_type, Value partial_memref,
                      MemRefType output_memref_type,
                      ArrayRef<unsigned> reduction_dims,
                      ArrayRef<int64_t> chunk_sizes, ArrayRef<Value> dim_ivs,
                      const ChunkPipelineConfig& cfg);

  // -------------------------------------------------------------------------
  // Emit the Compute stage inside a chunk pipeline.
  //
  // Reads the input chunk from fifo_in[0] unconditionally, then runs a first
  // cloned linalg.generic against a fresh tensor.empty to produce an
  // intermediate partial result.
  //
  // A runtime branch on condition then selects the final output:
  //   - condition == true  (first iteration):
  //       yield the first generic result directly (no prior partial).
  //   - condition == false (subsequent iterations):
  //       read the running partial from fifo_partial[0], run a second cloned
  //       linalg.generic with input=partial and output=first generic result,
  //       and yield the second generic result.
  //
  // The conditional result is written to fifo_out[0] (unconditional).
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token 0             : Load → Compute (depends_in).
  //   token n_tokens-1    : Compute → Store (depends_out).
  //
  // Parameters:
  //   condition               — i1 value; true on the first chunk iteration.
  //   chunk_input_tensor_type — tensor type for one input chunk.
  //   output_tensor_type      — tensor type for the reduction result.
  //   generic_op              — original linalg.generic to clone.
  //   cfg                     — slot layout and token count for this pipeline.
  // -------------------------------------------------------------------------
  void buildComputeStage(Value condition,
                         RankedTensorType chunk_input_tensor_type,
                         RankedTensorType output_tensor_type,
                         linalg::GenericOp generic_op,
                         const ChunkPipelineConfig& cfg);

  // -------------------------------------------------------------------------
  // Emit the Store stage inside a chunk pipeline.
  //
  // For each slot in cfg.out_slot_types: unconditionally reads the reduction
  // result from fifo_out[i] and writes it back to partial_memref (the local
  // memory partial-accumulation buffer).
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token n_tokens-1 : Compute → Store (depends_in).
  //
  // Parameters:
  //   partial_memref — local memory destination buffer (same as the one the
  //   Load
  //                    stage reads the running partial result from).
  //   cfg            — slot layout and token count for this pipeline.
  // -------------------------------------------------------------------------
  void buildStoreStage(Value partial_memref, const ChunkPipelineConfig& cfg);

  // -------------------------------------------------------------------------
  // Pipeline-shape query methods.
  //
  // These static methods assume the Load / Compute / Store three-stage
  // pipeline shape produced by StageCoarsening.
  // -------------------------------------------------------------------------

  /// Find the sibling stage of `compute_stage` inside `pipeline` that is
  /// upstream (the Load stage) — i.e. the sibling whose `depends_out` token
  /// appears in `compute_stage`'s `depends_in`.
  /// Returns a null StageOp if not found.
  static auto findLoadStage(ktdf::PipelineOp pipeline,
                            ktdf::StageOp compute_stage) -> StageOp;

  /// Find the sibling stage of `compute_stage` inside `pipeline` that is
  /// downstream (the Store stage) — i.e. the sibling that consumes a token
  /// from `compute_stage`'s `depends_out`.
  /// Returns a null StageOp if not found.
  static auto findStoreStage(ktdf::PipelineOp pipeline,
                             ktdf::StageOp compute_stage) -> StageOp;

  /// Trace the input data path for a reduction linalg.generic:
  ///   generic.inputs[0] ← ReadFromFifoOp(fifo_in) ←
  ///   DataTransferOp(src=input_memref, dst=fifo_in) in load_stage
  ///
  /// Returns {load_transfer, read_from_fifo}, or {nullptr, nullptr} on
  /// failure.
  static auto findLoadTransfer(ktdf::StageOp load_stage,
                               linalg::GenericOp generic_op)
      -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp>;

  /// Trace the output data path for a reduction linalg.generic:
  ///   generic.result(0) → WriteToFifoOp(fifo_out) →
  ///   DataTransferOp(src=fifo_out, dst=partial_memref) in store_stage
  ///
  /// Returns {store_transfer, write_to_fifo}, or {nullptr, nullptr} on
  /// failure.
  static auto findStoreTransfer(ktdf::StageOp compute_stage,
                                ktdf::StageOp store_stage,
                                linalg::GenericOp generic_op)
      -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp>;

 private:
  // -------------------------------------------------------------------------
  // Emit (batch_iv_ - 0) / 1 — the normalised affine form expected by the
  // backend for local-slot index arithmetic.
  //
  // c0 must be emitted by the caller inside the stage body (ktdf.pipeline
  // only allows ktdf.stage and ktdf.private as immediate children, so
  // constants cannot be emitted at pipeline scope).
  // -------------------------------------------------------------------------
  Value emitLocalSlotIdx(OpBuilder& b, Value c0);

  OpBuilder& pipe_bldr_;  // Builder positioned at the pipeline body end.
  Location loc_;          // Location for all emitted ops.
  Value batch_iv_;        // Outer batch scf.for induction variable.
  ArrayAttr load_units_;  // applicable_units for the Load stage (may be null).
  ArrayAttr
      compute_units_;  // applicable_units for the Compute stage (may be null).
  ArrayAttr
      store_units_;  // applicable_units for the Store stage (may be null).

  // Slot/token result wrapper bound via setSlots(); all stage builders address
  // slots and tokens through the FifoPrivateResult named accessors.
  FifoPrivateResult slots_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_
