//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD
#define DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def PathExpansionPass : Pass<"path-expansion", "mlir::ModuleOp"> {
  let summary = "Expand data movement paths to match architecture graph";
  let description = [{
    This pass legalizes data movement in ktdf.pipeline operations by expanding
    paths to match the architecture graph. It inserts intermediate transfer
    stages and resources where needed to ensure all data movement follows legal
    hardware routing.
  }];
  let constructor = "::scheduler::createPathExpansionPass()";
}

def NormalizeGridTo1DPass : Pass<"normalize-grid-to-1d", "mlir::ModuleOp"> {
  let summary = "Flatten N-D grids to 1-D and delinearize compute tile ids";
  let description = [{
    Rewrites each function's `grid` attribute from an N-D array to a single
    1-D entry equal to the row-major product of its dimensions, and converts
    the matching multi-result `ktdp.get_compute_tile_id` into a single-result
    flat id whose original per-dimension coordinates are reconstructed with
    row-major `arith.divui`/`arith.remui`. 1-D grids pass through unchanged.
  }];
  let constructor = "::scheduler::createNormalizeGridTo1DPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdp::KtdpDialect"
  ];
}

def NormalizeSCFForLoopsPass : Pass<"normalize-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Normalize scf.for loops to lb=0, step=1";
  let description = [{
    Rewrites every scf.for loop in the module so that it has lower bound 0 and
    step 1.  The induction variable is denormalized inside the body so that all
    existing uses continue to see the original iteration values:

      denormalized_iv = orig_lb + normalized_iv * orig_step

    The denormalization is emitted as an affine.apply immediately inside the
    loop body, then sunk to just before each use to minimise register pressure.
  }];
  let constructor = "::scheduler::createNormalizeSCFForLoopsPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect"
  ];
}

def TileSCFForLoopsPass : Pass<"tile-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Tile perfectly nested scf.for loops";
  let description = [{
    This pass tiles perfectly nested scf.for loops with four execution modes:
    
    1. Static mode with custom tiling (tile-sizes specified, fallback-scf-tiling=false):
       Uses command-line tile sizes with custom normalized tiling scheme
    2. Static mode with fallback SCF tiling (tile-sizes specified, fallback-scf-tiling=true):
       Uses command-line tile sizes with SCF's standard tiling (for testing)
    3. Dynamic mode with custom tiling (tile-sizes not specified, fallback-scf-tiling=false):
       Uses LoopTilingAnalysis to identify candidates and custom normalized tiling
    4. Dynamic mode with fallback SCF tiling (tile-sizes not specified, fallback-scf-tiling=true):
       Uses LoopTilingAnalysis with SCF's standard tiling (for testing)
    
    Custom tiling scheme example (uses ktdf.tiling.derive_size and ktdf.tiling.linearize_index):
    ```
    // Original normalized loops (lb=0, step=1)
    scf.for %i = %c0 to %c23 step %c1 {
      scf.for %j = %c0 to %c12 step %c1 {
        %sum = arith.addi %i, %j : index
      }
    }
    
    // After custom normalized tiling with tile sizes [5, 5]
    %n_tiles_i = arith.ceildivui %c23, %c5 : index
    scf.for %tile_i = %c0 to %n_tiles_i step %c1 {
      %n_tiles_j = arith.ceildivui %c12, %c5 : index
      scf.for %tile_j = %c0 to %n_tiles_j step %c1 {
        %tile_size_i = ktdf.tiling.derive_size [%tile_i : %c5], total_size = %c23
        %tile_size_j = ktdf.tiling.derive_size [%tile_j : %c5], total_size = %c12
        scf.for %point_i = %c0 to %tile_size_i step %c1 {
          scf.for %point_j = %c0 to %tile_size_j step %c1 {
            %i = ktdf.tiling.linearize_index[%tile_i : %c5], [%point_i : %c1]
            %j = ktdf.tiling.linearize_index[%tile_j : %c5], [%point_j : %c1]
            %sum = arith.addi %i, %j : index
          }
        }
      }
    }
    
    // After applying another round of custom tiling [3, 3] to innermost loops
    %n_tiles_i = arith.ceildivui %c23, %c5 : index
    scf.for %tile_i = %c0 to %n_tiles_i step %c1 {
      %n_tiles_j = arith.ceildivui %c12, %c5 : index
      scf.for %tile_j = %c0 to %n_tiles_j step %c1 {
        %tile_size_i = ktdf.tiling.derive_size [%tile_i : %c5], total_size = %c23
        %tile_size_j = ktdf.tiling.derive_size [%tile_j : %c5], total_size = %c12
        %n_tiles_pi = arith.ceildivui %tile_size_i, %c3 : index
        scf.for %tile_pi = %c0 to %n_tiles_pi step %c1 {
          %n_tiles_pj = arith.ceildivui %tile_size_j, %c3 : index
          scf.for %tile_pj = %c0 to %n_tiles_pj step %c1 {
            %tile_size_pi = ktdf.tiling.derive_size [%tile_pi : %c3], total_size = %tile_size_i
            %tile_size_pj = ktdf.tiling.derive_size [%tile_pj : %c3], total_size = %tile_size_j
            scf.for %point_pi = %c0 to %tile_size_pi step %c1 {
              scf.for %point_pj = %c0 to %tile_size_pj step %c1 {
                %point_i = ktdf.tiling.linearize_index[%tile_pi : %c3], [%point_pi : %c1]
                %point_j = ktdf.tiling.linearize_index[%tile_pj : %c3], [%point_pj : %c1]
                %i = ktdf.tiling.linearize_index[%tile_i : %c5], [%point_i : %c1]
                %j = ktdf.tiling.linearize_index[%tile_j : %c5], [%point_j : %c1]
                %sum = arith.addi %i, %j : index
              }
            }
          }
        }
      }
    }
    ```
    
    Standard SCF tiling example (fallback-scf-tiling=true):
    ```
    scf.for %i = 0 to 100 step 1 {
      scf.for %j = 0 to 100 step 1 { ... }
    }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %j0 = 0 to 100 step 32 {
        scf.for %i1 = %i0 to min(%i0+32, 100) step 1 {
          scf.for %j1 = %j0 to min(%j0+32, 100) step 1 { ... }
        }
      }
    }
    ```
  }];
  let constructor = "::scheduler::createTileSCFForLoopsPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for each loop dimension (for testing only)">,
    Option<"normalizeLoops", "normalize-loops", "bool",
           /*default=*/"true",
           "Normalize tiled loops to start at 0 and add outer IV to uses">,
    Option<"useFallbackSCFTiling", "fallback-scf-tiling", "bool",
           /*default=*/"false",
           "Use SCF's standard tiling instead of custom normalized tiling (for testing)">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def StripMineSCFForLoopsPass : Pass<"strip-mine-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Strip-mine scf.for loops";
  let description = [{
    This pass performs strip-mining on scf.for loops.
    Strip-mining splits a loop into an outer loop and an inner loop,
    where the outer loop iterates over blocks and the inner loop
    iterates within each block.
    
    When strip-mine-size is specified on the command line, it is used as a
    static value for all loops (testing mode). The size must be positive.
    
    When strip-mine-size is not specified, the pass uses LoopTilingAnalysis to
    identify strip-mining candidates and materializes ktdf.tiling.reserve_size 
    ops with metadata-based attributes for each candidate loop.
    
    Example transformation:
    ```
    scf.for %i = 0 to 100 step 1 { ... }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %i1 = %i0 to min(%i0+32, 100) step 1 { ... }
    }
    ```
  }];
  let constructor = "::scheduler::createStripMineSCFForLoopsPass()";
  let options = [
    ListOption<"stripMineSize", "strip-mine-size", "int64_t",
               "Strip-mine factor (block size, for testing only)">,
    Option<"normalizeLoops", "normalize-loops", "bool",
           /*default=*/"true",
           "Normalize loops to start at 0 and add outer IV to uses">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ParallelizeLoopsAcrossInstancesPass
    : Pass<"parallelize-loops-across-instances", "mlir::ModuleOp"> {
  let summary = "Map parallel loops onto hardware instances via ktdf.parallel";
  let description = [{
    For each ktdf.pipeline whose work is corelet-local (no core-shared
    units in the union of its immediate stages' applicable_units), find
    the outermost parallel-annotated scf.for in the pipeline's enclosing
    scope whose statically-known trip count permits a split across
    `num_instances` (preferring divisible trips; falling back to imbalanced
    when none qualifies), and rewrite that scf.for into a ktdf.parallel op
    with `distribute(num_instances = N)`. `N` is sourced from
    SchedulerExtContext::getNumCorelets() at pass construction; the
    pass itself is target-neutral.

    Body legality (cross-iteration dependencies, L1 sharing) is trusted
    via the loop_type=parallel_loop annotation set by upstream lowering.
    Reduction loops (loop_type=reduction_loop or unannotated) are not
    candidates.
  }];
  let constructor = "::scheduler::createParallelizeLoopsAcrossInstancesPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

def DoubleBufferingPass
    : Pass<"double-buffering", "mlir::ModuleOp"> {
  let summary = "Double-buffer scratchpad memrefs handed off between sibling stages";
  let description = [{
    For each ktdf.pipeline whose enclosing scope contains at least one scf.for
    and which has no pre-existing modulo attribute, find scratchpad memrefs
    yielded by the pipeline's ktdf.private region that are written by exactly
    one sibling stage and read by one or more sibling stages (via
    ktdf.data_transfer ops). For each such buffer, allocate two physical copies
    just outside the pipeline scope, insert one shared ktdf.buffer_phase and
    per-buffer ktdf.select_memref before the pipeline, rewrite the pipeline's
    references to use the selected memrefs, and set modulo(size: 2) on the
    pipeline.

    Memory-space eligibility is determined by querying the MemoryTree to check
    if the memory space is per-core scratchpad memory (e.g., L1 on Spyre).

    The pass leaves ktdf.private's signature consistent: dropped result slots
    are removed from the op and from ktdf.private_yield, and orphaned
    memref.alloc ops in the private region are erased.
  }];
  let constructor = "::scheduler::createDoubleBufferingPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect",
      "mlir::ktdp::KtdpDialect",
      "mlir::scf::SCFDialect",
      "mlir::memref::MemRefDialect"];
}

def AffineMinCanonicalizationPass : Pass<"affine-min-canonicalization", "mlir::ModuleOp"> {
  let summary = "Canonicalize arith.addi + arith.minsi + affine.apply to affine.min";
  let description = [{
    This pass transforms the pattern:
      %add = arith.addi %iv, %step
      %min = arith.minsi %add, %ub
      %result = affine.apply ()[%iv, %min] -> (-iv + min)
    
    Into the canonical form:
      %result = affine.min (%iv)[%ub, %step] -> (ub - iv, step)
    
    This simplifies the IR and makes it easier for subsequent passes like
    address-assignment to analyze bounds. The transformation is mathematically
    equivalent: both compute min(ub - iv, step).
  }];
  let constructor = "::scheduler::createAffineMinCanonicalizationPass()";
  let dependentDialects = [
      "mlir::affine::AffineDialect",
      "mlir::arith::ArithDialect"];
}

def AddressAssignmentPass : Pass<"address-assignment", "mlir::ModuleOp"> {
  let summary = "Analyze memory allocation and assign addresses";
  let description = [{
    This pass collects all memref.alloc operations in the module and groups
    them by memory space (L1, DDR), analyses their sizes and assigns addresses.

    Capacity and reservation are tracked in bytes, but the address an
    allocation is replaced with counts elements of its element type, matching
    the offset of `ktdp.construct_memory_view`. Each allocation is aligned to
    at least one element so the conversion is exact.
  }];

  let constructor = "::scheduler::createAddressAssignmentPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect",
      "mlir::memref::MemRefDialect",
      "mlir::affine::AffineDialect"];

  let statistics = [
    Statistic<"num_processed", "num-processed", "Number of processed allocations">,
    Statistic<"num_failed", "num-failed", "Number of failed allocations">
  ];
}

def AccessGranularityLegalizationPass
    : Pass<"access-granularity-legalization", "mlir::ModuleOp"> {
  let summary = "Enforce hardware physical access_granularity constraints on data_transfer ops";
  let description = [{
    Runs immediately after path-expansion and before splat-legalization.
    Enforces hardware physical access_granularity constraints across all
    ktdf.data_transfer operations (memory-to-memory and memory-to-FIFO),
    independently of whether a broadcast/splat is involved.

    For example, an MNILU transfer of [1, 1, 1, 1] may need to be widened to
    [1, 1, 1, 64] to satisfy DDR bus alignment constraints.

    Key responsibilities:
      1. Source buffer bounds checking: when widening a load transfer the pass
         inspects the source memref shape/allocation bounds and verifies the
         innermost dimension is large enough to support the widened physical
         read, or diagnoses unpadded small buffers that would cause
         out-of-bounds reads.
      2. Loads vs. stores: differentiates read widening (safe into padded
         intermediate buffers) from write widening (requires hardware
         byte-enables or masking).
      3. Integration with padding modes: leverages ZeroPadFront / ZeroPadBack
         transfer modes for sub-granule non-splat slices (boundary/tail tiles).
  }];
  let constructor = "::scheduler::createAccessGranularityLegalizationPass()";
}

def ChunkWideAccessesPass
    : Pass<"chunk-wide-accesses", "mlir::ModuleOp"> {
  let summary = "Split FIFO accesses wider than the throttle into a loop of throttle-sized accesses";
  let description = [{
    Finds every `ktdf.fifo.slot` allocated by a `ktdf.private` that holds more
    elements than the throttle -- the SIMD lane count of the default compute
    unit for the slot's element type. Such a slot is lowered to an
    `agen.vector_load` / `agen.vector_store` plus a `dataflow.send` /
    `dataflow.receive` of the whole slot, which the hardware cannot issue.

    The slot is shrunk to the throttle, and each access to it -- a
    `ktdf.write_to_fifo` or `ktdf.read_from_fifo` of a memref, or a
    `ktdf.data_transfer` between the slot and a memref -- is replaced by a
    throttle-sized `ktdf.data_transfer` enclosed in an `scf.for` nest that
    covers the original extent. The pieces, in loop order, are consecutive in
    row-major order of the original access, so both ends of the slot stay in
    step even when they are shaped differently.

    All the accesses of a slot are split together. A slot that cannot be split
    this way is diagnosed.
  }];
  let constructor = "::scheduler::createChunkWideAccessesPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def IndirectAddrBufFillLegalizationPass
    : Pass<"indirect-addr-buf-fill-legalization", "mlir::ModuleOp"> {
  let summary = "Stage indirect address buffer fills through a reachable memory";
  let description = [{
    Runs after path expansion. A `ktdf.data_transfer` that fills an indirect
    address buffer (a memory node carrying
    `ktdf_arch.feature.indirect_address_buffer`) is executed by the unit its
    stage is mapped to, so it can only read from a memory that has a datapath
    into that unit. A store unit, for instance, reads from L1 and cannot fill
    the buffer straight from global memory.

    For every illegal fill this pass:
      1. Allocates a buffer in the memory the filling unit does read from,
         adding it to the pipeline's `ktdf.private` resources.
      2. Copies the index data into that buffer from an earlier stage whose
         unit has a datapath from the original memory to the staging memory,
         under the same guard the fill carries.
      3. Repoints the fill at the staging buffer.

    Fills whose source memory already feeds the filling unit are left alone.
  }];
  let constructor =
      "::scheduler::createIndirectAddrBufFillLegalizationPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

def SplatLegalizationPass : Pass<"splat-legalization", "mlir::ModuleOp"> {
  let summary = "Validate splat loads, widen to access granularity, and name the sub-SIMD shuffle";
  let description = [{
    Runs after access-granularity-legalization. For each ktdf.data_transfer
    with a splat transfer mode whose destination is a FIFO:

      1. Validates that the load unit's SIMD feature declares the splat
         capability.
      2. Calls computeSplatGranularityElements to derive the effective load
         width from the load unit's access_granularity arch attribute.
      3. When that width exceeds the source element count -- when the hardware
         really does have to load more than was asked for -- legalizes the
         innermost static_source_sizes dimension to it.  The size operand
         carries the load width, so no attribute is emitted for it.
      4. In that same case, sets
         splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd> on each
         consuming ktdf.read_from_fifo.

    A legalized load splats the granule it read, not the single element: every
    sub-SIMD group ends up holding one live element instead of the whole vector
    holding one.  The splat mode says so, and names the shuffle that closes the
    gap; it is the pass's durable output.

    The shuffle itself is emitted by the lowering to DFIR, which reads the group
    width off the compute unit's sub_simd_lanes.  This pass queries the arch for
    the load width and nothing else.  Splat placement (ensuring the splat lands
    on the terminal FIFO load) is the responsibility of path expansion upstream.
  }];
  let constructor = "::scheduler::createSplatLegalizationPass()";
  let dependentDialects = [
    "mlir::ktdf::KTDFDialect"
  ];
}

def CustomLinalgBufferizationPass : Pass<"custom-linalg-bufferization", "mlir::ModuleOp"> {
  let summary = "Absorb bufferization.to_tensor into its buffer-mode consumer";
  let description = [{
    Rebuilds every `linalg.generic` that reads a `bufferization.to_tensor` so it
    operates on buffers, and drops the `to_tensor`.

    A `to_tensor` is how a device pattern hands a buffer it materialized to a
    consumer it cannot retype.  PDL replaces a value with one of the same type;
    it cannot reach into that consumer to change its operand types or drop its
    results.  So the pattern leaves a tensor view behind and this pass finishes
    the job.

    For each `bufferization.to_tensor` whose result feeds a `linalg.generic`:

      1. Allocates a `memref.alloc` for each of the generic's tensor results,
         in the memory local to the stage's unit -- naming the memory is what
         lets AddressAssignmentPass place the buffer.
      2. Re-emits every other tensor input as a memref-typed
         `ktdf.read_from_fifo`.  A tensor input from anything else is an error:
         there is no buffer to name for it.
      3. Rebuilds the generic in buffer semantics -- same indexing maps, same
         iterator types, same body, no results -- and re-points every consumer
         of its results at the matching output buffer.
      4. Erases the original generic, the `to_tensor`, and whatever they leave
         dead.

    Indexing maps are left alone.  The broadcast map a splat legalization
    installed already reads the element the splat made uniform, so there is
    nothing about the buffer that changes how the generic indexes it.

    The pass is not specific to splats: any `to_tensor` whose uses can be
    converted is absorbed.
  }];
  let constructor = "::scheduler::createCustomLinalgBufferizationPass()";

  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def EnsureDeviceDeclarationPass
    : Pass<"ensure-device-declaration", "mlir::ModuleOp"> {
  let summary = "Ensure that there is a ktdf_arch.device declaration";
  let description = [{
    Detect if there is a ktdf_arch.device declaration. 
    If not, insert a device declaration based on the option given to this pass.
    And also, ensure ktdf_arch::DeviceManager is instantiated.
  }];
  let constructor = "::scheduler::createEnsureDeviceDeclarationPass()";
  let options = [
    Option<"deviceFileName", "device-filename", "std::string",
           /*default=*/"",
           "Path to the file containing the architecture description">,
    Option<"deviceName", "device-name", "std::string",
           /*default=*/"",
           "Name of the device definition. If unspecified, the only device in the file is chosen">
  ]; 
  let dependentDialects = [
    "mlir::ktdf_arch::KTDFArchDialect"
  ];
}

def MaterializeRegistersPass : Pass<"materialize-registers", "mlir::ModuleOp"> {
  let summary = "Make the registers a generic body uses real, in front of it";
  let description = [{
    Visits every nested `linalg.generic` and makes the registers it uses real
    in front of it: an allocation the body carries, and an `arith.constant`
    with a `ktdf_arch.maps_to` attribute, which needs an allocation and a fill
    of its own.

    This requires the cardinality of the `linalg.generic` iteration domain to
    be statically known, and all consumers of the registers to be either
    `memref.load`, `memref.store` or `ktdf.opaque`. Register sizes are
    increased to match the iteration domain, but scalar users remain fixed to
    the first element of the allocated segment. A store of a value the body
    does not compute per element fills every element instead.

    Example:
    ```mlir
    linalg.generic ins(%in: tensor<1x64xf16>) ... {
    ^bb0(%arg0: f16);
      %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.0 : f16
      %reg = memref.alloc() : memref<f16>
      memref.store %value, %reg[] : ...
      %ktdf.opaque ins(%arg0, %cst: ...) outs(%reg: ...)
    }
    ```
    
    Is transformed into:
    ```mlir
    %cst0 = arith.constant 0 : index
    %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.0 : f16
    %cst_reg = memref.alloc() : memref<64xf16, "SFU_REG">
    linalg.fill ins(%cst: ...) outs(%cst_reg) ...
    %reg = memref.alloc() : memref<64xf16>
    linalg.generic ins(%in: tensor<1x64xf16>) ... {
    ^bb0(%arg0: f16);
      memref.store %value, %reg[%cst0] : ...
      %ktdf.opaque ins(%arg0, %cst_reg: ...) outs(%reg: ...)
    }
    ```
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def ApplyDevicePatternsPass : Pass<"apply-device-patterns", "mlir::func::FuncOp"> {
  let summary = "Applies patterns defined in the `ktdf_arch.device`";
  let description = [{
    Given a root operation that is mapped to a `ktdf_arch.device`, this pass
    applies all the (enabled groups of) patterns defined within the device to
    the IR nested below the root operation.
  }];

  let options = [
    ListOption<"enabled_groups", "groups", "std::string", 
      "List of pattern groups to enable, or empty for ungrouped patterns only">
  ];

  // Everything a device's patterns may name, plus what a device file itself
  // parses as.  A pattern creating an op of an unloaded dialect is a hard
  // error, and the device is imported while the pass is already running, so a
  // dialect missing here takes the whole run down.
  let dependentDialects = [
    "mlir::agen::AgenDialect",
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::func::FuncDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::ktdp::KtdpDialect",
    "mlir::memref::MemRefDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::vectorchain::VectorChainDialect"
  ];
}

def HoistInvariantsPass : Pass<"hoist-invariants", "mlir::func::FuncOp"> {
  let summary = "Hoist invariants out of pipelines and loops";
  let description = [{
    Given a pure operation that is invariant under the pipeline or loop 
    structure it is contained in, hoists the invariant to the surrounding scope.

    Example:
    ```mlir
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ... {
          ^bb0(%arg0: i32):
            %cst0 = arith.constant 0 : i32
            arith.addi %cst0, %arg0 : i32
          }
        }
      }
    }
    ```

    Is transformed into:
    ```mlir
    %cst0 = arith.constant 0 : i32
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ... {
          ^bb0(%arg0: i32):
            arith.addi %cst0, %arg0 : i32
          }
        }
      }
    }
    ```
  }];

  let statistics = [
    Statistic<"num_hoisted", "num-hoisted", "Number of hoisted operations">,
  ];
}

def HoistConstantStoragePass : Pass<"hoist-constant-storage", "mlir::func::FuncOp"> {
  let summary = "Hoists memory allocations with constant contents";
  let description = [{
    Given an allocation that is used to hold a constant value, hoists the 
    allocation and its write to the surrounding scope. If the write is in a
    pipeline, it will be hoisted into a sub-pipeline.

    Example:
    ```mlir
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          %cst = arith.constant 1.0 : f16
          %alloc = memref.alloc() : memref<f16, "UNIT">
          memref.store %cst, %alloc : memref<f16, "UNIT">
          linalg.generic ...
            ins(%alloc: memref<f16, "UNIT">)
            outs(...)
        } {applicable_units = ["UNIT"]}
      }
    }
    ```

    Is transformed into:
    ```mlir
    %cst = arith.constant 1.0 : f16
    %alloc = memref.alloc() : memref<f16, "UNIT">
    ktdf.pipeline {
      ktdf.stage {
        memref.store %cst, %alloc : memref<f16, "UNIT">
      } {applicable_units = ["UNIT"]}
    }
    scf.for ... {
      ktdf.pipeline {
        ktdf.stage {
          linalg.generic ...
            ins(%alloc: memref<f16, "UNIT">)
            outs(...)
        } {applicable_units = ["UNIT"]}
      }
    }
    ```
  }];

  let dependentDialects = [
    "mlir::ktdf::KTDFDialect"
  ];

  let statistics = [
    Statistic<"num_hoisted", "num-hoisted", "Number of hoisted constant storage allocations">,
  ];
}

def FoldFifoRelayStagesPass
    : Pass<"fold-fifo-relay-stages", "mlir::ModuleOp"> {
  let summary = "Fold transfers relaying memory to memory through FIFOs";
  let description = [{
    Folds a transfer from memory into a FIFO slot, and the transfer out of that
    slot into memory, into a single memory-to-memory transfer. Each such relay
    folds on its own, so the other transfers of its two stages may go to other
    stages and between other memories.

    Path expansion can only split stages, never merge them, so it expects
    every transfer between two memories in one stage. If no unit can perform
    such a transfer, path expansion splits it again.

    A relay can fold when:
    - Its two stages hold nothing but transfers.
    - Its slot is written once and read once, by different stages, and has no
      other users.
    - Its source and destination are in different memories.
    - What goes into the slot is what comes out of it (ie fifo-side sizes must match).
    - Its two transfers do not carry different throttles.

    The folded transfer goes into one of the two stages, the target, which
    also awaits what the other stage awaits. The other stage loses its half of
    the relay, and is erased if left empty. The stages that awaited an erased
    stage await what it awaited instead.

    Each transfer reads from one place and writes to another. For a memref,
    the place is its memory. For a FIFO slot, it is the resource that the
    slot's type names on the stage's side: the slot's source if the stage
    writes into it, and its destination if the stage reads from it. For
    example, a stage that writes into `!ktdf.fifo.slot<"MNILU" -> "L1LU", ...>`
    writes to "MNILU".

    The anchor of a stage is a place that all of its transfers read from, or
    a place that all of them write to. A stage that reads from DDR and a
    stage that writes to DDR have different anchors. If the transfers share
    both the place they read from and the place they write to, a memory wins
    over a FIFO. If both places are memories, or both are FIFOs, both are
    anchors.

    The fold must keep what path expansion expects:
    - The dependencies stay free of cycles.
    - Each of the two stages that still holds transfers has an anchor.
    - A stage only gains an anchor that no other stage has.
    - The stages that await the producer, other than the consumer, still see
      the source read. The stages that await the consumer still see the
      destination written.
    - The applicable units of the stages intersect, if both have any.

    The target is the stage whose fold leaves the other one empty if there is
    one, then the producer, then the consumer. Afterwards, stages are reordered
    so that each comes after the stages it awaits, and the dependency tokens
    are normalized (see `simplify-pipeline`).

    Example:
    ```mlir
    ktdf.stage depends_in(none) depends_out(%token) {
      ktdf.data_transfer from %ddr[%i, %j] size [1, 64] to %slot size [1, 64]
        : memref<1x64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    }
    ktdf.stage depends_in(%token) depends_out(none) {
      ktdf.data_transfer from %slot size [1, 64] to %l1[0, 0] size [1, 64]
        : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<1x64xf16, "L1">
    }
    ```

    Is transformed into:
    ```mlir
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %ddr[%i, %j] size [1, 64] to %l1[0, 0] size [1, 64]
        : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
    }
    ```
  }];

  let dependentDialects = [
    "mlir::ktdf::KTDFDialect"
  ];

  let statistics = [
    Statistic<"num_folded", "num-folded", "Number of folded relays">,
  ];
}

def OrderFifoSlotsByUsePass
    : Pass<"order-fifo-slots-by-use", "mlir::ModuleOp"> {
  let summary = "Write and read the slots of a FIFO in the order they are used";
  let description = [{
    The slots between the same pair of ends share one physical FIFO, whichever
    allocation and pipeline of the function they come from, and it delivers in
    the order it was filled. The
    lowering below makes each read a receive the consumer takes the value from
    at the op using it, so the slots have to be written in the order the
    consumer uses them, not merely read in the order they are written.

    For each such group of slots, this pass orders the writes in the producer
    and the reads in the consumer by where the consumer first uses each value:
    the position of the consuming op in the stage, then, for a linalg.generic,
    the position in its body of the first op using the operand.

    Reordering a FIFO changes the order it delivers in, so it is done only when
    that order is known: one block writes all of its slots, another block in a
    different stage reads all of them, and no other op of the function takes a
    slot of the FIFO. Each write and read then runs once per run of its block,
    so the FIFO carries the order of the writes over and over, and applying one
    permutation to the writes and the reads keeps both ends in step. A pipeline
    is a DAG of stages, so neither stage is on a cycle.

    A FIFO is left alone when:
    - a slot is not written and read exactly once;
    - its slots are written or read in more than one block, or written and
      read in the same stage;
    - another op of the function takes a slot between the same ends, e.g. in
      another stage, branch of the stage DAG, or nested pipeline;
    - a value is used outside the consumer's block;
    - a write would have to move above the definition of one of its operands,
      or above an op between the writes that touches memory.

    Example, where the consumer copies %b into a register before it adds %a:
    ```mlir
    ktdf.stage depends_in(none) depends_out(%t) {
      ktdf.data_transfer from %a_mem[...] to %slots#0 ...
      ktdf.data_transfer from %b_mem[...] to %slots#1 ...
    }
    ktdf.stage depends_in(%t) depends_out(none) {
      %a = ktdf.read_from_fifo %slots#0 ...
      %b = ktdf.read_from_fifo %slots#1 ...
      memref.copy %b, %reg ...
      linalg.generic ins(%a, %reg ...) ...
    }
    ```

    The write and the read of %slots#1 are moved in front of those of
    %slots#0.
  }];

  let dependentDialects = [
    "mlir::ktdf::KTDFDialect"
  ];

  let statistics = [
    Statistic<"num_reordered", "num-reordered",
              "Number of FIFOs whose slots were reordered">,
    Statistic<"num_skipped", "num-skipped",
              "Number of FIFOs left unchanged as the order they deliver in "
              "can't be told">,
  ];
}

def IndirectAddrBufLegalizationPass
    : Pass<"indirect-addr-buf-legalization", "mlir::ModuleOp"> {
  let summary = "Legalize indirect address buffer usage";
  let description = [{
    Materializes the ind_addr_buf window loop and per-entry loop required by
    the hardware indirect address buffer constraint.
    Outer scf.for over ind_addr_buf windows (annotated parallel), then inner
    scf.for over entries; the ind_addr_buf fill only runs on the entry loop's
    first iteration, guarded by an scf.if on the entry induction variable,
    with the ind_addr_buf memref view hoisted once above the entire
    window/entry loop nest instead of threaded as an scf.for iter-arg.
    No-op when no ktdp_lowering.construct_indirect_access_tile is present.
  }];
  let constructor = "::scheduler::createIndirectAddrBufLegalizationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::ktdf::KTDFDialect"
  ];
}

def IndirectAccessLoopMaterializationPass
    : Pass<"indirect-access-loop-materialization", "mlir::ModuleOp"> {
  let summary = "Materialize loops for non-contiguous direct subscripts of "
                "indirect access tiles";
  let description = [{
    Inspects the remaining direct-subscript intermediate variables of each
    ktdp_lowering.construct_indirect_access_tile. A variable is materialized
    into a scf.for loop when its dimension fails the dense-packing condition
    (stride[k] != size[k+1]*stride[k+1], or stride != 1 for the innermost
    dimension of $base) OR the full-coverage condition (variables_space_set
    bound < dimension size). Variables that satisfy both conditions are
    retained so that ConstructThreeStagePipeline can read the per-transfer
    tile size, as long as the retained region does not exceed the SIMD lane
    count of the default compute unit for $base's element type. Retained
    variables are admitted innermost-first; the first one that would push the
    region past the lane count, and every variable outside it, is
    materialized instead. The innermost variable is always retained. If the
    device has no default compute unit or SIMD lane count for the element
    type, `materialize-loops-without-simd-lanes` (default true) selects
    between materializing every variable but the innermost one and applying
    no lane limit. Materialized loops are annotated parallel_loop.
    No-op when all remaining variables are retained, or when no
    ktdp_lowering.construct_indirect_access_tile is present.
  }];
  let constructor = "::scheduler::createIndirectAccessLoopMaterializationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::ktdf::KTDFDialect"
  ];
}

def SimplifyLinalgPass : Pass<"simplify-linalg", "mlir::func::FuncOp"> {
  let summary = "Simplifies `linalg` operations";
  let description = [{
    Rewrites `linalg.generic` and adjacent `tensor` operations to simplify the
    computational demand for the device. In particular:

    - Reduces the working set as much as possible.

      By slicing the input and output operands, the working set of the `generic`
      op is reduced as much as possible. These are expected to be folded into 
      adjacent transfers, reducing the memory bandwidth and size of
      intermediates.

    - Normalizes to unit stride.

      A `generic` that accesses an operand with greather-than-unit strides has
      its operands sliced such that it works on unit stride. These are expected 
      to be folded into adjacent transfers, reducing the memory bandwidth and 
      size of intermediates, as well as exposing a dense computation.

    - Promoting constant fills to `arith.constant dense<...>`.

      Invariant inputs and yields are not hoisted out of the `generic`, so that
      broadcasts stay visible to the data transfers that implement them.

    - Demoting `out` to `in` operands.

      Outputs that aren't being modified in the body are turned into inputs,
      allowing further simplification.

    - The `linalg` canonicalizations.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_TRANSFORMS_PASSES_TD
