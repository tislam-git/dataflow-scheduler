add_subdirectory(Utils)

add_dataflow_scheduler_library(DataflowSchedulerTransforms
    PathExpansion/Materializer.cpp
    PathExpansion/Planner.cpp
    PathExpansion/PlannerFactories.cpp
    AccessGranularityLegalization.cpp
    CustomLinalgBufferization.cpp
    AddressAssignment.cpp
    IndirectAddrBufLegalization.cpp
    IndirectAccessLoopMaterialization.cpp
    AffineMinCanonicalization.cpp
    ApplyDevicePatterns.cpp
    ChunkWideAccesses.cpp
    DoubleBuffering.cpp
    EnsureDeviceDeclaration.cpp
    FoldFifoRelayStages.cpp
    HoistConstantStorage.cpp
    HoistInvariants.cpp
    IndirectAddrBufFillLegalization.cpp
    MaterializeRegisters.cpp
    NormalizeGridTo1D.cpp
    NormalizeSCFForLoops.cpp
    OrderFifoSlotsByUse.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PathExpansion.cpp
    SimplifyLinalg.cpp
    SplatLegalizationPass.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Transforms

  GROUP DIALECT
  EXPORT_NAME Transforms
  
  DEPENDS
    DataflowSchedulerTransformsPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::ConversionUtils
    DataflowScheduler::KTDFAnalysis
    DataflowScheduler::KTDFLoweringDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowSchedulerDialects::KTDFArchTransforms
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::KTDPLoweringDialect
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRIR
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRPass
    MLIRPresburger
    MLIRTransformUtils
    MLIRValueBoundsOpInterface
    MLIRAlignmentAttrInterface
)
