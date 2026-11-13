add_dataflow_scheduler_library(DataflowSchedulerKTDFLowToDFIR
    FoldSubViews.cpp
    KTDFLowToDFIR.cpp
    LogicalMemoryViewBuilder.cpp
    OperationLowerings.cpp
    ParallelLowering.cpp
    PreludeWorkPartition.cpp
    ProgramUnitBuilder.cpp
    QueryMapArithCollapse.cpp
    SymbolicStartAddress.cpp
    UnitTypeDiscovery.cpp
    Utils.cpp
    BufferPhaseLowering.cpp
    DataTransferLowering.cpp
    LinalgLowering.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR

  GROUP CONVERSION
  EXPORT_NAME KTDFLowToDFIR

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    KTIR::SpyreOpDialect
    DataflowSchedulerDialects::AgenDialect
    DataflowScheduler::ConversionUtils
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowScheduler::KTDFLoweringDialect
    DataflowScheduler::KTDFToKTDFLow
    DataflowScheduler::KTDFUtils
    DataflowScheduler::KTDPLoweringDialect
    DataflowScheduler::TransformUtils
    DataflowSchedulerDialects::SymbolDialect
    DataflowSchedulerDialects::UniformDialect
    DataflowScheduler::Utils
    DataflowSchedulerDialects::VectorChainDialect

  LINK_MLIR_LIBS
    MLIRAffineAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTransformUtils
    MLIRViewLikeInterface
)
