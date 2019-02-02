#pragma once

enum class EConvertTargetPolicy {
    CastFloat,
    UseClassNames,
    MakeClassNames
};

enum class EOverfittingDetectorType {
    Wilcoxon,
    IncToDec,
    Iter
};

enum class ESamplingFrequency {
    PerTree,
    PerTreeLevel
};

enum class ESamplingType {
    Groupwise,
    Docwise
};

enum class EFeatureType {
    Float,
    Categorical
};

enum EErrorType {
    PerObjectError,
    PairwiseError,
    QuerywiseError
};

enum class ETaskType {
    GPU,
    CPU
};

enum EBoostingType {
    Ordered,
    Plain
};

enum class EDataPartitionType {
    FeatureParallel,
    DocParallel
};

enum class ELoadUnimplementedPolicy {
    SkipWithWarning,
    Exception,
    ExceptionOnChange
};

enum class ELeavesEstimation {
    Gradient,
    Newton,
    //Use optimal leaves from structure search for model
    Simple
};

enum class EScoreFunction {
    SolarL2,
    Correlation,
    NewtonL2,
    NewtonCorrelation,
    LOOL2,
    SatL2,
    L2
};

enum class EBootstrapType {
    Poisson,
    Bayesian,
    Bernoulli,
    No
};

enum class EGrowingPolicy {
    ObliviousTree,
    Lossguide,
    Levelwise,
    Region
};

enum class ENanMode {
    Min,
    Max,
    Forbidden
};

enum class ECrossValidation {
    Classical,
    Inverted
};

enum class ELossFunction {
    /* binary classification errors */

    Logloss,
    CrossEntropy,
    CtrFactor,

    /* regression errors */

    RMSE,
    Lq,
    MAE,
    Quantile,
    LogLinQuantile,
    MAPE,
    Poisson,
    MSLE,
    MedianAbsoluteError,
    SMAPE,

    /* multiclassification errors */

    MultiClass,
    MultiClassOneVsAll,

    /* pair errors */

    PairLogit,
    PairLogitPairwise,

    /* ranking errors */
    YetiRank,
    YetiRankPairwise,
    QueryRMSE,
    QuerySoftMax,
    QueryCrossEntropy,

    /* user defined errors */

    Custom,
    UserPerObjMetric,
    UserQuerywiseMetric,

    /* regression metrics */

    R2,
    NumErrors,

    /* classification metrics */

    AUC,
    Accuracy,
    BalancedAccuracy,
    BalancedErrorRate,
    BrierScore,
    Precision,
    Recall,
    F1,
    TotalF1,
    MCC,
    ZeroOneLoss,
    HammingLoss,
    HingeLoss,
    Kappa,
    WKappa,
    LogLikelihoodOfPrediction,

    /* pair metrics */

    PairAccuracy,

    /* ranking metrics */
    AverageGain,
    QueryAverage,
    PFound,
    PrecisionAt,
    RecallAt,
    MAP,
    NDCG
};

enum class EHessianType {
    Symmetric,
    Diagonal
};


enum class ECounterCalc {
    Full,
    SkipTest
};

enum class EPredictionType {
    Probability,
    Class,
    RawFormulaVal,
    InternalRawFormulaVal
};

enum class EFstrType {
    PredictionValuesChange  /* "PredictionValuesChange", "FeatureImportance" */,
    LossFunctionChange,
    InternalFeatureImportance,
    Interaction,
    InternalInteraction,
    ShapValues
};

enum class EObservationsToBootstrap {
    LearnAndTest,
    TestOnly
};

enum class EGpuCatFeaturesStorage {
    CpuPinnedMemory,
    GpuRam
};

enum class EProjectionType {
    TreeCtr,
    SimpleCtr
};


enum class EPriorEstimation {
    No,
    BetaPrior
};

enum class ELaunchMode {
    Train,
    Eval,
    CV
};

enum class ENodeType {
    Master,
    SingleHost
};

enum class EModelType {
    CatboostBinary /* "CatboostBinary", "cbm", "catboost" */,
    AppleCoreML    /* "AppleCoreML", "coreml"     */,
    Cpp            /* "Cpp", "CPP", "cpp" */,
    Python         /* "Python", "python" */,
    Json           /* "Json", "json"       */,
    Onnx           /* "Onnx", "onnx" */
};

enum class EFinalCtrComputationMode {
    Skip,
    Default
};

enum class ELeavesEstimationStepBacktracking {
    None,
    AnyImprovment,
    Armijo
};

enum class EKappaMetricType {
    Cohen,
    Weighted
};

enum class ENdcgMetricType {
    Base,
    Exp
};

enum class EMetricBestValue {
    Max,
    Min,
    FixedValue,
    Undefined
};
