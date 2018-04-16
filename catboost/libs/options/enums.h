#pragma once

enum class EOverfittingDetectorType {
    Wilcoxon,
    IncToDec,
    Iter
};

enum class ESamplingFrequency {
    PerTree,
    PerTreeLevel
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

enum class ENanMode {
    Min,
    Max,
    Forbidden
};

enum class ELossFunction {
    /* binary classification errors */

    Logloss,
    CrossEntropy,
    CtrFactor,

    /* regression errors */

    RMSE,
    MAE,
    Quantile,
    LogLinQuantile,
    MAPE,
    Poisson,

    /* multiclassification errors */

    MultiClass,
    MultiClassOneVsAll,

    /* pair errors */

    PairLogit,

    /* ranking errors */
    YetiRank,
    QueryRMSE,
    QuerySoftMax,

    /* user defined errors */

    Custom,
    UserPerObjMetric,
    UserQuerywiseMetric,

    /* regression metrics */

    R2,

    /* classification metrics */

    AUC,
    Accuracy,
    Precision,
    Recall,
    F1,
    TotalF1,
    MCC,

    /* pair metrics */

    PairAccuracy,

    /* ranking metrics */
    QueryAverage,
    PFound
};

enum class ECounterCalc {
    Full,
    SkipTest
};

enum class EPredictionType {
    Probability,
    Class,
    RawFormulaVal
};

enum class EFstrType {
    FeatureImportance,
    InternalFeatureImportance,
    Interaction,
    InternalInteraction,
    Doc,
    ShapValues
};

enum class EObservationsToBootstrap {
    LearnAndTest,
    TestOnly
};

enum EGpuCatFeaturesStorage {
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
    Worker,
    SingleHost
};

enum class EModelType {
    CatboostBinary,
    AppleCoreML,
    CPP,
    Python
};
