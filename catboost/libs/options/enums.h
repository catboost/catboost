#pragma once

enum class EOverfittingDetectorType {
    Wilcoxon,
    IncToDec,
    Iter
};

enum class EWeightSamplingFrequency {
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
    Dynamic,
    Plain
};

enum class ELoadUnimplementedPolicy {
    SkipWithWarning,
    Exception,
    ExceptionOnChange
};

enum class ELeavesEstimation {
    Gradient,
    Newton
};

enum class EScoreFunction {
    SolarL2,
    Correlation
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

    /* regression errors */

    RMSE,
    MAE,
    Quantile,
    LogLinQuantile,
    MAPE,
    Poisson,
    QueryRMSE,

    /* multiclassification errors */

    MultiClass,
    MultiClassOneVsAll,

    /* pair errors */

    PairLogit,

    /* ranking errors */
    YetiRank,

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

    /* custom errors */

    Custom,
    UserPerObjErr,
    UserQuerywiseErr
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
    Doc
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
