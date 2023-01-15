#pragma once

#include <catboost/libs/model/enums.h>

#include <util/system/types.h>


enum class EOverfittingDetectorType {
    None,
    Wilcoxon,
    IncToDec,
    Iter
};

enum class ESamplingFrequency {
    PerTree,
    PerTreeLevel
};

enum class ESamplingUnit {
    Object,
    Group
};

enum class EFeatureType {
    Float,
    Categorical,
    Text
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
    Exact,
    //Use optimal leaves from structure search for model
    Simple
};

enum class EScoreFunction {
    SolarL2,
    Cosine,
    NewtonL2,
    NewtonCosine,
    LOOL2,
    SatL2,
    L2
};

enum class EModelShrinkMode {
    Constant,
    Decreasing
};

enum class EBootstrapType {
    Poisson,
    Bayesian,
    Bernoulli,
    MVS, // Minimal Variance Sampling, scheme of bootstrap with subsampling, which reduces variance in score approximation
    No
};

enum class EGrowPolicy {
    SymmetricTree,
    Lossguide,
    Depthwise,
    Region
};

enum class ENanMode {
    Min,
    Max,
    Forbidden
};

enum class ECrossValidation {
    Classical,
    Inverted,
    TimeSeries
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
    Expectile,
    LogLinQuantile,
    MAPE,
    Poisson,
    MSLE,
    MedianAbsoluteError,
    SMAPE,
    Huber,
    Tweedie,

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
    StochasticFilter,
    StochasticRank,

    /* user defined errors */

    PythonUserDefinedPerObject,
    PythonUserDefinedMultiRegression,
    UserPerObjMetric,
    UserQuerywiseMetric,

    /* regression metrics */

    R2,
    NumErrors,
    FairLoss,

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
    NormalizedGini,
    PRAUC,

    /* pair metrics */

    PairAccuracy,

    /* ranking metrics */
    AverageGain,
    QueryAverage,
    PFound,
    PrecisionAt,
    RecallAt,
    MAP,
    NDCG,
    DCG,
    FilteredDCG,

    /* multi-regression */
    MultiRMSE,

    Combination
};

enum class ERankingType {
    CrossEntropy,
    AbsoluteValue,
    Order
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
    LogProbability,
    Class,
    RawFormulaVal,
    Exponent,
    InternalRawFormulaVal
};

enum class EFstrType {
    PredictionValuesChange,
    LossFunctionChange,
    FeatureImportance,
    InternalFeatureImportance,
    Interaction,
    InternalInteraction,
    ShapValues,
    PredictionDiff,
    ShapInteractionValues
};

enum class EFstrCalculatedInFitType {
    PredictionValuesChange,
    LossFunctionChange,
    FeatureImportance
};

enum class EPreCalcShapValues {
    Auto,
    UsePreCalc,
    NoPreCalc
};

enum class ECalcTypeShapValues {
    Approximate,
    Regular,
    Exact
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

enum class EFinalCtrComputationMode {
    Skip,
    Default
};

enum class EFinalFeatureCalcersComputationMode {
    Skip,
    Default
};

enum class ELeavesEstimationStepBacktracking {
    No,
    AnyImprovement,
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

enum class ENdcgDenominatorType {
    LogPosition,
    Position
};

enum class EMetricBestValue {
    Max,
    Min,
    FixedValue,
    Undefined
};

enum class EFeatureCalcerType : ui32 {
//Examples
//    LinearModel,
//    TfIdf,
//    NGrams
    BoW,
    NaiveBayes,
    BM25,
    CosDistanceWithClassCenter,
    GaussianHomoscedasticModel,
    GaussianHeteroscedasticModel,
    EmbeddingDistanceToClass
};

enum class EAutoClassWeightsType {
    Balanced,
    SqrtBalanced,
    None
};

namespace NCB {
    enum class EFeatureEvalMode {
        OneVsNone,
        OneVsOthers,
        OneVsAll,
        OthersVsAll
    };

    enum class ERawTargetType : ui32 {
        Integer,
        Float,
        String,
        None
    };
}
