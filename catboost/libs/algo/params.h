#pragma once

#include "error_holder.h"
#include "ders_holder.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>

#include <catboost/libs/model/split.h>

#include <library/binsaver/bin_saver.h>
#include <library/grid_creator/binarization.h>
#include <library/json/json_reader.h>

#include <util/generic/algorithm.h>
#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/maybe.h>
#include <util/datetime/systime.h>

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

    /* multiclassification errors */

    MultiClass,
    MultiClassOneVsAll,

    /* pair errors */

    PairLogit,

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

    Custom
};

enum class ELeafEstimation {
    Gradient,
    Newton
};

enum class EFeatureType {
    Float,
    Categorical
};

enum class ECounterCalc {
    Full,
    SkipTest
};

constexpr int CB_THREAD_LIMIT = 56;

struct TCtrDescription {
    ECtrType CtrType = ECtrType::Borders;
    int TargetBorderCount = 1;
    EBorderSelectionType TargetBorderType = EBorderSelectionType::MinEntropy;

    TCtrDescription() = default;

    explicit TCtrDescription(const ECtrType& ctrType)
        : CtrType(ctrType)
    {
    }
};

struct TCtrParams {
    int CtrBorderCount = 16;
    int MaxCtrComplexity = 4;
    yvector<float> DefaultPriors = {0, 0.5, 1};
    yvector<std::pair<int, yvector<float>>> PerCtrPriors;
    yvector<std::pair<int, yvector<float>>> PerFeaturePriors;
    yvector<std::pair<std::pair<int, int>, yvector<float>>> PerFeatureCtrPriors;
    yvector<TCtrDescription> Ctrs = {TCtrDescription(), TCtrDescription(ECtrType::Counter)};
};

enum class EPredictionType {
    Probability,
    Class,
    RawFormulaVal
};

struct TCustomMetricDescriptor {
    void* CustomData;

    TErrorHolder (*EvalFunc)(const yvector<yvector<double>>& approx,
                             const yvector<float>& target,
                             const yvector<float>& weight,
                             int begin, int end, void* customData) = nullptr;
    TString (*GetDescriptionFunc)(void* customData) = nullptr;
    bool (*IsMaxOptimalFunc)(void* customData) = nullptr;
    double (*GetFinalErrorFunc)(const TErrorHolder& error, void* customData) = nullptr;
};

struct TCustomObjectiveDescriptor {
    void* CustomData;
    void (*CalcDersRange)(int count, const double* approxes, const float* targets,
                          const float* weights, TDer1Der2* ders, void* customData) = nullptr;
    void (*CalcDersMulti)(const yvector<double>& approx, float target, float weight,
                          yvector<double>* ders, TArray2D<double>* der2, void* customData) = nullptr;
};

enum class EFstrType {
    FeatureImportance,
    InternalFeatureImportance,
    Interaction,
    InternalInteraction,
    Doc
};

const int ParameterNotSet = -1;

struct TOverfittingDetectorParams {
    float AutoStopPval = 0;
    EOverfittingDetectorType OverfittingDetectorType = EOverfittingDetectorType::IncToDec;
    int OverfittingDetectorIterationsWait = 20;
};

class TFitParams {
public:
    int Iterations = 500;
    int ThreadCount = 8;

    // Objective, always maximized.
    // TODO(asaitgalin): Rename.
    // TODO(annaveronika): remove LossFunction.
    ELossFunction LossFunction;
    bool StoreExpApprox;
    ENanMode NanMode = ENanMode::Min;
    TString Objective = ToString<ELossFunction>(ELossFunction::RMSE);
    TMaybe<TCustomObjectiveDescriptor> ObjectiveDescriptor;

    // Custom metrics to calculate and log.
    // TODO(asaitgalin): Rename.
    yvector<TString> CustomLoss;
    // Main evaluation metric (used for overfitting detection and best model selection).
    // If not specified default metric for objective is used.
    TMaybe<TString> EvalMetric;
    TMaybe<TCustomMetricDescriptor> EvalMetricDescriptor;

    float Border = 0.5f;
    float LearningRate = 0.03f;
    int Depth = 6;
    size_t RandomSeed = GetCycleCount();
    int GradientIterations = 1;
    float Rsm = 1;
    int FoldPermutationBlockSize = ParameterNotSet;
    int BorderCount = 128;
    TCtrParams CtrParams;
    ECounterCalc CounterCalcMethod = ECounterCalc::Full;
    TOverfittingDetectorParams OdParams;
    bool UseBestModel = false;
    bool DetailedProfile = false;
    ELeafEstimation LeafEstimationMethod = ELeafEstimation::Gradient;
    yvector<int> IgnoredFeatures;
    TString TimeLeftLog = "time_left.tsv";
    int ClassesCount = 0;
    yvector<float> ClassWeights;
    yvector<TString> ClassNames;
    size_t OneHotMaxSize = 0;
    float RandomStrength = 1;
    float BaggingTemperature = 1.0f;

    TString LearnErrorLog = "learn_error.tsv";
    TString TestErrorLog = "test_error.tsv";

    EBorderSelectionType FeatureBorderType = EBorderSelectionType::MinEntropy;
    float L2LeafRegularizer = 3;
    bool Verbose = false;
    bool HasTime = false;
    TString Name = "experiment";

    TString MetaFileName = "meta.tsv";
    bool SaveSnapshot = false;
    TString SnapshotFileName = "experiment.cbsnapshot";
    TString TrainDir;
    EPredictionType PredictionType = EPredictionType::RawFormulaVal;

    float FoldLenMultiplier = 2;
    ui64 CtrLeafCountLimit = Max<ui64>();
    ui64 UsedRAMLimit = Max<ui64>();
    bool StoreAllSimpleCtr = false;
    bool PrintTrees = false;
    bool DeveloperMode = false;
    bool ApproxOnFullHistory = false;
    bool AllowWritingFiles = true;

    TFitParams() = default;

    TFitParams(const NJson::TJsonValue& tree,
               const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
               const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
               NJson::TJsonValue* resultingParams = nullptr)
        : ObjectiveDescriptor(objectiveDescriptor)
        , EvalMetricDescriptor(evalMetricDescriptor)
    {
        InitFromJson(tree, resultingParams);
        StoreExpApprox = EqualToOneOf(LossFunction,
                                      ELossFunction::Logloss,
                                      ELossFunction::LogLinQuantile,
                                      ELossFunction::Poisson,
                                      ELossFunction::CrossEntropy,
                                      ELossFunction::PairLogit);
    }

private:
    void InitFromJson(const NJson::TJsonValue& tree, NJson::TJsonValue* resultingParams = nullptr);
    void ParseCtrDescription(const NJson::TJsonValue& tree, ELossFunction lossFunction, yset<TString>* validKeys);
};

struct TCrossValidationParams {
    size_t FoldCount = 0;
    bool Inverted = false;
    int PartitionRandSeed = 0;
    bool Shuffle = true;
};

struct TCvDataPartitionParams {
    int FoldIdx = -1;
    size_t FoldCount = 0;
    bool Inverted = false;
    int RandSeed = 0;
};

struct TCmdLineParams {
    TCvDataPartitionParams CvParams;

    TString LearnFile;
    TString CdFile;
    TString TestFile;

    TString EvalFileName;
    TString ModelFileName = "model.bin";
    TString FstrRegularFileName;
    TString FstrInternalFileName;

    TString LearnPairsFile;
    TString TestPairsFile;

    bool HasHeaders = false;
    char Delimiter = '\t';
};

NJson::TJsonValue ReadTJsonValue(const TString& paramsJson);
void CheckFitParams(const NJson::TJsonValue& tree,
                     const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                     const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor);

inline bool IsClassificationLoss(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::Logloss ||
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll ||
            lossFunction == ELossFunction::AUC ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::MCC);
}

inline bool IsMultiClassError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll);
}

inline bool IsPairwiseError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::PairLogit);
}

ELossFunction GetLossType(const TString& lossDescription);
yhash<TString, float> GetLossParams(const TString& lossDescription);

inline bool IsClassificationLoss(const TString& lossFunction) {
    return IsClassificationLoss(GetLossType(lossFunction));
}
