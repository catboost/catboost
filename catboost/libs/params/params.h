#pragma once

#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/metrics/ders_holder.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/libs/model/split.h>
#include <catboost/libs/helpers/eval_helpers.h>

#include <library/binsaver/bin_saver.h>
#include <library/grid_creator/binarization.h>
#include <library/json/json_reader.h>

#include <util/generic/algorithm.h>
#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/maybe.h>
#include <util/datetime/systime.h>

using TIndexType = ui32;

enum class ENanMode {
    Min,
    Max,
    Forbidden
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

enum class ETaskType {
    CPU,
    GPU
};

enum class EWeightSamplingFrequency {
    PerTree,
    PerTreeLevel
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
    int CtrBorderCount = 15;
    int MaxCtrComplexity = 4;
    TVector<float> DefaultPriors = {0, 0.5, 1};
    TVector<float> DefaultCounterPriors = {0};
    TVector<std::pair<int, TVector<float>>> PerFeaturePriors;
    TVector<TCtrDescription> Ctrs = {TCtrDescription(), TCtrDescription(ECtrType::Counter)};
};

enum class EFstrType {
    FeatureImportance,
    InternalFeatureImportance,
    Interaction,
    InternalInteraction,
    Doc
};

const int FoldPermutationBlockSizeNotSet = -1;
inline int DefaultFoldPermutationBlockSize(int docCount) {
    return Min(256, docCount / 1000 + 1);
}

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
    TVector<TString> CustomLoss;
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
    int FoldPermutationBlockSize = FoldPermutationBlockSizeNotSet;
    int BorderCount = 128;
    TCtrParams CtrParams;
    ECounterCalc CounterCalcMethod = ECounterCalc::Full;
    TOverfittingDetectorParams OdParams;
    bool UseBestModel = false;
    bool DetailedProfile = false;
    ELeafEstimation LeafEstimationMethod = ELeafEstimation::Gradient;
    TVector<int> IgnoredFeatures;
    TString TimeLeftLog = "time_left.tsv";
    int ClassesCount = 0;
    TVector<float> ClassWeights;
    TVector<TString> ClassNames;
    size_t OneHotMaxSize = 0;
    float RandomStrength = 1;
    float BaggingTemperature = 1.0f;

    TString LearnErrorLog = "learn_error.tsv";
    TString TestErrorLog = "test_error.tsv";

    EBorderSelectionType FeatureBorderType = EBorderSelectionType::MinEntropy;
    float L2LeafRegularizer = 3;
    ELoggingLevel LoggingLevel = ELoggingLevel::Silent;
    bool HasTime = false;
    TString Name = "experiment";

    TString MetaFileName = "meta.tsv";
    bool SaveSnapshot = false;
    TString SnapshotFileName = "experiment.cbsnapshot";
    TString TrainDir;
    TVector<EPredictionType> PredictionTypes = {EPredictionType::RawFormulaVal};

    float FoldLenMultiplier = 2;
    ui64 CtrLeafCountLimit = Max<ui64>();
    ui64 UsedRAMLimit = Max<ui64>();
    bool StoreAllSimpleCtr = false;
    bool PrintTrees = false;
    bool DeveloperMode = false;
    bool ApproxOnFullHistory = false;
    bool AllowWritingFiles = true;
    EWeightSamplingFrequency WeightSamplingFrequency = EWeightSamplingFrequency::PerTreeLevel;

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

inline bool IsQuerywiseError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::QueryRMSE);
}

inline bool IsClassificationLoss(const TString& lossFunction) {
    return IsClassificationLoss(GetLossType(lossFunction));
}
