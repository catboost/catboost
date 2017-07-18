#pragma once

#include "error_holder.h"
#include "ders_holder.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>

#include <catboost/libs/model/projection.h>

#include <library/binsaver/bin_saver.h>
#include <library/grid_creator/binarization.h>
#include <library/json/json_reader.h>

#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/maybe.h>
#include <util/datetime/systime.h>

enum class ELossFunction {
    RMSE,
    MAE,
    Logloss,
    CrossEntropy,
    Quantile,
    LogLinQuantile,
    Poisson,
    MAPE,
    MultiClass,
    AUC,
    Accuracy,
    Precision,
    Recall,
    R2,
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

enum class ECtrType {
    Borders,
    Buckets,
    MeanValue,
    CounterTotal,
    CounterMax
};

bool IsCounter(ECtrType ctrType);

constexpr int CB_THREAD_LIMIT = 32;

struct TCtrDescription {
    ECtrType CtrType = ECtrType::Borders;
    int TargetBorderCount = 1;
    EBorderSelectionType TargetBorderType = EBorderSelectionType::MinEntropy;

    TCtrDescription() = default;

    explicit TCtrDescription(const ECtrType& ctrType)
        : CtrType(ctrType)
    { }
};

struct TCtrParams {
    int CtrBorderCount = 16;
    int MaxCtrComplexity = 4;
    yvector<float> DefaultPriors = {0, 0.5, 1};
    yvector<std::pair<int, yvector<float>>> PerFeaturePriors;
    yvector<TCtrDescription> Ctrs = { TCtrDescription(), TCtrDescription(ECtrType::CounterMax) };
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

class TFitParams {
public:
    int Iterations = 500;
    int ThreadCount = 8;

    // Objective, always maximized.
    // TODO(asaitgalin): Rename.
    // TODO(annaveronika): remove LossFunction.
    ELossFunction LossFunction;
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
    float AutoStopPval = 0;
    EOverfittingDetectorType OverfittingDetectorType = EOverfittingDetectorType::IncToDec;
    int OverfittingDetectorIterationsWait = 20;
    bool UseBestModel = false;
    bool DetailedProfile = false;
    ELeafEstimation LeafEstimationMethod = ELeafEstimation::Gradient;
    yvector<int> IgnoredFeatures;
    TString TimeLeftLog = "time_left.tsv";
    yvector<float> ClassWeights;
    int ClassesCount = 0;
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

    TFitParams() = default;

    explicit TFitParams(const NJson::TJsonValue& tree,
                        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
                        NJson::TJsonValue* resultingParams = nullptr)
        : ObjectiveDescriptor(objectiveDescriptor)
        , EvalMetricDescriptor(evalMetricDescriptor)
    {
        InitFromJson(tree, resultingParams);
    }

private:
    void InitFromJson(const NJson::TJsonValue& tree, NJson::TJsonValue* resultingParams = nullptr);
};

struct TCrossValidationParams {
    size_t FoldCount = 0;
    bool Inverted = false;
    int RandSeed = 0;
    bool Shuffle = true;
    int EvalPeriod = 1;
    bool EnableEarlyStopping = true;
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
    bool CalcFstr = false;
    bool HasHeaders = false;
    char Delimiter = '\t';
};

NJson::TJsonValue ReadTJsonValue(const TString& paramsJson);

inline bool IsClassificationLoss(const ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::Logloss ||
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::AUC ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall);
}

ELossFunction GetLossType(const TString& lossDescription);
yhash<TString, float> GetLossParams(const TString& lossDescription);
