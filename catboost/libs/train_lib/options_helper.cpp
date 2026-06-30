#include "options_helper.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/defaults_helper.h>

#include <util/generic/algorithm.h>
#include <util/generic/maybe.h>
#include <util/generic/ymath.h>
#include <util/string/builder.h>
#include <util/system/types.h>


static double Round(double number, int precision) {
    const double multiplier = pow(10, precision);
    return round(number * multiplier) / multiplier;
}

static void SetOneHotMaxSizeAndPrintNotice(
    TStringBuf message,
    ui32 value,
    NCatboostOptions::TOption<ui32>* oneHotMaxSizeOption) {

    oneHotMaxSizeOption->Set(value);
    CATBOOST_NOTICE_LOG << message << ". OneHotMaxSize set to " << oneHotMaxSizeOption->Get() << Endl;
}


void UpdateOneHotMaxSize(
    ui32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    NCatboostOptions::TCatBoostOptions* catBoostOptions) {

    if (!maxCategoricalFeaturesUniqValuesOnLearn) {
        return;
    }

    const auto taskType = catBoostOptions->GetTaskType();
    const auto lossFunction = catBoostOptions->LossFunctionDescription->GetLossFunction();

    NCatboostOptions::TOption<ui32>& oneHotMaxSizeOption = catBoostOptions->CatFeatureParams->OneHotMaxSize;

    if ((taskType == ETaskType::CPU) && IsPairwiseScoring(lossFunction)) {
        if ((maxCategoricalFeaturesUniqValuesOnLearn > 1) && oneHotMaxSizeOption.IsSet()) {
            CB_ENSURE(
                oneHotMaxSizeOption < 2,
                "Pairwise scoring loss functions on CPU do not support one hot features, so "
                " one_hot_max_size must be < 2 (all categorical features will be used in CTRs)."
            );
        } else {
            SetOneHotMaxSizeAndPrintNotice(
                "Pairwise scoring loss functions on CPU do not support one hot features",
                1,
                &oneHotMaxSizeOption);
        }
    }

    bool calcCtrs = maxCategoricalFeaturesUniqValuesOnLearn > oneHotMaxSizeOption;
    bool needTargetDataForCtrs = calcCtrs && CtrsNeedTargetData(catBoostOptions->CatFeatureParams);

    if (needTargetDataForCtrs && !hasLearnTarget) {
         CATBOOST_WARNING_LOG << "CTR features require Target data, but Learn dataset does not have it,"
             " so CTR features will not be calculated.\n";

        if ((taskType == ETaskType::GPU) && !oneHotMaxSizeOption.IsSet()) {
            SetOneHotMaxSizeAndPrintNotice("No Target data to calculate CTRs", 255, &oneHotMaxSizeOption);
        }
    }

    if (IsGroupwiseMetric(lossFunction) && !oneHotMaxSizeOption.IsSet()) {
        // TODO(akhropov): might be tuned in the future
        SetOneHotMaxSizeAndPrintNotice("Groupwise loss function", 10, &oneHotMaxSizeOption);
    }

    // TODO(akhropov): Tune OneHotMaxSize for regression. MLTOOLS-2503.
}

void UpdateYetiRankEvalMetric(
    const TMaybe<NCB::TTargetStats>& learnTargetStats,
    const TMaybe<NCB::TTargetStats>& testTargetStats,
    NCatboostOptions::TCatBoostOptions* catBoostOptions) {

    if (!IsYetiRankLossFunction(catBoostOptions->LossFunctionDescription.Get().LossFunction)) {
        return;
    }
    CB_ENSURE(learnTargetStats.Defined(),
        "Targets are required for " << catBoostOptions->LossFunctionDescription.Get().LossFunction << " loss function.");
    bool isPfoundMetricApplicable = 0 <= learnTargetStats->MinValue && learnTargetStats->MaxValue <= 1;
    if (testTargetStats.Defined()) {
        isPfoundMetricApplicable &= 0 <= testTargetStats->MinValue && testTargetStats->MaxValue <= 1;
    }
    if (!isPfoundMetricApplicable) {
        NCatboostOptions::TLossDescription lossDescription;
        lossDescription.Load(LossDescriptionToJson("NDCG"));
        catBoostOptions->MetricOptions.Get().ObjectiveMetric.Set(lossDescription);
    }
}

static void UpdateUseBestModel(
    bool hasTest,
    bool hasTestConstTarget,
    bool hasTestPairs,
    NCatboostOptions::TOutputFilesOptions* outputFilesOptions
) {
    if (outputFilesOptions->UseBestModel.NotSet() && hasTest && (!hasTestConstTarget || hasTestPairs)) {
        outputFilesOptions->UseBestModel = true;
    }
    if (!hasTest && outputFilesOptions->UseBestModel) {
        CATBOOST_WARNING_LOG << "You should provide test set for use best model. use_best_model parameter has been switched to false value." << Endl;
        outputFilesOptions->UseBestModel = false;
    }
}

namespace {
    struct TLearningRateCoefficients {
        // learning_rate = exp(B + A log size + C log iter - C log 1000);
        double DatasetSizeCoeff = 0; // A
        double DatasetSizeConst = 0; // B
        double IterCountCoeff = 0; // C
        double IterCountConst = 0; // D
    };

    enum class ETargetType {
        RMSE,
        Logloss,
        MultiClass,
        Unknown
    };

    enum class EUseBestModel {
        False,
        True
    };

    enum class EBoostFromAverage {
        False,
        True
    };

    struct TAutoLearningRateKey {
        ETargetType TargetType;
        ETaskType TaskType;
        EUseBestModel UseBestModel;
        EBoostFromAverage BoostFromAverage;

        TAutoLearningRateKey() {}

        TAutoLearningRateKey(ETargetType targetType, ETaskType taskType, EUseBestModel useBestModel,
                             EBoostFromAverage boostFromAverage)
            : TargetType(targetType), TaskType(taskType), UseBestModel(useBestModel),
              BoostFromAverage(boostFromAverage) {}

        TAutoLearningRateKey(ETargetType targetType, ETaskType taskType, bool useBestModel, bool boostFromAverage)
            : TargetType(targetType), TaskType(taskType),
              UseBestModel(useBestModel ? EUseBestModel::True : EUseBestModel::False),
              BoostFromAverage(boostFromAverage ? EBoostFromAverage::True : EBoostFromAverage::False) {}

        bool operator==(const TAutoLearningRateKey &rhs) const {
            return std::tie(TargetType, TaskType, UseBestModel, BoostFromAverage)
                   == std::tie(rhs.TargetType, rhs.TaskType, rhs.UseBestModel, rhs.BoostFromAverage);
        }

        size_t GetHash() const {
            return MultiHash(TargetType, TaskType, UseBestModel, BoostFromAverage);
        }
    };
};

template<>
struct THash<TAutoLearningRateKey> {
    inline size_t operator()(const TAutoLearningRateKey& lrKey) const noexcept {
        return lrKey.GetHash();
    }
};

namespace {
    struct TAutoLRParamsGuesser {
    private:

        static ETargetType GetTargetType(ELossFunction lossFunction) {
            switch (lossFunction) {
                case ELossFunction::Logloss:
                case ELossFunction::MultiLogloss:
                case ELossFunction::MultiCrossEntropy:
                    return ETargetType::Logloss;
                case ELossFunction::MultiClass:
                    return ETargetType::MultiClass;
                case ELossFunction::RMSE:
                    return ETargetType::RMSE;
                default:
                    return ETargetType::Unknown;
            }
        }

    public:
        TAutoLRParamsGuesser() {
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::CPU, EUseBestModel::True, EBoostFromAverage::True)] =
                {0.246, -5.127, -0.451, 0.978};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::CPU, EUseBestModel::False, EBoostFromAverage::True)] =
                {0.408, -7.299, -0.928, 2.701};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::CPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {0.247, -5.158, -0.435, 0.934};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::CPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {0.427, -7.525, -0.917, 2.63};

            Coefficients[TAutoLearningRateKey(ETargetType::MultiClass, ETaskType::CPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {0.02, -2.364, -0.382, 0.924};
            Coefficients[TAutoLearningRateKey(ETargetType::MultiClass, ETaskType::CPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {0.051, -2.889, -0.845, 2.928};

            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::CPU, EUseBestModel::True, EBoostFromAverage::True)] =
                {0.157, -4.062, -0.61, 1.557};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::CPU, EUseBestModel::False, EBoostFromAverage::True)] =
                {0.158, -4.287, -0.813, 2.571};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::CPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {0.189, -4.383, -0.623, 1.439};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::CPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {0.178, -4.473, -0.76, 2.133};

            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::GPU, EUseBestModel::True, EBoostFromAverage::True)] =
                {0.04, -3.226, -0.488, 0.758};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::GPU, EUseBestModel::False, EBoostFromAverage::True)] =
                {0.427, -7.316, -0.907, 2.354};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::GPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {-0.085, -2.055, -0.414, 0.427};
            Coefficients[TAutoLearningRateKey(ETargetType::Logloss, ETaskType::GPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {-0.055, -3.01, -0.896, 2.366};

            Coefficients[TAutoLearningRateKey(ETargetType::MultiClass, ETaskType::GPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {0.101, -2.95, -0.437, 1.136};
            Coefficients[TAutoLearningRateKey(ETargetType::MultiClass, ETaskType::GPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {0.204, -4.144, -0.833, 2.889};


            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::GPU, EUseBestModel::True, EBoostFromAverage::True)] =
                {0.108, -3.525, -0.285, 0.058};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::GPU, EUseBestModel::False, EBoostFromAverage::True)] =
                {0.131, -4.114, -0.597, 1.693};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::GPU, EUseBestModel::True, EBoostFromAverage::False)] =
                {0.051, -3.001, -0.449, 0.859};
            Coefficients[TAutoLearningRateKey(ETargetType::RMSE, ETaskType::GPU, EUseBestModel::False, EBoostFromAverage::False)] =
                {0.047, -3.034, -0.591, 1.554};
        }

        static bool NeedToUpdate(ETaskType taskType, ELossFunction lossFunction, bool useBestModel, bool boostFromAverage) {
            const auto& wat = Singleton<TAutoLRParamsGuesser>();
            return wat->Coefficients.contains(TAutoLearningRateKey(GetTargetType(lossFunction), taskType, useBestModel, boostFromAverage));
        }


        static double GetLearningRate(
            ETaskType taskType, ELossFunction lossFunction, bool useBestModel, bool boostFromAverage, double iterationCount, double learnObjectCount
        ) {
            const auto& wat = Singleton<TAutoLRParamsGuesser>();
            TLearningRateCoefficients& coeffs = wat->Coefficients.at(TAutoLearningRateKey(GetTargetType(lossFunction), taskType, useBestModel, boostFromAverage));

            const double customIterationConstant = exp(coeffs.IterCountCoeff * log(iterationCount) + coeffs.IterCountConst);
            const double defaultIterationConstant = exp(coeffs.IterCountCoeff * log(1000) + coeffs.IterCountConst);
            const double defaultLearningRate = exp(coeffs.DatasetSizeCoeff * log(learnObjectCount) + coeffs.DatasetSizeConst);
            return Round(Min(defaultLearningRate * customIterationConstant / defaultIterationConstant, 0.5), /*precision=*/6);
        }

    private:
        THashMap<TAutoLearningRateKey, TLearningRateCoefficients> Coefficients;
    };
};

static void UpdateLearningRate(ui32 learnObjectCount, bool useBestModel, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    const bool boostFromAverage = catBoostOptions->BoostingOptions->BoostFromAverage.Get();
    auto& learningRate = catBoostOptions->BoostingOptions->LearningRate;
    const int iterationCount = catBoostOptions->BoostingOptions->IterationCount;
    const auto lossFunction = catBoostOptions->LossFunctionDescription->GetLossFunction();
    const auto taskType = catBoostOptions->GetTaskType();

    if (
        learningRate.NotSet() &&
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationMethod.NotSet() &&
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationIterations.NotSet() &&
        catBoostOptions->ObliviousTreeOptions->L2Reg.NotSet()
    ) {
        TAutoLRParamsGuesser lrGuesser;
        if (lrGuesser.NeedToUpdate(taskType, lossFunction, useBestModel, boostFromAverage)) {
            learningRate = lrGuesser.GetLearningRate(taskType, lossFunction, useBestModel, boostFromAverage, iterationCount, learnObjectCount);
            CATBOOST_NOTICE_LOG << "Learning rate set to " << learningRate << Endl;
        }
    }
}

static void UpdateLeavesEstimationIterations(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    auto& leavesEstimationIterations = catBoostOptions->ObliviousTreeOptions->LeavesEstimationIterations;
    const bool hasFeaturesLayout = !!trainDataMetaInfo.FeaturesLayout;
    if (
        leavesEstimationIterations.NotSet() &&
        IsSmallIterationCount(catBoostOptions->BoostingOptions->IterationCount) &&
        hasFeaturesLayout && trainDataMetaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo().size() < 20
    ) {
        leavesEstimationIterations = 1;
    }
}

static bool IsConstTarget(const NCB::TDataMetaInfo& dataMetaInfo) {
    return dataMetaInfo.TargetStats.Defined() && dataMetaInfo.TargetStats->MinValue == dataMetaInfo.TargetStats->MaxValue;
}

static void UpdateAndValidateMonotoneConstraints(
    const NCB::TFeaturesLayout& featuresLayout,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    if (catBoostOptions->GetTaskType() != ETaskType::CPU) {
        return;
    }
    auto& monotoneConstraints = catBoostOptions->ObliviousTreeOptions->MonotoneConstraints.Get();

    TMap<ui32, int> floatFeatureMonotonicConstraints;
    for (auto [featureIdx, constraint] : monotoneConstraints) {
        if (constraint != 0) {
            CB_ENSURE(
                featuresLayout.GetExternalFeatureType(featureIdx) == EFeatureType::Float,
                "Monotone constraints may be imposed only on float features."
            );
            ui32 floatFeatureIdx = featuresLayout.GetInternalFeatureIdx(featureIdx);
            floatFeatureMonotonicConstraints[floatFeatureIdx] = constraint;
        }
    }
    monotoneConstraints = floatFeatureMonotonicConstraints;
}

static void DropModelShrinkageIfBaselineUsed(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    bool learningContinuation,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    if (catBoostOptions->GetTaskType() == ETaskType::CPU &&
        catBoostOptions->BoostingOptions->ModelShrinkRate.Get() != 0.0f
    ) {
        if (trainDataMetaInfo.BaselineCount != 0) {
            CATBOOST_WARNING_LOG << "Model shrinkage in combination with baseline column " <<
            "is not implemented yet. Reset model_shrink_rate to 0." << Endl;
            catBoostOptions->BoostingOptions->ModelShrinkRate.Set(0);
        }
        if (learningContinuation) {
            CATBOOST_WARNING_LOG << "Model shrinkage in combination with learning continuation " <<
            "is not implemented yet. Reset model_shrink_rate to 0." << Endl;
            catBoostOptions->BoostingOptions->ModelShrinkRate.Set(0);
        }
    }
}

static void AdjustBoostFromAverageDefaultValue(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo,
    bool continueFromModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    if (catBoostOptions->BoostingOptions->BoostFromAverage.IsSet()) {
        return;
    }
    if (catBoostOptions->SystemOptions->IsSingleHost()
        && !continueFromModel
        && EqualToOneOf(
            catBoostOptions->LossFunctionDescription->GetLossFunction(),
            ELossFunction::RMSE, ELossFunction::MAE, ELossFunction::Quantile, ELossFunction::MAPE,
            ELossFunction::MultiQuantile, ELossFunction::MultiRMSE, ELossFunction::MultiRMSEWithMissingValues)
    ) {
        catBoostOptions->BoostingOptions->BoostFromAverage.Set(true);
    }
    if (trainDataMetaInfo.BaselineCount != 0 || (testDataMetaInfo.Defined() && testDataMetaInfo->BaselineCount != 0)) {
        catBoostOptions->BoostingOptions->BoostFromAverage.Set(false);
    }
}

static void AdjustPosteriorSamplingDeafultValues(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    bool continueFromModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    if (!catBoostOptions->BoostingOptions->PosteriorSampling.GetUnchecked()) {
        return;
    }
    CB_ENSURE(!continueFromModel, "Model shrinkage and Posterior Sampling in combination with learning continuation " <<
        "is not implemented yet.");
    CB_ENSURE(trainDataMetaInfo.BaselineCount == 0, "Model shrinkage and Posterior Sampling in combination with baseline column " <<
        "is not implemented yet.");
    CB_ENSURE(catBoostOptions->BoostingOptions->ModelShrinkMode != EModelShrinkMode::Decreasing, "Posterior Sampling requires " <<
        "Constant Model Shrink Mode");
    catBoostOptions->BoostingOptions->ModelShrinkRate.Set(1 / (2. * trainDataMetaInfo.ObjectCount));
    catBoostOptions->BoostingOptions->DiffusionTemperature.Set(trainDataMetaInfo.ObjectCount);
}

static void UpdateDictionaryDefaults(
    ui64 learnPoolSize,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    const ui64 minTokenOccurence = (learnPoolSize < 1000) ? 1 : 5;
    auto& textProcessingOptions = catBoostOptions->DataProcessingOptions->TextProcessingOptions;
    textProcessingOptions->SetDefaultMinTokenOccurrence(minTokenOccurence);
}

void SetDataDependentDefaults(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo,
    bool continueFromModel,
    bool continueFromProgress,
    NCatboostOptions::TOutputFilesOptions* outputFilesOptions,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    const ui64 learnPoolSize = trainDataMetaInfo.ObjectCount;
    const ui64 testPoolSize = testDataMetaInfo.Defined() ? testDataMetaInfo->ObjectCount : 0;
    const bool isConstTestTarget = testDataMetaInfo.Defined() && IsConstTarget(*testDataMetaInfo);
    const bool hasTestPairs = testDataMetaInfo.Defined() && testDataMetaInfo->HasPairs;
    UpdateUseBestModel(testPoolSize, isConstTestTarget, hasTestPairs, outputFilesOptions);
    UpdateBoostingTypeOption(learnPoolSize, catBoostOptions);
    AdjustBoostFromAverageDefaultValue(trainDataMetaInfo, testDataMetaInfo, continueFromModel, catBoostOptions);
    UpdateLearningRate(learnPoolSize, outputFilesOptions->UseBestModel.Get(), catBoostOptions);
    UpdateOneHotMaxSize(
        trainDataMetaInfo.MaxCatFeaturesUniqValuesOnLearn,
        trainDataMetaInfo.TargetCount > 0,
        catBoostOptions
    );
    UpdateYetiRankEvalMetric(
        trainDataMetaInfo.TargetStats,
        testDataMetaInfo.Defined() ? testDataMetaInfo->TargetStats : Nothing(),
        catBoostOptions
    );
    UpdateLeavesEstimationIterations(trainDataMetaInfo, catBoostOptions);
    UpdateAndValidateMonotoneConstraints(*trainDataMetaInfo.FeaturesLayout.Get(), catBoostOptions);
    DropModelShrinkageIfBaselineUsed(trainDataMetaInfo, continueFromModel || continueFromProgress, catBoostOptions);
    UpdateDictionaryDefaults(learnPoolSize, catBoostOptions);
    UpdateSampleRateOption(learnPoolSize, catBoostOptions);
    AdjustPosteriorSamplingDeafultValues(trainDataMetaInfo, continueFromModel || continueFromProgress, catBoostOptions);
}
