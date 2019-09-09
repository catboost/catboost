#include "options_helper.h"

#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/defaults_helper.h>

#include <util/system/types.h>
#include <util/generic/ymath.h>
#include <util/generic/maybe.h>
#include <util/string/builder.h>


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


static void UpdateOneHotMaxSize(
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

static void UpdateUseBestModel(bool learningContinuation, bool hasTest, bool hasTestConstTarget, bool hasTestPairs, NCatboostOptions::TOption<bool>* useBestModel) {
    if (useBestModel->NotSet() && !learningContinuation && hasTest && (!hasTestConstTarget || hasTestPairs)) {
        *useBestModel = true;
    }
    if (learningContinuation && *useBestModel) {
        CATBOOST_WARNING_LOG << "Using best model is not supported for learning continuation. use_best_model parameter has been switched to false value." << Endl;
        *useBestModel = false;
    }
    if (!hasTest && *useBestModel) {
        CATBOOST_WARNING_LOG << "You should provide test set for use best model. use_best_model parameter has been switched to false value." << Endl;
        *useBestModel = false;
    }
}

static void UpdateLearningRate(ui32 learnObjectCount, bool useBestModel, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    auto& learningRate = catBoostOptions->BoostingOptions->LearningRate;
    const int iterationCount = catBoostOptions->BoostingOptions->IterationCount;
    const bool doUpdateLearningRate = (
        learningRate.NotSet() &&
        IsBinaryClassOnlyMetric(catBoostOptions->LossFunctionDescription->GetLossFunction()) &&
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationMethod.NotSet() &&
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationIterations.NotSet() &&
        catBoostOptions->ObliviousTreeOptions->L2Reg.NotSet()
    );
    if (doUpdateLearningRate) {
        double a = 0, b = 0, c = 0, d = 0;
        if (useBestModel) {
            a = 0.105;
            b = -3.276;
            c = -0.428;
            d = 0.911;
        } else {
            a = 0.283;
            b = -6.044;
            c = -0.891;
            d = 2.620;
        }
        // TODO(nikitxskv): Don't forget to change formula when add l2-leaf-reg depending on weights.
        const double customIterationConstant = exp(c * log(iterationCount) + d);
        const double defaultIterationConstant = exp(c * log(1000) + d);
        const double defaultLearningRate = exp(a * log(learnObjectCount) + b);
        learningRate = Min(defaultLearningRate * customIterationConstant / defaultIterationConstant, 0.5);
        learningRate = Round(learningRate, /*precision=*/6);

        CATBOOST_NOTICE_LOG << "Learning rate set to " << learningRate << Endl;
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
    TVector<int>* monotoneConstraints = &catBoostOptions->ObliviousTreeOptions->MonotoneConstraints.Get();
    CB_ENSURE(
        monotoneConstraints->size() <= featuresLayout.GetExternalFeatureCount(),
        "length of monotone constraints vector exceeds number of features."
    );
    CB_ENSURE(
        AllOf(
            xrange(monotoneConstraints->size()),
            [&] (int featureIndex) {
                return (
                    (*monotoneConstraints)[featureIndex] == 0 ||
                    featuresLayout.GetExternalFeatureType(featureIndex) == EFeatureType::Float
                );
            }
        ),
        "Monotone constraints may be imposed only on float features."
    );

    // Ensure that monotoneConstraints size is zero or equals to the number of float features.
    if (AllOf(*monotoneConstraints, [] (int constraint) { return constraint == 0; })) {
        monotoneConstraints->clear();
    } else {
        TVector<int> floatFeatureMonotonicConstraints(featuresLayout.GetFloatFeatureCount(), 0);
        for (ui32 floatFeatureId : xrange<ui32>(featuresLayout.GetFloatFeatureCount())) {
            floatFeatureMonotonicConstraints[floatFeatureId] = (*monotoneConstraints)[
                featuresLayout.GetExternalFeatureIdx(floatFeatureId, EFeatureType::Float)
            ];
        }
        *monotoneConstraints = floatFeatureMonotonicConstraints;
    }
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
    if (catBoostOptions->SystemOptions->IsSingleHost() && !continueFromModel) {
        catBoostOptions->BoostingOptions->BoostFromAverage.Set(true);
    }
    if (trainDataMetaInfo.BaselineCount != 0 || (testDataMetaInfo.Defined() && testDataMetaInfo->BaselineCount != 0)) {
        catBoostOptions->BoostingOptions->BoostFromAverage.Set(false);
    }
}

void SetDataDependentDefaults(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo,
    bool continueFromModel,
    bool continueFromProgress,
    NCatboostOptions::TOption<bool>* useBestModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    const ui64 learnPoolSize = trainDataMetaInfo.ObjectCount;
    const ui64 testPoolSize = testDataMetaInfo.Defined() ? testDataMetaInfo->ObjectCount : 0;
    const bool isConstTestTarget = testDataMetaInfo.Defined() && IsConstTarget(*testDataMetaInfo);
    const bool hasTestPairs = testDataMetaInfo.Defined() && testDataMetaInfo->HasPairs;
    UpdateUseBestModel(continueFromModel || continueFromProgress, testPoolSize, isConstTestTarget, hasTestPairs, useBestModel);
    UpdateBoostingTypeOption(learnPoolSize, catBoostOptions);
    UpdateLearningRate(learnPoolSize, useBestModel->Get(), catBoostOptions);
    UpdateOneHotMaxSize(
        trainDataMetaInfo.MaxCatFeaturesUniqValuesOnLearn,
        trainDataMetaInfo.HasTarget,
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
    AdjustBoostFromAverageDefaultValue(trainDataMetaInfo, testDataMetaInfo, continueFromModel, catBoostOptions);
}
