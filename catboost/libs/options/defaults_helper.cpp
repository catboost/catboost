#include "defaults_helper.h"

#include "enum_helpers.h"

#include <catboost/libs/ctr_description/ctr_type.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>


double Round(double number, int precision) {
    const double multiplier = pow(10, precision);
    return round(number * multiplier) / multiplier;
}


void SetOneHotMaxSizeAndPrintNotice(
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
