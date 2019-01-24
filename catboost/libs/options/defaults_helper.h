#pragma once

#include "enums.h"
#include "cat_feature_options.h"
#include "catboost_options.h"

#include <util/system/types.h>
#include <util/generic/ymath.h>

const ui32 FoldPermutationBlockSizeNotSet = 0;

double Round(double number, int precision);

inline int DefaultFoldPermutationBlockSize(int docCount) {
    return Min(256, docCount / 1000 + 1);
}

inline void UpdateCtrsTargetBordersOption(ELossFunction lossFunction, ui32 approxDim, NCatboostOptions::TCatFeatureParams* catFeatureParams) {
    const NCatboostOptions::TOption<ui32>& commonBorderCountOption
        = catFeatureParams->TargetBinarization->BorderCount;

    catFeatureParams->ForEachCtrDescription(
        [&] (NCatboostOptions::TCtrDescription* ctr) {
            if (NeedTargetClassifier(ctr->Type)) {
                NCatboostOptions::TOption<ui32>& borderCountOption = ctr->TargetBinarization->BorderCount;
                if (!borderCountOption.IsSet()) {
                    if (commonBorderCountOption.IsSet()) {
                        borderCountOption.Set(commonBorderCountOption.Get());
                    } else if (IsMultiClassMetric(lossFunction)) {
                        borderCountOption.Set(approxDim - 1);
                    }
                }
            }
        }
    );
}

inline void UpdateBoostingTypeOption(size_t learnSampleCount, NCatboostOptions::TOption<EBoostingType>* boostingTypeOption) {
    if (boostingTypeOption->NotSet() && learnSampleCount >= 50000) {
        *boostingTypeOption = EBoostingType::Plain;
    }
}

inline void UpdateUseBestModel(bool hasTest, bool hasTestConstTarget, bool hasTestPairs, NCatboostOptions::TOption<bool>* useBestModel) {
    if (useBestModel->NotSet() && hasTest && (!hasTestConstTarget || hasTestPairs)) {
        *useBestModel = true;
    }
    if (!hasTest && *useBestModel) {
        CATBOOST_WARNING_LOG << "You should provide test set for use best model. use_best_model parameter swiched to false value." << Endl;
        *useBestModel = false;
    }
}

inline void UpdateLearningRate(ui32 learnObjectCount, bool useBestModel, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    auto& learningRate = catBoostOptions->BoostingOptions->LearningRate;
    const int iterationCount = catBoostOptions->BoostingOptions->IterationCount;
    const bool doUpdateLearningRate = (
        learningRate.NotSet() &&
        IsBinaryClassMetric(catBoostOptions->LossFunctionDescription->GetLossFunction()) &&
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

void UpdateOneHotMaxSize(
    ui32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
);


inline void SetDataDependentDefaults(
    ui32 learnPoolSize,
    bool hasLearnTarget,
    ui32 maxCategoricalFeaturesUniqValuesOnLearn,
    ui32 testPoolSize,
    bool hasTestConstTarget,
    bool hasTestPairs,
    NCatboostOptions::TOption<bool>* useBestModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    UpdateUseBestModel(testPoolSize, hasTestConstTarget, hasTestPairs, useBestModel);
    UpdateBoostingTypeOption(learnPoolSize, &catBoostOptions->BoostingOptions->BoostingType);
    UpdateLearningRate(learnPoolSize, useBestModel->Get(), catBoostOptions);
    UpdateOneHotMaxSize(maxCategoricalFeaturesUniqValuesOnLearn, hasLearnTarget, catBoostOptions);
}
