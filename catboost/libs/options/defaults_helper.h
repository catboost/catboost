#pragma once

#include "enums.h"
#include "cat_feature_options.h"
#include "catboost_options.h"

#include <util/system/types.h>

const ui32 FoldPermutationBlockSizeNotSet = 0;

inline int DefaultFoldPermutationBlockSize(int docCount) {
    return Min(256, docCount / 1000 + 1);
}

inline void UpdateCtrTargetBordersOption(ELossFunction lossFunction, ui32 approxDim, NCatboostOptions::TCtrDescription* ctr) {
    if (NeedTargetClassifier(ctr->Type)) {
        if (IsMultiClassError(lossFunction)) {
            ctr->TargetBinarization->BorderCount = approxDim - 1;
        }
    }
}

inline void UpdateCtrsTargetBordersOption(ELossFunction lossFunction, ui32 approxDim, TVector<NCatboostOptions::TCtrDescription>* ctrs) {
    for (auto& ctr : *ctrs) {
        UpdateCtrTargetBordersOption(lossFunction, approxDim, &ctr);
    }
}

inline void UpdateCtrsTargetBordersOption(ELossFunction lossFunction, ui32 approxDim, NCatboostOptions::TCatFeatureParams* catFeatureParams) {
    UpdateCtrsTargetBordersOption(lossFunction, approxDim, &catFeatureParams->SimpleCtrs.Get());
    UpdateCtrsTargetBordersOption(lossFunction, approxDim, &catFeatureParams->CombinationCtrs.Get());
    for (auto& perFeatureCtr : catFeatureParams->PerFeatureCtrs.Get()) {
        UpdateCtrsTargetBordersOption(lossFunction, approxDim, &perFeatureCtr.second);
    }
}

inline void UpdateBoostingTypeOption(size_t learnSampleCount, NCatboostOptions::TOption<EBoostingType>* boostingTypeOption) {
    if (boostingTypeOption->NotSet()) {
        if (learnSampleCount >= 50000) {
            *boostingTypeOption = EBoostingType::Plain;
        } else {
            *boostingTypeOption = EBoostingType::Ordered;
        }
    }
}

inline void UpdateUseBestModel(bool hasTestLabels, NCatboostOptions::TOption<bool>* useBestModel) {
    if (useBestModel->NotSet() && hasTestLabels) {
        *useBestModel = true;
    }
}

inline void UpdateLeavesEstimation(bool hasWeights, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    if (
        hasWeights && IsClassificationLoss(catBoostOptions->LossFunctionDescription->GetLossFunction()) &&
        !IsMultiClassError(catBoostOptions->LossFunctionDescription->GetLossFunction())
    ) {
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationMethod = ELeavesEstimation::Gradient;
        catBoostOptions->ObliviousTreeOptions->LeavesEstimationIterations = 40;
    }
}

inline void SetDataDependantDefaults(
    int learnPoolSize,
    int testPoolSize,
    bool hasTestLabels,
    bool hasWeights,
    NCatboostOptions::TOption<bool>* useBestModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
) {
    if (testPoolSize != 0) {
        UpdateUseBestModel(hasTestLabels, useBestModel);
    }
    UpdateBoostingTypeOption(learnPoolSize, &catBoostOptions->BoostingOptions->BoostingType);

    // TODO(nikitxskv): Remove it when the l2 normalization will be added.
    UpdateLeavesEstimation(hasWeights, catBoostOptions);
}
