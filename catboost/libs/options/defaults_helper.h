#pragma once

#include "enums.h"
#include "cat_feature_options.h"
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
