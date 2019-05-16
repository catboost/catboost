#pragma once

#include "enums.h"
#include "cat_feature_options.h"
#include "catboost_options.h"

const ui32 FoldPermutationBlockSizeNotSet = 0;

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
                    } else if (IsMultiClassOnlyMetric(lossFunction)) {
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
