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

inline void UpdateBoostingTypeOption(size_t learnSampleCount, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    auto& boostingTypeOption = catBoostOptions->BoostingOptions->BoostingType;
    if (
        boostingTypeOption.NotSet() &&
        (learnSampleCount >= 50000 || catBoostOptions->BoostingOptions->IterationCount < 500) &&
        !(catBoostOptions->GetTaskType() == ETaskType::CPU && catBoostOptions->BoostingOptions->ApproxOnFullHistory.Get())
    ) {
        boostingTypeOption = EBoostingType::Plain;
    }
}

inline void UpdateSampleRateOption(size_t learnSampleCount, NCatboostOptions::TCatBoostOptions* catBoostOptions) {
    auto& takenFraction = catBoostOptions->ObliviousTreeOptions->BootstrapConfig->GetTakenFraction();
    if (takenFraction.NotSet() && learnSampleCount < 100) {
        takenFraction.SetDefault(1.0);
    }
}

void UpdateMetricPeriodOption(
    const NCatboostOptions::TCatBoostOptions& trainOptions,
    NCatboostOptions::TOutputFilesOptions* outputOptions
);
