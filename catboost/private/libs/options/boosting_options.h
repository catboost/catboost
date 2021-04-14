#pragma once

#include "enums.h"
#include "option.h"
#include "overfitting_detector_options.h"
#include "unimplemented_aware_option.h"

#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TBoostingOptions {
        explicit TBoostingOptions(ETaskType taskType);

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        bool operator==(const TBoostingOptions& rhs) const;
        bool operator!=(const TBoostingOptions& rhs) const;

        void Validate() const;

        TOption<float> LearningRate;
        TOption<float> FoldLenMultiplier;
        TOption<ui32> PermutationBlockSize;
        TOption<ui32> IterationCount;
        TOption<ui32> PermutationCount;
        TOption<TOverfittingDetectorOptions> OverfittingDetector;
        TOption<EBoostingType> BoostingType;
        TOption<bool> BoostFromAverage;
        TCpuOnlyOption<bool> ApproxOnFullHistory;
        TCpuOnlyOption<float> ModelShrinkRate;
        TCpuOnlyOption<EModelShrinkMode> ModelShrinkMode;
        TOption<bool> Langevin;
        TOption<float> DiffusionTemperature;

        TCpuOnlyOption<bool> PosteriorSampling;

        TGpuOnlyOption<ui32> MinFoldSize;
        TGpuOnlyOption<EDataPartitionType> DataPartitionType;
    };
}
