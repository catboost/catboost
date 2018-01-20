#pragma once

#include "option.h"
#include "json_helper.h"
#include "enums.h"
#include "overfitting_detector_options.h"
#include <catboost/libs/logging/logging_level.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    struct TBoostingOptions {
        explicit TBoostingOptions(ETaskType taskType)
            : LearningRate("learning_rate", 0.03)
            , FoldLenMultiplier("fold_len_multiplier", 2.0)
            , PermutationBlockSize("fold_permutation_block", 0)
            , IterationCount("iterations", 1000)
            , PermutationCount("permutation_count", 4)
            , OverfittingDetector("od_config", TOverfittingDetectorOptions())
            , BoostingType("boosting_type", EBoostingType::Dynamic)
            , ApproxOnFullHistory("approx_on_full_history", false, taskType)
            , MinFoldSize("min_fold_size", 100, taskType)
        {
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &LearningRate, &FoldLenMultiplier, &PermutationBlockSize, &IterationCount, &OverfittingDetector,
                        &BoostingType, &PermutationCount, &MinFoldSize, &ApproxOnFullHistory);

            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
                       BoostingType, PermutationCount, MinFoldSize, ApproxOnFullHistory);
        }

        bool operator==(const TBoostingOptions& rhs) const {
            return std::tie(LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
                            ApproxOnFullHistory, BoostingType, PermutationCount,
                            MinFoldSize) ==
                   std::tie(rhs.LearningRate, rhs.FoldLenMultiplier, rhs.PermutationBlockSize, rhs.IterationCount,
                            rhs.OverfittingDetector, rhs.ApproxOnFullHistory, rhs.BoostingType,
                            rhs.PermutationCount, rhs.MinFoldSize);
        }

        bool operator!=(const TBoostingOptions& rhs) const {
            return !(rhs == *this);
        }

        void Validate() const {
            OverfittingDetector->Validate();
            CB_ENSURE(FoldLenMultiplier.Get() > 1.0f, "fold len multiplier should be greater than 1");
            CB_ENSURE(IterationCount.Get() > 0, "Iterations count should be positive");

            CB_ENSURE(PermutationCount.Get() > 0, "Permutation count should be positive");
            CB_ENSURE(MinFoldSize.GetUnchecked() > 0, "Min fold size should be positive");
        }

        TOption<float> LearningRate;
        TOption<float> FoldLenMultiplier;
        TOption<ui32> PermutationBlockSize;
        TOption<ui32> IterationCount;
        TOption<ui32> PermutationCount;
        TOption<TOverfittingDetectorOptions> OverfittingDetector;
        TOption<EBoostingType> BoostingType;
        TCpuOnlyOption<bool> ApproxOnFullHistory;

        TGpuOnlyOption<ui32> MinFoldSize;
    };
}
