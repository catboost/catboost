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
            , BoostingType("boosting_type", EBoostingType::Ordered)
            , ApproxOnFullHistory("approx_on_full_history", false, taskType)
            , MinFoldSize("min_fold_size", 100, taskType)
            , DataPartitionType("data_partition", EDataPartitionType::FeatureParallel, taskType)
        {
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &LearningRate, &FoldLenMultiplier, &PermutationBlockSize, &IterationCount, &OverfittingDetector,
                        &BoostingType, &PermutationCount, &MinFoldSize, &ApproxOnFullHistory, &DataPartitionType);

            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
                       BoostingType, PermutationCount, MinFoldSize, ApproxOnFullHistory, DataPartitionType);
        }

        bool operator==(const TBoostingOptions& rhs) const {
            return std::tie(LearningRate, FoldLenMultiplier, PermutationBlockSize, IterationCount, OverfittingDetector,
                            ApproxOnFullHistory, BoostingType, PermutationCount,
                            MinFoldSize, DataPartitionType) ==
                   std::tie(rhs.LearningRate, rhs.FoldLenMultiplier, rhs.PermutationBlockSize, rhs.IterationCount,
                            rhs.OverfittingDetector, rhs.ApproxOnFullHistory, rhs.BoostingType,
                            rhs.PermutationCount, rhs.MinFoldSize, rhs.DataPartitionType);
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

            if (BoostingType.IsSet()) {
                if (DataPartitionType.GetUnchecked() == EDataPartitionType::DocParallel) {
                    CB_ENSURE(BoostingType == EBoostingType::Plain, "Can't use ordered boosting in doc-parallel mode");
                }
            }

            CB_ENSURE(!(ApproxOnFullHistory.GetUnchecked() && BoostingType.Get() == EBoostingType::Plain), "Can't use approx-on-full-history with Plain boosting-type");
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
        TGpuOnlyOption<EDataPartitionType> DataPartitionType;
    };
}
