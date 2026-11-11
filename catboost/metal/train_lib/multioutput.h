#pragma once

#include "initialization.h"

#include <catboost/libs/model/model.h>

namespace NCB {
    inline bool IsMetalMultiOutput(ELossFunction objective) {
        return objective == ELossFunction::MultiRMSE || objective == ELossFunction::RMSEWithUncertainty ||
               objective == ELossFunction::MultiLogloss || objective == ELossFunction::MultiCrossEntropy;
    }

    inline ui32 MetalMultiOutputObjective(ELossFunction objective) {
        switch (objective) {
            case ELossFunction::MultiRMSE: return 2;
            case ELossFunction::RMSEWithUncertainty: return 3;
            case ELossFunction::MultiLogloss: return 4;
            case ELossFunction::MultiCrossEntropy: return 5;
            default: CB_ENSURE(false, "Unsupported Metal multioutput objective");
        }
    }

    inline TVector<float> MetalMultiOutputTargets(
        TConstArrayRef<TConstArrayRef<float>> targets, ui32 rows, ui32 dimensions, ELossFunction objective)
    {
        const ui32 targetDimensions = objective == ELossFunction::RMSEWithUncertainty ? 1 : dimensions;
        CB_ENSURE(dimensions >= 2 && dimensions <= 64 && targets.size() == targetDimensions,
                  "Metal multioutput targets must match between 2 and 64 approximation dimensions");
        CB_ENSURE(ui64(rows) * targetDimensions * sizeof(float) <= (1ull << 30),
                  "Metal multioutput targets exceed the experimental 1 GiB limit");
        TVector<float> result;
        result.reserve(ui64(rows) * targetDimensions);
        for (const auto& column : targets) {
            CB_ENSURE(column.size() == rows, "Metal multioutput target columns must match the object count");
            result.insert(result.end(), column.begin(), column.end());
        }
        return result;
    }

    inline TVector<double> MetalMultiOutputBias(
        TConstArrayRef<TConstArrayRef<float>> targets, TConstArrayRef<float> weights,
        ui32 dimensions, ELossFunction objective, bool boostFromAverage)
    {
        TVector<double> result(dimensions, 0);
        if (boostFromAverage) {
            CB_ENSURE(objective == ELossFunction::MultiRMSE && targets.size() == dimensions,
                      "Metal multioutput boost_from_average is supported for MultiRMSE only");
            const auto rmse = NCatboostOptions::TLossDescription().CloneWithLossFunction(ELossFunction::RMSE);
            for (ui32 dimension = 0; dimension < dimensions; ++dimension) {
                result[dimension] = CalcMetalInitialBias(rmse, targets[dimension], weights);
            }
        }
        return result;
    }

    // Reuse normal constant-model application to initialize shared host metric
    // cursors with a per-output bias. This model has no learned trees.
    inline TFullModel MetalMultiOutputBiasModel(TConstArrayRef<double> bias) {
        TFullModel result;
        result.ModelTrees.GetMutable()->SetApproxDimension(bias.size());
        result.SetScaleAndBias({1.0, TVector<double>(bias.begin(), bias.end())});
        result.UpdateDynamicData();
        return result;
    }
}
