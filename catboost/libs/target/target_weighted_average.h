#pragma once

#include <catboost/libs/data/target.h>

namespace NCB {

    inline float CalculateWeightedTargetAverage(const TTargetDataProvider& targetDataProvider) {
        auto targetRef = targetDataProvider.GetTarget().GetOrElse(TConstArrayRef<float>());
        auto weightsRef = GetWeights(targetDataProvider);

        const double summaryWeight = weightsRef.empty() ? targetRef.size() : Accumulate(weightsRef, 0.0);
        double targetSum = 0.0;
        if (weightsRef.empty()) {
            targetSum = Accumulate(targetRef, 0.0);
        } else {
            for (size_t i = 0; i < targetRef.size(); ++i) {
                targetSum += targetRef[i] * weightsRef[i];
            }
        }
        return targetSum / summaryWeight;
    }

} // namespace NCB
