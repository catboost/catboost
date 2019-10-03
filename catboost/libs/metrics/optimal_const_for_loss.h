#pragma once

#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>


namespace NCB {
    inline float CalculateWeightedTargetAverage(TConstArrayRef<float> target, TConstArrayRef<float> weights) {
        const double summaryWeight = weights.empty() ? target.size() : Accumulate(weights, 0.0);
        double targetSum = 0.0;
        if (weights.empty()) {
            targetSum = Accumulate(target, 0.0);
        } else {
            Y_ASSERT(target.size() == weights.size());
            for (size_t i = 0; i < target.size(); ++i) {
                targetSum += target[i] * weights[i];
            }
        }
        return targetSum / summaryWeight;
    }

    //TODO(isaf27): add baseline to CalcOptimumConstApprox
    inline TMaybe<double> CalcOptimumConstApprox(
        ELossFunction lossFunction,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weights
    ) {
        // TODO(ilyzhin): refactor it (https://a.yandex-team.ru/review/969134/details#comment-1471301)
        switch(lossFunction) {
            case ELossFunction::RMSE:
                return CalculateWeightedTargetAverage(target, weights);
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
            {
                const double bestProbability = CalculateWeightedTargetAverage(target, weights);
                return Logit(bestProbability);
            }
            default:
                return Nothing();
        }
    }
}
