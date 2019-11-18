#pragma once

#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
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

    inline float CalculateWeightedTargetQuantile(
        TConstArrayRef<float> target,
        TConstArrayRef<float> weights,
        const NCatboostOptions::TLossDescription& lossDescription
    ) {
        const auto& params = lossDescription.GetLossParams();
        auto it = params.find("alpha");
        double alpha = it == params.end() ? 0.5 : FromString<double>(it->second);
        it = params.find("delta");
        double delta = it == params.end() ? 1e-6 : FromString<double>(it->second);

        const TVector<float> defaultWeights(target.size(), 1);

        return CalcSampleQuantile(target, weights.empty() ? MakeConstArrayRef(defaultWeights) : weights, alpha, delta);
    }

    //TODO(isaf27): add baseline to CalcOptimumConstApprox
    inline TMaybe<double> CalcOptimumConstApprox(
        const NCatboostOptions::TLossDescription& lossDescription,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weights
    ) {
        // TODO(ilyzhin): refactor it (https://a.yandex-team.ru/review/969134/details#comment-1471301)
        auto lossFunction = lossDescription.GetLossFunction();
        switch(lossFunction) {
            case ELossFunction::RMSE:
                return CalculateWeightedTargetAverage(target, weights);
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
            {
                const double bestProbability = CalculateWeightedTargetAverage(target, weights);
                return Logit(bestProbability);
            }
            case ELossFunction::Quantile:
            case ELossFunction::MAE: {
                return CalculateWeightedTargetQuantile(target, weights, lossDescription);
            }
            default:
                return Nothing();
        }
    }
}
