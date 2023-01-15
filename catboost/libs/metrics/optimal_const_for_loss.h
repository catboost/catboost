#pragma once

#include <catboost/libs/helpers/math_utils.h>
#include <catboost/libs/helpers/quantile.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/string/cast.h>


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
        double alpha,
        double delta
    ) {
        if (target.empty()) {
            return 0;
        }
        const TVector<float> defaultWeights(target.size(), 1);
        const auto weightsRef = weights.empty() ? MakeConstArrayRef(defaultWeights) : weights;
        double q = CalcSampleQuantile(target, weightsRef, alpha);

        // specific adjust according to delta parameter
        if (delta > 0) {
            const double totalWeight = weights.empty() ? static_cast<double>(target.size()) : Accumulate(weights, 0.0);
            const double needWeight = totalWeight * alpha;
            double lessWeight = 0;
            double equalWeight = 0;
            for (auto i : xrange(target.size())) {
                if (target[i] < q) {
                    lessWeight += weightsRef[i];
                } else if (target[i] == q) {
                    equalWeight += weightsRef[i];
                }
            }
            if (lessWeight + equalWeight * alpha >= needWeight - DBL_EPSILON) {
                q -= delta;
            } else {
                q += delta;
            }
        }

        return q;
    }

    inline float CalculateOptimalConstApproxForMAPE(
        TConstArrayRef<float> target,
        TConstArrayRef<float> weights
    ) {
        TVector<float> weightsWithTarget = weights.empty()
            ? TVector<float>(target.size(), 1.0)
            : TVector<float>(weights.begin(), weights.end());
        for (auto idx : xrange(target.size())) {
            weightsWithTarget[idx] /= Max(1.0f, Abs(target[idx]));
        }
        return CalcSampleQuantile(target, weightsWithTarget, 0.5);
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
                const auto& params = lossDescription.GetLossParamsMap();
                auto it = params.find("alpha");
                double alpha = it == params.end() ? 0.5 : FromString<double>(it->second);
                it = params.find("delta");
                double delta = it == params.end() ? 1e-6 : FromString<double>(it->second);
                return CalculateWeightedTargetQuantile(target, weights, alpha, delta);
            }
            case ELossFunction::MAPE:
                return CalculateOptimalConstApproxForMAPE(target, weights);
            default:
                return Nothing();
        }
    }
}
