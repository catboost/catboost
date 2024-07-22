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

    inline float CalculateWeightedTargetAverageWithMissingValues(TConstArrayRef<float> target, TConstArrayRef<float> weights) {
        double targetSum = 0.0;
        double summaryWeight = 0.0;

        if (weights.empty()) {
            for (size_t i = 0; i < target.size(); ++i) {
                if (!IsNan(target[i])) {
                    targetSum += target[i];
                    summaryWeight += 1;
                }
            }
        } else {
            for (size_t i = 0; i < target.size(); ++i) {
                if (!IsNan(target[i])) {
                    targetSum += target[i] * weights[i];
                    summaryWeight += weights[i];
                }
            }
        }
        return targetSum / summaryWeight;
    }

    inline float CalculateWeightedTargetVariance(TConstArrayRef<float> target, TConstArrayRef<float> weights, float mean) {
        const double summaryWeight = weights.empty() ? target.size() : Accumulate(weights, 0.0);
        double targetSum = 0.0;
        if (weights.empty()) {
            for (size_t i = 0; i < target.size(); ++i) {
                targetSum += Sqr(target[i] - mean);
            }
        } else {
            Y_ASSERT(target.size() == weights.size());
            for (size_t i = 0; i < target.size(); ++i) {
                targetSum += Sqr(target[i] - mean) * weights[i];
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
        const TVector<float> defaultWeights(target.size(), 1); // espetrov: replace with dispatch by weights.empty()
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
            : TVector<float>(weights.begin(), weights.end()); // espetrov: replace with dispatch by weights.empty()
        for (auto idx : xrange(target.size())) {
            weightsWithTarget[idx] /= Max(1.0f, Abs(target[idx]));
        }
        return CalcSampleQuantile(target, weightsWithTarget, 0.5);
    }

    inline float CalculateOptimalConstApproxForLogCosh(
        TConstArrayRef<float> target,
        TConstArrayRef<float> weights
    ) {
        const int BINSEARCH_ITERATIONS = 100;
        const double APPROX_PRECISION = 1e-9;

        if (target.empty()) {
            return 0;
        }

        auto func = [&] (double approx, auto hasWeights) {
            double res = 0;
            for (auto idx: xrange(target.size())) {
                res += tanh(approx - target[idx]) * (hasWeights ? weights[idx]: 1.);
            }
            return res;
        };

        auto res = std::minmax_element(target.begin(), target.end());
        double left = *res.first;
        double right = *res.second;

        for (auto id = 0; id < BINSEARCH_ITERATIONS && (right - left) > APPROX_PRECISION; id++) {
            Y_UNUSED(id);
            double m = (left + right) / 2;
            double value = weights.empty() ? func(m, std::false_type()) : func(m, std::true_type());
            if (value > 0) {
                right = m;
            }
            else {
                left = m;
            }
        }

        return left;
    }

    //TODO(isaf27): add baseline to CalcOptimumConstApprox
    inline TMaybe<double> CalcOneDimensionalOptimumConstApprox(
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
            case ELossFunction::GroupQuantile:
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
            case ELossFunction::LogCosh: {
                return CalculateOptimalConstApproxForLogCosh(target, weights);
            }
            default:
                return Nothing();
        }
    }

    inline TMaybe<TVector<double>> CalcOptimumConstApprox(
        const NCatboostOptions::TLossDescription& lossDescription,
        TConstArrayRef<TConstArrayRef<float>> target,
        TConstArrayRef<float> weights
    ) {
        auto lossFunction = lossDescription.GetLossFunction();
        switch (lossFunction) {
            case ELossFunction::RMSEWithUncertainty:
            {
                double mean = CalculateWeightedTargetAverage(target[0], weights);
                double var = CalculateWeightedTargetVariance(target[0], weights, mean);
                return TVector<double>({mean, 0.5 * log(var)});
            }
            case ELossFunction::MultiRMSE:
            {
                NCatboostOptions::TLossDescription singleRMSELoss;
                singleRMSELoss.LossFunction = ELossFunction::RMSE;
                TVector<double> startPoint(target.size());
                for (int dim : xrange(target.size())) {
                    startPoint[dim] = *CalcOneDimensionalOptimumConstApprox(singleRMSELoss, target[dim], weights);
                }
                return startPoint;
            }
            case ELossFunction::MultiLogloss:
            {
                NCatboostOptions::TLossDescription logloss;
                logloss.LossFunction = ELossFunction::Logloss;
                TVector<double> startPoint(target.size());
                for (int dim : xrange(target.size())) {
                    startPoint[dim] = *CalcOneDimensionalOptimumConstApprox(logloss, target[dim], weights);
                }
                return startPoint;
            }
            case ELossFunction::MultiQuantile:
            {
                auto params = lossDescription.GetLossParamsMap();
                const auto alpha = NCatboostOptions::GetAlphaMultiQuantile(params);
                NCatboostOptions::TLossDescription quantileDescription;
                quantileDescription.LossFunction = ELossFunction::Quantile;
                if (params.contains("delta")) {
                    quantileDescription.LossParams->Put("delta", params.at("delta"));
                }
                const auto quantileCount = alpha.size();
                TVector<double> startPoint(quantileCount);
                for (auto quantile : xrange(quantileCount)) {
                    quantileDescription.LossParams->Put("alpha", ToString(alpha[quantile]));
                    startPoint[quantile] = *CalcOneDimensionalOptimumConstApprox(
                        quantileDescription,
                        target[0],
                        weights);
                }
                return startPoint;
            }
            case ELossFunction::MultiRMSEWithMissingValues:
            {
                TVector<double> startPoint(target.size());
                for (int dim : xrange(target.size())) {
                    startPoint[dim] = CalculateWeightedTargetAverageWithMissingValues(target[dim], weights);
                }
                return startPoint;
            }
            default:
            {
                TMaybe<double> optimum = CalcOneDimensionalOptimumConstApprox(lossDescription, target[0], weights);
                if (optimum.Defined()) {
                    return TVector<double>(1, *optimum.Get());
                } else {
                    return Nothing();
                }
            }
        }
    }
}
