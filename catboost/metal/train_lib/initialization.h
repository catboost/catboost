#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/loss_description.h>
#include <library/cpp/accurate_accumulate/accurate_accumulate.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace NCB {
    // CUDA's host initialization is shared in optimal_const_for_loss.h. Keep
    // its float32 result and quantile delta convention, but replace bounded
    // value-space bisection with exact ordering of the observed float values.
    // Double products also avoid overflowing representable weighted means.
    inline float CalcMetalInitialBias(
        const NCatboostOptions::TLossDescription& loss,
        TConstArrayRef<float> targets, TConstArrayRef<float> weights)
    {
        CB_ENSURE(!targets.empty() && (weights.empty() || weights.size() == targets.size()),
                  "Metal bias initialization requires one weight per target");
        const auto objective = loss.GetLossFunction();
        TKahanAccumulator<double> totalWeight, targetSum;
        TVector<float> effectiveWeights(targets.size());
        for (size_t row = 0; row < targets.size(); ++row) {
            const float weight = weights.empty() ? 1.0f : weights[row];
            CB_ENSURE(std::isfinite(targets[row]) && std::isfinite(weight) && weight >= 0,
                      "Metal bias initialization requires finite targets and nonnegative weights");
            effectiveWeights[row] = objective == ELossFunction::MAPE
                ? weight / std::max(1.0f, std::abs(targets[row])) : weight;
            totalWeight += static_cast<double>(effectiveWeights[row]);
            targetSum += static_cast<double>(targets[row]) * weight;
        }
        CB_ENSURE(totalWeight.Get() > 0, "Metal bias initialization requires positive total weight");
        if (objective == ELossFunction::RMSE || objective == ELossFunction::Logloss ||
            objective == ELossFunction::CrossEntropy) {
            const float mean = static_cast<float>(targetSum.Get() / totalWeight.Get());
            if (objective == ELossFunction::RMSE) return mean;
            CB_ENSURE(mean > 0 && mean < 1,
                      "Metal binary boost_from_average requires a weighted target mean in (0, 1)");
            return static_cast<float>(std::log(static_cast<double>(mean) / (1.0 - mean)));
        }
        CB_ENSURE(objective == ELossFunction::Quantile || objective == ELossFunction::MAE ||
                  objective == ELossFunction::MAPE, "Metal boost_from_average is unsupported for this objective");
        const auto& params = loss.GetLossParamsMap();
        const double alpha = objective == ELossFunction::Quantile ? NCatboostOptions::GetAlpha(loss) : 0.5;
        const double delta = objective == ELossFunction::MAPE ? 0.0 :
            params.contains("delta") ? FromString<double>(params.at("delta")) : 1e-6;
        CB_ENSURE(std::isfinite(alpha) && alpha >= 0 && alpha <= 1 &&
                  std::isfinite(delta) && delta >= 0 && delta <= 0.01,
                  "Metal quantile initialization requires alpha in [0, 1] and delta in [0, 0.01]");
        TVector<size_t> order(targets.size());
        std::iota(order.begin(), order.end(), size_t(0));
        std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
            return targets[left] < targets[right];
        });
        size_t selected = 0;
        if (alpha >= 1) {
            selected = order.size() - 1;
            while (effectiveWeights[order[selected]] == 0) --selected;
        } else if (alpha > 0) {
            const double threshold = alpha * totalWeight.Get();
            double cumulative = 0;
            for (; selected + 1 < order.size(); ++selected) {
                cumulative += effectiveWeights[order[selected]];
                if (cumulative >= threshold) break;
            }
        }
        double quantile = targets[order[selected]];
        if (delta > 0) {
            TKahanAccumulator<double> below, equal;
            for (size_t row = 0; row < targets.size(); ++row) {
                if (targets[row] < quantile) below += effectiveWeights[row];
                else if (targets[row] == quantile) equal += effectiveWeights[row];
            }
            quantile += below.Get() + equal.Get() * alpha >= alpha * totalWeight.Get() -
                std::numeric_limits<double>::epsilon() ? -delta : delta;
        }
        return static_cast<float>(quantile);
    }
}
