#pragma once

#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/gpu_data/dataset_base.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/options/loss_description.h>

#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <cmath>

namespace NCatboostCuda {

    template <class TMapping>
    inline TMaybe<TVector<double>> CalcOptimumConstApproxFromGpuTarget(
        const TTarget<TMapping>& target,
        const NCatboostOptions::TLossDescription& lossDescription
    ) {
        const ELossFunction lossFunction = lossDescription.GetLossFunction();
        const auto& targets = target.GetTargets();
        const auto& weights = target.GetWeights();

        CB_ENSURE(
            targets.GetColumnCount() == 1,
            "GPU boost_from_average for multi-dimensional targets is not supported"
        );

        const double totalWeight = static_cast<double>(ReduceToHost(weights, EOperatorType::Sum));
        CB_ENSURE(totalWeight > 0.0, "All weights are zero");

        const double targetWeightedSum = static_cast<double>(DotProduct(targets, weights));
        const double mean = targetWeightedSum / totalWeight;

        switch (lossFunction) {
            case ELossFunction::RMSE:
                return TVector<double>{mean};
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
                return TVector<double>{NCB::Logit(mean)};
            case ELossFunction::RMSEWithUncertainty: {
                const double target2WeightedSum = static_cast<double>(DotProduct(targets, targets, &weights));
                const double mean2 = target2WeightedSum / totalWeight;
                double var = mean2 - mean * mean;
                if (var < 0.0) {
                    var = 0.0;
                }
                return TVector<double>{mean, 0.5 * std::log(var)};
            }
            default:
                return Nothing();
        }
    }

}
