#pragma once

// Pointwise target functions for CatBoost-MLX.
// RMSE (L2) is the first supported loss function.

#include <catboost/mlx/targets/target_func.h>

namespace NCatboostMlx {

    // RMSE (Root Mean Squared Error) / L2 loss.
    // Loss = 0.5 * (pred - target)^2
    // Gradient = pred - target
    // Hessian = 1.0 (constant)
    class TRMSETarget : public IMLXTargetFunc {
    public:
        void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,
            mx::array& gradients,
            mx::array& hessians
        ) const override {
            // Gradient = prediction - target
            gradients = mx::subtract(cursor, targets);

            // Apply sample weights
            gradients = mx::multiply(gradients, weights);

            // Hessian is constant = weight (for weighted case) or 1.0
            hessians = mx::copy(weights);

            TMLXDevice::EvalNow({gradients, hessians});
        }

        mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const override {
            // RMSE = sqrt(weighted mean of squared errors)
            auto diff = mx::subtract(cursor, targets);
            auto sqDiff = mx::multiply(diff, diff);
            auto weightedSqDiff = mx::multiply(sqDiff, weights);
            auto loss = mx::sqrt(mx::mean(weightedSqDiff));
            TMLXDevice::EvalNow(loss);
            return loss;
        }
    };

    // Logloss (Binary Cross-Entropy) - for Phase 7
    // class TLoglossTarget : public IMLXTargetFunc { ... };

}  // namespace NCatboostMlx
