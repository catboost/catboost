#pragma once

// Base interface for MLX GPU target (loss) functions.
// Each target computes gradients and hessians from the current prediction cursor.

#include <catboost/mlx/gpu_data/mlx_device.h>
#include <mlx/mlx.h>

namespace NCatboostMlx {
    namespace mx = mlx::core;

    class IMLXTargetFunc {
    public:
        virtual ~IMLXTargetFunc() = default;

        // Compute first-order gradient (derivative of loss w.r.t. prediction)
        // and second-order hessian for each document.
        virtual void ComputeDerivatives(
            const mx::array& cursor,   // [numDocs] current predictions
            const mx::array& targets,  // [numDocs] true values
            const mx::array& weights,  // [numDocs] sample weights
            mx::array& gradients,      // [numDocs] output: d(loss)/d(pred)
            mx::array& hessians        // [numDocs] output: d²(loss)/d(pred)²
        ) const = 0;

        // Compute the loss value (for metric reporting)
        virtual mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const = 0;
    };

}  // namespace NCatboostMlx
