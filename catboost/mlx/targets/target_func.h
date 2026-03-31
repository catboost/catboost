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

        // Number of approximation dimensions.
        // 1 for RMSE and Logloss; numClasses-1 for MultiClass.
        virtual ui32 GetApproxDimension() const = 0;

        // Compute first-order gradient and second-order hessian.
        // For single-dim (RMSE, Logloss): cursor/gradients/hessians are [numDocs].
        // For multi-dim  (MultiClass):    cursor/gradients/hessians are [K, numDocs].
        // Targets are always [numDocs] (class index for classification).
        virtual void ComputeDerivatives(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights,  // [numDocs] sample weights
            mx::array& gradients,
            mx::array& hessians
        ) const = 0;

        // Compute scalar loss value (for metric reporting).
        virtual mx::array ComputeLoss(
            const mx::array& cursor,
            const mx::array& targets,
            const mx::array& weights
        ) const = 0;
    };

}  // namespace NCatboostMlx
