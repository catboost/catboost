#pragma once

// Leaf value estimation for CatBoost-MLX.
// Computes optimal leaf values using Newton's method on accumulated gradient/hessian sums.

#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/gpu_data/mlx_device.h>

#include <mlx/mlx.h>

namespace NCatboostMlx {
    namespace mx = mlx::core;

    // Compute leaf values from histogram gradient/hessian sums.
    // Returns an array of leaf values: [numLeaves] float32.
    //
    // Newton step: leaf_value = -gradientSum / (hessianSum + lambda) * learningRate
    mx::array ComputeLeafValues(
        const mx::array& gradientSums,  // [numLeaves]
        const mx::array& hessianSums,   // [numLeaves]
        float l2RegLambda,
        float learningRate
    );

    // GPU-accelerated leaf sum accumulation via Metal kernel.
    // Replaces the CPU for-loop over documents with a parallel segmented reduction.
    // Output arrays are [approxDim * numLeaves] float32, zero-initialized.
    void ComputeLeafSumsGPU(
        const mx::array& gradients,    // [approxDim, numDocs] or [numDocs]
        const mx::array& hessians,     // [approxDim, numDocs] or [numDocs]
        const mx::array& partitions,   // [numDocs] uint32
        ui32 numDocs,
        ui32 numLeaves,
        ui32 approxDim,
        mx::array& outGradSums,        // [approxDim * numLeaves]
        mx::array& outHessSums         // [approxDim * numLeaves]
    );

}  // namespace NCatboostMlx
