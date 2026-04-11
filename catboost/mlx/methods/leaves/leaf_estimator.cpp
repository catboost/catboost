#include "leaf_estimator.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    mx::array ComputeLeafValues(
        const mx::array& gradientSums,
        const mx::array& hessianSums,
        float l2RegLambda,
        float learningRate
    ) {
        // Newton step: -gradSum / (hessSum + lambda) * lr
        // Using MLX ops (no custom kernel needed — numLeaves is small, typically 2-256).
        //
        // This function accepts any flat input shape — [numLeaves] for the scalar
        // (approxDim==1) path, or [approxDim * numLeaves] for the fused multiclass
        // path.  Either way the Newton step is element-wise so the same formula
        // applies regardless of how the caller has laid out the data.
        //
        // Returns a *lazy* MLX array — EvalNow is intentionally absent here.
        // The caller decides when to materialise (either at the next read site or
        // by batching the eval with other work).  This eliminates the K EvalNow
        // CPU-GPU round trips that the old per-dimension loop incurred.
        auto denominator = mx::add(hessianSums, mx::array(l2RegLambda));
        auto rawValues = mx::negative(mx::divide(gradientSums, denominator));
        return mx::multiply(rawValues, mx::array(learningRate));
    }

    void ComputeLeafSumsGPU(
        const mx::array& gradients,
        const mx::array& hessians,
        const mx::array& partitions,
        ui32 numDocs,
        ui32 numLeaves,
        ui32 approxDim,
        mx::array& outGradSums,
        mx::array& outHessSums
    ) {
        // Safety: kLeafAccumSource uses MAX_LEAVES=64 for shared memory allocation.
        // Exceeding this silently discards documents assigned to leaves >= 64,
        // corrupting leaf statistics. Fail fast with a clear error.
        CB_ENSURE(numLeaves <= 64,
            "CatBoost-MLX: ComputeLeafSumsGPU supports at most 64 leaves (max_depth=6). "
            "Got " << numLeaves << " leaves (max_depth=" << (ui32)__builtin_ctz(numLeaves) << "). "
            "Reduce max_depth to 6 or below.");

        // Flatten gradients/hessians to 1D: [approxDim * numDocs]
        auto flatGrads = mx::reshape(gradients, {static_cast<int>(approxDim * numDocs)});
        auto flatHess = mx::reshape(hessians, {static_cast<int>(approxDim * numDocs)});
        auto flatParts = mx::reshape(partitions, {static_cast<int>(numDocs)});

        auto numDocsArr = mx::array(static_cast<uint32_t>(numDocs), mx::uint32);
        auto numLeavesArr = mx::array(static_cast<uint32_t>(numLeaves), mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);

        auto kernel = mx::fast::metal_kernel(
            "leaf_accumulate",
            /*input_names=*/{"gradients", "hessians", "partitions",
                "numDocs", "numLeaves", "approxDim"},
            /*output_names=*/{"gradSums", "hessSums"},
            /*source=*/KernelSources::kLeafAccumSource,
            /*header=*/KernelSources::kLeafAccumHeader,
            /*ensure_row_contiguous=*/true,
            // BUG-001 FIX: atomic_outputs=false — the redesigned single-threadgroup
            // kernel writes directly (non-atomically). No other threadgroup touches the
            // same output slot so atomic writes are unnecessary.
            /*atomic_outputs=*/false
        );

        // BUG-001 FIX: Single-threadgroup dispatch.
        // The kernel iterates over all numDocs with stride LEAF_BLOCK_SIZE internally.
        // Eliminates cross-threadgroup atomic_fetch_add non-determinism on leaf slots.
        auto grid = std::make_tuple(256, 1, 1);
        auto tg = std::make_tuple(256, 1, 1);

        auto results = kernel(
            /*inputs=*/{flatGrads, flatHess, flatParts,
                numDocsArr, numLeavesArr, approxDimArr},
            /*output_shapes=*/{{static_cast<int>(approxDim * numLeaves)},
                               {static_cast<int>(approxDim * numLeaves)}},
            /*output_dtypes=*/{mx::float32, mx::float32},
            grid, tg,
            /*template_args=*/{},
            /*init_value=*/0.0f,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        outGradSums = results[0];
        outHessSums = results[1];

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: ComputeLeafSumsGPU: "
            << numDocs << " docs, " << numLeaves << " leaves, "
            << approxDim << " dims" << Endl;
    }

}  // namespace NCatboostMlx
