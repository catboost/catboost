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

    // ---------------------------------------------------------------------------
    // Single-pass leaf accumulation (numLeaves <= 64, depth <= 6)
    // ---------------------------------------------------------------------------
    static void ComputeLeafSumsGPUSinglePass(
        const mx::array& flatGrads,
        const mx::array& flatHess,
        const mx::array& flatParts,
        ui32 numDocs,
        ui32 numLeaves,
        ui32 approxDim,
        mx::array& outGradSums,
        mx::array& outHessSums
    ) {
        auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),   mx::uint32);
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
            /*atomic_outputs=*/false
        );

        auto grid = std::make_tuple(256, 1, 1);
        auto tg   = std::make_tuple(256, 1, 1);

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
    }

    // ---------------------------------------------------------------------------
    // Multi-pass leaf accumulation (numLeaves > 64, depth 7-10)
    //
    // Each pass handles a chunk of LEAF_CHUNK_SIZE=64 leaves.  The chunked kernel
    // keeps the private per-thread array at LEAF_PRIV_SIZE (5 KB) by limiting the
    // working set to 64 leaves per pass — no register spill at any depth.
    //
    // numPasses = ceil(numLeaves / 64).  At depth=10 that is 16 passes (cheap
    // relative to the histogram build which dominates iteration time).
    // ---------------------------------------------------------------------------
    static void ComputeLeafSumsGPUMultiPass(
        const mx::array& flatGrads,
        const mx::array& flatHess,
        const mx::array& flatParts,
        ui32 numDocs,
        ui32 numLeaves,
        ui32 approxDim,
        mx::array& outGradSums,
        mx::array& outHessSums
    ) {
        constexpr ui32 kChunkSize = 64;

        auto chunkedKernel = mx::fast::metal_kernel(
            "leaf_accumulate_chunked",
            /*input_names=*/{"gradients", "hessians", "partitions",
                "numDocs", "chunkBase", "chunkSize", "approxDim"},
            /*output_names=*/{"gradSums", "hessSums"},
            /*source=*/KernelSources::kLeafAccumChunkedSource,
            /*header=*/KernelSources::kLeafAccumHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto grid = std::make_tuple(256, 1, 1);
        auto tg   = std::make_tuple(256, 1, 1);

        auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),   mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);

        // Allocate full output arrays (zero-initialised); fill chunk-by-chunk.
        std::vector<float> gradBuf(approxDim * numLeaves, 0.0f);
        std::vector<float> hessBuf(approxDim * numLeaves, 0.0f);

        for (ui32 chunkBase = 0; chunkBase < numLeaves; chunkBase += kChunkSize) {
            const ui32 chunkSize = std::min(kChunkSize, numLeaves - chunkBase);

            auto chunkBaseArr = mx::array(static_cast<uint32_t>(chunkBase), mx::uint32);
            auto chunkSizeArr = mx::array(static_cast<uint32_t>(chunkSize), mx::uint32);

            auto results = chunkedKernel(
                /*inputs=*/{flatGrads, flatHess, flatParts,
                    numDocsArr, chunkBaseArr, chunkSizeArr, approxDimArr},
                /*output_shapes=*/{{static_cast<int>(approxDim * chunkSize)},
                                   {static_cast<int>(approxDim * chunkSize)}},
                /*output_dtypes=*/{mx::float32, mx::float32},
                grid, tg,
                /*template_args=*/{},
                /*init_value=*/0.0f,
                /*verbose=*/false,
                /*stream=*/mx::Device::gpu
            );

            // Materialise before reading back to host.
            mx::eval({results[0], results[1]});

            const float* gp = results[0].data<float>();
            const float* hp = results[1].data<float>();

            // Copy chunk slice into the full output buffers.
            // Layout: [approxDim * numLeaves], dim-major.
            for (ui32 k = 0; k < approxDim; ++k) {
                for (ui32 li = 0; li < chunkSize; ++li) {
                    gradBuf[k * numLeaves + chunkBase + li] = gp[k * chunkSize + li];
                    hessBuf[k * numLeaves + chunkBase + li] = hp[k * chunkSize + li];
                }
            }
        }

        outGradSums = mx::array(gradBuf.data(),
            {static_cast<int>(approxDim * numLeaves)}, mx::float32);
        outHessSums = mx::array(hessBuf.data(),
            {static_cast<int>(approxDim * numLeaves)}, mx::float32);
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
        // Support depth 1-10 (2-1024 leaves).  The old CB_ENSURE(numLeaves <= 64)
        // guard is removed: single-pass handles depth <= 6 (64 leaves); multi-pass
        // handles depth 7-10 (128-1024 leaves) via chunked kernel dispatches.
        CB_ENSURE(numLeaves >= 2 && numLeaves <= 1024,
            "CatBoost-MLX: ComputeLeafSumsGPU supports depth 1-10 (2-1024 leaves). "
            "Got " << numLeaves << " leaves.");

        // Flatten to 1D: [approxDim * numDocs]
        auto flatGrads = mx::reshape(gradients, {static_cast<int>(approxDim * numDocs)});
        auto flatHess  = mx::reshape(hessians,  {static_cast<int>(approxDim * numDocs)});
        auto flatParts = mx::reshape(partitions, {static_cast<int>(numDocs)});

        if (numLeaves <= 64) {
            ComputeLeafSumsGPUSinglePass(
                flatGrads, flatHess, flatParts,
                numDocs, numLeaves, approxDim,
                outGradSums, outHessSums);
        } else {
            ComputeLeafSumsGPUMultiPass(
                flatGrads, flatHess, flatParts,
                numDocs, numLeaves, approxDim,
                outGradSums, outHessSums);
        }

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: ComputeLeafSumsGPU: "
            << numDocs << " docs, " << numLeaves << " leaves, "
            << approxDim << " dims" << Endl;
    }

}  // namespace NCatboostMlx
