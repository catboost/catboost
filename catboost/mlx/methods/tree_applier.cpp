#include "tree_applier.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/logging/logging.h>

#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    void ApplyObliviousTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& splits,
        const mx::array& leafValues,
        ui32 approxDimension
    ) {
        const ui32 numDocs  = dataset.GetNumDocs();
        const ui32 depth    = static_cast<ui32>(splits.size());
        const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();
        const ui32 lineSize = dataset.GetCompressedIndex().GetNumUi32PerDoc();

        // Pack per-level split descriptors into flat uint32 arrays.
        // Five parallel arrays of length `depth` — one entry per split level.
        // For depth=0: produce a single-element placeholder so MLX sees a
        // non-empty buffer; the Metal kernel checks (d < depth) so the split
        // loop body never executes, and all docs land on leaf 0.
        const ui32 bufDepth = (depth == 0) ? 1u : depth;
        TVector<uint32_t> colIdxVec(bufDepth, 0u), shiftVec(bufDepth, 0u),
                          maskVec(bufDepth, 0u),    threshVec(bufDepth, 0u),
                          isOneHotVec(bufDepth, 0u);
        for (ui32 d = 0; d < depth; ++d) {
            colIdxVec[d]   = splits[d].FeatureColumnIdx;
            shiftVec[d]    = splits[d].Shift;
            maskVec[d]     = splits[d].Mask;
            threshVec[d]   = splits[d].BinThreshold;
            isOneHotVec[d] = splits[d].IsOneHot ? 1u : 0u;
        }

        auto splitColIdxArr = mx::array(reinterpret_cast<const int32_t*>(colIdxVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitShiftArr  = mx::array(reinterpret_cast<const int32_t*>(shiftVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitMaskArr   = mx::array(reinterpret_cast<const int32_t*>(maskVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitThreshArr = mx::array(reinterpret_cast<const int32_t*>(threshVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);
        auto splitOneHotArr = mx::array(reinterpret_cast<const int32_t*>(isOneHotVec.data()),
                                        {static_cast<int>(bufDepth)}, mx::uint32);

        auto numDocsArr   = mx::array(static_cast<uint32_t>(numDocs),          mx::uint32);
        auto depthArr     = mx::array(static_cast<uint32_t>(depth),             mx::uint32);
        auto lineSizeArr  = mx::array(static_cast<uint32_t>(lineSize),          mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDimension),   mx::uint32);

        // Flatten compressedData to 1D [numDocs * lineSize].
        auto flatData = mx::reshape(compressedData,
            {static_cast<int>(numDocs * lineSize)});

        // Flatten leafValues to [numLeaves * approxDimension] row-major.
        // approxDimension == 1: [numLeaves] — reshape is a no-op.
        // approxDimension >  1: [numLeaves, K] — flatten to [numLeaves * K].
        const ui32 numLeaves    = 1u << depth;
        auto flatLeafValues     = mx::reshape(leafValues,
            {static_cast<int>(numLeaves * approxDimension)});

        // Flatten the existing cursor to [approxDimension * numDocs] for the kernel.
        // This is the read-only cursorIn — the kernel outputs cursorOut = cursorIn + delta.
        auto& cursor     = dataset.GetCursor();
        auto cursorShape = cursor.shape();
        auto flatCursor  = mx::reshape(cursor,
            {static_cast<int>(approxDimension * numDocs)});

        // ---- Metal kernel dispatch ----
        // One thread per document; no shared memory; no atomics.
        // Thread d computes leafIdx from all split levels then writes:
        //   cursorOut[k * numDocs + d] = cursorIn[k * numDocs + d]
        //                               + leafValues[leafIdx * approxDim + k]
        // for all k in [0, approxDimension).
        auto kernel = mx::fast::metal_kernel(
            "apply_oblivious_tree",
            /*input_names=*/{
                "compressedData", "splitColIdx", "splitShift", "splitMask",
                "splitThreshold", "splitIsOneHot", "leafValues", "cursorIn",
                "numDocs", "depth", "lineSize", "approxDim"
            },
            /*output_names=*/{"cursorOut"},
            /*source=*/KernelSources::kTreeApplySource,
            /*header=*/KernelSources::kTreeApplyHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(numDocs) + blockSize - 1) / blockSize;
        auto grid = std::make_tuple(numBlocks * blockSize, 1, 1);
        auto tg   = std::make_tuple(blockSize, 1, 1);

        auto results = kernel(
            /*inputs=*/{
                flatData, splitColIdxArr, splitShiftArr, splitMaskArr,
                splitThreshArr, splitOneHotArr, flatLeafValues, flatCursor,
                numDocsArr, depthArr, lineSizeArr, approxDimArr
            },
            /*output_shapes=*/{{static_cast<int>(approxDimension * numDocs)}},
            /*output_dtypes=*/{mx::float32},
            grid, tg,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        // Restore the original cursor shape and write back to the dataset.
        cursor = mx::reshape(results[0], cursorShape);

        // Update partition assignments: each document's leaf index.
        // Computed via MLX ops (O(depth) dispatches over a small depth <= 6).
        // This mirrors the old implementation's approach and keeps the partition
        // logic identical to structure_searcher.cpp's incremental bit-OR pattern.
        auto leafIndices = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
        for (ui32 level = 0; level < depth; ++level) {
            const auto& split = splits[level];

            auto column = mx::slice(compressedData,
                {0, static_cast<int>(split.FeatureColumnIdx)},
                {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)});
            column = mx::reshape(column, {static_cast<int>(numDocs)});

            auto featureValues = mx::bitwise_and(
                mx::right_shift(column, mx::array(static_cast<int>(split.Shift))),
                mx::array(static_cast<int>(split.Mask)));

            mx::array goRight;
            if (split.IsOneHot) {
                goRight = mx::equal(featureValues,
                    mx::array(static_cast<int>(split.BinThreshold)));
            } else {
                goRight = mx::greater(featureValues,
                    mx::array(static_cast<int>(split.BinThreshold)));
            }
            auto bits   = mx::astype(goRight, mx::uint32);
            bits        = mx::left_shift(bits, mx::array(static_cast<int>(level)));
            leafIndices = mx::bitwise_or(leafIndices, bits);
        }
        dataset.GetPartitions() = leafIndices;

        TMLXDevice::EvalNow({cursor, dataset.GetPartitions()});

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Applied tree depth=" << depth
            << " to " << numDocs << " documents (Metal kernel)" << Endl;
    }

}  // namespace NCatboostMlx
