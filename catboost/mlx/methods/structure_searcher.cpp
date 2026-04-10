#include "structure_searcher.h"
#include <catboost/mlx/methods/score_calcer.h>
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    TPartitionLayout ComputePartitionLayout(
        const mx::array& partitions, ui32 numDocs, ui32 numPartitions
    ) {
        // GPU-resident bucket sort — zero CPU-GPU syncs.
        //
        // Step 1: stable sort doc indices by their partition assignment.
        //   MLX argsort on uint32 uses a stable merge sort.
        //   Docs within the same partition appear in ascending docIdx order,
        //   matching the original CPU scatter-sort behaviour.
        auto docIndices = mx::astype(
            mx::argsort(partitions, /*axis=*/0), mx::uint32
        );

        // Step 2: count docs per partition via scatter-add of ones.
        //   Reuse the float32 scatter pattern from ComputeLeafSumsGPU.
        //   float32 is exact for integer values up to 2^24 (~16M docs).
        CB_ENSURE(numDocs < (1u << 24),
            "ComputePartitionLayout: numDocs (" << numDocs
            << ") exceeds float32 exact integer range (2^24 = 16777216). "
            "Bucket counts via scatter_add_axis would incur rounding errors.");
        auto onesF = mx::ones({static_cast<int>(numDocs)}, mx::float32);
        auto partSizesF = mx::scatter_add_axis(
            mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
            partitions, onesF, 0
        );

        // Step 3: exclusive prefix sum for partition start offsets.
        //   MLX cumsum(a, axis, reverse=false, inclusive=false) gives exclusive form directly.
        auto partOffsetsF = mx::cumsum(partSizesF, /*axis=*/0, /*reverse=*/false, /*inclusive=*/false);

        TPartitionLayout layout;
        layout.DocIndices  = docIndices;
        layout.PartSizes   = mx::astype(partSizesF, mx::uint32);
        layout.PartOffsets = mx::astype(partOffsetsF, mx::uint32);

        // No EvalNow here — arrays are consumed lazily by the histogram kernel
        // in the same MLX graph, avoiding an unnecessary CPU-GPU sync point.
        return layout;
    }

    TObliviousTreeStructure SearchTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda,
        ui32 approxDimension
    ) {
        TObliviousTreeStructure result;
        result.Splits.reserve(maxDepth);
        result.SplitProperties.reserve(maxDepth);

        const auto& features = dataset.GetCompressedIndex().GetFeatures();
        const ui32 numDocs = dataset.GetNumDocs();

        for (ui32 depth = 0; depth < maxDepth; ++depth) {
            const ui32 numPartitions = 1u << depth;  // 2^depth leaves at current level

            CATBOOST_DEBUG_LOG << "CatBoost-MLX: Searching depth " << depth
                << " (" << numPartitions << " partitions)" << Endl;

            // Compute partition layout from current assignments
            auto layout = ComputePartitionLayout(
                dataset.GetPartitions(), numDocs, numPartitions);

            // Step 1-2: Compute histograms per dimension; compute partition stats
            // once for all dims in a single GPU dispatch (OPT-1: eliminates
            // approxDim CPU-GPU round trips per depth level).
            TVector<THistogramResult> perDimHistograms;
            perDimHistograms.reserve(approxDimension);

            for (ui32 k = 0; k < approxDimension; ++k) {
                // Slice gradients and hessians for dimension k (histogram only)
                mx::array dimGrads, dimHess;
                if (approxDimension == 1) {
                    dimGrads = mx::reshape(dataset.GetGradients(), {static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dataset.GetHessians(),  {static_cast<int>(numDocs)});
                } else {
                    dimGrads = mx::slice(dataset.GetGradients(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimGrads = mx::reshape(dimGrads, {static_cast<int>(numDocs)});
                    dimHess  = mx::slice(dataset.GetHessians(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimHess  = mx::reshape(dimHess, {static_cast<int>(numDocs)});
                }

                auto histResult = ComputeHistograms(
                    dataset, dimGrads, dimHess,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    numPartitions
                );
                perDimHistograms.push_back(std::move(histResult));
            }

            // Single GPU dispatch for all-dimension partition stats (OPT-1).
            // ComputeLeafSumsGPU handles approxDim internally via the [approxDim,numDocs]
            // layout — no CPU readback needed here; GPU arrays passed directly to
            // FindBestSplitGPU overload that accepts mx::array partition totals.
            mx::array allGradSums, allHessSums;
            ComputeLeafSumsGPU(
                dataset.GetGradients(), dataset.GetHessians(),
                dataset.GetPartitions(),
                numDocs, numPartitions, approxDimension,
                allGradSums, allHessSums
            );
            // allGradSums / allHessSums: [approxDim * numPartitions] float32 — stays on GPU

            // Step 3: Find best split on GPU using pre-computed GPU partition sums
            // and binToFeature lookup table (OPT-1 + OPT-2 combined path).
            auto bestSplit = FindBestSplitGPU(
                perDimHistograms,
                allGradSums, allHessSums,
                features,
                l2RegLambda,
                numPartitions,
                approxDimension
            );

            if (!bestSplit.Defined()) {
                CATBOOST_INFO_LOG << "CatBoost-MLX: No valid split found at depth " << depth
                    << ", stopping tree growth" << Endl;
                break;
            }

            CATBOOST_DEBUG_LOG << "CatBoost-MLX: Best split at depth " << depth
                << ": feature=" << bestSplit.FeatureId << " bin=" << bestSplit.BinId
                << " gain=" << bestSplit.Gain << Endl;

            // Build the split level descriptor
            const auto& feat = features[bestSplit.FeatureId];
            TObliviousSplitLevel split;
            split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            split.Shift = feat.Shift;
            split.Mask = feat.Mask >> feat.Shift;
            split.BinThreshold = bestSplit.BinId;
            split.IsOneHot = feat.OneHotFeature;

            result.Splits.push_back(split);
            result.SplitProperties.push_back(bestSplit);

            // Apply this split level to update partition assignments for next depth.
            // leafIdx |= (featureValue > threshold) << depth
            {
                auto& partitions = dataset.GetPartitions();
                const auto& compressedData = dataset.GetCompressedIndex().GetCompressedData();

                auto column = mx::slice(compressedData,
                    {0, static_cast<int>(split.FeatureColumnIdx)},
                    {static_cast<int>(numDocs), static_cast<int>(split.FeatureColumnIdx + 1)}
                );
                column = mx::reshape(column, {static_cast<int>(numDocs)});

                auto featureValues = mx::bitwise_and(
                    mx::right_shift(column, mx::array(static_cast<int>(split.Shift))),
                    mx::array(static_cast<int>(split.Mask))
                );

                // Compare: OneHot uses equality, ordinal uses greater-than
                mx::array goRight;
                if (feat.OneHotFeature) {
                    goRight = mx::equal(featureValues,
                        mx::array(static_cast<int>(split.BinThreshold)));
                } else {
                    goRight = mx::greater(featureValues,
                        mx::array(static_cast<int>(split.BinThreshold)));
                }
                auto bits = mx::astype(goRight, mx::uint32);
                bits = mx::left_shift(bits, mx::array(static_cast<int>(depth)));

                partitions = mx::bitwise_or(partitions, bits);
                TMLXDevice::EvalNow(partitions);
            }
        }

        return result;
    }

}  // namespace NCatboostMlx
