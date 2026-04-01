#include "structure_searcher.h"
#include <catboost/mlx/methods/score_calcer.h>
#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    TPartitionLayout ComputePartitionLayout(
        const mx::array& partitions, ui32 numDocs, ui32 numPartitions
    ) {
        TPartitionLayout layout;
        layout.PartSizesHost.resize(numPartitions, 0);
        layout.PartOffsetsHost.resize(numPartitions, 0);

        // Read partition assignments to CPU
        TMLXDevice::EvalNow(partitions);
        const uint32_t* partData = partitions.data<uint32_t>();

        // Count docs per partition
        for (ui32 d = 0; d < numDocs; ++d) {
            ui32 p = partData[d];
            if (p < numPartitions) {
                layout.PartSizesHost[p]++;
            }
        }

        // Compute prefix-sum offsets
        for (ui32 p = 1; p < numPartitions; ++p) {
            layout.PartOffsetsHost[p] = layout.PartOffsetsHost[p - 1] + layout.PartSizesHost[p - 1];
        }

        // Build sorted doc indices: scatter each doc to its partition's slot
        TVector<ui32> sortedDocIndices(numDocs);
        TVector<ui32> writePos(layout.PartOffsetsHost);  // copy offsets as write cursors
        for (ui32 d = 0; d < numDocs; ++d) {
            ui32 p = partData[d];
            if (p < numPartitions) {
                sortedDocIndices[writePos[p]++] = d;
            }
        }

        // Transfer to GPU
        layout.DocIndices = mx::array(
            reinterpret_cast<const int32_t*>(sortedDocIndices.data()),
            {static_cast<int>(numDocs)}, mx::uint32
        );
        layout.PartOffsets = mx::array(
            reinterpret_cast<const int32_t*>(layout.PartOffsetsHost.data()),
            {static_cast<int>(numPartitions)}, mx::uint32
        );
        layout.PartSizes = mx::array(
            reinterpret_cast<const int32_t*>(layout.PartSizesHost.data()),
            {static_cast<int>(numPartitions)}, mx::uint32
        );
        TMLXDevice::EvalNow({layout.DocIndices, layout.PartOffsets, layout.PartSizes});

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

            // Step 1-2: Compute histograms and partition stats per dimension
            TVector<THistogramResult> perDimHistograms;
            TVector<TVector<TPartitionStatistics>> perDimPartStats;
            perDimHistograms.reserve(approxDimension);
            perDimPartStats.reserve(approxDimension);

            TMLXDevice::EvalNow({dataset.GetGradients(), dataset.GetHessians(), dataset.GetPartitions()});
            const uint32_t* parts = dataset.GetPartitions().data<uint32_t>();

            for (ui32 k = 0; k < approxDimension; ++k) {
                // Slice gradients and hessians for dimension k
                mx::array dimGrads, dimHess;
                if (approxDimension == 1) {
                    dimGrads = mx::reshape(dataset.GetGradients(), {static_cast<int>(numDocs)});
                    dimHess = mx::reshape(dataset.GetHessians(), {static_cast<int>(numDocs)});
                } else {
                    dimGrads = mx::slice(dataset.GetGradients(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimGrads = mx::reshape(dimGrads, {static_cast<int>(numDocs)});
                    dimHess = mx::slice(dataset.GetHessians(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimHess = mx::reshape(dimHess, {static_cast<int>(numDocs)});
                }

                auto histResult = ComputeHistograms(
                    dataset, dimGrads, dimHess,
                    layout.DocIndices, layout.PartOffsets, layout.PartSizes,
                    numPartitions
                );
                perDimHistograms.push_back(std::move(histResult));

                // Compute partition statistics for this dimension
                TMLXDevice::EvalNow(dimGrads);
                const float* grads = dimGrads.data<float>();
                TMLXDevice::EvalNow(dimHess);
                const float* hess = dimHess.data<float>();

                TVector<TPartitionStatistics> dimPartStats(numPartitions);
                for (ui32 d = 0; d < numDocs; ++d) {
                    ui32 p = parts[d];
                    if (p < numPartitions) {
                        dimPartStats[p].Sum += grads[d];
                        dimPartStats[p].Weight += hess[d];
                        dimPartStats[p].Count += 1.0;
                    }
                }
                perDimPartStats.push_back(std::move(dimPartStats));
            }

            // Step 3: Find best split across all features, summing gain across dims
            auto bestSplit = FindBestSplitMultiDim(
                perDimHistograms,
                perDimPartStats,
                features,
                l2RegLambda,
                numPartitions
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
