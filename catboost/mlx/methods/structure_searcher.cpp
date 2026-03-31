#include "structure_searcher.h"
#include <catboost/mlx/methods/score_calcer.h>
#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    TObliviousTreeStructure SearchTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda
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

            // Compute partition offsets and sizes from current partition assignments
            // For oblivious trees, partitions are indexed 0..2^depth-1
            TVector<ui32> partOffsets(numPartitions, 0);
            TVector<ui32> partSizes(numPartitions, 0);

            // TODO: Compute actual partition sizes from dataset.GetPartitions()
            // For now, use uniform partition (all docs in leaf 0 at depth 0)
            if (depth == 0) {
                partOffsets[0] = 0;
                partSizes[0] = numDocs;
            } else {
                // After splits, need to sort/reorder docs by partition
                // or compute offsets from the partition array
                // This will be filled in during integration testing
                for (ui32 p = 0; p < numPartitions; ++p) {
                    partSizes[p] = numDocs / numPartitions;  // placeholder uniform split
                    partOffsets[p] = p * partSizes[p];
                }
            }

            auto partOffsetsArr = mx::array(
                reinterpret_cast<const int32_t*>(partOffsets.data()),
                {static_cast<int>(numPartitions)}, mx::uint32
            );
            auto partSizesArr = mx::array(
                reinterpret_cast<const int32_t*>(partSizes.data()),
                {static_cast<int>(numPartitions)}, mx::uint32
            );

            // Step 1: Compute histograms
            auto histResult = ComputeHistograms(
                dataset,
                partOffsetsArr,
                partSizesArr,
                numPartitions,
                /*useWeights=*/false  // RMSE has constant hessian=1
            );

            // Step 2: Compute partition statistics for scoring
            TVector<TPartitionStatistics> partStats(numPartitions);
            for (ui32 p = 0; p < numPartitions; ++p) {
                partStats[p] = TPartitionStatistics(
                    static_cast<double>(partSizes[p]),  // weight = count for uniform weights
                    0.0,  // sum (gradient sum — computed from histogram totals)
                    static_cast<double>(partSizes[p])
                );
            }

            // Step 3: Find best split across all features
            auto bestSplit = FindBestSplit(
                histResult,
                partStats,
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

            // Convert best split to oblivious split level.
            // TCFeature stores a pre-shifted mask (e.g. 0xFF000000 for byte at offset 24).
            // TObliviousSplitLevel needs the post-shift mask (e.g. 0xFF) because the tree
            // applier does: featureValue = (packed >> shift) & mask.
            const auto& feat = features[bestSplit.FeatureId];
            TObliviousSplitLevel split;
            split.FeatureColumnIdx = static_cast<ui32>(feat.Offset);
            split.Shift = feat.Shift;
            split.Mask = feat.Mask >> feat.Shift;  // unshift: 0xFF for 1-byte, 0xF for half-byte
            split.BinThreshold = bestSplit.BinId;

            result.Splits.push_back(split);
            result.SplitProperties.push_back(bestSplit);
        }

        return result;
    }

}  // namespace NCatboostMlx
