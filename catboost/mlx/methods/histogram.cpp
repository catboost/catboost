#include "histogram.h"

#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>

namespace NCatboostMlx {

    namespace {
        // Dispatch histogram computation for one feature group (4 packed features).
        // This is a placeholder — the actual Metal kernel will be dispatched here.
        // TODO(Phase 7): Replace with direct Metal kernel dispatch via mx::fast::metal_kernel()
        void DispatchHistogramGroup(
            const mx::array& compressedData,
            const mx::array& stats,
            const mx::array& partOffsets,
            const mx::array& partSizes,
            ui32 featureColumnIdx,
            ui32 lineSize,
            ui32 maxBlocksPerPart,
            const mx::array& foldCounts,
            const mx::array& firstFoldIndices,
            ui32 totalBinFeatures,
            ui32 numStats,
            ui32 numPartitions,
            mx::array& histogram
        ) {
            // The actual Metal kernel (hist.metal) will be dispatched here using:
            //   mlx::core::fast::metal_kernel("histogram_one_byte_features", ...)
            //
            // The kernel source is already written in catboost/mlx/kernels/hist.metal.
            // Integration requires:
            //   1. Loading the .metal source
            //   2. Calling metal_kernel() to register it
            //   3. Invoking the returned function with proper grid/threadgroup dims
            //   4. Binding all buffer arguments

            Y_UNUSED(compressedData);
            Y_UNUSED(stats);
            Y_UNUSED(partOffsets);
            Y_UNUSED(partSizes);
            Y_UNUSED(featureColumnIdx);
            Y_UNUSED(lineSize);
            Y_UNUSED(maxBlocksPerPart);
            Y_UNUSED(foldCounts);
            Y_UNUSED(firstFoldIndices);
            Y_UNUSED(totalBinFeatures);
            Y_UNUSED(numStats);
            Y_UNUSED(numPartitions);
            Y_UNUSED(histogram);
        }
    }  // anonymous namespace

    mx::array CreateZeroHistogram(ui32 numPartitions, ui32 numStats, ui32 totalBinFeatures) {
        auto hist = mx::zeros(
            {static_cast<int>(numPartitions * numStats * totalBinFeatures)},
            mx::float32
        );
        TMLXDevice::EvalNow(hist);
        return hist;
    }

    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions,
        bool useWeights
    ) {
        const auto& compressedIndex = dataset.GetCompressedIndex();
        const auto& features = compressedIndex.GetFeatures();
        const ui32 numDocs = compressedIndex.GetNumDocs();
        const ui32 lineSize = compressedIndex.GetNumUi32PerDoc();
        const ui32 numStats = useWeights ? 2 : 1;

        // Count total bin features
        ui32 totalBinFeatures = 0;
        for (const auto& feat : features) {
            totalBinFeatures += feat.Folds;
        }

        CATBOOST_DEBUG_LOG << "CatBoost-MLX: Computing histograms for "
            << features.size() << " features, " << totalBinFeatures << " bin-features, "
            << numPartitions << " partitions, " << numStats << " stats" << Endl;

        // Allocate output histogram
        auto histogram = CreateZeroHistogram(numPartitions, numStats, totalBinFeatures);

        // For now, use a conservative single block per partition
        // (Phase 7 optimization: dynamic block count based on partition size)
        const ui32 maxBlocksPerPart = 1;

        // Process features in groups of 4 (one-byte packing)
        const ui32 numFeatures = features.size();
        const ui32 numFeatureGroups = (numFeatures + 3) / 4;

        for (ui32 groupIdx = 0; groupIdx < numFeatureGroups; ++groupIdx) {
            const ui32 featureStart = groupIdx * 4;
            const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);

            // Build per-feature fold counts and first fold indices for this group
            TVector<ui32> foldCounts(4, 0);
            TVector<ui32> firstFoldIndices(4, 0);
            for (ui32 f = 0; f < featuresInGroup; ++f) {
                foldCounts[f] = features[featureStart + f].Folds;
                firstFoldIndices[f] = features[featureStart + f].FirstFoldIndex;
            }

            // Create MLX arrays for kernel parameters
            auto foldCountsArr = mx::array(
                reinterpret_cast<const int32_t*>(foldCounts.data()),
                {4}, mx::uint32
            );
            auto firstFoldArr = mx::array(
                reinterpret_cast<const int32_t*>(firstFoldIndices.data()),
                {4}, mx::uint32
            );

            // Build the stats array: for gradient-only, just pass gradients
            // For gradient+weight, stack as [numStats, numDocs]
            auto statsArr = dataset.GetGradients();
            if (useWeights) {
                statsArr = mx::concatenate({
                    mx::reshape(dataset.GetGradients(), {1, static_cast<int>(numDocs)}),
                    mx::reshape(dataset.GetWeights(), {1, static_cast<int>(numDocs)})
                }, 0);
                statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});
            } else {
                statsArr = mx::reshape(statsArr, {static_cast<int>(numDocs)});
            }

            DispatchHistogramGroup(
                compressedIndex.GetCompressedData(),
                statsArr,
                partitionOffsets,
                partitionSizes,
                groupIdx,       // feature column index
                lineSize,
                maxBlocksPerPart,
                foldCountsArr,
                firstFoldArr,
                totalBinFeatures,
                numStats,
                numPartitions,
                histogram
            );
        }

        TMLXDevice::EvalNow(histogram);

        return THistogramResult{
            .Histograms = histogram,
            .NumPartitions = numPartitions,
            .NumStats = numStats,
            .TotalBinFeatures = totalBinFeatures
        };
    }

}  // namespace NCatboostMlx
