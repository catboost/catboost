#include "histogram.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    namespace {
        // Dispatch histogram computation for one feature group (4 packed features).
        // Returns a fresh histogram array with this group's contributions.
        mx::array DispatchHistogramGroup(
            const mx::array& compressedData,
            const mx::array& stats,
            const mx::array& docIndices,
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
            ui32 totalNumDocs,
            const mx::Shape& histShape
        ) {
            // Create scalar mx::array inputs for constant uint& parameters.
            // 0-dim arrays become `const constant T&` in the generated Metal signature.
            auto featureColArr = mx::array(static_cast<uint32_t>(featureColumnIdx), mx::uint32);
            auto lineSizeArr   = mx::array(static_cast<uint32_t>(lineSize), mx::uint32);
            auto maxBlocksArr  = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
            auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
            auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
            auto totalDocsArr  = mx::array(static_cast<uint32_t>(totalNumDocs), mx::uint32);

            // Flatten compressed data to 1D for linear doc*lineSize indexing in kernel
            auto flatCompressed = mx::reshape(compressedData, {-1});

            // Register kernel (MLX caches compiled kernels internally by name)
            auto kernel = mx::fast::metal_kernel(
                "histogram_one_byte_features",
                /*input_names=*/{
                    "compressedIndex", "stats", "docIndices",
                    "partOffsets", "partSizes",
                    "featureColumnIdx", "lineSize", "maxBlocksPerPart",
                    "foldCounts", "firstFoldIndices",
                    "totalBinFeatures", "numStats", "totalNumDocs"
                },
                /*output_names=*/{"histogram"},
                /*source=*/KernelSources::kHistOneByteSource,
                /*header=*/KernelSources::kHistHeader,
                /*ensure_row_contiguous=*/true,
                /*atomic_outputs=*/false
            );

            // Grid: total threads to launch (MLX divides by threadgroup to get threadgroup count)
            // (256 * maxBlocksPerPart, numPartitions, numStats) gives
            // (maxBlocksPerPart, numPartitions, numStats) threadgroups of 256 threads each.
            auto grid = std::make_tuple(
                static_cast<int>(256 * maxBlocksPerPart),
                static_cast<int>(numPartitions),
                static_cast<int>(numStats)
            );
            auto threadgroup = std::make_tuple(256, 1, 1);

            auto results = kernel(
                /*inputs=*/{
                    flatCompressed, stats, docIndices,
                    partOffsets, partSizes,
                    featureColArr, lineSizeArr, maxBlocksArr,
                    foldCounts, firstFoldIndices,
                    totalBinsArr, numStatsArr, totalDocsArr
                },
                /*output_shapes=*/{histShape},
                /*output_dtypes=*/{mx::float32},
                grid,
                threadgroup,
                /*template_args=*/{},
                /*init_value=*/0.0f,
                /*verbose=*/false,
                /*stream=*/mx::Device::gpu
            );

            return results[0];
        }

        // Common histogram dispatch logic shared by both ComputeHistograms overloads.
        THistogramResult ComputeHistogramsImpl(
            const TMLXCompressedIndex& compressedIndex,
            const TVector<TCFeature>& features,
            const mx::array& statsArr,
            const mx::array& docIndices,
            const mx::array& partitionOffsets,
            const mx::array& partitionSizes,
            ui32 numDocs,
            ui32 lineSize,
            ui32 numStats,
            ui32 totalBinFeatures,
            ui32 numPartitions
        ) {
            mx::Shape histShape = {static_cast<int>(numPartitions * numStats * totalBinFeatures)};
            const ui32 maxBlocksPerPart = 1;

            const ui32 numFeatures = features.size();
            const ui32 numFeatureGroups = (numFeatures + 3) / 4;

            mx::array histogram;

            for (ui32 groupIdx = 0; groupIdx < numFeatureGroups; ++groupIdx) {
                const ui32 featureStart = groupIdx * 4;
                const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);

                TVector<ui32> foldCountsVec(4, 0);
                TVector<ui32> firstFoldIndicesVec(4, 0);
                for (ui32 f = 0; f < featuresInGroup; ++f) {
                    foldCountsVec[f] = features[featureStart + f].Folds;
                    firstFoldIndicesVec[f] = features[featureStart + f].FirstFoldIndex;
                }

                auto foldCountsArr = mx::array(
                    reinterpret_cast<const int32_t*>(foldCountsVec.data()),
                    {4}, mx::uint32
                );
                auto firstFoldArr = mx::array(
                    reinterpret_cast<const int32_t*>(firstFoldIndicesVec.data()),
                    {4}, mx::uint32
                );

                auto groupResult = DispatchHistogramGroup(
                    compressedIndex.GetCompressedData(),
                    statsArr,
                    docIndices,
                    partitionOffsets,
                    partitionSizes,
                    groupIdx,
                    lineSize,
                    maxBlocksPerPart,
                    foldCountsArr,
                    firstFoldArr,
                    totalBinFeatures,
                    numStats,
                    numPartitions,
                    numDocs,
                    histShape
                );

                if (groupIdx == 0) {
                    histogram = groupResult;
                } else {
                    histogram = mx::add(histogram, groupResult);
                }
            }

            TMLXDevice::EvalNow(histogram);

            return THistogramResult{
                .Histograms = histogram,
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = totalBinFeatures
            };
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
        const mx::array& docIndices,
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

        if (totalBinFeatures == 0) {
            return THistogramResult{
                .Histograms = mx::zeros({1}, mx::float32),
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = 0
            };
        }

        // Build the stats array: gradient only, or gradient + hessian stacked
        auto statsArr = dataset.GetGradients();
        if (useWeights) {
            statsArr = mx::concatenate({
                mx::reshape(dataset.GetGradients(), {1, static_cast<int>(numDocs)}),
                mx::reshape(dataset.GetHessians(), {1, static_cast<int>(numDocs)})
            }, 0);
            statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});
        } else {
            statsArr = mx::reshape(statsArr, {static_cast<int>(numDocs)});
        }

        return ComputeHistogramsImpl(
            compressedIndex, features, statsArr,
            docIndices, partitionOffsets, partitionSizes,
            numDocs, lineSize, numStats, totalBinFeatures, numPartitions
        );
    }

    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& gradients,
        const mx::array& hessians,
        const mx::array& docIndices,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions
    ) {
        const auto& compressedIndex = dataset.GetCompressedIndex();
        const auto& features = compressedIndex.GetFeatures();
        const ui32 numDocs = compressedIndex.GetNumDocs();
        const ui32 lineSize = compressedIndex.GetNumUi32PerDoc();
        const ui32 numStats = 2;  // always grad + hess

        ui32 totalBinFeatures = 0;
        for (const auto& feat : features) {
            totalBinFeatures += feat.Folds;
        }

        if (totalBinFeatures == 0) {
            return THistogramResult{
                .Histograms = mx::zeros({1}, mx::float32),
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = 0
            };
        }

        // Build stats array from provided gradient + hessian
        auto statsArr = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians, {1, static_cast<int>(numDocs)})
        }, 0);
        statsArr = mx::reshape(statsArr, {static_cast<int>(numStats * numDocs)});

        return ComputeHistogramsImpl(
            compressedIndex, features, statsArr,
            docIndices, partitionOffsets, partitionSizes,
            numDocs, lineSize, numStats, totalBinFeatures, numPartitions
        );
    }

}  // namespace NCatboostMlx
