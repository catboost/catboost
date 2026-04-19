#include "histogram.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

namespace NCatboostMlx {

    namespace {
        // Threadgroup-memory budget guard for the L1a histogram kernel (Sprint 18).
        // The kernel declares `threadgroup float simdHist[NUM_SIMD_GROUPS][HIST_PER_SIMD]`
        // in kernel_sources.h; its byte size must stay within Apple Silicon's 32 KB
        // threadgroup-memory limit. MSL does not support static_assert in shader source,
        // so we mirror the constants here and assert on the host side. Any future bump
        // to the SIMD-group count or per-SIMD histogram size must re-tile the layout
        // (e.g. split the 4-tile reduction into more tiles) before tripping this assert.
        constexpr unsigned kHistSimdSize         = 32;
        constexpr unsigned kHistBlockSize        = 256;
        constexpr unsigned kHistFeaturesPerPack  = 4;
        constexpr unsigned kHistBinsPerByte      = 256;
        constexpr unsigned kHistNumSimdGroups    = kHistBlockSize / kHistSimdSize;            // 8
        constexpr unsigned kHistPerSimd          = kHistFeaturesPerPack * kHistBinsPerByte;    // 1024
        constexpr unsigned kHistThreadgroupBytes = kHistNumSimdGroups * kHistPerSimd * sizeof(float); // 32768
        constexpr unsigned kAppleSiliconTgLimit  = 32768;
        static_assert(kHistThreadgroupBytes <= kAppleSiliconTgLimit,
                      "L1a histogram kernel exceeds Apple Silicon's 32 KB threadgroup limit; "
                      "bumping NUM_SIMD_GROUPS or HIST_PER_SIMD requires re-tiling the reduction.");

        // Batched histogram dispatch — one Metal dispatch covers ALL feature groups.
        //
        // DEC-015 (Sprint 19): `compressedData` is the col-major transposed view
        // [numUi32PerDoc * numDocs] from TMLXCompressedIndex::GetCompressedDataTransposed().
        // The kernel load address uses `featureColumnIdx * totalNumDocs + docIdx`
        // (col-major) instead of `docIdx * lineSize + featureColumnIdx` (row-major).
        // This collapses 32-doc batch reads from ~25 cache lines (row-major) to
        // 1 cache line per 32-doc batch, eliminating the 12.78 ms gather-latency
        // bottleneck measured in S19-01b.
        //
        // Blast-radius note: only the L1a histogram kernel reads from the transposed
        // view.  All other consumers (leaves.metal, tree_applier.cpp, etc.) continue
        // to read from GetCompressedData() (row-major). CompressedData_ is unchanged.
        mx::array DispatchHistogramBatched(
            const mx::array& compressedDataTransposed, // [numUi32PerDoc * numDocs], col-major (DEC-015)
            const mx::array& stats,
            const mx::array& docIndices,
            const mx::array& partOffsets,
            const mx::array& partSizes,
            const mx::array& featureColumnIndices,     // [numGroups] — group g reads column g
            ui32 lineSize,
            ui32 maxBlocksPerPart,
            ui32 numGroups,
            const mx::array& foldCountsFlat,           // [numGroups * 4]
            const mx::array& firstFoldIndicesFlat,     // [numGroups * 4]
            ui32 totalBinFeatures,
            ui32 numStats,
            ui32 numPartitions,
            ui32 totalNumDocs,
            const mx::Shape& histShape
        ) {
            // Scalar uniforms → 0-dim arrays → `const constant T&` in Metal signature.
            auto lineSizeArr   = mx::array(static_cast<uint32_t>(lineSize), mx::uint32);
            auto maxBlocksArr  = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
            auto numGroupsArr  = mx::array(static_cast<uint32_t>(numGroups), mx::uint32);
            auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
            auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
            auto totalDocsArr  = mx::array(static_cast<uint32_t>(totalNumDocs), mx::uint32);

            // Input names match kHistOneByteSource variable names exactly.
            // The kernel body reads `featureColumnIndices[groupIdx]` (array) and
            // `numGroups` (scalar) — both must be present in input_names.
            auto kernel = mx::fast::metal_kernel(
                "histogram_one_byte_features",
                /*input_names=*/{
                    "compressedIndex", "stats", "docIndices",
                    "partOffsets", "partSizes",
                    "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
                    "foldCountsFlat", "firstFoldIndicesFlat",
                    "totalBinFeatures", "numStats", "totalNumDocs"
                },
                /*output_names=*/{"histogram"},
                /*source=*/KernelSources::kHistOneByteSource,
                /*header=*/KernelSources::kHistHeader,
                /*ensure_row_contiguous=*/true,
                /*atomic_outputs=*/true   // writeback uses atomic_fetch_add_explicit
            );

            // Grid: (256 * maxBlocksPerPart * numGroups, numPartitions, numStats).
            // Dividing by threadgroup (256,1,1) gives
            // (maxBlocksPerPart * numGroups, numPartitions, numStats) threadgroups.
            // Each threadgroup handles one (group, block, partition, stat) tuple.
            auto grid = std::make_tuple(
                static_cast<int>(256 * maxBlocksPerPart * numGroups),
                static_cast<int>(numPartitions),
                static_cast<int>(numStats)
            );
            auto threadgroup = std::make_tuple(256, 1, 1);

            auto results = kernel(
                /*inputs=*/{
                    compressedDataTransposed, stats, docIndices,
                    partOffsets, partSizes,
                    featureColumnIndices, lineSizeArr, maxBlocksArr, numGroupsArr,
                    foldCountsFlat, firstFoldIndicesFlat,
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
            const ui32 numGroups   = (numFeatures + 3) / 4;

            // Build flat fold metadata for all groups (4 slots per group).
            // featureColumnIndices[g] = g (group g reads the g-th ui32 column).
            TVector<ui32> foldCountsFlatVec(numGroups * 4, 0u);
            TVector<ui32> firstFoldIndicesFlatVec(numGroups * 4, 0u);
            TVector<ui32> featureColumnIndicesVec(numGroups, 0u);

            for (ui32 g = 0; g < numGroups; ++g) {
                featureColumnIndicesVec[g] = g;
                const ui32 featureStart    = g * 4;
                const ui32 featuresInGroup = std::min(4u, numFeatures - featureStart);
                for (ui32 slot = 0; slot < featuresInGroup; ++slot) {
                    foldCountsFlatVec[g * 4 + slot]      = features[featureStart + slot].Folds;
                    firstFoldIndicesFlatVec[g * 4 + slot] = features[featureStart + slot].FirstFoldIndex;
                }
            }

            auto foldCountsArr = mx::array(
                reinterpret_cast<const int32_t*>(foldCountsFlatVec.data()),
                {static_cast<int>(numGroups * 4)}, mx::uint32
            );
            auto firstFoldArr = mx::array(
                reinterpret_cast<const int32_t*>(firstFoldIndicesFlatVec.data()),
                {static_cast<int>(numGroups * 4)}, mx::uint32
            );
            auto featureColsArr = mx::array(
                reinterpret_cast<const int32_t*>(featureColumnIndicesVec.data()),
                {static_cast<int>(numGroups)}, mx::uint32
            );

            // DEC-015: col-major view — 1 cache line per 32-doc batch vs 25 row-major.
            const mx::array& compressedTransposed = compressedIndex.GetCompressedDataTransposed();

            auto histogram = DispatchHistogramBatched(
                compressedTransposed,
                statsArr,
                docIndices,
                partitionOffsets,
                partitionSizes,
                featureColsArr,
                lineSize,
                maxBlocksPerPart,
                numGroups,
                foldCountsArr,
                firstFoldArr,
                totalBinFeatures,
                numStats,
                numPartitions,
                numDocs,
                histShape
            );

            // No EvalNow here — histogram is consumed lazily as an input to the
            // suffix_sum_histogram Metal kernel in FindBestSplitGPU.  MLX will
            // materialise the full graph in that same command buffer, avoiding an
            // unnecessary CPU-GPU sync point.

            return THistogramResult{
                .Histograms = histogram,
                .NumPartitions = numPartitions,
                .NumStats = numStats,
                .TotalBinFeatures = totalBinFeatures
            };
        }
    }  // anonymous namespace

    mx::array CreateZeroHistogram(ui32 numPartitions, ui32 numStats, ui32 totalBinFeatures) {
        // Return a lazy zero array — no EvalNow needed.
        // mx::zeros() produces a trivially lazy expression; any downstream consumer
        // (e.g. scatter-add) will materialise it as part of that operation's graph.
        return mx::zeros(
            {static_cast<int>(numPartitions * numStats * totalBinFeatures)},
            mx::float32
        );
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
