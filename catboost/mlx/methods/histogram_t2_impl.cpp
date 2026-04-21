// histogram_t2_impl.cpp — Production T2 sort-by-bin histogram dispatch.
//
// Intentionally minimal dependencies: only mlx.h + kernel_sources.h.
// This file is compiled:
//   (a) as part of the CatBoost-MLX library (linked with histogram.cpp), and
//   (b) directly by catboost/mlx/tests/bench_boosting.cpp for standalone builds.
//
// The separation from histogram.cpp avoids pulling the full CatBoost header
// tree (TMLXDataSet, TVector, CB_ENSURE, etc.) into the bench's standalone
// compilation unit.
//
// Public API: NCatboostMlx::DispatchHistogramT2 (declared in histogram.h).

#include <catboost/mlx/kernels/kernel_sources.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

using ui32 = uint32_t;

namespace NCatboostMlx {
namespace mx = mlx::core;

    // =========================================================================
    // T2 kernel registration — module-scope statics, initialized once per process.
    //
    // Kernel names bumped to s23d0 (Commit 3) after numTGs removal (NIT-5).
    // The name change invalidates MLX's kernel cache vs the s22d2 registration,
    // ensuring the updated input list is used (not a stale compiled version).
    // =========================================================================

    namespace {
        mx::fast::CustomKernelFunction& GetT2SortKernel() {
            // NIT-5 applied: numTGs removed (it was never read by the kernel body).
            static auto k = mx::fast::metal_kernel(
                "t2_sort_s23d0",
                /*input_names=*/{
                    "compressedIndex", "docIndices", "partOffsets", "partSizes",
                    "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
                    "numPartitions", "numStats", "totalNumDocs"
                },
                /*output_names=*/{"sortedDocs", "binOffsets"},
                /*source=*/KernelSources::kT2SortSource,
                /*header=*/KernelSources::kHistHeader,
                /*ensure_row_contiguous=*/true,
                /*atomic_outputs=*/false
            );
            return k;
        }

        mx::fast::CustomKernelFunction& GetT2AccumKernel() {
            // NIT-5 applied: numTGs removed (it was never read by the kernel body).
            static auto k = mx::fast::metal_kernel(
                "t2_accum_s23d0",
                /*input_names=*/{
                    "sortedDocs", "binOffsets", "compressedIndex", "stats",
                    "featureColumnIndices", "foldCountsFlat", "firstFoldIndicesFlat",
                    "partOffsets",
                    "lineSize", "maxBlocksPerPart", "numGroups",
                    "numPartitions", "numStats",
                    "totalBinFeatures", "totalNumDocs"
                },
                /*output_names=*/{"histogram"},
                /*source=*/KernelSources::kT2AccumSource,
                /*header=*/KernelSources::kHistHeader,
                /*ensure_row_contiguous=*/true,
                /*atomic_outputs=*/true
            );
            return k;
        }
    }  // anonymous namespace

    // =========================================================================
    // DispatchHistogramT2 — production T2 sort-by-bin histogram dispatch.
    //
    // Preconditions (callers must enforce before calling):
    //   - maxBlocksPerPart == 1  (NIT-4: T2 kernels dispatch one block per partition;
    //     any value > 1 silently wastes TGs).  Enforced by CB_ENSURE in histogram.cpp
    //     when called via ComputeHistogramsImpl; bench enforces via its own guard.
    //   - maxFoldCount <= 127  (DEC-016 T1 envelope).  Caller responsibility.
    //
    // Grid geometry: identical to DispatchHistogramBatched (T1):
    //   (256 * maxBlocksPerPart * numGroups, numPartitions, numStats)
    //
    // Buffer sizes at gate config (50k/RMSE/d6/128b, 50 features, numStats=2):
    //   sortedDocs: 13 groups × 2 stats × 50000 docs × 4 B = 5.2 MB
    //   binOffsets: 13 × 64 parts × 2 stats × 129 entries × 4 B ≈ 0.86 MB
    // =========================================================================

    mx::array DispatchHistogramT2(
        const mx::array& compressedData,
        const mx::array& stats,
        const mx::array& docIndices,
        const mx::array& partOffsets,
        const mx::array& partSizes,
        const mx::array& featureColIndices,
        const mx::array& foldCountsFlat,
        const mx::array& firstFoldFlat,
        ui32 lineSize,
        ui32 maxBlocksPerPart,
        ui32 numGroups,
        ui32 numPartitions,
        ui32 numStats,
        ui32 totalBinFeatures,
        ui32 totalNumDocs,
        const mx::Shape& histShape
    ) {
        // Scalar uniforms — 0-dim arrays become `const constant T&` in Metal signature.
        // NIT-5: numTGs removed (was never read by the kernel body; s23d0 kernel registration
        //   does not include it in input_names).
        auto flatCompressed = mx::reshape(compressedData, {-1});
        auto lineSizeArr    = mx::array(static_cast<uint32_t>(lineSize),         mx::uint32);
        auto maxBlocksArr   = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
        auto numGroupsArr   = mx::array(static_cast<uint32_t>(numGroups),        mx::uint32);
        auto numPartsArr    = mx::array(static_cast<uint32_t>(numPartitions),    mx::uint32);
        auto numStatsArr    = mx::array(static_cast<uint32_t>(numStats),         mx::uint32);
        auto totalBinsArr   = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
        auto totalDocsArr   = mx::array(static_cast<uint32_t>(totalNumDocs),     mx::uint32);
        const ui32 numTGsVal = numGroups * numPartitions * numStats;  // used for buffer size only

        // Grid: same geometry as DispatchHistogramBatched (T1).
        // T2 maxBlocksPerPart is always 1, so X = 256 * numGroups.
        auto grid = std::make_tuple(
            static_cast<int>(256 * maxBlocksPerPart * numGroups),
            static_cast<int>(numPartitions),
            static_cast<int>(numStats)
        );
        auto threadgroup = std::make_tuple(256, 1, 1);

        // sortedDocs: one slab per (groupIdx, statIdx) of size totalNumDocs.
        // binOffsets: 129 entries per TG (BIN_OFFSETS_STRIDE from kHistHeader).
        mx::Shape sortedDocsShape = {static_cast<int>(numGroups * numStats * totalNumDocs)};
        mx::Shape binOffsetsShape  = {static_cast<int>(numTGsVal * 129u)};

        // --- T2-sort dispatch ---
        // MLX lazy: sortOut is a lazy expression; not evaluated until consumed.
        auto sortOut = GetT2SortKernel()(
            /*inputs=*/{
                flatCompressed, docIndices, partOffsets, partSizes,
                featureColIndices, lineSizeArr, maxBlocksArr, numGroupsArr,
                numPartsArr, numStatsArr, totalDocsArr
            },
            /*output_shapes=*/{sortedDocsShape, binOffsetsShape},
            /*output_dtypes=*/{mx::uint32, mx::uint32},
            grid, threadgroup,
            /*template_args=*/{},
            /*init_value=*/0.0f,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        // --- T2-accum dispatch ---
        // sortOut[0] and sortOut[1] as inputs → MLX graph ensures sort runs before accum.
        auto accumOut = GetT2AccumKernel()(
            /*inputs=*/{
                sortOut[0], sortOut[1], flatCompressed, stats,
                featureColIndices, foldCountsFlat, firstFoldFlat,
                partOffsets,
                lineSizeArr, maxBlocksArr, numGroupsArr,
                numPartsArr, numStatsArr,
                totalBinsArr, totalDocsArr
            },
            /*output_shapes=*/{histShape},
            /*output_dtypes=*/{mx::float32},
            grid, threadgroup,
            /*template_args=*/{},
            /*init_value=*/0.0f,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        return accumOut[0];
    }

}  // namespace NCatboostMlx
