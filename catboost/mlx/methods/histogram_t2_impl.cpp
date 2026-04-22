// histogram_t2_impl.cpp — Production T2 histogram dispatch (S24 D0 v5).
//
// S24 D0 v5 (DEC-023 complete fix): T2-accum now uses T1-style SIMD-shuffle
// accumulation for ALL features (0-3) reading from docIndices.  The T2-sort
// kernel is no longer called.  This produces ULP=0 vs T1 by construction.
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
    // T2 kernel registration — module-scope static, initialized once per process.
    //
    // S24 D0 v5: only GetT2AccumKernel() registered.  Kernel name t2_accum_s24d0_v5
    // invalidates any in-process cache from prior v4/v3/s23d0 registrations.
    // =========================================================================

    namespace {
        // GetT2SortKernel() removed (S24 D0 v5): T2-sort is no longer dispatched.
        // kT2SortSource is retained in kernel_sources.h for reference; the sort kernel
        // registration is removed here because the sort outputs (sortedDocs, binOffsets)
        // are no longer consumed by T2-accum.

        mx::fast::CustomKernelFunction& GetT2AccumKernel() {
            // NIT-5 applied: numTGs removed (it was never read by the kernel body).
            // Kernel name bumped to s24d0_v5 (DEC-023 complete fix): all four features
            // (0-3) now use T1-style SIMD-shuffle accumulation reading from docIndices.
            // sortedDocs and binOffsets removed from inputs (T2-accum no longer reads them);
            // partSizes added to supply totalDocsInPart directly.
            // This produces ULP=0 vs T1 for all features by construction.
            static auto k = mx::fast::metal_kernel(
                "t2_accum_s24d0_v5",
                /*input_names=*/{
                    "compressedIndex", "stats",
                    "docIndices", "partSizes",
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
    // DispatchHistogramT2 — production T2 histogram dispatch (S24 D0 v5).
    //
    // S24 D0 v5: T2-sort is no longer called from this function.  All four
    // features (0-3) use T1-style SIMD-shuffle accumulation in T2-accum, reading
    // directly from docIndices.  The sort step served only to populate sortedDocs
    // for the feature-0 bin-range scan; with that scan removed, the sort is dead.
    //
    // GetT2SortKernel() is retained in the anonymous namespace for reference and
    // potential future use; it is not called here.
    //
    // Preconditions (callers must enforce before calling):
    //   - maxBlocksPerPart == 1  (NIT-4: T2 kernels dispatch one block per partition;
    //     any value > 1 silently wastes TGs).  Enforced by CB_ENSURE in histogram.cpp
    //     when called via ComputeHistogramsImpl; bench enforces via its own guard.
    //   - maxFoldCount <= 127  (DEC-016 T1 envelope).  Caller responsibility.
    //
    // Grid geometry: identical to DispatchHistogramBatched (T1):
    //   (256 * maxBlocksPerPart * numGroups, numPartitions, numStats)
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
        auto flatCompressed = mx::reshape(compressedData, {-1});
        auto lineSizeArr    = mx::array(static_cast<uint32_t>(lineSize),         mx::uint32);
        auto maxBlocksArr   = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
        auto numGroupsArr   = mx::array(static_cast<uint32_t>(numGroups),        mx::uint32);
        auto numPartsArr    = mx::array(static_cast<uint32_t>(numPartitions),    mx::uint32);
        auto numStatsArr    = mx::array(static_cast<uint32_t>(numStats),         mx::uint32);
        auto totalBinsArr   = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
        auto totalDocsArr   = mx::array(static_cast<uint32_t>(totalNumDocs),     mx::uint32);

        // Grid: same geometry as DispatchHistogramBatched (T1).
        // T2 maxBlocksPerPart is always 1, so X = 256 * numGroups.
        auto grid = std::make_tuple(
            static_cast<int>(256 * maxBlocksPerPart * numGroups),
            static_cast<int>(numPartitions),
            static_cast<int>(numStats)
        );
        auto threadgroup = std::make_tuple(256, 1, 1);

        // --- T2-accum dispatch ---
        // S24 D0 v5: T2-sort removed; all features use T1-style SIMD accumulation.
        // partSizes supplies totalDocsInPart per partition (previously read from
        // binOffsets sentinel).  Input order matches GetT2AccumKernel() input_names.
        auto accumOut = GetT2AccumKernel()(
            /*inputs=*/{
                flatCompressed, stats,
                docIndices, partSizes,
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
