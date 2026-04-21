#pragma once

// ============================================================================
// G1 Path 5 histogram dispatch — Sprint 25 DEC-026 G1 measurement scaffold
//
// Reconstructs the pre-v5 T2-sort + T2-accum path with features 1-3 swapped
// for int-atomic fixed-point accumulation (making features 1-3 deterministic
// at the bit level while keeping feature-0's bin-range scan topology — the
// Value-B producing kernel we want to characterise for G1).
//
// NOT production code.  Only called from g1_gain_dump.cpp when
// --kernel=t2_path5 is passed.  Dispatches side-by-side with the production
// v5 T2-accum path when --kernel=t1 (= production v5, which is bit-identical
// to T1 at the 18 DEC-008 configs).
//
// Kernel registrations are module-scope statics (one per process).  Names
// use the `_g1_` tag to avoid MLX kernel-cache collisions with production.
// ============================================================================

#include <cstdint>
#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <catboost/mlx/kernels/kernel_sources.h>  // kHistHeader, kScoreHeader
#include "../kernels/g1_kernels.h"                // G1Kernels::*

namespace G1Dispatch {

namespace mx = mlx::core;
using ui32 = uint32_t;

namespace {
    inline mx::fast::CustomKernelFunction& GetT2SortPath5Kernel() {
        static auto k = mx::fast::metal_kernel(
            "t2_sort_s25_g1_path5",
            /*input_names=*/{
                "compressedIndex", "docIndices", "partOffsets", "partSizes",
                "featureColumnIndices", "lineSize", "maxBlocksPerPart", "numGroups",
                "numPartitions", "numStats", "totalNumDocs"
            },
            /*output_names=*/{"sortedDocs", "binOffsets"},
            /*source=*/G1Kernels::kT2SortPath5Source,
            /*header=*/NCatboostMlx::KernelSources::kHistHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );
        return k;
    }

    inline mx::fast::CustomKernelFunction& GetT2AccumPath5Kernel() {
        static auto k = mx::fast::metal_kernel(
            "t2_accum_s25_g1_path5",
            /*input_names=*/{
                "sortedDocs", "binOffsets", "compressedIndex", "stats",
                "featureColumnIndices", "foldCountsFlat", "firstFoldIndicesFlat",
                "partOffsets",
                "lineSize", "maxBlocksPerPart", "numGroups",
                "numPartitions", "numStats",
                "totalBinFeatures", "totalNumDocs"
            },
            /*output_names=*/{"histogram"},
            /*source=*/G1Kernels::kT2AccumPath5Source,
            /*header=*/NCatboostMlx::KernelSources::kHistHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/true
        );
        return k;
    }
}  // anonymous namespace

// ============================================================================
// DispatchHistogramT2Path5 — Path 5 histogram dispatch (G1 only).
//
// Signature is identical to NCatboostMlx::DispatchHistogramT2 so g1_gain_dump
// can swap between the two via a flag.
//
// Grid geometry: same as production T1/T2 (256 × maxBlocksPerPart × numGroups,
// numPartitions, numStats) with threadgroup (256, 1, 1).
// ============================================================================
inline mx::array DispatchHistogramT2Path5(
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
    auto flatCompressed = mx::reshape(compressedData, {-1});
    auto lineSizeArr    = mx::array(static_cast<uint32_t>(lineSize),         mx::uint32);
    auto maxBlocksArr   = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
    auto numGroupsArr   = mx::array(static_cast<uint32_t>(numGroups),        mx::uint32);
    auto numPartsArr    = mx::array(static_cast<uint32_t>(numPartitions),    mx::uint32);
    auto numStatsArr    = mx::array(static_cast<uint32_t>(numStats),         mx::uint32);
    auto totalBinsArr   = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto totalDocsArr   = mx::array(static_cast<uint32_t>(totalNumDocs),     mx::uint32);
    const ui32 numTGsVal = numGroups * numPartitions * numStats;

    auto grid = std::make_tuple(
        static_cast<int>(256 * maxBlocksPerPart * numGroups),
        static_cast<int>(numPartitions),
        static_cast<int>(numStats)
    );
    auto threadgroup = std::make_tuple(256, 1, 1);

    // Output shapes for T2-sort: slab layout matching production kT2SortSource.
    //   sortedDocs: numGroups × numStats × totalNumDocs   (uint32)
    //   binOffsets: numGroups × numPartitions × numStats × 129  (uint32)
    mx::Shape sortedDocsShape = {static_cast<int>(numGroups * numStats * totalNumDocs)};
    mx::Shape binOffsetsShape = {static_cast<int>(numTGsVal * 129u)};

    // --- T2-sort dispatch (Path 5 deterministic serial scatter) ---
    auto sortOut = GetT2SortPath5Kernel()(
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

    // --- T2-accum dispatch (Path 5: feat-0 bin-range scan + int-atomic feats 1-3) ---
    auto accumOut = GetT2AccumPath5Kernel()(
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

}  // namespace G1Dispatch
