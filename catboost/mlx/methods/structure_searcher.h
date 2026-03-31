#pragma once

// Tree structure searcher for CatBoost-MLX.
// Builds an oblivious (symmetric) tree by greedily selecting the best split at each depth level.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/methods/histogram.h>
#include <catboost/mlx/methods/tree_applier.h>

namespace NCatboostMlx {

    // Partition layout: sorted doc indices + partition offsets/sizes.
    // Used to feed the histogram kernel which expects docs grouped by partition.
    struct TPartitionLayout {
        mx::array DocIndices;       // [numDocs] uint32 — doc indices sorted by partition
        mx::array PartOffsets;      // [numPartitions] uint32 — start offset per partition
        mx::array PartSizes;        // [numPartitions] uint32 — doc count per partition
        TVector<ui32> PartSizesHost;   // CPU-side copy for scoring
        TVector<ui32> PartOffsetsHost; // CPU-side copy
    };

    // Compute partition layout from the current partition assignments.
    // Sorts doc indices by partition on CPU and returns GPU arrays for kernel dispatch.
    TPartitionLayout ComputePartitionLayout(
        const mx::array& partitions, ui32 numDocs, ui32 numPartitions);

    // Result of tree structure search: the splits that define the tree.
    struct TObliviousTreeStructure {
        TVector<TObliviousSplitLevel> Splits;  // [depth] — one split per level
        TVector<TBestSplitProperties> SplitProperties;  // metadata per split
    };

    // Search for the best oblivious tree structure using greedy level-wise splits.
    //
    // At each depth level:
    //   1. Compute histograms for all features across current partitions
    //   2. Evaluate all (feature, bin) split candidates
    //   3. Pick the single best split (same split applied to all leaves at this level)
    //   4. Update partition assignments
    //
    // For approxDimension > 1 (multi-class), runs histograms per-dimension
    // and sums gains across all dimensions to find the shared best split.
    //
    // Returns the tree structure (splits) — leaf values are computed separately.
    TObliviousTreeStructure SearchTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda,
        ui32 approxDimension = 1
    );

}  // namespace NCatboostMlx
