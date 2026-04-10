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
    // All arrays are GPU-resident (mx::array); no CPU mirrors are maintained.
    struct TPartitionLayout {
        mx::array DocIndices;   // [numDocs] uint32 — doc indices sorted by partition
        mx::array PartOffsets;  // [numPartitions] uint32 — start offset per partition
        mx::array PartSizes;    // [numPartitions] uint32 — doc count per partition
    };

    // Compute partition layout entirely on GPU — no CPU-GPU sync.
    //
    // Algorithm:
    //   DocIndices  = argsort(partitions)              — stable sort by partition ID
    //   PartSizes   = scatter_add(ones, partitions)    — count docs per partition
    //   PartOffsets = exclusive_cumsum(PartSizes)      — prefix-sum offsets
    //
    // No EvalNow is issued; arrays are returned as lazy MLX expressions
    // and are materialized when first consumed by the histogram kernel.
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
