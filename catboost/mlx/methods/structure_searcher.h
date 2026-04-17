#pragma once

// Tree structure searcher for CatBoost-MLX.
// Builds an oblivious (symmetric) tree by greedily selecting the best split at each depth level.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/methods/histogram.h>
#include <catboost/mlx/methods/stage_profiler.h>
#include <catboost/mlx/methods/tree_applier.h>

#include <unordered_map>

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
        ui32 approxDimension = 1,
        TStageProfiler* profiler = nullptr
    );

    // Result of a depthwise (non-symmetric) tree structure search.
    // NodeSplits[i] is the split for internal node i in BFS order:
    //   depth 0: node 0 (root)
    //   depth 1: nodes 1, 2
    //   depth d: nodes [2^d-1 .. 2^(d+1)-2]
    // NumNodes = 2^depth - 1  (total internal nodes for a full tree of depth `depth`).
    // NumLeaves = 2^depth.
    struct TDepthwiseTreeStructure {
        TVector<TObliviousSplitLevel> NodeSplits;  // [numNodes] — one split per internal node (BFS order)
        ui32 Depth = 0;
    };

    // Search for the best depthwise tree structure.
    //
    // At each depth level, picks the best split **per leaf** (per partition).
    // Each partition gets its own (feature, bin) split — different from the
    // oblivious case where all partitions at a depth level share one split.
    //
    // The resulting tree is non-symmetric: nodes at the same depth can have
    // different split rules.  This matches XGBoost's `grow_policy=depthwise`.
    //
    // Returns the tree structure with NodeSplits in BFS order.
    TDepthwiseTreeStructure SearchDepthwiseTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda,
        ui32 approxDimension = 1,
        TStageProfiler* profiler = nullptr
    );

    // Result of a lossguide (best-first leaf-wise) tree structure search.
    //
    // Lossguide grows the tree one leaf at a time, always splitting the leaf with
    // the highest loss reduction gain.  The resulting tree is unbalanced — leaves
    // at different depths.  Complexity is controlled by maxLeaves (not maxDepth).
    //
    // Internal representation:
    //   NodeSplitMap   — sparse map: BFS node index → split descriptor.
    //                    Contains only internal (split) nodes; leaf nodes are absent.
    //                    Using a hash map avoids O(2^depth) allocation for unbalanced trees.
    //   NumLeaves      — number of terminal leaves (= number of splits + 1).
    //   LeafBfsIds     — BFS node index for each dense leaf (size = NumLeaves).
    //                    Dense leaf index k corresponds to BFS node LeafBfsIds[k].
    //   LeafDocIds     — per-document dense leaf assignment [numDocs] uint32.
    //                    Updated incrementally during search; consumed by leaf estimator.
    struct TLossguideTreeStructure {
        // Sparse map from BFS node index → split descriptor (only internal nodes).
        // Using an unordered_map avoids allocating O(2^depth) entries for unbalanced trees.
        std::unordered_map<ui32, TObliviousSplitLevel> NodeSplitMap;  // bfsIdx → split
        TVector<ui32>                 LeafBfsIds;    // [NumLeaves] BFS indices of leaf nodes
        ui32                          NumLeaves = 1; // always >= 1 (root starts as one leaf)
        mx::array                     LeafDocIds;    // [numDocs] uint32 — dense leaf per doc
    };

    // Search for the best lossguide tree structure (best-first / leaf-wise).
    //
    // Algorithm:
    //   1. Start with the root as the single leaf (leaf 0).
    //   2. Maintain a priority queue of (gain, leafId) — pops the highest-gain leaf.
    //   3. Compute histograms for the chosen leaf, find its best split.
    //   4. Create two children; update LeafDocIds for docs in this leaf.
    //   5. Repeat until maxLeaves reached or no valid split remains.
    //
    // maxLeaves controls tree complexity (analogous to LightGBM's num_leaves).
    // maxDepth is an optional secondary limit (0 = no depth limit).
    //
    // Returns the tree structure.  Leaf values are computed by the caller using
    // LeafDocIds as the partition array.
    TLossguideTreeStructure SearchLossguideTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxLeaves,
        float l2RegLambda,
        ui32 approxDimension = 1,
        ui32 maxDepth = 0,
        TStageProfiler* profiler = nullptr
    );

}  // namespace NCatboostMlx
