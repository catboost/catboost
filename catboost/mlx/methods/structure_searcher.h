#pragma once

// Tree structure searcher for CatBoost-MLX.
// Builds an oblivious (symmetric) tree by greedily selecting the best split at each depth level.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/methods/histogram.h>
#include <catboost/mlx/methods/tree_applier.h>

namespace NCatboostMlx {

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
    // Returns the tree structure (splits) — leaf values are computed separately.
    TObliviousTreeStructure SearchTreeStructure(
        TMLXDataSet& dataset,
        ui32 maxDepth,
        float l2RegLambda
    );

}  // namespace NCatboostMlx
