#pragma once

// Tree application for CatBoost-MLX.
// Evaluates an oblivious or depthwise tree on all documents and updates the prediction cursor.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>

namespace NCatboostMlx {

    // Description of one level of an oblivious tree split (or one internal node for depthwise).
    struct TObliviousSplitLevel {
        ui32 FeatureColumnIdx;  // which ui32 column in compressed index
        ui32 Shift;             // bit shift to extract feature value
        ui32 Mask;              // bit mask after shift
        ui32 BinThreshold;      // threshold: value > threshold goes right (ordinal), value == threshold goes right (OneHot)
        bool IsOneHot = false;  // true for categorical OneHot splits (equality), false for ordinal (threshold)
    };

    // Apply an oblivious tree to the dataset:
    //   1. For each document, compute leaf index from split conditions
    //   2. Add leafValues[leafIdx] to cursor[doc]
    //   3. Update partition (leaf) assignments
    //
    // For approxDimension > 1, leafValues shape is [numLeaves, K].
    // For approxDimension == 1, leafValues shape is [numLeaves].
    void ApplyObliviousTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& splits,  // [depth] — one split per level
        const mx::array& leafValues,                    // [2^depth] or [2^depth, K]
        ui32 approxDimension = 1
    );

    // Apply a depthwise (non-symmetric) tree to the dataset.
    //
    // nodeSplits: splits for internal nodes in BFS order (size = 2^depth - 1).
    //   depth 0: node 0 (root)
    //   depth 1: nodes 1, 2
    //   ...
    // leafValues: [numLeaves] or [numLeaves, K] (numLeaves = 2^depth).
    // depth: actual depth of the tree (nodeSplits.size() must equal 2^depth - 1).
    void ApplyDepthwiseTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& nodeSplits,  // [numNodes] BFS order
        const mx::array& leafValues,                        // [2^depth] or [2^depth, K]
        ui32 depth,
        ui32 approxDimension = 1
    );

}  // namespace NCatboostMlx
