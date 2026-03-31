#pragma once

// Tree application for CatBoost-MLX.
// Evaluates an oblivious tree on all documents and updates the prediction cursor.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>

namespace NCatboostMlx {

    // Description of one level of an oblivious tree split.
    struct TObliviousSplitLevel {
        ui32 FeatureColumnIdx;  // which ui32 column in compressed index
        ui32 Shift;             // bit shift to extract feature value
        ui32 Mask;              // bit mask after shift
        ui32 BinThreshold;      // threshold: value > threshold goes right
    };

    // Apply an oblivious tree to the dataset:
    //   1. For each document, compute leaf index from split conditions
    //   2. Add leafValues[leafIdx] to cursor[doc]
    //   3. Update partition (leaf) assignments
    void ApplyObliviousTree(
        TMLXDataSet& dataset,
        const TVector<TObliviousSplitLevel>& splits,  // [depth] — one split per level
        const mx::array& leafValues                    // [2^depth]
    );

}  // namespace NCatboostMlx
