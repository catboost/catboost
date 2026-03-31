#pragma once

// Histogram computation dispatch for CatBoost-MLX.
// Orchestrates Metal kernel calls to compute per-feature, per-bin gradient/weight histograms.

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/gpu_data/gpu_structures.h>

#include <mlx/mlx.h>

namespace NCatboostMlx {
    namespace mx = mlx::core;

    // Result of histogram computation: a flat buffer of gradient (and optionally weight) sums
    // per feature-bin, per partition.
    // Layout: [numPartitions, numStats, totalBinFeatures]
    struct THistogramResult {
        mx::array Histograms;   // float32
        ui32 NumPartitions;
        ui32 NumStats;
        ui32 TotalBinFeatures;
    };

    // Compute histograms for all features across all leaf partitions.
    //
    // For each partition (leaf), for each feature, for each bin:
    //   histogram[part][stat][binFeature] = sum of stat[doc] for docs in partition where feature_bin == bin
    //
    // Where stat=0 is gradient, stat=1 is weight (for weighted GBDT).
    //
    // This dispatches the appropriate Metal kernel variant based on feature encoding:
    //   - One-byte: histogram_one_byte_features (most common)
    //   - Half-byte: histogram_half_byte_features
    //   - Binary: histogram_binary_features
    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& docIndices,         // [numDocs] uint32 — sorted doc indices by partition
        const mx::array& partitionOffsets,   // [numPartitions] uint32 — doc offset per leaf
        const mx::array& partitionSizes,     // [numPartitions] uint32 — doc count per leaf
        ui32 numPartitions,
        bool useWeights = false               // if true, compute weight histograms too (numStats=2)
    );

    // Overload accepting explicit gradient/hessian arrays (for multi-dimensional approx).
    // Allows the caller to pass a single dimension's gradient+hessian slice
    // without modifying the dataset's internal buffers.
    THistogramResult ComputeHistograms(
        const TMLXDataSet& dataset,
        const mx::array& gradients,          // [numDocs] single dimension gradient
        const mx::array& hessians,           // [numDocs] single dimension hessian
        const mx::array& docIndices,
        const mx::array& partitionOffsets,
        const mx::array& partitionSizes,
        ui32 numPartitions
    );

    // Zero-initialize a histogram buffer
    mx::array CreateZeroHistogram(ui32 numPartitions, ui32 numStats, ui32 totalBinFeatures);

}  // namespace NCatboostMlx
