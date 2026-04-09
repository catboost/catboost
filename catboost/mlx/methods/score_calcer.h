#pragma once

// Split score computation for CatBoost-MLX.
// Given histograms and partition statistics, finds the best split per leaf.

#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/gpu_data/mlx_device.h>
#include <catboost/mlx/methods/histogram.h>

namespace NCatboostMlx {

    struct TSplitCandidate {
        TBestSplitProperties Properties;
        ui32 PartitionIdx;
    };

    // Find the best split for each partition pair using histogram data.
    // Returns the best (feature, bin) split for each active leaf.
    //
    // For an oblivious tree at depth d, there are 2^d leaves.
    // We evaluate all possible splits and pick the single best one
    // (since oblivious trees use the same split at each level).
    TBestSplitProperties FindBestSplit(
        const THistogramResult& histograms,
        const TVector<TPartitionStatistics>& partitionStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    );

    // Multi-dimensional variant: sums gain across all K dimensions.
    // Each dimension has its own histogram and partition statistics.
    // For approxDim == 1, delegates to FindBestSplit.
    TBestSplitProperties FindBestSplitMultiDim(
        const TVector<THistogramResult>& perDimHistograms,
        const TVector<TVector<TPartitionStatistics>>& perDimPartStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    );

    // GPU-accelerated split scoring: two-phase suffix-sum + score reduction.
    // Keeps histogram data on GPU, eliminating the CPU readback sync point.
    TBestSplitProperties FindBestSplitGPU(
        const TVector<THistogramResult>& perDimHistograms,
        const TVector<TVector<TPartitionStatistics>>& perDimPartStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    );

    // GPU-accelerated overload accepting partition totals already on GPU.
    // partGradSums and partHessSums are [approxDim * numPartitions] float32 arrays
    // produced by ComputeLeafSumsGPU. Avoids the CPU readback round trip that the
    // TVector overload requires.
    TBestSplitProperties FindBestSplitGPU(
        const TVector<THistogramResult>& perDimHistograms,
        const mx::array& partGradSums,   // [approxDim * numPartitions]
        const mx::array& partHessSums,   // [approxDim * numPartitions]
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions,
        ui32 approxDim
    );

}  // namespace NCatboostMlx
