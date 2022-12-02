#pragma once

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/gpu_data/splitter.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/partitions.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/reorder_bins.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/methods/pointwise_kernels.h>
#include <catboost/cuda/methods/pairwise_kernels.h>
#include <catboost/cuda/targets/non_diag_target_der.h>

namespace NCatboostCuda {
    class TPairwiseOptimizationSubsets {
    public:
        TPairwiseOptimizationSubsets(TNonDiagQuerywiseTargetDers&& target, // size can exceed 32-bits
                                     ui32 maxDepth)
            : PairwiseTarget(std::move(target)) // size can exceed 32-bits
        {
            const ui32 maxPointwisePartCount = (1ULL << maxDepth);
            const ui32 maxPairwisePartCount = maxPointwisePartCount * maxPointwisePartCount;

            PairParts.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPairwisePartCount));
            PairPartStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPairwisePartCount));

            PointPartitions.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPointwisePartCount));
            PointPartitionStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(maxPointwisePartCount));
            SetZeroLevel();
        }

        ui32 GetCurrentDepth() const {
            return CurrentDepth;
        }

        const TNonDiagQuerywiseTargetDers& GetPairwiseTarget() const {
            return PairwiseTarget;
        }

        const TStripeBuffer<TDataPartition>& GetPairPartitions() const {
            return PairParts;
        }

        const TStripeBuffer<TPartitionStatistics>& GetPairPartStats() const {
            return PairPartStats;
        }

        const TStripeBuffer<ui32>& GetPairBins() const {
            return PairBins;
        }

        const TStripeBuffer<ui32>& GetPointBins() const {
            return PointBins;
        }

        const TStripeBuffer<TDataPartition>& GetPointPartitions() const {
            return PointPartitions;
        }

        const TStripeBuffer<TPartitionStatistics>& GetPointPartitionStats() const {
            return PointPartitionStats;
        }

        void Split(const TStripeBuffer<ui32>& cindex,
                   const NCudaLib::TDistributedObject<TCFeature>& feature,
                   ui32 bin);

    private:
        void SetZeroLevel() {
            PairBins.Reset(PairwiseTarget.Pairs.GetMapping());
            PointBins.Reset(PairwiseTarget.Docs.GetMapping());
            FillBuffer(PairBins, 0u);
            FillBuffer(PointBins, 0u);
            CurrentDepth = 0;
            RebuildStats();
        }

        void RebuildStats();

    private:
        TNonDiagQuerywiseTargetDers PairwiseTarget; // size can exceed 32-bits

        TStripeBuffer<TDataPartition> PairParts;
        TStripeBuffer<TPartitionStatistics> PairPartStats;

        TStripeBuffer<ui32> PairBins; // size can exceed 32-bits
        TStripeBuffer<ui32> PointBins;

        TStripeBuffer<TDataPartition> PointPartitions;
        TStripeBuffer<TPartitionStatistics> PointPartitionStats;

        ui32 CurrentDepth = 0;
    };

}
