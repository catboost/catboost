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
#include <catboost/cuda/targets/pairwise_target.h>

namespace NCatboostCuda {
    class TPairwiseOptimizationSubsets {
    public:
        TPairwiseOptimizationSubsets(TPairwiseTarget&& target,
                                     ui32 maxDepth)
            : PairwiseTarget(std::move(target))
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

        const TPairwiseTarget& GetPairwiseTarget() const {
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
                   ui32 bin) {
            auto& profiler = NCudaLib::GetProfiler();
            //pointwise split
            {
                auto guard = profiler.Profile(TStringBuilder() << "Update bins");

                auto tempIndices = TStripeBuffer<ui32>::CopyMapping(PointBins);
                MakeSequence(tempIndices);

                UpdateBinFromCompressedIndex(cindex,
                                             feature,
                                             bin,
                                             PairwiseTarget.Docs,
                                             CurrentDepth,
                                             PointBins);

                ReorderBins(PointBins,
                            tempIndices,
                            CurrentDepth,
                            1);

                {
                    auto tmp = TStripeBuffer<ui32>::CopyMapping(PointBins);
                    tmp.Copy(PairwiseTarget.Docs);
                    Gather(PairwiseTarget.Docs, tmp, tempIndices);
                }
                {
                    auto tmp = TStripeBuffer<float>::CopyMapping(PointBins);
                    tmp.Copy(PairwiseTarget.PointTarget);
                    Gather(PairwiseTarget.PointTarget, tmp, tempIndices);

                    if (PairwiseTarget.PointWeights.GetObjectsSlice().Size()) {
                        tmp.Copy(PairwiseTarget.PointWeights);
                        Gather(PairwiseTarget.PointWeights, tmp, tempIndices);
                    }
                }
            }
            //pairwise split
            {
                auto tempIndices = TStripeBuffer<ui32>::CopyMapping(PairwiseTarget.Pairs);
                MakeSequence(tempIndices);
                UpdatePairwiseBins(cindex, feature, bin, CurrentDepth, PairwiseTarget.Pairs, PairBins);
                ReorderBins(PairBins, tempIndices, 2 * CurrentDepth, 2);
                {
                    auto tempPairs = TStripeBuffer<uint2>::CopyMapping(PairwiseTarget.Pairs);
                    tempPairs.Copy(PairwiseTarget.Pairs);
                    Gather(PairwiseTarget.Pairs, tempPairs, tempIndices);
                }
                {
                    auto tempWeights = TStripeBuffer<float>::CopyMapping(PairwiseTarget.PairWeights);
                    tempWeights.Copy(PairwiseTarget.PairWeights);
                    Gather(PairwiseTarget.PairWeights, tempWeights, tempIndices);
                }
            }

            ++CurrentDepth;
            RebuildStats();
        }

    private:
        void SetZeroLevel() {
            PairBins.Reset(PairwiseTarget.Pairs.GetMapping());
            PointBins.Reset(PairwiseTarget.Docs.GetMapping());
            FillBuffer(PairBins, 0u);
            FillBuffer(PointBins, 0u);
            CurrentDepth = 0;
            RebuildStats();
        }
        void RebuildStats() {
            const ui32 numLeaves = 1ULL << CurrentDepth;
            const ui32 pairLeavesCount = numLeaves * numLeaves;
            PairParts.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(pairLeavesCount));
            PairPartStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(pairLeavesCount));

            PointPartitions.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(numLeaves));
            PointPartitionStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(numLeaves));

            UpdatePartitionDimensions(PointBins, PointPartitions);
            UpdatePartitionDimensions(PairBins, PairParts);

            UpdatePartitionStats(PointPartitionStats,
                                 PointPartitions,
                                 PairwiseTarget.PointTarget,
                                 PairwiseTarget.PointWeights);

            UpdatePartitionStatsWeightsOnly(PairPartStats,
                                            PairParts,
                                            PairwiseTarget.PairWeights);
        }
    private:
        TPairwiseTarget PairwiseTarget;

        TStripeBuffer<TDataPartition> PairParts;
        TStripeBuffer<TPartitionStatistics> PairPartStats;

        TStripeBuffer<ui32> PairBins;
        TStripeBuffer<ui32> PointBins;

        TStripeBuffer<TDataPartition> PointPartitions;
        TStripeBuffer<TPartitionStatistics> PointPartitionStats;

        ui32 CurrentDepth = 0;


    };

}
