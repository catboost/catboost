#include "pairwise_optimization_subsets.h"

void NCatboostCuda::TPairwiseOptimizationSubsets::Split(const TStripeBuffer<ui32>& cindex,
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
            tmp.Copy(PairwiseTarget.PointWeightedDer);
            Gather(PairwiseTarget.PointWeightedDer, tmp, tempIndices);

            if (PairwiseTarget.PointDer2OrWeights.GetObjectsSlice().Size()) {
                tmp.Copy(PairwiseTarget.PointDer2OrWeights);
                Gather(PairwiseTarget.PointDer2OrWeights, tmp, tempIndices);
            }
        }
    }
    //pairwise split
    {
        auto tempIndices = TStripeBuffer<ui64>::CopyMapping(PairwiseTarget.Pairs);
        MakeSequence(tempIndices);
        UpdatePairwiseBins(cindex, feature, bin, CurrentDepth, PairwiseTarget.Pairs, PairBins);
        ReorderBins(PairBins, tempIndices, 2 * CurrentDepth, 2);
        {
            auto tempPairs = TStripeBuffer<uint2>::CopyMapping(PairwiseTarget.Pairs);
            tempPairs.Copy(PairwiseTarget.Pairs);
            Gather(PairwiseTarget.Pairs, tempPairs, tempIndices);
        }
        {
            auto tempWeights = TStripeBuffer<float>::CopyMapping(PairwiseTarget.PairDer2OrWeights);
            tempWeights.Copy(PairwiseTarget.PairDer2OrWeights);
            Gather(PairwiseTarget.PairDer2OrWeights, tempWeights, tempIndices);
        }
    }

    ++CurrentDepth;
    RebuildStats();
}

void NCatboostCuda::TPairwiseOptimizationSubsets::RebuildStats() {
    const ui32 numLeaves = 1ULL << CurrentDepth;
    const ui32 pairLeavesCount = numLeaves * numLeaves;

    PairParts.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(pairLeavesCount));
    PairPartStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(pairLeavesCount));

    PointPartitions.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(numLeaves));
    PointPartitionStats.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(numLeaves));

    UpdatePartitionDimensions(PointBins,
                              PointPartitions);

    UpdatePartitionDimensions(PairBins,
                              PairParts);

    UpdatePartitionStats(PointPartitionStats,
                         PointPartitions,
                         PairwiseTarget.PointWeightedDer,
                         PairwiseTarget.PointDer2OrWeights);

    UpdatePartitionStatsWeightsOnly(PairPartStats,
                                    PairParts,
                                    PairwiseTarget.PairDer2OrWeights);
}
