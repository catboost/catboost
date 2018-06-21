#include "leaves_estimation_helper.h"
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/methods/pairwise_kernels.h>
#include <catboost/cuda/gpu_data/non_zero_filter.h>
#include <catboost/cuda/models/add_bin_values.h>

namespace NCatboostCuda {
    void ReorderPairs(TStripeBuffer<ui32>* pairBins,
                      ui32 binCount,
                      TStripeBuffer<uint2>* pairs,
                      TStripeBuffer<float>* pairWeights) {
        auto sortedPairs = TStripeBuffer<uint2>::CopyMapping(*pairs);
        auto sortedPairWeights = TStripeBuffer<float>::CopyMapping(*pairWeights);

        auto indices = TStripeBuffer<ui32>::CopyMapping(pairs);
        MakeSequence(indices);
        const ui32 depth = IntLog2(binCount);
        RadixSort(*pairBins, indices, false, 0, depth * 2);

        Gather(sortedPairs, *pairs, indices);
        Gather(sortedPairWeights, *pairWeights, indices);

        TStripeBuffer<uint2>::Swap(sortedPairs, *pairs);
        TStripeBuffer<float>::Swap(sortedPairWeights, *pairWeights);
    }

    TStripeBuffer<ui32> ComputeBinOffsets(const TStripeBuffer<ui32>& sortedBins,
                                          ui32 binCount) {
        TStripeBuffer<ui32> offsets;
        auto offsetsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(binCount + 1);
        offsets.Reset(offsetsMapping);
        UpdatePartitionOffsets(sortedBins,
                               offsets);
        return offsets;
    }

    void FilterZeroLeafBins(const TStripeBuffer<const ui32>& bins,
                            TStripeBuffer<uint2>* pairs,
                            TStripeBuffer<float>* pairWeights) {
        ZeroSameLeafBinWeights(bins,
                               *pairs,
                               pairWeights);

        auto indices = TStripeBuffer<ui32>::CopyMapping(*pairs);
        MakeSequence(indices);

        FilterZeroEntries(pairWeights,
                          &indices);

        auto supportPairs = TStripeBuffer<uint2>::CopyMapping(indices);
        Gather(supportPairs, *pairs, indices);

        TStripeBuffer<uint2>::Swap(*pairs, supportPairs);
    }

    TVector<float> ComputeBinStatisticsForParts(const TStripeBuffer<float>& stat,
                                                const TStripeBuffer<ui32>& partOffsets,
                                                ui32 partCount) {
        auto reducedStatsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(partCount);
        auto reducedStat = TStripeBuffer<float>::Create(reducedStatsMapping);
        SegmentedReduceVector(stat, partOffsets, reducedStat, EOperatorType::Sum);
        return ReadReduce(reducedStat);
    }

    void ComputeByLeafOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                            TStripeBuffer<ui32>* binOffsets,
                            TStripeBuffer<ui32>* indices) {
        auto orderedBins = TStripeBuffer<ui32>::CopyMapping(bins);
        indices->Reset(bins.GetMapping());

        orderedBins.Copy(bins);
        MakeSequence(*indices);

        const ui32 depth = IntLog2(binCount);
        RadixSort(orderedBins, *indices, false, 0, depth);
        (*binOffsets) = ComputeBinOffsets(orderedBins, binCount);
    }

    void MakeSupportPairsMatrix(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                TStripeBuffer<uint2>* pairs,
                                TStripeBuffer<float>* pairWeights,
                                TStripeBuffer<ui32>* pairPartOffsets,
                                TVector<float>* partLeafWeights) {
        FilterZeroLeafBins(bins,
                           pairs,
                           pairWeights);

        auto pairBins = TStripeBuffer<ui32>::CopyMapping(*pairs);

        FillPairBins(bins,
                     binCount,
                     *pairs,
                     &pairBins);

        ReorderPairs(&pairBins,
                     binCount,
                     pairs,
                     pairWeights);

        (*pairPartOffsets) = ComputeBinOffsets(pairBins,
                                               binCount * binCount);

        (*partLeafWeights) = ComputeBinStatisticsForParts(*pairWeights,
                                                          *pairPartOffsets,
                                                          binCount * binCount);
    }

    void MakePointwiseComputeOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                   const TStripeBuffer<const float>& weights,
                                   TStripeBuffer<ui32>* orderByPart,
                                   TStripeBuffer<ui32>* partOffsets,
                                   TVector<float>* pointLeafWeights) {
        ComputeByLeafOrder(bins,
                           binCount,
                           partOffsets,
                           orderByPart);
        auto tmp = TStripeBuffer<float>::CopyMapping(weights);
        Gather(tmp, weights, *orderByPart);
        (*pointLeafWeights) = ComputeBinStatisticsForParts(tmp, *partOffsets, binCount);
    }

    void TEstimationTaskHelper::MoveToPoint(const TMirrorBuffer<float>& point, ui32 stream) {
        Cursor.Copy(Baseline, stream);

        AddBinModelValues(point,
                          Bins,
                          Cursor,
                          stream);
    }

}
