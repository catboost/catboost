#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/cuda_util/segmented_scan.h>
#include <catboost/cuda/cuda_util/segmented_sort.h>
#include <catboost/cuda/methods/exact_estimation.h>
#include <catboost/cuda/targets/target_func.h>

namespace NCatboostCuda {
    void ReorderPairs(TStripeBuffer<ui32>* pairBins,
                      ui32 binCount,
                      TStripeBuffer<uint2>* pairs,
                      TStripeBuffer<float>* pairWeights);

    template <typename TMapping>
    TCudaBuffer<ui32, TMapping> ComputeBinOffsets(const TCudaBuffer<ui32, TMapping>& sortedBins, ui32 binCount) {
        TCudaBuffer<ui32, TMapping> offsets;
        auto offsetsMapping = sortedBins.GetMapping().RepeatOnAllDevices(binCount + 1);
        offsets.Reset(offsetsMapping);
        UpdatePartitionOffsets(sortedBins, offsets);

        return offsets;
    }

    void FilterZeroLeafBins(const TStripeBuffer<const ui32>& bins,
                            TStripeBuffer<uint2>* pairs,
                            TStripeBuffer<float>* pairWeights);

    TVector<double> ComputeBinStatisticsForParts(const TStripeBuffer<float>& stat,
                                                 const TStripeBuffer<ui32>& partOffsets,
                                                 ui32 partCount);

    template <typename TMapping>
    void ComputeByLeafOrder(const TCudaBuffer<const ui32, TMapping>& bins, ui32 binCount,
                            TCudaBuffer<ui32, TMapping>* binOffsets,
                            TCudaBuffer<ui32, TMapping>* indices) {
        auto orderedBins = TCudaBuffer<ui32, TMapping>::CopyMapping(bins);
        indices->Reset(bins.GetMapping());

        orderedBins.Copy(bins);
        MakeSequence(*indices);

        const ui32 depth = NCB::IntLog2(binCount);
        RadixSort(orderedBins, *indices, false, 0, depth);
        (*binOffsets) = ComputeBinOffsets(orderedBins, binCount);
    }

    void MakeSupportPairsMatrix(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                TStripeBuffer<uint2>* pairs,
                                TStripeBuffer<float>* pairWeights,
                                TStripeBuffer<ui32>* pairPartOffsets,
                                TVector<double>* partLeafWeights);

    void MakePointwiseComputeOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                                   const TStripeBuffer<const float>& weights,
                                   TStripeBuffer<ui32>* orderByPart,
                                   TStripeBuffer<ui32>* partOffsets,
                                   TVector<double>* pointLeafWeights);

    // This method works with the assumption that the weights are positive
    template <typename TMapping>
    void ComputeWeightedQuantile(const TCudaBuffer<ui32, TMapping>& bins,
                                 const TCudaBuffer<float, TMapping>& targets,
                                 const TCudaBuffer<float, TMapping>& weights,
                                 ui32 binCount,
                                 TVector<float>& point,
                                 const NCatboostOptions::TLossDescription& lossDescription,
                                 ui32 binarySearchIterations) {
        const auto &params = lossDescription.GetLossParamsMap();
        auto it = params.find("alpha");
        float alpha = it == params.end() ? 0.5 : FromString<float>(it->second);

        auto singleDevMapping = NCudaLib::TSingleMapping(0, targets.GetObjectsSlice().Size());

        //
        const bool shouldCompress = false;
        auto singleDevBins = TSingleBuffer<const ui32>::Create(singleDevMapping);
        Reshard(bins, singleDevBins, 0u, shouldCompress);

        auto singleDevTargets = TSingleBuffer<float>::Create(singleDevMapping);
        Reshard(targets, singleDevTargets, 0u, shouldCompress);

        auto singleDevWeights = TSingleBuffer<float>::Create(singleDevMapping);
        Reshard(weights, singleDevWeights, 0u, shouldCompress);
        //

        //
        auto indices = TSingleBuffer<ui32>::CopyMapping(singleDevBins);
        TSingleBuffer<ui32> binsOffsets;
        ComputeByLeafOrder(singleDevBins,
                           binCount,
                           &binsOffsets,
                           &indices);

        auto orderedTargets = TSingleBuffer<float>::Create(singleDevMapping);
        Gather(orderedTargets, singleDevTargets, indices);

        auto orderedWeights = TSingleBuffer<float>::Create(singleDevMapping);
        Gather(orderedWeights, singleDevWeights, indices);
        //

        //
        auto tmpTargets = TSingleBuffer<float>::CopyMapping(orderedTargets);
        auto tmpWeights = TSingleBuffer<float>::CopyMapping(orderedWeights);
        FillBuffer(tmpTargets, 0.0f);
        FillBuffer(tmpWeights, 0.0f);
        SegmentedRadixSort(orderedTargets, orderedWeights, tmpTargets,
                           tmpWeights, binsOffsets, binCount, 10, 32);
        //

        //
        auto endOfBinsFlags = TSingleBuffer<ui32>::Create(singleDevMapping);
        FillBuffer(endOfBinsFlags, ui32(0));
        MakeEndOfBinsFlags(binsOffsets, &endOfBinsFlags, binCount);

        auto weightsPrefixSum = TSingleBuffer<float>::Create(singleDevMapping);
        SegmentedScanVector(orderedWeights, endOfBinsFlags, weightsPrefixSum, true, 1);
        //

        //
        auto pointMapping = NCudaLib::TSingleMapping(0, binCount);
        auto needWeights = TSingleBuffer<float>::Create(pointMapping);
        FillBuffer(needWeights, 0.0f);

        ComputeNeedWeights(orderedTargets,
                           orderedWeights,
                           binsOffsets,
                           &needWeights,
                           alpha);
        //

        auto result = TSingleBuffer<float>::Create(pointMapping);
        CalculateQuantileWithBinarySearch(orderedTargets,
                                          weightsPrefixSum,
                                          needWeights,
                                          binsOffsets,
                                          binCount,
                                          &result,
                                          alpha,
                                          binarySearchIterations);

        result.Read(point);
    }

    template <typename TMapping>
    void ComputeExactApprox(const TCudaBuffer<ui32, TMapping>& bins,
                            const TCudaBuffer<float, TMapping>& targets,
                            const TCudaBuffer<float, TMapping>& weights,
                            ui32 binCount,
                            TVector<float>& point,
                            const NCatboostOptions::TLossDescription& lossDescription,
                            ui32 binarySearchIterations = 16) {
        switch (lossDescription.GetLossFunction()) {
            case ELossFunction::Quantile:
            case ELossFunction::MAE: {
                ComputeWeightedQuantile(bins,
                                        targets,
                                        weights,
                                        binCount,
                                        point,
                                        lossDescription,
                                        binarySearchIterations);
                return;
            }
            case ELossFunction::MAPE: {
                auto weightsWithTargets = TCudaBuffer<float, TMapping>::CopyMapping(weights);
                ComputeWeightsWithTargets(targets, weights, &weightsWithTargets);

                ComputeWeightedQuantile(bins,
                                        targets,
                                        weightsWithTargets,
                                        binCount,
                                        point,
                                        lossDescription,
                                        binarySearchIterations);
                return;
            }
            default: {
                CB_ENSURE(false, "Only MAPE, MAE and Quantile are supported for Exact leaves estimation on GPU");
            }
        }
    }

}
