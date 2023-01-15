#pragma once

#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/targets/permutation_der_calcer.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>

namespace NCatboostCuda {
    void ReorderPairs(TStripeBuffer<ui32>* pairBins,
                      ui32 binCount,
                      TStripeBuffer<uint2>* pairs,
                      TStripeBuffer<float>* pairWeights);

    TStripeBuffer<ui32> ComputeBinOffsets(const TStripeBuffer<ui32>& sortedBins,
                                          ui32 binCount);

    void FilterZeroLeafBins(const TStripeBuffer<const ui32>& bins,
                            TStripeBuffer<uint2>* pairs,
                            TStripeBuffer<float>* pairWeights);

    TVector<double> ComputeBinStatisticsForParts(const TStripeBuffer<float>& stat,
                                                 const TStripeBuffer<ui32>& partOffsets,
                                                 ui32 partCount);

    void ComputeByLeafOrder(const TStripeBuffer<const ui32>& bins, ui32 binCount,
                            TStripeBuffer<ui32>* binOffsets,
                            TStripeBuffer<ui32>* indices);

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

}
