#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

    void MakePairwiseDerivatives(const float* histogram, int leavesCount, int firstMatrix, int matricesCount, int histLineSize, float* linearSystem,
                                 TCudaStream stream);

    void MakePointwiseDerivatives(const float* pointwiseHist, int pointwiseHistLineSize,
                                  const TPartitionStatistics* partStats,
                                  bool hasPointwiseWeights,
                                  int rowSize,
                                  int firstMatrixIdx,
                                  int matricesCount,
                                  float* linearSystem,
                                  TCudaStream stream);

    void UpdateBinsPairs(TCFeature feature, ui32 bin,
                         const ui32* compressedIndex,
                         const uint2* pairs,
                         ui32 pairCount,
                         ui32 depth,
                         ui32* bins,
                         TCudaStream stream);

    void SelectBestSplit(const float* scores,
                         const TCBinFeature* binFeature, int size,
                         double scoreBeforeSplit, const float* featureWeights,
                         int bestIndexBias, TBestSplitPropertiesWithIndex* best,
                         TCudaStream stream);

    void FillPairDer2Only(const float* ders2,
                          const float* groupDers2,
                          const ui32* qids,
                          const uint2* pairs,
                          ui32 pairCount,
                          float* pairDer2,
                          TCudaStream stream);

    void FillPairBins(const uint2* pairs,
                      const ui32* bins,
                      ui32 binCount,
                      ui32 pairCount,
                      ui32* pairBins,
                      TCudaStream stream);

    void ZeroSameLeafBinWeights(const uint2* pairs,
                                const ui32* bins,
                                ui32 pairCount,
                                float* pairWeights,
                                TCudaStream stream);

}
