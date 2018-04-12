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
                         int bestIndexBias, TBestSplitPropertiesWithIndex* best,
                         TCudaStream stream);

}
