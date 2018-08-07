#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/libs/options/enums.h>

namespace NKernel {


    void ComputeOptimalSplits(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                              const float* histograms,
                              const double* partStats, int statCount,
                              ui32* partIds, int partBlockSize, int partBlockCount,
                              TBestSplitProperties* result, ui32 argmaxBlockCount,
                              EScoreFunction scoreFunction,
                              bool multiclassOptimization,
                              double l2,
                              bool normalize,
                              double scoreStdDev,
                              ui64 seed,
                              TCudaStream stream);

    void ComputeTargetVariance(const float* stats,
                               ui32 size,
                               ui32 statCount,
                               ui64 statLineSize,
                               bool isMulticlasss,
                               double* aggregatedStats,
                               TCudaStream stream);
};
