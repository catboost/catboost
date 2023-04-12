#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/private/libs/options/enums.h>

namespace NKernel {

    void UpdatePartitionProps(const float* target,
                              const float* weights,
                              const float* counts,
                              const struct TDataPartition* parts,
                              struct TPartitionStatistics* partStats,
                              int partsCount,
                              TCudaStream stream);


    void GatherHistogramByLeaves(const float* histogram,
                                 const ui32 binFeatureCount,
                                 const ui32 histCount,
                                 const ui32 leafCount,
                                 const ui32 foldCount,
                                 float* result,
                                 TCudaStream stream
    );

    void FindOptimalSplit(const TCBinFeature* binaryFeatures,ui32 binaryFeatureCount,
                          const float* catFeaturesWeights,
                          const float* binFeaturesWeights, ui32 binaryFeatureWeightsCount,
                          const float* splits, const TPartitionStatistics* parts, ui32 pCount, ui32 foldCount,
                          double scoreBeforeSplit,
                          TBestSplitProperties* result, ui32 resultSize,
                          EScoreFunction scoreFunction, double l2, double metaL2Exponent, double metaL2Frequency, bool normalize,
                          double scoreStdDev, ui64 seed, bool gatheredByLeaves,
                          TCudaStream stream);


};
