#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/private/libs/options/enums.h>

namespace NKernel {


    void ComputeOptimalSplits(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                              const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                              const float* histograms,
                              const double* partStats, int statCount,
                              const ui32* partIds, int partBlockSize, int partBlockCount,
                              const ui32* restPartIds, int restPartCount,
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

    void ComputeOptimalSplitsRegion(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                                    const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                                    const float* histograms,
                                    const double* partStats, int statCount,
                                    const ui32* partIds, int partCount,
                                    TBestSplitProperties* result, ui32 argmaxBlockCount,
                                    EScoreFunction scoreFunction,
                                    bool multiclassOptimization,
                                    double l2,
                                    bool normalize,
                                    double scoreStdDev,
                                    ui64 seed,
                                    TCudaStream stream);

    void ComputeOptimalSplit(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                             const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                             const float* histograms,
                             const double* partStats, int statCount,
                             const ui32 firstPartId, const ui32 maybeSecondPartId,
                             TBestSplitProperties* result, ui32 argmaxBlockCount,
                             EScoreFunction scoreFunction,
                             bool multiclassOptimization,
                             double l2,
                             bool normalize,
                             double scoreStdDev,
                             ui64 seed,
                             TCudaStream stream);


    void ComputeTreeScore(const double* partStats, int statCount,
                          const ui32* allPartIds, int allPartCount,
                          EScoreFunction scoreFunction,
                          bool multiclassOptimization,
                          double l2,
                          bool normalize,
                          double scoreStdDev,
                          ui64 seed,
                          double* result,
                          TCudaStream stream);


};
