#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

/*
 * All routines here assume histograms are zeroed externally
 */

namespace NKernel {


    void ComputeHistBinary(const TFeatureInBlock* features,
                           const int fCount,
                           const TDataPartition* parts,
                           const ui32 partId,
                           const ui32* cindex,
                           const int* indices,
                           const float* stats,
                           ui32 numStats,
                           ui32 statLineSize,
                           float* histograms,
                           TCudaStream stream);

    void ComputeHistBinary(const TFeatureInBlock* features,
                           const int fCount,
                           const TDataPartition* parts,
                           const ui32 partId,
                           const ui32* bins,
                           ui32 binsLineSize,
                           const float* stats,
                           ui32 numStats,
                           ui32 statLineSize,
                           float* histograms,
                           TCudaStream stream);



    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* features,
                            const int fCount,
                            const TDataPartition* parts,
                            const ui32 partId,
                            const ui32* bins,
                            ui32 binsLineSize,
                            const float* stats,
                            ui32 numStats,
                            ui32 statLineSize,
                            float* histograms,
                            TCudaStream stream);


    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* groups,
                            const int fCount,
                            const TDataPartition* parts,
                            const ui32 partId,
                            const ui32* cindex,
                            const int* indices,
                            const float* stats,
                            ui32 numStats,
                            ui32 statLineSize,
                            float* histograms,
                            TCudaStream stream);


    void ComputeHistHalfByte(const TFeatureInBlock* features,
                             const int fCount,
                             const TDataPartition* parts,
                             const ui32 partId,
                             const ui32* bins,
                             ui32 binsLineSize,
                             const float* stats,
                             ui32 numStats,
                             ui32 statLineSize,
                             float* histograms,
                             TCudaStream stream);

    void ComputeHistHalfByte(const TFeatureInBlock* features,
                             const int fCount,
                             const TDataPartition* parts,
                             const ui32 partId,
                             const ui32* cindex,
                             const int* indices,
                             const float* stats,
                             ui32 numStats,
                             ui32 statLineSize,
                             float* histograms,
                             TCudaStream stream);








}
