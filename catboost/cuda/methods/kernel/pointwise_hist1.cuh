#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>


namespace NKernel {


    void ComputeHist1Binary(const TCFeature* bFeatures, ui32 bCount,
                            const ui32* cindex,
                            const float* target,
                            const ui32* indices,
                            ui32 size,
                            const TDataPartition* partition,
                            ui32 partsCount,
                            ui32 foldCount,
                            bool fullPass,
                            ui32 histLineSize,
                            float* binSums,
                            TCudaStream stream);


    void ComputeHist1HalfByte(const TCFeature* halfByteFeatures, ui32 halfByteFeaturesCount,
                              const ui32* cindex,
                              const float* target,
                              const ui32* indices,
                              ui32 size,
                              const TDataPartition* partition,
                              ui32 partsCount,
                              ui32 foldCount,
                              bool fullPass,
                              ui32 histLineSize,
                              float* binSums,
                              TCudaStream stream);

    void ComputeHist1NonBinary(const TCFeature* nbFeatures, ui32 nbCount,
                               const ui32* cindex,
                               const float* target,
                               const ui32* indices,
                               ui32 size,
                               const TDataPartition* partition,
                               ui32 partCount,
                               ui32 foldCount,
                               bool fullPass,
                               ui32 histLineSize,
                               float* binSums,
                               TCudaStream stream);
}
