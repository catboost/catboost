#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>



//hist2 is sums for 2 elements.
namespace NKernel {

    void BindPointwiseTextureData(const float* targets, const float* weights, ui32 size);

    void UpdateFoldBins(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                        ui32 loadBit, ui32 foldBits, TCudaStream stream);

    void ComputeHist2Binary(const TCFeature* bFeatures,
                            int bCount,
                            const ui32* cindex,
                            const float* target, const float* weight, const ui32* indices,ui32 size,
                            const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                            float* binSums, bool fullPass,
                            TCudaStream stream);

    void ComputeHist2NonBinary(const TCFeature* nbFeatures, int nbCount,
                               const ui32* cindex,
                               const float* target, const float* weight, const ui32* indices, ui32 size,
                               const TDataPartition* partition, ui32 partCount, ui32 foldCount,
                               float* binSums, const int binFeatureCount,
                               bool fullPass,
                               TCudaStream stream);

    void ComputeHist2HalfByte(const TCFeature* halfByteFeatures, int halfByteFeaturesCount,
                              const ui32* cindex,
                              const float* target, const float* weight,  const ui32* indices, ui32 size,
                              const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                              float* binSums, const int binFeatureCount,
                              bool fullPass,
                              TCudaStream stream);

};
