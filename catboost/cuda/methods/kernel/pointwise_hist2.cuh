#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>



//hist2 is sums for 2 elements.
namespace NKernel {

    void UpdatePointwiseHistograms(float* histograms,
                                   int firstBinFeature, int binFeatureCount,
                                   int partCount,
                                   int foldCount,
                                   int histCount,
                                   int histLineSize,
                                   const TDataPartition* parts,
                                   TCudaStream stream);

    void UpdateFoldBins(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                        ui32 loadBit, ui32 foldBits, TCudaStream stream);

    void ComputeHist2Binary(const TCFeature* features, ui32 featureCount,
                            const ui32* cindex,
                            const float* target,
                            const float* weight,
                            const ui32* indices,
                            ui32 size,
                            const TDataPartition* partition,
                            ui32 partsCount,
                            ui32 foldCount,
                            bool fullPass,
                            ui32 histLineSize,
                            float* binSums,
                            TCudaStream stream);

    template <int Bits>
    void ComputeHist2NonBinary(const TCFeature* features, ui32 featureCount,
                               const ui32* cindex,
                               const float* target,
                               const float* weight,
                               const ui32* indices,
                               ui32 size,
                               const TDataPartition* partition,
                               ui32 partsCount,
                               ui32 foldCount,
                               bool fullPass,
                               ui32 histLineSize,
                               float* binSums,
                               ui32 featureCountForBits,
                               TCudaStream stream);

    void ComputeHist2HalfByte(const TCFeature* features, ui32 featureCount,
                              const ui32* cindex,
                              const float* target,
                              const float* weight,
                              const ui32* indices,
                              ui32 size,
                              const TDataPartition* partition,
                              ui32 partsCount,
                              ui32 foldCount,
                              bool fullPass,
                              ui32 histLineSize,
                              float* binSums,
                              TCudaStream stream);

    void ScanPointwiseHistograms(const TCFeature* features,
                                 int featureCount,
                                 int partCount, int foldCount,
                                 int histLineSize, bool fullPass,
                                 int histCount,
                                 float* binSums,
                                 TCudaStream stream);
};
