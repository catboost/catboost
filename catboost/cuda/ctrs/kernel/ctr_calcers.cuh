#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>

namespace NKernel {

    void UpdateBordersMask(const ui32* bins,  const ui32* prevBins, ui32* indices, ui32 size, TCudaStream stream);

    void ExtractBorderMasks(const ui32* indices, ui32* dst, ui32 size, bool startSegment, TCudaStream stream);

    void ComputeWeightedBinFreqCtr(const ui32* writeIdx, const ui32* bins,
                                   const float* binSums,
                                   float totalWeight, float prior, float priorObservations,
                                   float* dst,
                                   ui32 size, TCudaStream stream);

    void ComputeNonWeightedBinFreqCtr(const ui32* writeIdx, const ui32* bins,
                                      const ui32* binOffsets, ui32 size,
                                      float prior, float priorObservations,
                                      float* dst, TCudaStream stream);

    void MergeBinsKernel(ui32* bins, const ui32* prev, ui32 shift, ui32 size, TCudaStream stream);

    void FillBinarizedTargetsStats(const ui8* sample, const float* sampleWeights, ui32 size,
                                   float* sums, ui32 binIndex, bool borders,
                                   TCudaStream stream);


    void MakeMeans(float* sums, const float* weights, ui32 size,
                   float sumPrior, float weightPrior,
                   TCudaStream stream);

    void MakeMeansAndScatter(const float* sums, const float* weights, ui32 size,
                             float sumPrior, float weightPrior,
                             const ui32* map, ui32 mask,
                             float* dst,
                             TCudaStream stream);

    void GatherTrivialWeights(const ui32* indices, ui32 size,
                              ui32 firstZeroIndex, bool writeSegmentStartFloatMask,
                              float* dst, TCudaStream stream);

    void WriteMask(const ui32* indices, ui32 size,
                   float* dst,
                   TCudaStream stream);

    void ApplyGroupwiseCtrFix(ui32 size,
                              const ui32* fixIndices,
                              float* ctr,
                              TCudaStream stream);

    void CreateFixedIndices(const ui32* bins,
                            const ui32* binIndices,
                            ui32 mask, const ui32* indicesWithMask,
                            ui32 size,
                            ui32* fixedIndices,
                            TCudaStream stream);

    void FillBinIndices(ui32 mask, const ui32* indices,
                        const ui32* bins,
                        ui32 size,
                        ui32* binIndices,
                        TCudaStream stream);

    void MakeGroupStarts(ui32 mask,
                         const ui32* indices,
                         const ui32* groupIds,
                         ui32 size,
                         ui32* flags,
                         TCudaStream stream);

}
