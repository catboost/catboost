#pragma once


#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

void AddBinModelValue(const float* binValues, ui32 binCount,
                      const ui32* bins,
                      const ui32* readIndices,
                      const ui32* writeIndices,
                      float* cursor, ui32 size,
                      TCudaStream stream);


void AddObliviousTree(const TCFeature* features, const ui8* bins, const float* leaves, ui32 depth,
                      const ui32* cindex,
                      const ui32* readIndices,
                      const ui32* writeIndices,
                      float* cursor,
                      ui32 size,
                      TCudaStream stream);

void ComputeObliviousTreeBins(const TCFeature* features, const ui8* bins,
                              ui32 depth,
                              const ui32* cindex,
                              const ui32* readIndices,
                              const ui32* writeIndices,
                              ui32* cursor,
                              ui32 size,
                              TCudaStream stream);
}
