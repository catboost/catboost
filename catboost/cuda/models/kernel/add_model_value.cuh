#pragma once


#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>

namespace NKernel {

void AddBinModelValue(const float* binValues, ui32 binCount,
                      const ui32* bins,
                      const ui32* readIndices,
                      const ui32* writeIndices,
                      float* cursor, ui32 size,
                      TCudaStream stream);
}
