#pragma once
#include <util/system/types.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>



namespace NKernel {

    void UpdatePartitionDimensions(TDataPartition* parts, ui32 partCount,
                                   const ui32* sortedBins, ui32 size, TCudaStream stream);

    void UpdatePartitionOffsets(ui32* offsets, ui32 partCount,
                                const ui32* sortedBins, ui32 size, TCudaStream stream);

    void ComputeSegmentSizes(const ui32* offsets, ui32 size,
                             float* dst, TCudaStream stream);

}
