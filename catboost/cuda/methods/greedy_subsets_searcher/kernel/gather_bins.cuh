#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>


namespace NKernel {


    void GatherCompressedIndex(const TFeatureInBlock* feature,
                               int fCount,
                               int featuresPerBlock,
                               const TDataPartition* parts,
                               const ui32* partIds,
                               const int partCount,
                               const ui32* indices,
                               const ui32* cindex,
                               ui32 gatheredIndexLineSize,
                               ui32* gatheredIndex,
                               TCudaStream stream);

    void GatherCompressedIndex(const TFeatureInBlock* feature,
                               int fCount,
                               int featuresPerBlock,
                               const TDataPartition* parts,
                               const ui32 partId,
                               const ui32* indices,
                               const ui32* cindex,
                               ui32 gatheredIndexLineSize,
                               ui32* gatheredIndex,
                               TCudaStream stream);
}
