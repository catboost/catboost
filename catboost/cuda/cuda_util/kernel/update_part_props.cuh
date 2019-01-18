#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>

namespace NKernel {

    struct TPartStatsContext : public IKernelContext {
      ui64 TempVarsCount = 0;
      TDevicePointer<double> PartResults;
    };

    ui32 GetTempVarsCount(ui32 statCount, ui32 partCount);

    void UpdatePartitionsPropsForSplit(const TDataPartition* parts,
                                       const ui32* leftPartIds,
                                       const ui32* rightPartIds,
                                       ui32 partCount,
                                       const float* source,
                                       ui32 statCount, ui64 statLineSize,
                                       ui32 tempVarsCount, double* partResults,
                                       double* statSums,
                                       TCudaStream stream);

    void UpdatePartitionsPropsForSingleSplit(const TDataPartition* parts,
                                       const ui32 leftPartId,
                                       const ui32 rightPartId,
                                       const float* source,
                                       ui32 statCount,
                                       ui64 statLineSize,
                                       ui32 tempVarsCount,
                                       double* partResults,
                                       double* statSums,
                                       TCudaStream stream);


    void UpdatePartitionsProps(const TDataPartition* parts,
                               const ui32* partIds, ui32 partCount,
                               const float* source,
                               ui32 statCount, ui64 statLineSize,
                               ui32 tempVarsCount, double* partResults,
                               double* statSums,
                               TCudaStream stream);

    void UpdatePartitionsPropsForOffsets(const ui32* offsets, ui32 count,
                                         const float* source,
                                         ui32 statCount,
                                         ui64 statLineSize,
                                         ui32 tempVarsCount, double* partResults,
                                         double* statSums,
                                         TCudaStream stream
    );

    void CopyFloatToDouble(const float* src, ui32 size, double* dst, TCudaStream stream);
}
