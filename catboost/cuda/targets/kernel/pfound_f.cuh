#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TPFoundFContext : public IKernelContext {
        TDevicePointer<ui32> QidCursor;
    };
    void ComputeMatrixSizes(const ui32* queryOffsets,
                            ui32 qCount,
                            ui32* matrixSize,
                            TCudaStream stream);


    void MakePairs(const ui32* qOffsets,
                   const ui64* matrixOffsets,
                   ui32 qCount,
                   uint2* pairs,
                   TCudaStream stream);

    void PFoundFGradient(ui64 seed,
                         float decaySpeed,
                         ui32 bootstrapIter,
                         const ui32* queryOffsets,
                         ui32* qidCursor,
                         ui32 qCount,
                         const ui32* qids,
                         const ui64* matrixOffsets,
                         const float* expApprox,
                         const float* relev,
                         ui32 size,
                         float* weightMatrixDst, //should contain zeroes
                         TCudaStream stream);

    void MakeFinalTarget(const ui32* docIds,
                         const float* expApprox,
                         const float* querywiseWeights,
                         const float* relevs,
                         float* nzPairWeights,
                         ui32 nzPairCount,
                         float* resultDers,
                         uint2* nzPairs,
                         TCudaStream stream);
    void SwapWrongOrderPairs(const float* relevs,
                             ui32 nzPairCount,
                             uint2* nzPairs,
                             TCudaStream stream);
}
