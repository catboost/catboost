#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TYetiRankContext: public IKernelContext {
        TDevicePointer<float> QueryMeans;
        TDevicePointer<float> Approxes;
        TDevicePointer<float> TempDers;
        TDevicePointer<float> TempWeights;
        TDevicePointer<ui32> Qids;
        TDevicePointer<ui32> LastProceededQid;
    };

    void RemoveQueryMeans(const int* qids, int size, const float* queryMeans,
                          float* approx, TCudaStream stream);

    void YetiRankGradient(ui64 seed,
                          float decaySpeed,
                          ui32 bootstrapIter,
                          const ui32* queryOffsets,
                          int* qidCursor,
                          ui32 qOffsetsBias, ui32 qCount,
                          const int* qids,
                          const float* approx,
                          const float* relev,
                          const float* querywiseWeights,
                          ui32 size,
                          float* targetDst,
                          float* weightDst,
                          TCudaStream stream);
}
