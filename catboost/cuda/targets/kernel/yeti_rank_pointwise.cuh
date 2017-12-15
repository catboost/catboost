#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TYetiRankContext: public IKernelContext {
        float* QueryMeans;
        float* Approxes;
        float* TempDers;
        float* TempWeights;
        ui32* Qids;
        ui32* LastProceededQid;
    };

    void RemoveQueryMeans(const int* qids, int size, const float* queryMeans,
                          float* approx, TCudaStream stream);

    void YetiRankGradient(ui64 seed,
                          ui32 bootstrapIter,
                          const ui32* queryOffsets,
                          int* qidCursor,
                          ui32 qOffsetsBias, ui32 qCount,
                          const int* qids,
                          const float* approx,
                          const float* relev,
                          ui32 size,
                          float* targetDst,
                          float* weightDst,
                          TCudaStream stream);
}
