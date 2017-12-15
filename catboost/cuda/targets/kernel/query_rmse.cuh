#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TQueryRmseContext: public IKernelContext {
        float* QueryMeans;
        float* MseDer;
        ui32* Qids;
    };

    void ComputeGroupIds(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst,
                         TCudaStream stream);

    void ComputeGroupMeans(const float* target, const float* weights,
                          const ui32* qOffsets, ui32 qOffsetsBis,
                          const ui32* qSizes, ui32 qCount,
                          float* result,
                          TCudaStream stream);

    void ApproximateQueryRmse(const float* diffs, const float* weights,
                              const ui32* qids, ui32 size,
                              const float* queryMeans,
                              const ui32* writeMap,
                              float* functionValue,
                              float* der,
                              float* der2,
                              TCudaStream stream);

}
