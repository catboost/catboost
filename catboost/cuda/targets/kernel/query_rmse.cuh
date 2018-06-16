#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TQueryRmseContext: public IKernelContext {
       TDevicePointer<float> QueryMeans;
       TDevicePointer<float> MseDer;
       TDevicePointer<ui32> Qids;
    };



    void ApproximateQueryRmse(const float* diffs, const float* weights,
                              const ui32* qids, ui32 size,
                              const float* queryMeans,
                              const ui32* writeMap,
                              float* functionValue,
                              float* der,
                              float* der2,
                              TCudaStream stream);

}
