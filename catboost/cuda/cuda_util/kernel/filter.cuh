#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    template <typename TResult>
    void Filter(const float* weights,
                const ui64 size,
                TResult* result,
                TCudaStream stream);
}
