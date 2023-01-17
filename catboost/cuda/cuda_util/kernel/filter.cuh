#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    void Filter(const float* weights,
                const ui32 size,
                ui32* result,
                TCudaStream stream);
}
