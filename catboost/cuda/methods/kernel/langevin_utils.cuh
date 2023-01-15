#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    void AddLangevinNoise(ui64* seeds,
                          float* values,
                          ui32 objectsCount,
                          float coefficient,
                          TCudaStream stream);
};
