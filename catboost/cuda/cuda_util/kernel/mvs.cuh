#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

void CalculateMvsThreshold(const float takenFraction, float* candidates, ui32 size, float* threshold, TCudaStream stream);

void MvsBootstrapRadixSort(
    const float takenFraction,
    const float lambda,
    float* weights,
    const float* ders,
    ui32 size,
    const ui64* seeds,
    ui32 seedSize,
    TCudaStream stream
);

} //namespace NKernel
