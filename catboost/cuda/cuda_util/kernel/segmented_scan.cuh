#pragma once

#include "scan.cuh"
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <vector>

namespace NKernel {


    template <class T>
    ui64 SegmentedScanVectorTempSize(ui32 size, bool inclusive);

    template <typename T>
    cudaError_t SegmentedScanCub(const T* input, const ui32* flags, ui32 flagMask,
                                 T* output,
                                 ui32 size, bool inclusive,
                                 TScanKernelContext<T, T>& context,
                                 TCudaStream stream);

}

