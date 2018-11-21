#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

#include <util/system/types.h>

namespace NKernel {
    template <typename I, typename T>
    void MakeDcgDecay(const I* offsets, T* decay, ui64 size, TCudaStream stream);

    template <typename I, typename T>
    void MakeDcgExponentialDecay(const I* offsets, T* decay, ui64 size, T base, TCudaStream stream);
}
