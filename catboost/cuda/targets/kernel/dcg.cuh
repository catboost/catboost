#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

#include <util/system/types.h>

namespace NKernel {
    template <typename I, typename T>
    void MakeDcgDecay(const I* offsets, T* decay, ui64 size, TCudaStream stream);

    template <typename I, typename T>
    void MakeDcgExponentialDecay(const I* offsets, T* decay, ui64 size, T base, TCudaStream stream);

    void FuseUi32AndFloatIntoUi64(const ui32* ui32s, const float* floats, ui64 size, ui64* fused, bool negateFloats,  TCudaStream stream);
    void FuseUi32AndTwoFloatsIntoUi64(const ui32* ui32s, const float* floats1, const float* floats2, ui64 size, ui64* fused, bool negateFloats1, bool negateFloats2, TCudaStream stream);

    template <typename T, typename U>
    void GetBits(const T* src, U* dst, ui64 size, ui32 bitsOffset, ui32 bitsCount, TCudaStream stream);
}
