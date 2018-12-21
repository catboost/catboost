#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

#include <util/system/types.h>

namespace NKernel {
    template <typename I, typename T>
    void MakeDcgDecays(const I* offsets, T* decays, ui64 size, TCudaStream stream);

    template <typename I, typename T>
    void MakeDcgExponentialDecays(const I* offsets, T* decays, ui64 size, T base, TCudaStream stream);

    void FuseUi32AndFloatIntoUi64(const ui32* ui32s, const float* floats, ui64 size, ui64* fused, bool negateFloats,  TCudaStream stream);
    void FuseUi32AndTwoFloatsIntoUi64(const ui32* ui32s, const float* floats1, const float* floats2, ui64 size, ui64* fused, bool negateFloats1, bool negateFloats2, TCudaStream stream);

    template <typename T>
    void MakeElementwiseOffsets(const T* sizes, const T* biasedOffsets, ui64 size, T offsetsBias, T* elementwiseOffsets, ui64 elementwiseOffsetsSize, TCudaStream stream);

    template <typename T>
    void MakeEndOfGroupMarkers(const T* sizes, const T* biasedOffsets, ui64 size, T offsetsBias, T* endOfGroupMarkers, ui64 endOfGroupMarkersSize, TCudaStream stream);

    template <typename T, typename I>
    void GatherBySizeAndOffset(const T* src, const I* sizes, const I* biasedOffsets, ui64 size, I offsetsBias,I maxSize, T* dst, TCudaStream stream);

    template <typename T, typename I>
    void RemoveGroupMean(const T* values, ui64 valuesSize, const I* sizes, const I* biasedOffsets, ui64 size, I offsetsBias, T* normalized, TCudaStream stream);
}
