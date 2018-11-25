#include "dcg.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

#include <util/generic/va_args.h>
#include <util/system/types.h>

#include <cuda_fp16.h>

namespace NKernel {

// MakeDcgDecay

template <typename I, typename T>
__global__ void MakeDcgDecayImpl(const I* const offsets, T* const decay, const ui64 size) {
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        decay[i] = static_cast<T>(1) / log2(static_cast<T>(i - offsets[i] + 2));
        i += gridDim.x * blockDim.x;
    }
}

template <typename I, typename T>
void MakeDcgDecay(const I* offsets, T* decay, ui64 size, TCudaStream stream) {
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    MakeDcgDecayImpl<<<numBlocks, blockSize, 0, stream>>>(offsets, decay, size);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T) \
    template void MakeDcgDecay<I, T>(const I* offsets, T* decay, ui64 size, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeDcgExponentialDecay

template <typename I, typename T>
__global__ void MakeDcgExponentialDecayImpl(
    const I* const offsets, T* const decay, const ui64 size,
    const T base)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        decay[i] = pow(base, i - offsets[i]);
        i += gridDim.x * blockDim.x;
    }
}

template <typename I, typename T>
void MakeDcgExponentialDecay(const I* offsets, T* decay, ui64 size, T base, TCudaStream stream) {
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    MakeDcgExponentialDecayImpl<<<numBlocks, blockSize, 0, stream>>>(offsets, decay, size, base);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T) \
    template void MakeDcgExponentialDecay<I, T>(const I* offsets, T* decay, ui64 size, T base, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// FuseUi32AndFloatIntoUi64

__global__ void FuseUi32AndFloatIntoUi64Impl(
    const ui32* ui32s,
    const float* floats,
    const ui64 size,
    ui64* fused,
    bool negateFloats)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        const ui32 casted = *reinterpret_cast<const ui32*>(&floats[i]);
        const ui32 mask = -i32(casted >> 31) | (i32(1) << 31);
        ui64 value = static_cast<ui64>(ui32s[i]) << 32;
        value |= negateFloats ? static_cast<ui32>(~(casted ^ mask)) : static_cast<ui32>(casted ^ mask);
        fused[i] = value;
        i += gridDim.x * blockDim.x;
    }
}

void FuseUi32AndFloatIntoUi64(
    const ui32* ui32s,
    const float* floats,
    ui64 size,
    ui64* fused,
    bool negateFloats,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    FuseUi32AndFloatIntoUi64Impl<<<numBlocks, blockSize, 0, stream>>>(ui32s, floats, size, fused, negateFloats);
}

// GetBits

template <typename T, typename U>
__global__ void GetBitsImpl(
    const T* src,
    U* dst,
    const ui64 size,
    const ui32 bitsOffset,
    const ui32 bitsCount)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dst[i] = (src[i] << (sizeof(T) - bitsOffset + bitsCount)) >> (sizeof(T) + bitsCount);
        i += gridDim.x * blockDim.x;
    }
}

template <typename T, typename U>
void GetBits(const T* src, U* dst, ui64 size, ui32 bitsOffset, ui32 bitsCount, TCudaStream stream) {
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    GetBitsImpl<<<numBlocks, blockSize, 0, stream>>>(src, dst, size, bitsOffset, bitsCount);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, U) \
    template void GetBits<T, U>(const T* src, U* dst, ui64 size, ui32 bitsOffset, ui32 bitsCount, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui64, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// FuseUi32AndTwoFloatsIntoUi64

__global__ void FuseUi32AndTwoFloatsIntoUi64Impl(
    const ui32* ui32s,
    const float* floats1,
    const float* floats2,
    const ui64 size,
    ui64* fused,
    bool negateFloats1,
    bool negateFloats2)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        ui64 value = static_cast<ui64>(ui32s[i]) << 32;
        {
            const auto half = __float2half_rz(floats1[i]);
            const auto casted = *reinterpret_cast<const ui16*>(&half);
            const ui16 mask = -i16(casted >> 15) | (i16(1) << 15);
            value |= static_cast<ui64>(negateFloats1 ? static_cast<ui16>(~(casted ^ mask)) : static_cast<ui16>(casted ^ mask)) << 16;
        }
        {
            const auto half = __float2half_rz(floats2[i]);
            const auto casted = *reinterpret_cast<const ui16*>(&half);
            const ui16 mask = -i16(casted >> 15) | (i16(1) << 15);
            value |= static_cast<ui64>(negateFloats2 ? static_cast<ui16>(~(casted ^ mask)) : static_cast<ui16>(casted ^ mask));
        }
        fused[i] = value;
        i += gridDim.x * blockDim.x;
    }
}

void FuseUi32AndTwoFloatsIntoUi64(
    const ui32* ui32s,
    const float* floats1,
    const float* floats2,
    const ui64 size,
    ui64* fused,
    bool negateFloats1,
    bool negateFloats2,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    FuseUi32AndTwoFloatsIntoUi64Impl<<<numBlocks, blockSize, 0, stream>>>(ui32s, floats1, floats2, size, fused, negateFloats1, negateFloats2);
}

}
