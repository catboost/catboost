#include "dcg.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <util/generic/va_args.h>
#include <util/system/types.h>

#include <cuda_fp16.h>

namespace NKernel {

// MakeDcgDecays

template <typename I, typename T>
__global__ void MakeDcgDecaysImpl(const I* const offsets, T* const decays, const ui64 size) {
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        decays[i] = static_cast<T>(1) / log2(static_cast<T>(i - __ldg(offsets + i) + 2));
        i += gridDim.x * blockDim.x;
    }
}

template <typename I, typename T>
void MakeDcgDecays(const I* offsets, T* decays, ui64 size, TCudaStream stream) {
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    MakeDcgDecaysImpl<<<numBlocks, blockSize, 0, stream>>>(offsets, decays, size);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T) \
    template void MakeDcgDecays<I, T>(const I* offsets, T* decays, ui64 size, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeDcgExponentialDecays

template <typename I, typename T>
__global__ void MakeDcgExponentialDecaysImpl(
    const I* const offsets, T* const decays, const ui64 size,
    const T base)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        decays[i] = pow(base, i - __ldg(offsets + i));
        i += gridDim.x * blockDim.x;
    }
}

template <typename I, typename T>
void MakeDcgExponentialDecays(const I* offsets, T* decays, ui64 size, T base, TCudaStream stream) {
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    MakeDcgExponentialDecaysImpl<<<numBlocks, blockSize, 0, stream>>>(offsets, decays, size, base);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T) \
    template void MakeDcgExponentialDecays<I, T>(const I* offsets, T* decays, ui64 size, T base, TCudaStream stream);

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
        const float original = __ldg(floats + i);
        const ui32 casted = *reinterpret_cast<const ui32*>(&original);
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
    // uses __float2half_rz (round-towards-zero) because:
    // - this should be the fastest one (it simply drops unused bits)
    // - this was the only one available float16 rounding mode on CPU (see library/cpp/float16) at the
    //   moment

    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        ui64 value = static_cast<ui64>(__ldg(ui32s + i)) << 32;
        {
            const auto half = __float2half_rz(__ldg(floats1 + i));
            const auto casted = *reinterpret_cast<const ui16*>(&half);
            const ui16 mask = -i16(casted >> 15) | (i16(1) << 15);
            value |= static_cast<ui64>(negateFloats1 ? static_cast<ui16>(~(casted ^ mask)) : static_cast<ui16>(casted ^ mask)) << 16;
        }
        {
            const auto half = __float2half_rz(__ldg(floats2 + i));
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

// MakeElementwiseOffsets

template <ui32 LogicalWarpSize, typename T>
__global__ void MakeElementwiseOffsetsImpl(
    const T* const sizes,
    const T* const offsets,
    const ui64 size,
    const T offsetsBias,
    T* const elementwiseOffsets)
{
    ui64 i = blockIdx.x * (blockDim.x / LogicalWarpSize) + (threadIdx.x / LogicalWarpSize);
    while (i < size) {
        const ui32 groupSize = __ldg(sizes + i);
        const ui32 groupOffset = __ldg(offsets + i) - offsetsBias;
        for (ui32 j = threadIdx.x & (LogicalWarpSize - 1); j < groupSize; j += LogicalWarpSize) {
            elementwiseOffsets[groupOffset + j] = groupOffset;
        }

        i += gridDim.x * (blockDim.x / LogicalWarpSize);
    }
}

static ui64 GetBlockCount(ui32 logicalWarpSize, ui32 blockSize, ui64 objectCount) {
    return Min<ui64>(
        (logicalWarpSize * objectCount + blockSize - 1) / blockSize,
        TArchProps::MaxBlockCount());
}

template <typename T>
void MakeElementwiseOffsets(
    const T* const sizes,
    const T* const biasedOffsets,
    const ui64 size,
    const T offsetsBias,
    T* const elementwiseOffsets,
    ui64 elementwiseOffsetsSize,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const auto avgElementsPerGroup = elementwiseOffsetsSize / size;

#define Y_RUN_KERNEL(LOGICAL_KERNEL_SIZE)                                        \
        const ui32 logicalWarpSize = LOGICAL_KERNEL_SIZE;                        \
        const auto blockCount = GetBlockCount(logicalWarpSize, blockSize, size); \
        MakeElementwiseOffsetsImpl<logicalWarpSize><<<blockCount, blockSize, 0, stream>>>(sizes, biasedOffsets, size, offsetsBias, elementwiseOffsets);

    if (avgElementsPerGroup <= 2) {
        Y_RUN_KERNEL(2);
    } else if (avgElementsPerGroup <= 4) {
        Y_RUN_KERNEL(4);
    } else if (avgElementsPerGroup <= 8) {
        Y_RUN_KERNEL(8);
    } else if (avgElementsPerGroup <= 16) {
        Y_RUN_KERNEL(16);
    } else {
        Y_RUN_KERNEL(32);
    }

#undef Y_RUN_KERNEL
}

#define Y_CATBOOST_CUDA_F_IMPL(T) \
    template void MakeElementwiseOffsets<T>(const T* sizes, const T* biasedOffsets, ui64 size, T offsetsBias, T* elementwiseOffsets, ui64 elementwiseOffsetsSize, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    ui32);

#undef Y_CATBOOST_CUDA_F_IMPL

// MakeEndOfGroupMarkers

template <typename T>
__global__ void MakeEndOfGroupMarkersImpl(
    const T* const sizes,
    const T* const biasedOffsets,
    const ui64 size,
    const T offsetsBias,
    T* const endOfGroupMarkers,
    ui64 endOfGroupMarkersSize)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        if (i == 0) {
            endOfGroupMarkers[0] = 1;
        }

        const auto offset = __ldg(biasedOffsets + i) - offsetsBias + __ldg(sizes + i);
        if (offset < endOfGroupMarkersSize) {
            endOfGroupMarkers[offset] = 1;
        }

        i += gridDim.x * blockDim.x;
    }
}

template <typename T>
void MakeEndOfGroupMarkers(
    const T* const sizes,
    const T* const biasedOffsets,
    const ui64 size,
    const T offsetsBias,
    T* const endOfGroupMarkers,
    const ui64 endOfGroupMarkersSize,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    MakeEndOfGroupMarkersImpl<<<numBlocks, blockSize, 0, stream>>>(sizes, biasedOffsets, size, offsetsBias, endOfGroupMarkers, endOfGroupMarkersSize);
}

#define Y_CATBOOST_CUDA_F_IMPL(T) \
    template void MakeEndOfGroupMarkers<T>(const T* sizes, const T* biasedOffsets, ui64 size, T offsetsBias, T* endOfGroupMarkers, ui64 endOfGroupMarkersSize, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    ui32);

#undef Y_CATBOOST_CUDA_F_IMPL

// GatherBySizeAndOffset

template <typename T, typename I>
__global__ void GatherBySizeAndOffsetImpl(
    const T* const src,
    const I* const sizes,
    const I* const biasedOffsets,
    const ui64 size,
    const I offsetsBias,
    const I maxSize,
    T* const dst)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        dst[i] = __ldg(src + __ldg(biasedOffsets + i) - offsetsBias + min(__ldg(sizes + i), maxSize) - 1);
        i += gridDim.x * blockDim.x;
    }
}

template <typename T, typename I>
void GatherBySizeAndOffset(
    const T* const src,
    const I* const sizes,
    const I* const biasedOffsets,
    const ui64 size,
    const I offsetsBias,
    const I maxSize,
    T* const dst,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const ui64 numBlocks = Min(
        (size + blockSize - 1) / blockSize,
        (ui64)TArchProps::MaxBlockCount());
    GatherBySizeAndOffsetImpl<<<numBlocks, blockSize, 0, stream>>>(src, sizes, biasedOffsets, size, offsetsBias, maxSize, dst);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, I) \
    template void GatherBySizeAndOffset<T, I>(const T* src, const I* sizes, const I* biasedOffsets, ui64 size, I offsetsBias, I maxSize, T* dst, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// RemoveGroupMean

template <ui32 BlockSize, ui32 LogicalWarpSize, typename T, typename I>
__global__ void RemoveGroupMeanImpl(
    const T* values,
    const I* const sizes,
    const I* const biasedOffsets,
    const ui64 size,
    const I offsetsBias,
    T* normalized)
{
    const ui32 groupsPerBlock = BlockSize / LogicalWarpSize;
    const ui32 localGroupIdx = threadIdx.x / LogicalWarpSize;
    const ui32 groupIdx = blockIdx.x * groupsPerBlock + localGroupIdx;

    const ui32 groupOffset = groupIdx < size ? (__ldg(biasedOffsets + groupIdx) - offsetsBias) : 0;
    const ui32 groupSize = groupIdx < size ? __ldg(sizes + groupIdx) : 0;
    values += groupOffset;
    normalized += groupOffset;

    const ui32 localThreadIdx = threadIdx.x & (LogicalWarpSize - 1);

    T localMean = 0;
    for (ui32 i = localThreadIdx; i < groupSize; i += LogicalWarpSize) {
        localMean += __ldg(values + i) / groupSize;
    }

    T mean = ShuffleReduce<T>(localThreadIdx, localMean, LogicalWarpSize);
    mean = __shfl_sync(0xFFFFFFFF, mean, 0, LogicalWarpSize);

    for (ui32 i = localThreadIdx; i < groupSize; i += LogicalWarpSize) {
        normalized[i] = __ldg(values + i) - mean;
    }
}

template <typename T, typename I>
void RemoveGroupMean(
    const T* const values,
    const ui64 valuesSize,
    const I* const sizes,
    const I* const biasedOffsets,
    const ui64 size,
    const I offsetsBias,
    T* const normalized,
    TCudaStream stream)
{
    const ui32 blockSize = 512;
    const auto avgElementsPerGroup = valuesSize / size;

#define Y_RUN_KERNEL(LOGICAL_KERNEL_SIZE)                                        \
        const ui32 logicalWarpSize = LOGICAL_KERNEL_SIZE;                        \
        const auto blockCount = GetBlockCount(logicalWarpSize, blockSize, size); \
        RemoveGroupMeanImpl<blockSize, logicalWarpSize><<<blockCount, blockSize, 0, stream>>>(values, sizes, biasedOffsets, size, offsetsBias, normalized);

    if (avgElementsPerGroup <= 2) {
        Y_RUN_KERNEL(2);
    } else if (avgElementsPerGroup <= 4) {
        Y_RUN_KERNEL(4);
    } else if (avgElementsPerGroup <= 8) {
        Y_RUN_KERNEL(8);
    } else if (avgElementsPerGroup <= 16) {
        Y_RUN_KERNEL(16);
    } else {
        Y_RUN_KERNEL(32);
    }

#undef Y_RUN_KERNEL
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, I) \
    template void RemoveGroupMean<T, I>(const T* values, ui64 valuesSize, const I* sizes, const I* biasedOffsets, ui64 size, I offsetsBias, T* normalized, TCudaStream stream);

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, ui32));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

}
