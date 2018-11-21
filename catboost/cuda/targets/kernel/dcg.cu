#include "dcg.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

#include <util/generic/va_args.h>
#include <util/system/types.h>

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

}
