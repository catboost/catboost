#include "reduce.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

namespace NKernel {

template <typename T>
__global__ void ReduceBinaryImpl(T* dst,
                                 const T* sourceLeft, const T* sourceRight,
                                 ui64 size)
{
    ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
       dst[i] = sourceLeft[i] + sourceRight[i];
       i += gridDim.x * blockDim.x;
    }
}

template <typename T>
void ReduceBinary(T* dst, const T* sourceLeft, const T* sourceRight, ui64 size, TCudaStream stream) {
    const ui64 blockSize = 128;
    const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());
    ReduceBinaryImpl<T><<<numBlocks, blockSize, 0,  stream>>>(dst, sourceLeft, sourceRight, size);
}

template void ReduceBinary<float>(float* dst, const float* sourceLeft, const float* sourceRight, ui64 size, TCudaStream stream);

}
