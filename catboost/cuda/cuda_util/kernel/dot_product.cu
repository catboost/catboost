#include "dot_product.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

namespace NKernel {

    template <typename T, int BLOCK_SIZE>
    __global__ void DotProductImpl(const T *x, const T *y, T *partResults, ui64 size) {
        __shared__ T sdata[BLOCK_SIZE];
        ui32 tid = threadIdx.x;
        ui32 i = blockIdx.x * BLOCK_SIZE * 2 + tid;


        T valx = i < size ? __ldg(x + i) : 0;
        T valy = i < size ? __ldg(y + i) : 0;

        T val2x = i + BLOCK_SIZE < size ? __ldg(x + i + BLOCK_SIZE) : 0;
        T val2y = i + BLOCK_SIZE < size ? __ldg(y + i + BLOCK_SIZE) : 0;

        sdata[tid] =  valx * valy + val2x * val2y;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            partResults[blockIdx.x] = sdata[0];
        }
    }

    template <typename T>
    void DotProduct(const T *x, const T *y, TDotProductContext<T>& context, TCudaStream stream) {
        const ui32 blockSize = GetDotProductBlockSize();
        DotProductImpl<T, blockSize> << < context.NumBlocks, blockSize, 0, stream >> > (x, y, context.PartResults.Get(), context.Size);
    }

    template <typename T, int BLOCK_SIZE>
    __global__ void WeightedDotProductImpl(const T *x, const T *weights, const T *y, T *partResults, ui64 size) {
        __shared__
        T sdata[BLOCK_SIZE];
        ui32 tid = threadIdx.x;
        ui32 i = blockIdx.x * BLOCK_SIZE * 2 + tid;

        T valx = i < size ? __ldg(x + i) : 0;
        T valy = i < size ? __ldg(y + i) : 0;
        T weight = i < size ? __ldg(weights + i) : 0;

        T val2x = i + BLOCK_SIZE < size ? __ldg(x + i + BLOCK_SIZE) : 0.;
        T val2y = i + BLOCK_SIZE < size ? __ldg(y + i + BLOCK_SIZE) : 0;
        T weight2 = i + BLOCK_SIZE < size ? __ldg(weights + i + BLOCK_SIZE) : 0;
        sdata[tid] = weight * valx * valy + weight2 * val2x * val2y;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partResults[blockIdx.x] = sdata[0];
    }

    template <typename T>
    void WeightedDotProduct(const T *x, const T *weights, const T *y, TDotProductContext<T>& context, TCudaStream stream) {
        const ui32 blockSize = GetDotProductBlockSize();
        WeightedDotProductImpl<T, blockSize> << < context.NumBlocks, blockSize, 0, stream >> > (x, weights, y, context.PartResults.Get(), context.Size);
    }

    template void DotProduct<int>(const int *x, const int *y, TDotProductContext<int>& ctx, TCudaStream stream);

    template void DotProduct<double>(const double *x, const double *y, TDotProductContext<double>& ctx, TCudaStream stream);

    template void DotProduct<ui32>(const ui32 *x, const ui32 *y, TDotProductContext<ui32>& ctx, TCudaStream stream);

    template void DotProduct<float>(const float *x, const float *y, TDotProductContext<float>& ctx, TCudaStream stream);

    template void WeightedDotProduct<float>(const float *x, const float *weight, const float *y, TDotProductContext<float>& ctx, TCudaStream stream);

    template void WeightedDotProduct<double>(const double *x, const double *weight, const double *y, TDotProductContext<double>& ctx, TCudaStream stream);
}
