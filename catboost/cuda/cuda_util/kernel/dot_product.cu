#include "dot_product.cuh"
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

namespace NKernel {

    template<typename T, int BLOCK_SIZE>
    __global__ void DotProductImpl(const T *x, const T *y, T *partResults, ui64 size) {
        __shared__
        T sdata[BLOCK_SIZE];
        ui32 tid = threadIdx.x;
        ui32 i = blockIdx.x * BLOCK_SIZE * 2 + tid;

        T valx = i < size ? x[i] : 0;
        T valy = i < size ? y[i] : 0;
        sdata[tid] = valx * valy;
        __syncthreads();

        valx = i + BLOCK_SIZE < size ? x[i + BLOCK_SIZE] : 0;
        valy = i + BLOCK_SIZE < size ? y[i + BLOCK_SIZE] : 0;
        sdata[tid] += valx * valy;
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

    template<typename T>
    void DotProduct(const T *x, const T *y, TDotProductContext<T>& context, TCudaStream stream) {
        const ui32 blockSize = GetDotProductBlockSize();
        DotProductImpl<T, blockSize> << < context.NumBlocks, blockSize, 0, stream >> > (x, y, context.PartResults, context.Size);
    }

    template<typename T, int BLOCK_SIZE>
    __global__ void WeightedDotProductImpl(const T *x, const T *weights, const T *y, T *partResults, ui64 size) {
        __shared__
        T sdata[BLOCK_SIZE];
        ui32 tid = threadIdx.x;
        ui32 i = blockIdx.x * BLOCK_SIZE * 2 + tid;

        T valx = i < size ? x[i] : 0;
        T valy = i < size ? y[i] : 0;
        T weight = i < size ? weights[i] : 0;
        sdata[tid] = weight * valx * valy;
        __syncthreads();

        valx = i + BLOCK_SIZE < size ? x[i + BLOCK_SIZE] : 0.;
        valy = i + BLOCK_SIZE < size ? y[i + BLOCK_SIZE] : 0;
        weight = i + BLOCK_SIZE < size ? weights[i + BLOCK_SIZE] : 0;
        sdata[tid] += weight * valx * valy;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partResults[blockIdx.x] = sdata[0];
    }

    template<typename T>
    void WeightedDotProduct(const T *x, const T *weights, const T *y, TDotProductContext<T>& context, TCudaStream stream) {
        const ui32 blockSize = GetDotProductBlockSize();
        WeightedDotProductImpl<T, blockSize> << < context.NumBlocks, blockSize, 0, stream >> > (x, weights, y, context.PartResults, context.Size);
    }

    template void DotProduct<int>(const int *x, const int *y, TDotProductContext<int>& ctx, TCudaStream stream);

    template void DotProduct<double>(const double *x, const double *y, TDotProductContext<double>& ctx, TCudaStream stream);

    template void DotProduct<ui32>(const ui32 *x, const ui32 *y, TDotProductContext<ui32>& ctx, TCudaStream stream);

    template void DotProduct<float>(const float *x, const float *y, TDotProductContext<float>& ctx, TCudaStream stream);

    template void WeightedDotProduct<float>(const float *x, const float *weight, const float *y, TDotProductContext<float>& ctx, TCudaStream stream);

    template void WeightedDotProduct<double>(const double *x, const double *weight, const double *y, TDotProductContext<double>& ctx, TCudaStream stream);
}
