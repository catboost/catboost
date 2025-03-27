#include "binarize.cuh"
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

#include <cub/block/block_radix_sort.cuh>

namespace NKernel {

    __global__ void WriteCompressedIndexImpl(TCFeature feature, const ui8* bins, ui32 docCount, ui32* cindex) {

        cindex += feature.Offset;
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i  < docCount) {
            const ui32 bin = (((ui32)bins[i]) & feature.Mask) << feature.Shift;
            cindex[i] = cindex[i] | bin;
            i += blockDim.x * gridDim.x;
        }
    }

    void WriteCompressedIndex(TCFeature feature,
                              const ui8* bins, ui32 docCount,
                              ui32* cindex,
                              TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = (docCount + blockSize - 1) / blockSize;

        WriteCompressedIndexImpl<< < numBlocks, blockSize, 0, stream >> > (feature, bins, docCount, cindex);
    }



    template <bool ATOMIC_UPDATE, int BLOCK_SIZE, int DOCS_PER_THREAD>
    __launch_bounds__(BLOCK_SIZE, CUDA_MAX_THREADS_PER_SM / BLOCK_SIZE)
    __global__ void BinarizeFloatFeatureImpl(TCFeature feature, const float* values, ui32 docCount,
                                             const float* borders,
                                             const ui32* gatherIndex, ui32* dst) {

        const ui32 i = (blockIdx.x * BLOCK_SIZE * DOCS_PER_THREAD + threadIdx.x);

        __shared__ float sharedBorders[256];
        sharedBorders[0] = borders[0];
        __syncthreads();
        const int bordersCount = static_cast<int>(sharedBorders[0]);
        __syncthreads();
        dst += feature.Offset;

        if (threadIdx.x < bordersCount) {
            sharedBorders[threadIdx.x] = LdgWithFallback(borders, threadIdx.x + 1);
        }
        __syncthreads();

        ui32 index[DOCS_PER_THREAD];
        float featureValues[DOCS_PER_THREAD];

        #pragma unroll
        for (int j = 0; j < DOCS_PER_THREAD; ++j) {
            index[j] = 0;
            const int idx = i + j * BLOCK_SIZE;

            if (idx < docCount) {
                const ui32 readIdx = gatherIndex ? StreamLoad(gatherIndex + idx) : idx;
                featureValues[j] = StreamLoad(values + readIdx);
            }
        }

        #pragma unroll
        for (int border = 0; border < bordersCount; ++border)
        {
            const float borderValue = sharedBorders[border];
            #pragma unroll
            for (int j = 0; j < DOCS_PER_THREAD; ++j)
            {
                if (featureValues[j] > borderValue)
                {
                    ++index[j];
                }
            }
        }


        #pragma unroll
        for (int j = 0; j < DOCS_PER_THREAD; ++j)
        {
            const int idx = i + j * BLOCK_SIZE;

            if (idx < docCount) {

                if (ATOMIC_UPDATE)
                {
                    atomicOr(dst + idx, (index[j] & feature.Mask) << feature.Shift);
                } else {
                    ui32 bin = dst[idx];
                    bin |= (index[j] & feature.Mask) << feature.Shift;
                    dst[idx] = bin;
                }
            }
        }
    }

    //smth like bootstrap for quantiles estimation
    template <ui32 BLOCK_SIZE>
    __global__ void FastGpuBordersImpl(const float* values, ui32 size, float* borders, ui32 bordersCount) {

        const int valuesPerThread = 2;
        using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, 2>;
        const int tid = threadIdx.x;
        float vals[valuesPerThread];

        if (tid == 0 && blockIdx.x == 0) {
            borders[0] = bordersCount;
        }

        ui64 seed = (blockIdx.x  * 6364136223846793005 + 1442695040888963407) + (1664525 * threadIdx.x + 1013904223) & 0xFFFFFF;

        for (int i = 0; i < valuesPerThread; ++i) {
            const int idx = static_cast<int>(AdvanceSeed(&seed) % size);
            vals[i] = StreamLoad(values + idx);
        }

        {
            using TTempStorage = typename BlockRadixSort::TempStorage;
            __shared__ TTempStorage temp;
            BlockRadixSort(temp).Sort(vals);
        }

        float sum = 0;
        float weight = 0;
        for (int i = 0; i < valuesPerThread; ++i) {
            sum += vals[i];
            weight += 1.0f;
        }

        __shared__ float localBorders[BLOCK_SIZE];
        localBorders[tid] = sum / weight;
        __syncthreads();

        if (tid < bordersCount) {
            const ui32 offset = static_cast<ui32>((tid + 1.0f) * BLOCK_SIZE / bordersCount - 1e-5f);
            atomicAdd(borders + tid + 1, (localBorders[offset]) * 0.9999 / gridDim.x);
        }
    }

    __global__ void SortBordersImpl(float* borders, ui32 bordersCount)
    {

        using BlockRadixSort = cub::BlockRadixSort<float, 256, 1>;
        ui32 tid = threadIdx.x;
        float val[1];
        val[0] = tid < bordersCount ? borders[tid] : PositiveInfty();
        using TTempStorage = typename BlockRadixSort::TempStorage;
        __shared__ TTempStorage temp;
        BlockRadixSort(temp).Sort(val);
        if (tid < bordersCount) {
            borders[tid] = val[0];
        }
    }

    void FastGpuBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream) {
        FillBuffer(borders, 0.0f, bordersCount + 1, stream);
        const ui32 blockSize = 1024;
        const ui32 valuesPerBlock = 2 * blockSize;
        const ui32 numBlocks = min(CeilDivide(size, valuesPerBlock), 15);
        FastGpuBordersImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(values, size, borders, bordersCount);
        SortBordersImpl<<<1, 256, 0, stream>>>(borders + 1, bordersCount);
    }

    __global__ void QuantileBordersImpl(const float* sortedValues, ui32 size, float* borders, ui32 bordersCount) {
        const ui32 tid = threadIdx.x;
        __shared__ float localBorders[256];

        if (tid < bordersCount) {
            const ui32 offset = static_cast<ui32>((tid + 1.0) * size / (bordersCount + 1));
            localBorders[tid] = LdgWithFallback(sortedValues, offset);
        }
        __syncthreads();

        if (tid <(bordersCount + 1)) {
            borders[tid] = tid == 0 ? bordersCount : localBorders[tid -  1];
        }
    }


    __global__ void UniformBordersImpl(const float* values, ui32 size, float* borders, ui32 bordersCount) {

        const ui32 tid = threadIdx.x;
        const int blockSize = 1024;

        __shared__ float localMin[blockSize];
        __shared__ float localMax[blockSize];

        float minValue = PositiveInfty();
        float maxValue = NegativeInfty();

        ui64 seed = (1664525 * threadIdx.x + 1013904223) & 0xFFFFFF;

        #pragma unroll 16
        for (int i = 0; i < 32; ++i) {
            const int idx = static_cast<int>(AdvanceSeed(&seed) % size);
            float val = StreamLoad(values + idx);
            minValue = val < minValue ? val : minValue;
            maxValue = val > maxValue ? val : maxValue;
        }

        localMin[tid] = minValue * 0.999;
        localMax[tid] = maxValue * 1.001;
        __syncthreads();

        for (ui32 s = blockSize >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                localMin[tid] = min(localMin[tid], localMin[tid + s]);
                localMax[tid] = max(localMax[tid], localMax[tid + s]);
            }
            __syncthreads();
        }
        minValue = localMin[0];
        maxValue = localMax[0];

        if (tid < (bordersCount + 1)) {
            const float borderIdx = tid * 1.0f / bordersCount;
            //emulate ui8 rounding in CPU
            const float val =  (minValue + borderIdx * (maxValue - minValue)) * 0.9999;
            borders[tid] =  tid == 0 ? bordersCount : val;
        }
    }

    void ComputeQuantileBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream) {
        QuantileBordersImpl<<< 1, 256, 0, stream >>> (values, size, borders, bordersCount);
    }

    void ComputeUniformBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream) {
        UniformBordersImpl<<< 1, 1024, 0, stream >>> (values, size, borders, bordersCount);
    }

    void BinarizeFloatFeature(const float* values, ui32 docCount,
                              const float* borders,
                              TCFeature feature,
                              ui32* dst,
                              const ui32* gatherIndex,
                              bool atomicUpdate,
                              TCudaStream stream) {

        const ui32 blockSize = 1024;
        const ui32 docsPerThread = 8;
        const ui32 numBlocks = (docCount + docsPerThread * blockSize - 1) / (docsPerThread * blockSize);

        if (atomicUpdate)
        {
            BinarizeFloatFeatureImpl<true, blockSize, docsPerThread> << < numBlocks, blockSize, 0, stream >> > (feature, values, docCount,
                    borders, gatherIndex,
                    dst);
        } else {
            BinarizeFloatFeatureImpl<false, blockSize, docsPerThread> << < numBlocks, blockSize, 0, stream >> > (feature, values, docCount,
                    borders, gatherIndex,
                    dst);
        }
    }


}
