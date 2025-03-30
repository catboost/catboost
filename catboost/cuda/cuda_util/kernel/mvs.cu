#include "mvs.cuh"
#include "kernel_helpers.cuh"
#include "random_gen.cuh"
#include "reduce.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>

#include <cub/cub.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>


namespace NKernel {

__forceinline__ __device__ float GetSingleProbability(
    float derivativeAbsoluteValue,
    float threshold
) {
    return (derivativeAbsoluteValue > threshold) ? 1.0f : __fdividef(derivativeAbsoluteValue, threshold);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void GetThreshold(
    float takenFraction,
    float (&candidates)[ITEMS_PER_THREAD],
    float (&prefixSum)[ITEMS_PER_THREAD],
    ui32 size,
    float* threshold
) {
    const ui32 thisBlockSize = min(BLOCK_THREADS * ITEMS_PER_THREAD, size);
    const float sampleSize = thisBlockSize * takenFraction;
    __shared__ ui32 argMinBorder[BLOCK_THREADS];
    __shared__ float minBorder[BLOCK_THREADS];
    argMinBorder[threadIdx.x] = 0;
    minBorder[threadIdx.x] = thisBlockSize;
    __shared__ bool exit;
    if (ITEMS_PER_THREAD * threadIdx.x <= thisBlockSize - 1 &&
    ITEMS_PER_THREAD * (threadIdx.x + 1) > thisBlockSize - 1) {
        const ui32 localId = thisBlockSize - 1 - threadIdx.x * ITEMS_PER_THREAD;
        #pragma unroll
        for (int idx = 0; idx < ITEMS_PER_THREAD; ++idx) {
            if (idx == localId) {
                if (candidates[idx] <= prefixSum[idx] / sampleSize) {
                    *threshold = prefixSum[idx] / sampleSize;
                    exit = true;
                } else {
                    exit = false;
                }
            }
        }
    }
    __syncthreads();
    if (exit) {
        return;
    }

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        // Here cub::BlockRadixsort and cub::BlockScan numeration is used
        const ui32 i = k + ITEMS_PER_THREAD * threadIdx.x;

        if (i < thisBlockSize) {
            const float takenSize = prefixSum[k] / candidates[k] + thisBlockSize - i - 1;
            if (takenSize >= sampleSize) { // takenSize is non-growing function
                minBorder[threadIdx.x] = takenSize;
                argMinBorder[threadIdx.x] = i;
            }
        }
    }
    __syncthreads();

    #pragma  unroll
    for (int s = BLOCK_THREADS >> 1; s >= 32; s >>= 1) {
        if (threadIdx.x < s)
        {
            if (minBorder[threadIdx.x + s] < minBorder[threadIdx.x]) {
                argMinBorder[threadIdx.x] = argMinBorder[threadIdx.x + s];
                minBorder[threadIdx.x] = minBorder[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        __syncwarp();
        #pragma unroll
        for (int s = 32 >> 1; s > 0; s >>= 1) {
            if (minBorder[threadIdx.x + s] < minBorder[threadIdx.x]) {
                argMinBorder[threadIdx.x] = argMinBorder[threadIdx.x + s];
                minBorder[threadIdx.x] = minBorder[threadIdx.x + s];
            }
            __syncwarp();
        }
    }
    __syncthreads();

    if (
        ITEMS_PER_THREAD * threadIdx.x <= argMinBorder[0] &&
        ITEMS_PER_THREAD * (threadIdx.x + 1) > argMinBorder[0]
    ) {
        const int localId = argMinBorder[0] - threadIdx.x * ITEMS_PER_THREAD;
        const int denom = sampleSize - (thisBlockSize - argMinBorder[0] - 1);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            minBorder[i] = prefixSum[i];
        }
        *threshold = minBorder[localId] / (denom);
    }
}


template <int ITEMS_PER_THREAD, int BLOCK_THREADS>
__device__ __forceinline__ void CalculateThreshold(
    float takenFraction,
    const float* candidates,
    ui32 size,
    float* threshold
) {
    const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockScan =      cub::BlockScan<float, BLOCK_THREADS>;

    __shared__ union {
        typename BlockRadixSort::TempStorage Sort;
        typename BlockScan::TempStorage      Scan;
    } tempStorage;

    // Our current block's offset
    int blockOffset = blockIdx.x * TILE_SIZE;

    // Per-thread tile items
    float items[ITEMS_PER_THREAD];
    float scanItems[ITEMS_PER_THREAD];

    // Load items into a blocked arrangement

    int idx = blockOffset + threadIdx.x;
    const float inf = std::numeric_limits<float>::max();
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        if (idx < size) {
            items[k] = StreamLoad(candidates + idx);
        } else {
            items[k] = inf;
        }
        idx += BLOCK_THREADS;
    }
    __syncthreads();

    BlockRadixSort(tempStorage.Sort).Sort(items, 8);
    __syncthreads();

    BlockScan(tempStorage.Scan).InclusiveSum(items, scanItems);
    __syncthreads();

    GetThreshold<BLOCK_THREADS, ITEMS_PER_THREAD>(
        takenFraction,
        items,
        scanItems,
        size - blockOffset,
        threshold
    );
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void CalculateThresholdImpl(
    float takenFraction,
    float* candidates,
    ui32 size,
    float* threshold
) {
    CalculateThreshold<ITEMS_PER_THREAD, BLOCK_THREADS>(
        takenFraction, candidates, size, threshold + blockIdx.x
    );
}

template <int ITEMS_PER_THREAD, int BLOCK_THREADS>
__global__ void MvsBootstrapRadixSortImpl(
    float takenFraction,
    float lambda,
    float* weights,
    const float* ders,
    ui32 size,
    const ui64* seeds,
    ui32 seedSize
) {
    const int blockOffset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockScan =      cub::BlockScan<float, BLOCK_THREADS>;

    __shared__ union {
        typename BlockRadixSort::TempStorage Sort;
        typename BlockScan::TempStorage      Scan;
    } tempStorage;

    // Per-thread tile items
    float weightsPerThread[ITEMS_PER_THREAD];
    float items[ITEMS_PER_THREAD];
    float scanItems[ITEMS_PER_THREAD];

    const int idx = blockOffset + threadIdx.x;
    const float inf = sqrtf(std::numeric_limits<float>::max()) - 2 * lambda;
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inputIterator(ders);
    cub::LoadDirectWarpStriped(
        idx,
        inputIterator,
        weightsPerThread,
        size,
        inf
    );

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        weightsPerThread[k] = sqrtf(
            fmaf(weightsPerThread[k], weightsPerThread[k], lambda)
        );
        items[k] = weightsPerThread[k];
    }
    __syncthreads();

    BlockRadixSort(tempStorage.Sort).Sort(items, 8);
    __syncthreads();

    BlockScan(tempStorage.Scan).InclusiveSum(items, scanItems);
    __syncthreads();

    __shared__ float threshold;
    GetThreshold<BLOCK_THREADS, ITEMS_PER_THREAD>(
        takenFraction,
        items,
        scanItems,
        size - blockOffset,
        &threshold
    );
    __syncthreads();

    // Set Mvs weights
    ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
    ui64 s = __ldg(seeds + i % seedSize) + blockIdx.x;
    const float eps = std::numeric_limits<float>::epsilon();

    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        const float probability = GetSingleProbability(weightsPerThread[k], threshold);
        weightsPerThread[k] = (probability > eps && NextUniformF(&s) < probability)
            ? __fdividef(1.0f, probability)
            : 0.0f;
    }

    cub::CacheModifiedOutputIterator<cub::STORE_CS, float> outputIterator(weights);
    cub::StoreDirectWarpStriped(
        idx,
        outputIterator,
        weightsPerThread,
        size
    );
}

void MvsBootstrapRadixSort(
    const float takenFraction,
    const float lambda,
    float* weights,
    const float* ders,
    ui32 size,
    const ui64* seeds,
    ui32 seedSize,
    TCudaStream stream
) {
    const ui32 blockThreads = 512;
    const ui32 SCAN_ITEMS_PER_THREAD = 8192 / blockThreads;
    const ui32 numBlocks = CeilDivide(size, blockThreads * SCAN_ITEMS_PER_THREAD);

    {
        MvsBootstrapRadixSortImpl<SCAN_ITEMS_PER_THREAD, blockThreads> <<< numBlocks, blockThreads, 0, stream >>> (
            takenFraction, lambda, weights, ders, size, seeds, seedSize
        );
    }
}

void CalculateMvsThreshold(
    const float takenFraction,
    float* candidates,
    ui32 size,
    float* threshold,
    TCudaStream stream
) {
    const ui32 blockThreads = 512;
    const ui32 SCAN_ITEMS_PER_THREAD = 8192 / blockThreads;
    const ui32 numBlocks = CeilDivide(size, blockThreads * SCAN_ITEMS_PER_THREAD);
    {
        CalculateThresholdImpl<SCAN_ITEMS_PER_THREAD, blockThreads> <<< numBlocks, blockThreads, 0, stream >>> (
            takenFraction, candidates, size, threshold
        );
    }
}

}
