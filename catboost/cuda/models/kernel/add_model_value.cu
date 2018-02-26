#include "add_model_value.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>


namespace NKernel {

    //it may be faster to cache in shared memory binValues, but for fold + bin + depth > 10 it'll be slower and may fail on shared memory allocation
    //so current impl more conservative with const-load cache
    template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
    __launch_bounds__(BLOCK_SIZE, 2048 / BLOCK_SIZE)
    __global__ void AddBinModelValueImpl(const float* binValues, ui32 binCount,
                                         const ui32* bins, ui32 size,
                                         const ui32* readIndices, const ui32* writeIndices,
                                         float* cursor) {
        const ui32 i = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x;

        ui32 writeIndicesLocal[ELEMENTS_PER_THREAD];
        ui32 binsLocal[ELEMENTS_PER_THREAD];

        #pragma unroll ELEMENTS_PER_THREAD
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = i + j * BLOCK_SIZE;
            const ui32 readIdx = idx < size ? (readIndices ? LdgWithFallback(readIndices, idx) : idx) : (ui32)-1;;
            writeIndicesLocal[j] = idx < size ? (writeIndices ? LdgWithFallback(writeIndices, idx) : idx) : (ui32)-1;
            binsLocal[j] = idx < size ? LdgWithFallback(bins, readIdx) : 0;
        }

        float binsValuesLocal[ELEMENTS_PER_THREAD];

        #pragma unroll ELEMENTS_PER_THREAD
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = i + j * BLOCK_SIZE;
            binsValuesLocal[j] = idx < size ? LdgWithFallback(binValues, binsLocal[j])  : 0;
        }

        #pragma unroll ELEMENTS_PER_THREAD
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int idx = i + j * BLOCK_SIZE;
            if (idx < size) {
                cursor[writeIndicesLocal[j]] += binsValuesLocal[j];
            }
        }
    }

    void AddBinModelValue(const float* binValues, ui32 binCount,
                          const ui32* bins,
                          const ui32* readIndices, const ui32* writeIndices,
                          float* cursor, ui32 size,
                          TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 10;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize * elementsPerThreads);
        AddBinModelValueImpl<blockSize, elementsPerThreads> << <numBlocks, blockSize, 0, stream>>>(binValues, binCount, bins, size, readIndices, writeIndices, cursor);
    }



    __global__ void AddObliviousTreeImpl(const TCFeature* features, const ui8* bins, const float* leaves, ui32 depth,
                                         const ui32* cindex,
                                         const ui32* readIndices,
                                         const ui32* writeIndices,
                                         float* cursor,
                                         ui32 size) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ ui32 masksLocal[32];
        __shared__ ui32 valuesLocal[32];
        __shared__ ui64 offsetsLocal[32];
        __shared__ ui32 takeEqual[32];

        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            const ui32 value =(ui32)(bins[level]) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqual[level] = feature.OneHotFeature;
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            ui32 bin = 0;
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg((cindex + offsetsLocal[level]) + loadIdx) & mask;
                const ui32 split = (takeEqual[level] ? (featureVal == value) : featureVal > value);
                bin |= split << level;
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;
            cursor[writeIdx] += __ldg(leaves + bin);
            tid += blockDim.x  * gridDim.x;
        }
    }


    __global__ void ComputeObliviousTreeBinsImpl(const TCFeature* features, const ui8* bins,  ui32 depth,
                                                 const ui32* cindex,
                                                 const ui32* readIndices,
                                                 const ui32* writeIndices,
                                                 ui32* cursor,
                                                 ui32 size) {

        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ ui32 masksLocal[32];
        __shared__ ui32 valuesLocal[32];
        __shared__ ui64 offsetsLocal[32];
        __shared__ ui32 takeEqual[32];

        if (threadIdx.x < depth) {
            const int level = threadIdx.x;
            TCFeature feature = features[level];
            const ui32 value =(ui32)(bins[level]) << feature.Shift;
            const ui32 mask = feature.Mask << feature.Shift;

            masksLocal[level] = mask;
            valuesLocal[level] = value;
            takeEqual[level] = feature.OneHotFeature;
            offsetsLocal[level] = feature.Offset;
        }
        __syncthreads();

        while (tid < size) {
            ui32 bin = 0;
            const ui32 loadIdx = readIndices ? readIndices[tid] : tid;

            #pragma unroll 8
            for (ui32 level = 0; level < depth; ++level) {
                const ui32 value = valuesLocal[level];
                const ui32 mask = masksLocal[level];
                const ui32 featureVal = __ldg(cindex + offsetsLocal[level] + loadIdx) & mask;
                const ui32 split = (takeEqual[level] ? (featureVal == value) : featureVal > value);
                bin |= split << level;
            }
            const ui32 writeIdx = writeIndices ? writeIndices[tid] : tid;
            cursor[writeIdx] = bin;
            tid += blockDim.x  * gridDim.x;
        }
    }


    //doc parallel routines
    void AddObliviousTree(const TCFeature* features, const ui8* bins, const float* leaves, ui32 depth,
                          const ui32* cindex,
                          const ui32* readIndices,
                          const ui32* writeIndices,
                          float* cursor,
                          ui32 size,
                          TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
        AddObliviousTreeImpl<< <numBlocks, blockSize, 0, stream>>>(features, bins, leaves, depth, cindex, readIndices, writeIndices, cursor, size);
    }


    void ComputeObliviousTreeBins(const TCFeature* features, const ui8* bins, ui32 depth,
                                  const ui32* cindex,
                                  const ui32* readIndices,
                                  const ui32* writeIndices,
                                  ui32* cursor,
                                  ui32 size,
                                  TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize);
       ComputeObliviousTreeBinsImpl<< <numBlocks, blockSize, 0, stream>>>(features, bins, depth, cindex, readIndices, writeIndices, cursor, size);
    }

}
