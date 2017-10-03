#include "add_model_value.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


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
        const uint blockSize = 256;
        const ui32 elementsPerThreads = 10;
        const ui32 numBlocks = CeilDivide<ui32>(size, blockSize * elementsPerThreads);
        AddBinModelValueImpl<blockSize, elementsPerThreads> << <numBlocks, blockSize, 0, stream>>>(binValues, binCount, bins, size, readIndices, writeIndices, cursor);
    }

}
