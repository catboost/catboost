#include "add_model_value.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


namespace NKernel {

    //it may be faster to cache in shared memory binValues, but for fold + bin + depth > 10 it'll be slower and may fail on shared memory allocation
    //so current impl more conservative with const-load cache
    __global__ void AddBinModelValueImpl(const float* binValues, ui32 binCount,
                                         const ui32* bins, ui32 size,
                                         const ui32* readIndices, const ui32* writeIndices,
                                         float* cursor) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            ui32 readIdx =  readIndices ? LdgWithFallback(readIndices, i) : i;
            ui32 writeIdx = writeIndices ? LdgWithFallback(writeIndices, i) : i;

            ui32 bin = LdgWithFallback(bins, readIdx);
            cursor[writeIdx] += LdgWithFallback(binValues, bin);
        }
    }

    void AddBinModelValue(const float* binValues, ui32 binCount,
                          const ui32* bins,
                          const ui32* readIndices, const ui32* writeIndices,
                          float* cursor, ui32 size,
                          TCudaStream stream) {
        const uint blockSize = 1024;
        const uint numBlocks = (size + blockSize - 1) / blockSize;
        AddBinModelValueImpl << <numBlocks, blockSize, 0, stream>>>(binValues, binCount, bins, size, readIndices, writeIndices, cursor);
    }

}
