#include "reorder_one_bit.cuh"
#include "reorder_one_bit_impl.cuh"

#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <cub/device/device_scan.cuh>

namespace NKernel {


    template <class T>
    void ReorderOneBit(
        ui32 size,
        TReorderOneBitContext<ui32, T> context,
        ui32* keys,
        T* values,
        int bit,
        TCudaStream stream) {

        if (size) {
            cudaMemcpyAsync(context.TempValues.Get(), values, sizeof(T) * size, cudaMemcpyDefault, stream);
            cudaMemcpyAsync(context.TempKeys.Get(), keys, sizeof(ui32) * size, cudaMemcpyDefault, stream);

            {
                using TInput = TScanBitIterator<ui32>;
                TInput inputIter(context.TempKeys.Get(), bit);
                cub::DeviceScan::ExclusiveSum < TInput, int*> (context.ScanTempBuffer.Get(),
                    context.ScanTempBufferSize,
                    inputIter,
                    context.Offsets.Get(),
                    size,
                    stream);
            }

            const int blockSize = 512;
            const int N = 1;
            const int numBlocks = (size + (N * blockSize) - 1) / (N * blockSize);
            ReorderOneBitImpl<ui32, ui32, N, blockSize> << < numBlocks, blockSize, 0, stream >> > (
                context.TempKeys,
                context.TempValues,
                context.Offsets,
                bit,
                keys,
                values,
                size);
        }
    }

    ui64 ReorderBitTempSize(ui32 size) {
        ui64 sizeInBytes = 0;
        using TInput =  TScanBitIterator<ui32>;
        TInput fakeInput(nullptr, 0);
        cub::DeviceScan::ExclusiveSum< TInput, int * > (nullptr,
            sizeInBytes,
            fakeInput,
            nullptr,
            size);
        return sizeInBytes;
    }

    template void ReorderOneBit<ui32>(
        ui32 size,
        TReorderOneBitContext<ui32, ui32> context,
        ui32* keys,
        ui32* values,
        int bit,
        TCudaStream stream);

}



