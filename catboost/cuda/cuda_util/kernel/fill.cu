#include "fill.cuh"
#include "kernel_helpers.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

#include <util/generic/cast.h>

namespace NKernel
{

    template <typename T>
    __global__ void FillBufferImpl(T* buffer, T value, ui64  size, ui64 alignSize)
    {
        buffer += blockIdx.y * alignSize;
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            WriteThrough(buffer + i, value);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void FillBuffer(T* buffer, T value, ui64 size, ui32 columnCount, ui64 alignSize, TCudaStream stream) {
        if (size > 0) {
            dim3 numBlocks;
            const ui32 blockSize = 128;
            numBlocks.x = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));
            numBlocks.y = columnCount;
            numBlocks.z = 1;
            FillBufferImpl<T> << < numBlocks, blockSize, 0, stream>> > (buffer, value, size, alignSize);
        }
    }



    template <typename T>
    __global__ void MakeSequenceImpl(T offset, T* buffer, ui64  size)
    {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            WriteThrough(buffer + i, (T)(offset + i));
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void MakeSequence(T offset, T* buffer, ui64  size, TCudaStream stream)
    {
        if (size > 0)
        {
            const ui32 blockSize = 512;
            const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                         (ui64)TArchProps::MaxBlockCount()));
            MakeSequenceImpl<T> << < numBlocks, blockSize, 0, stream >> > (offset, buffer, size);
        }
    }

    template <typename T>
    __global__ void InversePermutationImpl(const T* indices, T* dst, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            dst[indices[i]] = i;
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void InversePermutation(const T* order, T* inverseOrder, ui64 size, TCudaStream stream)
    {
        if (size > 0)
        {
            const ui32 blockSize = 512;
            const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                       (ui64)TArchProps::MaxBlockCount()));
            InversePermutationImpl<T> << < numBlocks, blockSize, 0, stream >> > (order, inverseOrder, size);
        }
    }


    #define FILL_BUFFER(Type)\
    template void FillBuffer<Type>(Type* buffer, Type value, ui64  size, ui32 columnCount, ui64 alignSize, TCudaStream stream);

    FILL_BUFFER(char) // i8 and char are distinct types
    FILL_BUFFER(i8)
    FILL_BUFFER(ui8)
    FILL_BUFFER(i16)
    FILL_BUFFER(ui16)
    FILL_BUFFER(i32)
    FILL_BUFFER(ui32)
    FILL_BUFFER(i64)
    FILL_BUFFER(ui64)
    FILL_BUFFER(float)
    FILL_BUFFER(double)
    FILL_BUFFER(bool)
    FILL_BUFFER(TCBinFeature)

    #undef FILL_BUFFER


    template void MakeSequence<int>(int offset, int* buffer, ui64  size, TCudaStream stream);
    template void MakeSequence<ui32>(ui32 offset, ui32* buffer, ui64  size, TCudaStream stream);
    template void MakeSequence<ui64>(ui64 offset, ui64* buffer, ui64  size, TCudaStream stream);

    template void InversePermutation<ui32>(const ui32* order, ui32* inverseOrder, ui64 size, TCudaStream stream);
    template void InversePermutation<int>(const int* order, int* inverseOrder, ui64 size, TCudaStream stream);
}
