#include "fill.cuh"
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>

namespace NKernel
{

    template<typename T>
    __global__ void FillBufferImpl(T* buffer, T value, ui64  size)
    {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size)
        {
            buffer[i] = value;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void FillBuffer(T* buffer, T value, ui64 size, TCudaStream stream)
    {
        if (size > 0)
        {
            const ui32 blockSize = 512;
            const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                         (ui64)TArchProps::MaxBlockCount());
            FillBufferImpl<T> << < numBlocks, blockSize, 0, stream>> > (buffer, value, size);
        }
    }

    template<typename T>
    __global__ void MakeSequenceImpl(T offset, T* buffer, ui64  size)
    {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            buffer[i] = offset + i;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void MakeSequence(T offset, T* buffer, ui64  size, TCudaStream stream)
    {
        if (size > 0)
        {
            const ui32 blockSize = 512;
            const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                         (ui64)TArchProps::MaxBlockCount());
            MakeSequenceImpl<T> << < numBlocks, blockSize, 0, stream >> > (offset, buffer, size);
        }
    }

    template<typename T>
    __global__ void InversePermutationImpl(const T* indices, T* dst, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            dst[indices[i]] = i;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void InversePermutation(const T* order, T* inverseOrder, ui64 size, TCudaStream stream)
    {
        if (size > 0)
        {
            const ui32 blockSize = 512;
            const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                       (ui64)TArchProps::MaxBlockCount());
            InversePermutationImpl<T> << < numBlocks, blockSize, 0, stream >> > (order, inverseOrder, size);
        }
    }



    template void FillBuffer<char>(char* buffer, char value, ui64  size, TCudaStream stream);

    template void FillBuffer<unsigned char>(unsigned char* buffer, unsigned char value, ui64  size, TCudaStream stream);

    template void FillBuffer<short>(short* buffer, short value, ui64  size, TCudaStream stream);

    template void FillBuffer<ui16>(ui16* buffer, ui16 value, ui64  size, TCudaStream stream);

    template void FillBuffer<int>(int* buffer, int value, ui64  size, TCudaStream stream);

    template void FillBuffer<ui32>(ui32* buffer, ui32 value, ui64  size, TCudaStream stream);

    template void FillBuffer<float>(float* buffer, float value, ui64  size, TCudaStream stream);

    template void FillBuffer<double>(double* buffer, double value, ui64  size, TCudaStream stream);

    template void FillBuffer<long>(long* buffer, long value, ui64  size, TCudaStream stream);

    template void FillBuffer<ui64>(ui64* buffer, ui64 value, ui64  size, TCudaStream stream);

    template void MakeSequence<int>(int offset, int* buffer, ui64  size, TCudaStream stream);
    template void MakeSequence<ui32>(ui32 offset, ui32* buffer, ui64  size, TCudaStream stream);
    template void MakeSequence<ui64>(ui64 offset, ui64* buffer, ui64  size, TCudaStream stream);

    template void InversePermutation<ui32>(const ui32* order, ui32* inverseOrder, ui64 size, TCudaStream stream);
    template void InversePermutation<int>(const int* order, int* inverseOrder, ui64 size, TCudaStream stream);
}
