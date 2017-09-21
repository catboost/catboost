#include "transform.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <contrib/libs/cub/cub/block/block_radix_sort.cuh>


namespace NKernel {

    template<typename T>
    __global__ void AddVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T y0 = y[i];
            const T x0 = x[i];
            const T r0 = y0 + x0;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void AddVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        AddVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }


    template<typename T>
    __global__ void AddVectorImpl(T *x, const T y, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T x0 = x[i];
            const T r0 = y + x0;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void AddVector(T *x, const T y, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        AddVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template<typename T>
    __global__ void SubtractVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T y0 = y[i];
            const T x0 = x[i];
            const T r0 = x0 - y0;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void SubtractVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        SubtractVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template<typename T>
    __global__ void MultiplyVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;

        while (i < size) {
            const T y0 = y[i];
            const T x0 = x[i];
            const T r0 = y0 * x0;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void MultiplyVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        MultiplyVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template<typename T>
    __global__ void MultiplyVectorImpl(T *x, const T c, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T x0 = x[i];
            T r0 = x0 * c;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void MultiplyVector(T *x, const T c, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        MultiplyVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, c, size);
    }


    template<typename T>
    __global__ void DivideVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T x0 = x[i];
            T y0 = y[i];
            T r0 = (y0 < 1e-10f && y0 > -1e-10f) ? 0 : 1.0f * x0 / y0;
            x[i] = r0;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void DivideVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount());
        DivideVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template<typename T>
    __global__ void ExpVectorImpl(T *x, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T val = x[i];
            x[i] = exp(val);
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void ExpVector(T *x, ui64 size, TCudaStream stream) {
        const uint blockSize = 512;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());
        ExpVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, size);
    }

    template<typename T, typename Index>
    __global__ void GatherImpl(T *dst, const T *src, const Index *map, Index size) {
        Index i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i);
            WriteThrough(dst + i, StreamLoad(src + m));
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T, typename Index>
    void Gather(T *dst, const T *src, const Index* map, ui64 size, TCudaStream stream) {
        const ui64 blockSize = 256;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());

        if (numBlocks) {
            GatherImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size);
        }
    }


    template<typename T, typename Index>
    __global__ void GatherWithMaskImpl(T *dst, const T *src, const Index *map, Index size, Index mask) {
        Index i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i) & mask;
            WriteThrough(dst + i, StreamLoad(src + m));
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T, typename Index>
    void GatherWithMask(T *dst, const T *src, const Index* map, ui64 size, Index mask, TCudaStream stream) {
        const ui64 blockSize = 256;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());

        if (numBlocks) {
            GatherWithMaskImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, mask);
        }
    }


    template<typename T, typename Index>
    __global__ void ScatterImpl(T* dst, const T* src, const Index* map, Index size) {
        Index i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i);
            WriteThrough(dst + m, StreamLoad(src + i));
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T, typename Index>
    void Scatter(T *dst, const T *src, const Index* map, ui64 size, TCudaStream stream) {
        const uint blockSize = 256;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());
        if (numBlocks) {
            ScatterImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size);
        }
    }


    template<typename T, typename Index>
    __global__ void ScatterWithMaskImpl(T* dst, const T* src, const Index* map, Index size, Index mask) {
        Index i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i) & mask;
            WriteThrough(dst + m, StreamLoad(src + i));
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T, typename Index>
    void ScatterWithMask(T *dst, const T *src, const Index* map, ui64 size, Index mask, TCudaStream stream) {
        const uint blockSize = 256;
        const ui64 numBlocks = min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());
        if (numBlocks) {
            ScatterWithMaskImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, mask);
        }
    }

    template<typename T>
    __global__ void ReverseImpl(T *data, ui64 size) {
        ui64 i = blockIdx.x * blockDim.x + threadIdx.x;
        ui64 half = size / 2;
        while (i < half) {
            T a = data[i];
            T b = data[size - i - 1];
            data[i] = b;
            data[size - i - 1] = a;
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename T>
    void Reverse(T* data, ui64 size, TCudaStream stream) {
        const uint blockSize = 256;
        const ui64 numBlocks = min(((size + 1) / 2 + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount());
        ReverseImpl<T> << < numBlocks, blockSize, 0, stream >> > (data, size);
    }


    template void AddVector<float>(float *x, const float *y, ui64 size, TCudaStream stream);

    template void AddVector<int>(int *x, const int *y, ui64 size, TCudaStream stream);

    template void AddVector<float>(float *x, const float y, ui64 size, TCudaStream stream);

    template void AddVector<int>(int *x, const int y, ui64 size, TCudaStream stream);

    template void AddVector<ui32>(ui32 *x, const ui32 y, ui64 size, TCudaStream stream);

    template void AddVector<uint>(uint *x, const uint *y, ui64 size, TCudaStream stream);

    template void SubtractVector<float>(float *x, const float *y, ui64 size, TCudaStream stream);

    template void SubtractVector<int>(int *x, const int *y, ui64 size, TCudaStream stream);

    template void SubtractVector<uint>(uint *x, const uint *y, ui64 size, TCudaStream stream);

    template void MultiplyVector<float>(float *x, const float *y, ui64 size, TCudaStream stream);

    template void MultiplyVector<int>(int *x, const int *y, ui64 size, TCudaStream stream);

    template void MultiplyVector<uint>(uint *x, const uint *y, ui64 size, TCudaStream stream);

    template void MultiplyVector<float>(float *x, const float y, ui64 size, TCudaStream stream);

    template void MultiplyVector<int>(int *x, const int y, ui64 size, TCudaStream stream);

    template void MultiplyVector<uint>(uint *x, const uint y, ui64 size, TCudaStream stream);

    template void DivideVector<float>(float *x, const float *y, ui64 size, TCudaStream stream);

    template void DivideVector<int>(int *x, const int *y, ui64 size, TCudaStream stream);

    template void DivideVector<uint>(uint *x, const uint *y, ui64 size, TCudaStream stream);

//    template void Reduce<float>(float *data, ui64 size, uint partCount, TCudaStream stream);

    template void ExpVector<float>(float *x, ui64 size, TCudaStream stream);

    template void ExpVector<double>(double *x, ui64 size, TCudaStream stream);

    template void Gather<int, uint>(int *dst, const int *src, const uint* map, ui64 size, TCudaStream stream);
    template void Gather<ui8, uint>(ui8 *dst, const ui8 *src, const uint* map, ui64 size, TCudaStream stream);

    template void Gather<uint, uint>(uint *dst, const uint *src, const uint* map, ui64 size, TCudaStream stream);

    template void Gather<uint2, uint>(uint2 *dst, const uint2 *src, const uint* map, ui64 size, TCudaStream stream);

    template void Gather<float, uint>(float *dst, const float *src, const uint* map, ui64 size, TCudaStream stream);

    template void Scatter<int, uint>(int *dst, const int *src, const uint* map, ui64 size, TCudaStream stream);

    template void Scatter<uint, uint>(uint *dst, const uint *src, const uint* map, ui64 size, TCudaStream stream);
    template void Scatter<ui8, uint>(ui8 *dst, const ui8 *src, const uint* map, ui64 size, TCudaStream stream);

    template void Scatter<uint2, uint>(uint2 *dst, const uint2 *src, const uint* map, ui64 size, TCudaStream stream);

    template void Scatter<float, uint>(float *dst, const float *src, const uint* map, ui64 size, TCudaStream stream);

    template void GatherWithMask<float, ui32>(float *dst, const float *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void ScatterWithMask<float, ui32>(float *dst, const float *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);


    template void GatherWithMask<int, ui32>(int *dst, const int *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);
    template void GatherWithMask<ui8, ui32>(ui8 *dst, const ui8 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void ScatterWithMask<int, ui32>(int* dst, const int *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);
    template void ScatterWithMask<ui8, ui32>(ui8* dst, const ui8 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void GatherWithMask<uint2, ui32>(uint2 *dst, const uint2 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void ScatterWithMask<uint2, ui32>(uint2 *dst, const uint2 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void GatherWithMask<ui32, ui32>(ui32 *dst, const ui32 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void ScatterWithMask<ui32, ui32>(ui32 *dst, const ui32 *src, const ui32* map, ui64 size, ui32 mask, TCudaStream stream);

    template void Reverse<char>(char *data, ui64 size, TCudaStream stream);

    template void Reverse<unsigned char>(unsigned char *data, ui64 size, TCudaStream stream);

    template void Reverse<short>(short *data, ui64 size, TCudaStream stream);

    template void Reverse<ushort>(ushort *data, ui64 size, TCudaStream stream);

    template void Reverse<int>(int *data, ui64 size, TCudaStream stream);

    template void Reverse<uint>(uint *data, ui64 size, TCudaStream stream);

    template void Reverse<float>(float *data, ui64 size, TCudaStream stream);

}
