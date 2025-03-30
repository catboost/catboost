#include "transform.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/operators.cuh>

#include <util/generic/cast.h>

#include <cub/block/block_radix_sort.cuh>


namespace NKernel {


    template <typename T>
    __global__ void AddVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T y0 = __ldg(y + i);
            const T x0 = __ldg(x + i);
            const T r0 = y0 + x0;
            WriteThrough(x + i, r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void AddVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        AddVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }


    template <typename T>
    __global__ void AddVectorImpl(T *x, const T y, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T x0 = __ldg(x + i);
            const T r0 = y + x0;
            WriteThrough(x + i, r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void AddVector(T *x, const T y, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        AddVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template <typename T>
    __global__ void SubtractVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T y0 = __ldg(y + i);
            const T x0 = __ldg(x + i);
            const T r0 = x0 - y0;
            WriteThrough(x + i, r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    __global__ void SubtractVectorImpl(T *x, const T y, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            const T x0 = __ldg(x + i);
            const T r0 = x0 - y;
            WriteThrough(x + i, r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void SubtractVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        SubtractVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template <typename T>
    void SubtractVector(T *x, const T y, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        SubtractVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template <typename T>
    __global__ void MultiplyVectorImpl(T *x, const T *y, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;

        while (i < size) {
            const T y0 = __ldg(y + i);
            const T x0 = __ldg(x + i);
            const T r0 = y0 * x0;
            WriteThrough(x + i,  r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void MultiplyVector(T *x, const T *y, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        MultiplyVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, size);
    }

    template <typename T>
    __global__ void MultiplyVectorImpl(T *x, const T c, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T x0 = __ldg(x + i);
            T r0 = x0 * c;
            WriteThrough(x + i, r0);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void MultiplyVector(T *x, const T c, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        MultiplyVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, c, size);
    }


    template <typename T>
    __global__ void DivideVectorImpl(T *x, const T *y, bool skipZeroes, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T x0 = x[i];
            T y0 = y[i];
            T r0 = ZeroAwareDivide(x0, y0, skipZeroes);
            x[i] = r0;
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    __global__ void DivideVectorImpl(T *x, const T y, bool skipZeroes, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T x0 = x[i];
            T r0 = ZeroAwareDivide(x0, y, skipZeroes);
            x[i] = r0;
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void DivideVector(T *x, const T *y, ui64 size, bool skipZeroes, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        DivideVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, skipZeroes, size);
    }

    template <typename T>
    void DivideVector(T *x, const T y, ui64 size, bool skipZeroes, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize,
                                   (ui64)TArchProps::MaxBlockCount()));
        DivideVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, y, skipZeroes, size);
    }

    template <typename T>
    __global__ void ExpVectorImpl(T *x, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            T val = __ldg(x + i);
            x[i] = __expf(val);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void ExpVector(T *x, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));
        ExpVectorImpl<T> << < numBlocks, blockSize, 0, stream >> > (x, size);
    }

    template <typename T, typename Index>
    __global__ void GatherImpl(T *dst, const T *src, const Index *map, Index size,
                               int columnCount, ui64 dstColumnAlignSize, ui64 srcColumnAlignSize) {
        Index i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = __ldg(map + i);
            for (int column = 0; column < columnCount; ++column) {
                WriteThrough(dst + i + column * dstColumnAlignSize, StreamLoad(src + m + column * srcColumnAlignSize));
            }
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T, typename Index>
    void Gather(T *dst, const T *src, const Index* map, ui64 size, int columnCount, ui64 dstColumnAlignSize, ui64 srcColumnAlignSize, TCudaStream stream) {
        const ui64 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));

        if (numBlocks) {
            GatherImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, columnCount, dstColumnAlignSize, srcColumnAlignSize);
        }
    }


    template <typename T, typename Index>
    __global__ void GatherWithMaskImpl(T *dst, const T *src, const Index *map, Index size, Index mask) {
        Index i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i) & mask;
            WriteThrough(dst + i, StreamLoad(src + m));
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T, typename Index>
    void GatherWithMask(T *dst, const T *src, const Index* map, ui64 size, Index mask, TCudaStream stream) {
        const ui64 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));

        if (numBlocks) {
            GatherWithMaskImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, mask);
        }
    }


    template <typename T, typename Index>
    __global__ void ScatterImpl(T* dst, const T* src, const Index* map, Index size, int columnCount, ui64 dstColumnAlignSize, ui64 srcColumnALignSize) {
        Index i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = __ldg(map + i);
            for (int column = 0; column < columnCount; ++column) {
                WriteThrough(dst + m + dstColumnAlignSize * column, StreamLoad(src + i + srcColumnALignSize * column));
            }
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T, typename Index>
    void Scatter(T *dst, const T *src, const Index* map, ui64 size,  int columnCount, ui64 dstColumnAlignSize, ui64 srcColumnAlignSize, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));
        if (numBlocks) {
            ScatterImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, columnCount, dstColumnAlignSize, srcColumnAlignSize);
        }
    }


    template <typename T, typename Index>
    __global__ void ScatterWithMaskImpl(T* dst, const T* src, const Index* map, Index size, Index mask) {
        Index i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            Index m = StreamLoad(map + i) & mask;
            WriteThrough(dst + m, StreamLoad(src + i));
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T, typename Index>
    void ScatterWithMask(T *dst, const T *src, const Index* map, ui64 size, Index mask, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min((size + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));
        if (numBlocks) {
            ScatterWithMaskImpl<T, Index> << < numBlocks, blockSize, 0, stream >> > (dst, src, map, (Index)size, mask);
        }
    }

    template <typename T>
    __global__ void ReverseImpl(T *data, ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        ui64 half = size / 2;
        while (i < half) {
            T a = data[i];
            T b = data[size - i - 1];
            data[i] = b;
            data[size - i - 1] = a;
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void Reverse(T* data, ui64 size, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = SafeIntegerCast<ui32>(min(((size + 1) / 2 + blockSize - 1) / blockSize, (ui64)TArchProps::MaxBlockCount()));
        ReverseImpl<T> << < numBlocks, blockSize, 0, stream >> > (data, size);
    }




    #define BIN_OP_VECTOR_TEMPL(Type) \
    template void AddVector<Type>(Type *x, const Type *y, ui64 size, TCudaStream stream);\
    template void AddVector<Type>(Type *x, Type y, ui64 size, TCudaStream stream);\
    template void SubtractVector<Type>(Type *x, const Type *y, ui64 size, TCudaStream stream);\
    template void SubtractVector<Type>(Type *x, Type y, ui64 size, TCudaStream stream); \
    template void MultiplyVector<Type>(Type *x, const Type* y, ui64 size, TCudaStream stream);\
    template void MultiplyVector<Type>(Type *x, Type y, ui64 size, TCudaStream stream);\
    template void DivideVector<Type>(Type *x, const Type* y, ui64 size, bool skipZeroes, TCudaStream stream);\
    template void DivideVector<Type>(Type *x, Type y, ui64 size, bool skipZeroes, TCudaStream stream);\


    BIN_OP_VECTOR_TEMPL(int)
    BIN_OP_VECTOR_TEMPL(float)
    BIN_OP_VECTOR_TEMPL(ui32)
    BIN_OP_VECTOR_TEMPL(double)
    BIN_OP_VECTOR_TEMPL(ui8)
    BIN_OP_VECTOR_TEMPL(uint2)
    BIN_OP_VECTOR_TEMPL(ui16)

    #define FUNC_VECTOR_TEMPL(Type) \
    template void ExpVector<Type>(Type *x, ui64 size, TCudaStream stream);\

    FUNC_VECTOR_TEMPL(float)



    #define GATHER_SCATTER_TEMPL(Type, IndexType) \
    template void Gather<Type, IndexType>(Type *dst, const Type *src, const IndexType* map, ui64 size, int columntCount, ui64, ui64, TCudaStream stream); \
    template void Scatter<Type, IndexType>(Type *dst, const Type *src, const IndexType* map, ui64 size, int, ui64, ui64, TCudaStream stream); \
    template void GatherWithMask<Type, IndexType>(Type *dst, const Type *src, const IndexType* map, ui64 size, IndexType mask, TCudaStream stream); \
    template void ScatterWithMask<Type, IndexType>(Type *dst, const Type *src, const IndexType* map, ui64 size, IndexType mask, TCudaStream stream);


    GATHER_SCATTER_TEMPL(int, ui32)
    GATHER_SCATTER_TEMPL(ui8, ui32)
    GATHER_SCATTER_TEMPL(uint2, ui32)
    GATHER_SCATTER_TEMPL(uint2, ui64)
    GATHER_SCATTER_TEMPL(ui32, ui32)
    GATHER_SCATTER_TEMPL(float, ui32)
    GATHER_SCATTER_TEMPL(float, ui64)
    GATHER_SCATTER_TEMPL(bool, ui32)

    #define REVERSE_VECTOR_TEMPL(Type) \
    template void Reverse<Type>(Type *x, ui64 size, TCudaStream stream);

    REVERSE_VECTOR_TEMPL(char)
    REVERSE_VECTOR_TEMPL(float)
    REVERSE_VECTOR_TEMPL(unsigned char)
    REVERSE_VECTOR_TEMPL(short)
    REVERSE_VECTOR_TEMPL(ui16)
    REVERSE_VECTOR_TEMPL(int)
    REVERSE_VECTOR_TEMPL(ui32)

    // PowVector

    template <typename T>
    __global__ void PowVectorImpl(T* const x, const T base, const ui64 size) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            x[i] = pow(base, x[i]);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void PowVector(T* const x, const ui64 size, const T base, const TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(Min(
            (size + blockSize - 1) / blockSize,
            (ui64)TArchProps::MaxBlockCount()));
        PowVectorImpl<T><<<numBlocks, blockSize, 0, stream>>>(x, base, size);
    }

#define Y_CATBOOST_CUDA_F_IMPL(T) \
        template void PowVector<T>(T* x, ui64 size, T base, TCudaStream stream);

    Y_MAP_ARGS(
        Y_CATBOOST_CUDA_F_IMPL,
        float);

#undef Y_CATBOOST_CUDA_F_IMPL

    // PowVector

    template <typename T>
    __global__ void PowVectorImpl(const T* const x, const T base, const ui64 size, T* y) {
        ui64 i = (ui64)blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            y[i] = pow(base, x[i]);
            i += (ui64)gridDim.x * blockDim.x;
        }
    }

    template <typename T>
    void PowVector(const T* x, const ui64 size, const T base, T* y, const TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 numBlocks = SafeIntegerCast<ui32>(Min(
            (size + blockSize - 1) / blockSize,
            (ui64)TArchProps::MaxBlockCount()));
        PowVectorImpl<T><<<numBlocks, blockSize, 0, stream>>>(x, base, size, y);
    }

#define Y_CATBOOST_CUDA_F_IMPL(T) \
        template void PowVector<T>(const T* x, ui64 size, T base, T* y, TCudaStream stream);

    Y_MAP_ARGS(
        Y_CATBOOST_CUDA_F_IMPL,
        float);

#undef Y_CATBOOST_CUDA_F_IMPL
}
