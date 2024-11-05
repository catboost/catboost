#pragma once

#include "kernel.cuh"
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cooperative_groups.h>

template<int Alignment, typename T>
__forceinline__ __device__ __host__ T AlignBy(T x) {
    return NKernel::CeilDivide<size_t>(x, Alignment) * Alignment;
}

__forceinline__ __device__ int Align32(int i) {
    return AlignBy<32>(i);
}

template <class T>
__device__ __forceinline__ bool ExtractSignBit(T val)  {
    static_assert(sizeof(T) == sizeof(ui32), "Error: this works only for 4byte types");
    return (*reinterpret_cast<ui32*>(&val)) >> 31;
}

template <>
__device__ __forceinline__ bool ExtractSignBit(double val)  {
    return (*reinterpret_cast<ui64*>(&val)) >> 63;
}

template <class T>
__device__ __forceinline__ T OrSignBit(T val, bool flag)  {
    static_assert(sizeof(T) == sizeof(ui32), "Error: this works only for 4byte types");
    ui32 raw = (*reinterpret_cast<ui32*>(&val) | (flag << 31));
    return *reinterpret_cast<T*>(&raw);
}

template <>
__device__ __forceinline__ double OrSignBit(double val, bool flag)  {
    ui32 raw = (*reinterpret_cast<ui64*>(&val) | (static_cast<ui64>(flag) << 63));
    return *reinterpret_cast<ui64*>(&raw);
}

__forceinline__  __device__ ui32 RotateRight(ui32 bin, int bits) {
    return (bin << bits) | (bin >> (32 - bits));
}


__device__ __forceinline__  float PositiveInfty() {
    return __int_as_float(0x7f800000);
}

__device__ __forceinline__   float NegativeInfty() {
    return -PositiveInfty();
}

template <class T>
struct TCudaAdd {
    __forceinline__ __device__ T operator()(const T &left, const T &right) {
        return left + right;
    }

};

template <class T>
struct TCudaMultiply {
    __forceinline__ __device__ T operator()(const T &left, const T &right) {
        return left * right;
    }
};

template <class T>
struct TCudaMax {
    __forceinline__ __device__ T operator()(const T &left, const T &right) {
        return max(left, right);
    }

};

template <class T, class TOp = TCudaAdd<T> >
__forceinline__ __device__ T WarpReduce(int x, T val, int reduceSize, TOp op = TOp()) {
    __syncwarp();
    #pragma unroll
    for (int s = reduceSize >> 1; s > 0; s >>= 1) {
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, s));
    }
    return val;
}


template <typename T, int BLOCK_SIZE>
__forceinline__ __device__ T Reduce(volatile T* data) {
    const int x = threadIdx.x;

    #pragma  unroll
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (x < s)
        {
            data[x] += data[x + s];
        }
        __syncthreads();
    }
    T result = data[0];
    __syncthreads();
    return result;
}

template <class T,  class TOp = TCudaAdd<T>>
__forceinline__ __device__ T FastInBlockReduce(int x, volatile T* data, int reduceSize) {
    if (reduceSize > 32) {
        TOp op;

        #pragma  unroll
        for (int s = reduceSize >> 1; s >= 32; s >>= 1) {
            if (x < s) {
                T tmp1 = data[x];
                T tmp2 = data[x + s];
                data[x] = op(tmp1, tmp2);
            }
            __syncthreads();
        }
    }
    if (x < 32) {
        return WarpReduce<T, TOp>(x, data[x], min(reduceSize, 32));
    } else {
        return 0;
    }
}


template <class T, int TileSize, class TOp = TCudaAdd<T>>
__forceinline__ __device__ T TileReduce(cooperative_groups::thread_block_tile<TileSize> tile, const T threadValue) {
    tile.sync();
    T val = threadValue;
    TOp op;
    for (int s = tile.size() / 2; s > 0; s >>= 1) {
        val = op(val, tile.shfl_down(val, s));
    }
    return val;
};


template <typename T, typename TOffset = int>
__forceinline__ __device__ T Ldg(const T* data, TOffset offset = 0) {
    return cub::ThreadLoad<cub::LOAD_LDG>(data + offset);
}


template <typename T>
__forceinline__ __device__ T StreamLoad(const T* data) {
    return cub::ThreadLoad<cub::LOAD_CS>(data);
}

template <typename T>
__forceinline__ __device__ void WriteThrough(T* data, T val) {
    cub::ThreadStore<cub::STORE_WT>(data, val);
}

template <typename T>
__forceinline__ __device__ void StoreCS(T* data, T val) {
    cub::ThreadStore<cub::STORE_CS>(data, val);
}

template <class T>
struct TAtomicAdd {
    static __forceinline__ __device__ T Add(T* dst, T val) {
        return atomicAdd(dst, val);
    }
    static __forceinline__ __device__ T AddBlock(T* dst, T val) {
#if __CUDA_ARCH__ < 600
        return TAtomicAdd<T>::Add(dst, val);
#else
        return atomicAdd_block(dst, val);
#endif
    }
};

/*
 * For old devices
 */
template <>
struct TAtomicAdd<double> {
    static __forceinline__ __device__ double Add(double* address, double val) {
        #if __CUDA_ARCH__ < 600
        unsigned long long int* address_as_ull =
                (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                                 __longlong_as_double(assumed)));

            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
        #else
        return atomicAdd(address, val);
        #endif
    }
    static __forceinline__ __device__ double AddBlock(double* dst, double val) {
#if __CUDA_ARCH__ < 600
        return TAtomicAdd<double>::Add(dst, val);
#else
        return atomicAdd_block(dst, val);
#endif
    }
};


template <int BlockSize>
__forceinline__ __device__ void WriteToGlobalMemory(const float* values, float* dst, int dim) {
    for (int i = threadIdx.x; i < dim; i += BlockSize) {
        WriteThrough(dst + i,  values[i]);
    }
    __syncthreads();
}

__forceinline__ __device__ void Swap(float** left, float** right) {
    float* tmp = *left;
    *left = *right;
    *right = tmp;
}

template <int TileSize, class TOp = TCudaAdd<float>, typename TReduceType = float4>
__forceinline__ __device__ TReduceType TileReduce4(cooperative_groups::thread_block_tile<TileSize> tile, const TReduceType threadValue) {
    TOp op;
    tile.sync();
    TReduceType val = threadValue;
    for (int s = tile.size() / 2; s > 0; s >>= 1) {
        val.x = op(val.x, tile.shfl_down(val.x, s));
        val.y = op(val.y, tile.shfl_down(val.y, s));
        val.z = op(val.z, tile.shfl_down(val.z, s));
        val.w = op(val.w, tile.shfl_down(val.w, s));
    }
    return val;
};


template <int TileSize, class TOp = TCudaAdd<float>>
__forceinline__ __device__ float4 WarpReduce4(const float4 threadValue) {
    #define FULL_MASK 0xffffffff
    TOp op;
    __syncwarp();
    float4 val = threadValue;
    for (int s = TileSize / 2; s > 0; s >>= 1) {
        val.x = op(val.x, __shfl_down_sync(FULL_MASK, val.x, s));
        val.y = op(val.y, __shfl_down_sync(FULL_MASK, val.y, s));
        val.z = op(val.z, __shfl_down_sync(FULL_MASK, val.z, s));
        val.w = op(val.w, __shfl_down_sync(FULL_MASK, val.w, s));
    }
    return val;
};



template <int TileSize>
struct TTileReducer {
    template <class T>
    __forceinline__ __device__ static T Reduce(T val) {
        auto tile = cooperative_groups::tiled_partition<TileSize>(cooperative_groups::this_thread_block());;
        return TileReduce<T, TileSize>(tile, val);
    }
};

#define UNIMPL(K)\
    template <>\
    struct TTileReducer<K> {\
        template <class T>\
        __forceinline__ __device__ static T Reduce(T val) {\
            asm("trap;");\
            return 0;\
        }\
    };
UNIMPL(64)
UNIMPL(128)
UNIMPL(256)
#undef UNIMPL




__forceinline__ __device__ float4 operator*(float4 left, float4 right) {
    float4 result;
    result.x = left.x * right.x;
    result.y = left.y * right.y;
    result.z = left.z * right.z;
    result.w = left.w * right.w;
    return result;
}

__forceinline__ __device__ float4 operator*(float4 left, float scale) {
    float4 result;
    result.x = left.x * scale;
    result.y = left.y * scale;
    result.z = left.z * scale;
    result.w = left.w * scale;
    return result;
}

__forceinline__ __device__ float4 operator+(float4 left, float4 right) {
    float4 result;
    result.x = left.x + right.x;
    result.y = left.y + right.y;
    result.z = left.z + right.z;
    result.w = left.w + right.w;
    return result;
}

__forceinline__ __device__ float4 operator/(float4 left, float4 right) {
    float4 result;
    result.x = left.x / right.x;
    result.y = left.y / right.y;
    result.z = left.z / right.z;
    result.w = left.w / right.w;
    return result;
}

__forceinline__ __device__ float4 operator-(float4 left, float4 right) {
    float4 result;
    result.x = left.x - right.x;
    result.y = left.y - right.y;
    result.z = left.z - right.z;
    result.w = left.w - right.w;
    return result;
}


__forceinline__ __device__ float4 operator+(float4 left, float right) {
    float4 result;
    result.x = left.x + right;
    result.y = left.y + right;
    result.z = left.z + right;
    result.w = left.w + right;
    return result;
}

__forceinline__ __device__ float4 operator/(float4 left, float right) {
    float4 result;
    result.x = left.x / right;
    result.y = left.y / right;
    result.z = left.z / right;
    result.w = left.w / right;
    return result;
}

__forceinline__ __device__ float4 operator-(float4 left, float right) {
    float4 result;
    result.x = left.x - right;
    result.y = left.y - right;
    result.z = left.z - right;
    result.w = left.w - right;
    return result;
}

__forceinline__ __device__ float4& operator+=(float4& left, float4 right) {
    left = left + right;
    return left;
}

__forceinline__ __device__ float4& operator-=(float4& left, float4 right) {
    left = left - right;
    return left;
}

__forceinline__ __device__ float4& operator+=(float4& left, float right) {
    left = left + right;
    return left;
}

__forceinline__ __device__ float4& operator-=(float4& left, float right) {
    left = left - right;
    return left;
}

__forceinline__ __device__ float4& operator*=(float4& left, float4 right) {
    left = left * right;
    return left;
}

__forceinline__ __device__ float4& operator*=(float4& left, float right) {
    left = left * right;
    return left;
}

__forceinline__ __device__ float4& operator/=(float4& left, float4 right) {
    left = left / right;
    return left;
}

__forceinline__ __device__ float4& operator/=(float4& left, float right) {
    left = left / right;
    return left;
}


__forceinline__ __device__ float Reduce4(float4 val) {
    return (val.x + val.y) + (val.z + val.w);
}


__forceinline__ __device__ float Dot4(float4 left, float4 right) {
//    return fmaf(left.x, right.x, left.y * right.y) + fmaf(left.z, right.z , left.w * right.w);
    return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
}

__forceinline__ __device__ float4 FMA4(float alpha, float4 x, float4 y) {
    float4 result;
    result.x = fmaf(alpha, x.x, y.x);
    result.y = fmaf(alpha, x.y, y.y);
    result.z = fmaf(alpha, x.z, y.z);
    result.w = fmaf(alpha, x.w, y.w);
    return result;
}

__forceinline__ __device__ float4 FMA4(float4 alpha, float4 x, float4 y) {
    float4 result;
    result.x = fmaf(alpha.x, x.x, y.x);
    result.y = fmaf(alpha.y, x.y, y.y);
    result.z = fmaf(alpha.z, x.z, y.z);
    result.w = fmaf(alpha.w, x.w, y.w);
    return result;
}

__forceinline__ __device__ float4 Max4(float4 left, float right) {
    float4 result;
    result.x = max(left.x, right);
    result.y = max(left.y, right);
    result.z = max(left.z, right);
    result.w = max(left.w, right);
    return result;
}

__forceinline__ __device__ float4 Sqrt4(float4 left) {
    float4 result;
    result.x = sqrtf(left.x);
    result.y = sqrtf(left.y);
    result.z = sqrtf(left.z);
    result.w = sqrtf(left.w);
    return result;
}


__forceinline__ __device__ float4 BroadCast4(float val) {
    float4 result;
    result.x = result.y = result.z = result.w = val;
    return result;
}




template <int MaxDim>
__forceinline__ __device__ void Float4ToSharedMemory(float4 val, float* sharedMemory, int i) {
    sharedMemory[i] = val.x;
    sharedMemory[i + MaxDim] = val.y;
    sharedMemory[i + MaxDim * 2] = val.z;
    sharedMemory[i + MaxDim * 3] = val.w;
}

template <int MaxDim>
__forceinline__ __device__ float4 Float4FromSharedMemory(const float* sharedMemory, int i) {
    float4 val;
    val.x =  sharedMemory[i];
    val.y =  sharedMemory[i + MaxDim];
    val.z =  sharedMemory[i + MaxDim * 2];
    val.w =  sharedMemory[i + MaxDim * 3];
    return val;
}


__forceinline__ __device__ float4 Float4FromSharedMemory(int maxDim, const float* sharedMemory, int i) {
    float4 val;
    val.x =  sharedMemory[i];
    val.y =  sharedMemory[i + maxDim];
    val.z =  sharedMemory[i + maxDim * 2];
    val.w =  sharedMemory[i + maxDim * 3];
    return val;
}


__forceinline__ __device__ float4 LoadFloat4(const float* memory, int index, int lineSize) {
    float4 result;
    result.x = memory[index];
    result.y = memory[index + lineSize];
    result.z = memory[index + lineSize * 2];
    result.w = memory[index + lineSize * 3];
    return result;
}


template <int BlockSize,
          class TLeftLoader,
          class TRightLoader>
__forceinline__ __device__ float4 DotProduct4(
    const TLeftLoader& left,
    const TRightLoader& right,
    int dim,
    float* tmp) {

    float4 sum = BroadCast4(0.0f);

    for (int i = threadIdx.x; i < dim; i += BlockSize) {
        float4 l = left.Load(i);
        float4 r = right.Load(i);
        sum = FMA4(l, r, sum);
    }

    Float4ToSharedMemory<BlockSize>(sum, tmp, threadIdx.x);

    __syncthreads();

    for (int s = BlockSize / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                tmp[threadIdx.x + k * BlockSize] += tmp[threadIdx.x + s + k * BlockSize];
            }
        }
        __syncthreads();
    }

    float4 result = Float4FromSharedMemory<BlockSize>(tmp, 0);
    __syncthreads();
    return result;
}



template <int BlockSize,
          int N,
          class T,
          class TOp = TCudaAdd<T>>
__forceinline__ __device__ void WarpReduceN(int x, volatile T* data, int reduceSize, TOp op = TOp()) {
    __syncwarp();

    float val[N];
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        val[k] = data[x + k * BlockSize];
    }

    #pragma unroll
    for (int s = reduceSize >> 1; s > 0; s >>= 1) {
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            val[k] = op(val[k], __shfl_down_sync(0xFFFFFF, val[k], s));
        }
    }

    if (x == 0) {
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            data[k * BlockSize] = val[k];
        }
    }
}


template <int BlockSize,
          int N,
          class T,
          class TOp = TCudaAdd<T>>
__forceinline__ __device__ void BlockReduceN(volatile T* data, int reduceSize, TOp op = TOp()) {
    if (reduceSize > 32) {

        #pragma  unroll
        for (int s = reduceSize >> 1; s >= 32; s >>= 1) {
            if (threadIdx.x < s) {
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    T tmp1 = data[threadIdx.x + k * BlockSize];
                    T tmp2 = data[threadIdx.x + k * BlockSize + s];
                    data[threadIdx.x + k * BlockSize] = op(tmp1, tmp2);
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x < 32) {
        WarpReduceN<BlockSize, N>(threadIdx.x, data, min(reduceSize, 32), op);
    }
}



__forceinline__ __device__ int RoundUpToPowerTwo(int dim) {
    return 1 << (33 - __clz(dim));
}


template <int BlockSize, class TLoader>
__forceinline__ __device__ float4 ComputeSum2(
    TLoader& inputProvider,
    float* tmp,
    int dim) {

    float4 sum2 = BroadCast4(0.0f);

    for (int i = threadIdx.x; i < dim; i += BlockSize) {
        float4 point = inputProvider.Load(i);
        point = point * point;
        sum2 = sum2 + point;
    }

    Float4ToSharedMemory<BlockSize>(sum2, tmp, threadIdx.x);

    __syncthreads();

    for (int s = BlockSize / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                tmp[threadIdx.x + k * BlockSize] += tmp[threadIdx.x + s + k * BlockSize];
            }
        }
        __syncthreads();
    }

    float4 result;
    result.x = tmp[0];
    result.y = tmp[BlockSize];
    result.z = tmp[BlockSize * 2];
    result.w = tmp[BlockSize * 3];
    __syncthreads();
    return result;
}


__forceinline__ __device__ float AtomicFMA(float* dst, float alpha, float val) {
    int* dst_as_int = (int*)(dst);

    int assumed = *dst_as_int;
    int old = assumed;

    do {
        assumed = old;
        float newVal = alpha * __int_as_float(assumed) + val;
        old = atomicCAS(dst_as_int, assumed, __float_as_int(newVal));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __int_as_float(old);
}



template <int BlockSize>
__forceinline__ __device__ float4 SharedReduce4(float4 val, float* tmp) {
    Float4ToSharedMemory<BlockSize>(val, tmp, threadIdx.x);
    __syncthreads();
    if (BlockSize > 32) {
        for (int s = BlockSize / 2; s >= 32; s >>= 1) {
            if (threadIdx.x < s) {
                for (int k = 0; k < 4; ++k) {
                    tmp[threadIdx.x + BlockSize * k] += tmp[threadIdx.x + s + BlockSize * k];
                }
            }
            __syncthreads();
        }
    }
    for (int s = 16; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int k = 0; k < 4; ++k) {
                tmp[threadIdx.x + BlockSize * k] += tmp[threadIdx.x + s + BlockSize * k];
            }
        }
        __syncwarp();

    }
    __syncthreads();
    float4 result = Float4FromSharedMemory<BlockSize>(tmp, 0);
    __syncthreads();
    return result;
}


template <int BlockSize>
__forceinline__ __device__ void SharedReduce8(float4* val0, float4* val1, float* tmp) {
    Float4ToSharedMemory<BlockSize>(*val0, tmp, threadIdx.x);
    Float4ToSharedMemory<BlockSize>(*val1, tmp + 4 * BlockSize, threadIdx.x);
    __syncthreads();
    if (BlockSize > 32) {
        for (int s = BlockSize / 2; s >= 32; s >>= 1) {
            if (threadIdx.x < s) {
                for (int k = 0; k < 8; ++k) {
                    tmp[threadIdx.x + BlockSize * k] += tmp[threadIdx.x + s + BlockSize * k];
                }
            }
            __syncthreads();
        }
    }
    for (int s = 16; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            for (int k = 0; k < 8; ++k) {
                tmp[threadIdx.x + BlockSize * k] += tmp[threadIdx.x + s + BlockSize * k];
            }
        }
        __syncwarp();

    }
    __syncthreads();
    (*val0) = Float4FromSharedMemory<BlockSize>(tmp, 0);
    (*val1) = Float4FromSharedMemory<BlockSize>(tmp + 4 * BlockSize, 0);
    __syncthreads();
}


template <int BlockSize>
__forceinline__ __device__ void SharedPartReduce4(float4 val0, float4 val1, float* tmp, int tileSize) {
    Float4ToSharedMemory<BlockSize>(val0, tmp, threadIdx.x);
    Float4ToSharedMemory<BlockSize>(val1, tmp + 4 * BlockSize, threadIdx.x);
    __syncthreads();

    for (int s = BlockSize / 2; s >= tileSize; s >>= 1) {
        if (threadIdx.x < s) {
            for (int k = 0; k < 8; ++k) {
                tmp[threadIdx.x + BlockSize * k] += tmp[threadIdx.x + s + BlockSize * k];
            }
        }
        __syncthreads();
    }
}



__forceinline__ __device__ float2 operator*(float2 left, float2 right) {
    float2 result;
    result.x = left.x * right.x;
    result.y = left.y * right.y;
    return result;
}

__forceinline__ __device__ float2 operator*(float2 left, float scale) {
    float2 result;
    result.x = left.x * scale;
    result.y = left.y * scale;
    return result;
}

__forceinline__ __device__ float2 operator+(float2 left, float2 right) {
    float2 result;
    result.x = left.x + right.x;
    result.y = left.y + right.y;
    return result;
}

__forceinline__ __device__ float2 operator/(float2 left, float2 right) {
    float2 result;
    result.x = left.x / right.x;
    result.y = left.y / right.y;
    return result;
}

__forceinline__ __device__ float2 operator-(float2 left, float2 right) {
    float2 result;
    result.x = left.x - right.x;
    result.y = left.y - right.y;
    return result;
}


__forceinline__ __device__ float2 operator+(float2 left, float right) {
    float2 result;
    result.x = left.x + right;
    result.y = left.y + right;
    return result;
}

__forceinline__ __device__ float2 operator/(float2 left, float right) {
    float2 result;
    result.x = left.x / right;
    result.y = left.y / right;
    return result;
}

__forceinline__ __device__ float2 operator-(float2 left, float right) {
    float2 result;
    result.x = left.x - right;
    result.y = left.y - right;
    return result;
}

__forceinline__ __device__ float2& operator+=(float2& left, float2 right) {
    left = left + right;
    return left;
}

__forceinline__ __device__ float2& operator-=(float2& left, float2 right) {
    left = left - right;
    return left;
}

__forceinline__ __device__ float2& operator+=(float2& left, float right) {
    left = left + right;
    return left;
}

__forceinline__ __device__ float2& operator-=(float2& left, float right) {
    left = left - right;
    return left;
}

__forceinline__ __device__ float2& operator*=(float2& left, float2 right) {
    left = left * right;
    return left;
}

__forceinline__ __device__ float2& operator*=(float2& left, float right) {
    left = left * right;
    return left;
}

__forceinline__ __device__ float2& operator/=(float2& left, float2 right) {
    left = left / right;
    return left;
}

__forceinline__ __device__ float2& operator/=(float2& left, float right) {
    left = left / right;
    return left;
}


__forceinline__ __device__ float2 Sqrt2(float2 left) {
    float2 result;
    result.x = sqrtf(left.x);
    result.y = sqrtf(left.y);
    return result;
}
