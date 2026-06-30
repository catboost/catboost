#pragma once
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cooperative_groups.h>

// if ptxas warns like this
// 'Value of threads per SM for entry ... is out of range. .minnctapersm will be ignored'
// check that CUDA_MAX_THREADS_PER_SM is consistent with
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif

namespace NKernel {
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


    __device__ __forceinline__  float PositiveInfty()
    {
        return __int_as_float(0x7f800000);
    }

    __device__ __forceinline__   float NegativeInfty()
    {
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
    __forceinline__ __device__ T ShuffleReduce(int x, T val, int reduceSize, TOp op = TOp()) {
        __syncwarp();
        #pragma unroll
        for (int s = reduceSize >> 1; s > 0; s >>= 1) {
            val = op(val, __shfl_down_sync(0xFFFFFFFF, val, s));
        }
        return val;
    }

    template <class T, class TOp = TCudaAdd<T> >
    __forceinline__ __device__ T WarpReduce(int x, volatile T* data, int reduceSize, TOp op = TOp()) {
        __syncwarp();
        #if __CUDA_ARCH__ >= 350
        T val = data[x];

        #pragma unroll
        for (int s = reduceSize >> 1; s > 0; s >>= 1){
            val = op(val, __shfl_down_sync(0xFFFFFFFF, val, s));
        }
        if (x == 0) {
            data[x] = val;
        }
        __syncwarp();
        return val;
        #else
        //unsafe optimization
        #pragma unroll
        for (int s = reduceSize >> 1; s > 0; s >>= 1) {
            if (x < s) {
                const T tmp1 = data[x];
                const T tmp2 = data[x + s];
                data[x] = op(tmp1, tmp2);
            }
            __syncwarp();
        }
        return data[x];
        #endif
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
            return WarpReduce<T, TOp>(x, data, min(reduceSize, 32));
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

    template <typename T>
    __forceinline__ __device__ T LdgWithFallback(const T* data, ui64 offset) {
        return cub::ThreadLoad<cub::LOAD_LDG>(data + offset);
    }

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

    template <class U, class V>
    struct TPair {
        U First;
        V Second;

        __forceinline__ TPair() = default;

        __host__ __device__ __forceinline__ TPair(const U& first, const V& second)
                : First(first)
                , Second(second) {

        }
    };

    __forceinline__ __device__ bool NotZero(float val) {
        return fabs(val) > 1e-20f;
    }

    __forceinline__ __device__ unsigned int GetPairIndex(ui32 i, ui32 j) {
        return ((j * (j - 1)) >> 1) + i;
    }

    __forceinline__ __device__ uint2 GetPair(ui32 idx) {
        uint2 pair;
        pair.y = ui32((1.0f + sqrt(8.0f * idx + 1.0f)) * 0.5f);
        pair.x = idx - ((pair.y * (pair.y - 1)) >> 1);
        return pair;
    }

    __forceinline__ __device__ float ClipProb(float p) {
        return max(min(p, 1.0f - 1e-7f), 1e-7f);
    }

    template <class T>
    struct TAtomicAdd {
        static __forceinline__ __device__ T Add(T* dst, T val) {
            return atomicAdd(dst, val);
        }
    };

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
    };

    static __forceinline__ __host__ bool IsGridEmpty(const dim3& grid) {
        return grid.x == 0 || grid.y == 0 || grid.z == 0;
    }

}
