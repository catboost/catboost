#pragma once
#include <contrib/libs/cub/cub/thread/thread_load.cuh>
#include <contrib/libs/cub/cub/thread/thread_store.cuh>


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

        __forceinline__ __device__ T operator()(volatile T &left, volatile T &right) {
            return left + right;
        }
    };

    template <class T>
    struct TCudaMax {
        __forceinline__ __device__ T operator()(const T &left, const T &right) {
            return max(left, right);
        }

        __forceinline__ __device__ T operator()(volatile T &left, volatile T &right) {
            return max(left, right);
        }
    };

    template <class T, class TOp = TCudaAdd<T>>
    __forceinline__ __device__ T WarpReduce(int x, volatile T* data, int reduceSize, TOp op = TOp()) {
        #if __CUDA_ARCH__ >= 350
        T val = data[x];
        #pragma unroll
        for (int s = reduceSize >> 1; s > 0; s >>= 1)
        {
            val = op(val, __shfl_down(val, s));
        }
        if (x == 0) {
            data[x] = val;
        }
        return val;
        #else
        //unsafe optimization
        #pragma unroll
        for (int s = reduceSize >> 1; s > 0; s >>= 1)
        {
            if (x < s)
            {
                data[x] = op(data[x], data[x + s]);
            }
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

    template <class T>
    __forceinline__ __device__ T FastInBlockReduce(int x, volatile T* data, int reduceSize) {
        if (reduceSize > 32) {
            #pragma  unroll
            for (int s = reduceSize >> 1; s >= 32; s >>= 1) {
                if (x < s)
                {
                    data[x] += data[x + s];
                }
                __syncthreads();
            }
        }
        if (x < 32)
        {
            return WarpReduce(x, data, min(reduceSize, 32));
        } else {
            return 0;
        }
    }

    template <typename T>
    __forceinline__ __device__ T LdgWithFallback(const T* data, ui64 offset) {
        return cub::ThreadLoad<cub::LOAD_LDG>(data + offset);
    }


    #if __CUDA_ARCH__ < 350
    template <typename T>
    __forceinline__ __device__ T __ldg(const T* data) {
        return cub::ThreadLoad<cub::LOAD_LDG>(data);
    }
    #endif


    template <typename T>
    __forceinline__ __device__ T StreamLoad(const T* data) {
        return cub::ThreadLoad<cub::LOAD_CS>(data);
    }

    template <typename T>
    __forceinline__ __device__ void WriteThrough(T* data, T val) {
        cub::ThreadStore<cub::STORE_WT>(data, val);
    }

    template <class U, class V>
    struct TPair {
        U First;
        V Second;

        __host__ __device__ __forceinline__ TPair() = default;

        __host__ __device__ __forceinline__ TPair(const U& first, const V& second)
                : First(first)
                , Second(second) {

        }
    };

}
