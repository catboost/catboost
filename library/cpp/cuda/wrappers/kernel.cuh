#pragma once
#include <cuda_runtime.h>
#include <util/system/types.h>
#include <util/generic/array_ref.h>


namespace NKernel {

    typedef unsigned char uchar;
    typedef cudaStream_t TCudaStream;


    template <class T>
    __host__ __device__ inline T CeilDivide(T x, T y) {
        return (x + y - 1) / y;
    }

    template <class T>
    inline cudaError_t SetBitsToOneAsync(TArrayRef<T> data, TCudaStream stream) {
        return cudaMemsetAsync(reinterpret_cast<char*>(data.data()), 255, data.size() * sizeof(T), stream);
    }

    template <class T>
    inline cudaError_t ClearMemory(TArrayRef<T> data, cudaStream_t stream) {
        return cudaMemsetAsync(reinterpret_cast<char*>(data.data()), 0, data.size() * sizeof(T), stream);
    }

#if __CUDA_ARCH__ < 350
    template <typename T>
    __forceinline__ __device__ T __ldg(const T* data) {
        return data[0];
    }
#endif

}
