#pragma once
#include <cuda_runtime.h>
#include <util/system/types.h>

namespace NKernel {

    typedef unsigned char uchar;
    typedef cudaStream_t TCudaStream;

    struct IKernelContext {
        virtual ~IKernelContext() {
        }
    };

    template <class T>
    __host__ __device__ inline T CeilDivide(T x, T y) {
        return (x + y - 1) / y;
    }
}

