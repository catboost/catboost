#pragma once
#include <cuda_runtime.h>
#include <util/system/types.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/remote_objects.h>

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

    template <class T>
    using TDevicePointer = NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaDevice>;

    template <class T>
    using THostPointer =  NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost>;
}

