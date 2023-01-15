#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/remote_objects.h>

#include <library/cpp/cuda/wrappers/kernel.cuh>

namespace NKernel {
    using EPtrType = NCudaLib::EPtrType;

    struct IKernelContext {
        virtual ~IKernelContext() {
        }
    };

    template <class T>
    using TDevicePointer = NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaDevice>;

    template <class T>
    using THostPointer =  NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost>;

}
