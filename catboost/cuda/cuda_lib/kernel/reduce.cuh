#pragma once

#include "kernel.cuh"

namespace NKernel {

    template <typename T>
    void ReduceBinary(T* dst, const T* sourceLeft, const T* sourceRight, ui64 size, TCudaStream stream);


    template <class T>
    struct TKernelWithTempBufferContext : public IKernelContext {
        //if we don't have peer access
        TDevicePointer<T> TempBuffer;
    };

}


