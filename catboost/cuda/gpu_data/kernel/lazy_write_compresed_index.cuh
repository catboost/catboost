#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    struct TLazyWirteCompressedIndexKernelContext : public IKernelContext {
        TDevicePointer<ui8> TempStorage;
    };
}
