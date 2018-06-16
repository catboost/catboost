#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    struct TCubKernelContext : public IKernelContext {
        ui64 TempStorageSize = 0;
        TDevicePointer<char> TempStorage;
        bool Initialized = false;
    };

}
