#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {


    struct TPair2 {
        uint2 x;
        uint2 y;
    };

    struct TPair4 {
        uint2 x;
        uint2 y;
        uint2 z;
        uint2 w;
    };

    __forceinline__  __device__ uint2 ZeroPair() {
        uint2 pair;
        pair.x = pair.y = 0;
        return pair;
    }
}
