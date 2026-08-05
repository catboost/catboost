#pragma once

#include <util/system/types.h>

// hipCUB/rocPRIM's ThreadLoad default-constructs the loaded type in device code,
// so TDataPartition's constructor must be callable there. This header is
// included by plain host .cpp before any CUDA/HIP runtime, so annotate via a
// self-contained macro that is empty for the host compiler and host/device for
// nvcc/hipcc (avoids depending on __host__/__device__ being defined yet).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define Y_CUDA_HOST_DEVICE __host__ __device__
#else
#define Y_CUDA_HOST_DEVICE
#endif

struct TDataPartition {
    ui32 Offset;
    ui32 Size;

    Y_CUDA_HOST_DEVICE TDataPartition(ui32 offset = 0,
                   ui32 size = 0)
        : Offset(offset)
        , Size(size)
    {
    }
};

#undef Y_CUDA_HOST_DEVICE
