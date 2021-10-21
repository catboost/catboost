#pragma once

#include <cuda.h>

#if CUDA_VERSION < 11000
#define _CUB_INCLUDE(x) <contrib/libs/nvidia/cub/x>
#else
#define _CUB_INCLUDE(x) <x>
#endif
