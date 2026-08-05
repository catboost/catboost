#pragma once
// HIP compatibility shim: CUDA <math_constants.h> -> HIP math constants.
#include <hip/hip_math_constants.h>
#ifndef CUDART_PI_F
#define CUDART_PI_F HIP_PI_F
#endif
#ifndef CUDART_PI
#define CUDART_PI HIP_PI
#endif
#ifndef CUDART_INF_F
#define CUDART_INF_F HIP_INF_F
#endif
#ifndef CUDART_INF
#define CUDART_INF HIP_INF
#endif
