/*
*  Copyright 2021 NVIDIA Corporation
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/

#pragma once
#pragma clang system_header


#include <cub/util_compiler.cuh>

/**
 * @def CUB_EXEC_CHECK_DISABLE
 * Wrapper around `#pragma nv_exec_check_disable`.
 */

// #pragma nv_exec_check_disable is only recognized by NVCC.
#if defined(__CUDACC__) && \
    !defined(_NVHPC_CUDA) && \
    !(defined(__CUDA__) && defined(__clang__))

#if CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC
#define CUB_EXEC_CHECK_DISABLE __pragma("nv_exec_check_disable")
#else // // !MSVC
#define CUB_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#endif // MSVC

#else // !NVCC

#define CUB_EXEC_CHECK_DISABLE

#endif // NVCC
