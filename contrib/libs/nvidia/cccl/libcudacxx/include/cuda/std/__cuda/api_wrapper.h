//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__STD__CUDA_API_WRAPPER_H
#define _CUDA__STD__CUDA_API_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/cuda_error.h>

#define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)                \
  do                                                        \
  {                                                         \
    const ::cudaError_t __status = _NAME(__VA_ARGS__);      \
    switch (__status)                                       \
    {                                                       \
      case ::cudaSuccess:                                   \
        break;                                              \
      default:                                              \
        ::cudaGetLastError(); /* clear CUDA error state */  \
        ::cuda::__throw_cuda_error(__status, _MSG, #_NAME); \
    }                                                       \
  } while (0)

#define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)                         \
  do                                                                    \
  {                                                                     \
    [[maybe_unused]] const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    ::cudaGetLastError(); /* clear CUDA error state */                  \
    _CCCL_ASSERT(__status == cudaSuccess, _MSG);                        \
  } while (0)

#define _CCCL_LOG_CUDA_API(_NAME, _MSG, ...)                                       \
  [&]() {                                                                          \
    const ::cudaError_t __status = _NAME(__VA_ARGS__);                             \
    if (__status != ::cudaSuccess)                                                 \
    {                                                                              \
      ::cuda::__detail::__msg_storage __msg_buffer;                                \
      ::cuda::__detail::__format_cuda_error(__msg_buffer, __status, _MSG, #_NAME); \
      ::fprintf(stderr, "%s\n", __msg_buffer.__buffer);                            \
      ::fflush(stderr);                                                            \
    }                                                                              \
    ::cudaGetLastError(); /* clear CUDA error state */                             \
    return __status;                                                               \
  }()

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
