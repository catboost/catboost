//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H
#define _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Prevent resetting of the diagnostic state by guarding the push/pop with a macro
#if _CCCL_HAS_NVFP16()
_CCCL_DIAG_PUSH
#  include <cuda_fp16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8()
_CCCL_DIAG_PUSH
#  include <cuda_fp8.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP8()

#if _CCCL_HAS_NVFP6()
_CCCL_DIAG_PUSH
#  include <cuda_fp6.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP6()

#if _CCCL_HAS_NVFP4()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_MSVC(4100) // unreferenced formal parameter
#  include <cuda_fp4.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP4()

#endif // _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H
