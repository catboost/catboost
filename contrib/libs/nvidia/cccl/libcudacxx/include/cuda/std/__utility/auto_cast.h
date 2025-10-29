// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_AUTO_CAST_H
#define _LIBCUDACXX___UTILITY_AUTO_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>

#if _CCCL_STD_VER < 2020 && _CCCL_COMPILER(MSVC)
#  define _LIBCUDACXX_AUTO_CAST(expr) (_CUDA_VSTD::decay_t<decltype((expr))>) (expr)
#else
#  define _LIBCUDACXX_AUTO_CAST(expr) static_cast<_CUDA_VSTD::decay_t<decltype((expr))>>(expr)
#endif

#endif // _LIBCUDACXX___UTILITY_AUTO_CAST_H
