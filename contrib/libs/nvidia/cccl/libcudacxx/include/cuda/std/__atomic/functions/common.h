//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_FUNCTIONS_COMMON_H
#define _LIBCUDACXX___ATOMIC_FUNCTIONS_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __atomic_ptr_skip
{
  static constexpr auto __skip = 1;
};

template <typename _Tp>
struct __atomic_ptr_skip<_Tp*>
{
  static constexpr auto __skip = sizeof(_Tp);
};

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __atomic_ptr_skip<_Tp[]>
{};
template <typename _Tp, int n>
struct __atomic_ptr_skip<_Tp[n]>
{};

template <typename _Tp>
using __atomic_ptr_skip_t = __atomic_ptr_skip<remove_cvref_t<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_FUNCTIONS_COMMON_H
