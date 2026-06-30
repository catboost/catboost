//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_TYPES_H
#define __LIBCUDACXX___ATOMIC_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/types/base.h>
#include <cuda/std/__atomic/types/locked.h>
#include <cuda/std/__atomic/types/reference.h>
#include <cuda/std/__atomic/types/small.h>
#include <cuda/std/__type_traits/conditional.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __atomic_traits
{
  static constexpr bool __atomic_requires_lock      = !__atomic_is_always_lock_free<_Tp>::__value;
  static constexpr bool __atomic_requires_small     = sizeof(_Tp) < 4;
  static constexpr bool __atomic_supports_reference = __atomic_is_always_lock_free<_Tp>::__value && sizeof(_Tp) <= 8;
};

template <typename _Tp>
using __atomic_storage_t =
  _If<__atomic_traits<_Tp>::__atomic_requires_small,
      __atomic_small_storage<_Tp>,
      _If<__atomic_traits<_Tp>::__atomic_requires_lock, __atomic_locked_storage<_Tp>, __atomic_storage<_Tp>>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __LIBCUDACXX___ATOMIC_TYPES_H
