// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_DESTRUCT_N_H
#define _LIBCUDACXX___MEMORY_DESTRUCT_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __destruct_n
{
private:
  size_t __size_;

  template <class _Tp>
  _CCCL_API inline void __process(_Tp* __p, false_type) noexcept
  {
    for (size_t __i = 0; __i < __size_; ++__i, ++__p)
    {
      __p->~_Tp();
    }
  }

  template <class _Tp>
  _CCCL_API inline void __process(_Tp*, true_type) noexcept
  {}

  _CCCL_API inline void __incr(false_type) noexcept
  {
    ++__size_;
  }
  _CCCL_API inline void __incr(true_type) noexcept {}

  _CCCL_API inline void __set(size_t __s, false_type) noexcept
  {
    __size_ = __s;
  }
  _CCCL_API inline void __set(size_t, true_type) noexcept {}

public:
  _CCCL_API inline explicit __destruct_n(size_t __s) noexcept
      : __size_(__s)
  {}

  template <class _Tp>
  _CCCL_API inline void __incr() noexcept
  {
    __incr(integral_constant<bool, is_trivially_destructible<_Tp>::value>());
  }

  template <class _Tp>
  _CCCL_API inline void __set(size_t __s, _Tp*) noexcept
  {
    __set(__s, integral_constant<bool, is_trivially_destructible<_Tp>::value>());
  }

  template <class _Tp>
  _CCCL_API inline void operator()(_Tp* __p) noexcept
  {
    __process(__p, integral_constant<bool, is_trivially_destructible<_Tp>::value>());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_DESTRUCT_N_H
