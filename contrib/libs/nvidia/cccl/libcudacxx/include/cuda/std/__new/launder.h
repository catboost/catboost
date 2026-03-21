// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_LAUNDER_H
#define _LIBCUDACXX___NEW_LAUNDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* launder(_Tp* __p) noexcept
{
  static_assert(!_CCCL_TRAIT(is_function, _Tp), "can't launder functions");
  static_assert(!_CCCL_TRAIT(is_same, void, remove_cv_t<_Tp>), "can't launder cv-void");
#if defined(_CCCL_BUILTIN_LAUNDER)
  return _CCCL_BUILTIN_LAUNDER(__p);
#else
  return __p;
#endif // _CCCL_BUILTIN_LAUNDER
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NEW_LAUNDER_H
