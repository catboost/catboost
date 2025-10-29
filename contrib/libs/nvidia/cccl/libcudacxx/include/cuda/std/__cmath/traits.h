// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_TRAITS_H
#define _LIBCUDACXX___CMATH_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// isgreater

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isgreater(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x > __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool isgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isgreater((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isgreater((type) __x, (type) __y);))
}

// isgreaterequal

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isgreaterequal(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x >= __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool isgreaterequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isgreaterequal((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isgreaterequal((type) __x, (type) __y);))
}

// isless

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isless(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x < __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool isless(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isless((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isless((type) __x, (type) __y);))
}

// islessequal

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_islessequal(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x <= __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool islessequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::islessequal((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_islessequal((type) __x, (type) __y);))
}

// islessgreater

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_islessgreater(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x < __y || __x > __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool islessgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::islessgreater((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_islessgreater((type) __x, (type) __y);))
}

// isunordered

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline bool isunordered(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  return _CUDA_VSTD::isnan((type) __x) || _CUDA_VSTD::isnan((type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_TRAITS_H
