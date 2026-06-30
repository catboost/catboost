//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_TRAITS_H
#define _LIBCUDACXX___COMPLEX_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
inline constexpr bool __is_complex_float_v =
  _CCCL_TRAIT(is_floating_point, _Tp) || _CCCL_TRAIT(__is_extended_floating_point, _Tp);

template <class _Tp>
inline constexpr size_t __complex_alignment_v = 2 * sizeof(_Tp);

#define _LIBCUDACXX_COMPLEX_ALIGNAS _CCCL_ALIGNAS(__complex_alignment_v<_Tp>)

template <class _Tp>
struct __type_to_vector;

template <class _Tp>
using __type_to_vector_t = typename __type_to_vector<_Tp>::__type;

template <class _Tp, class = void>
inline constexpr bool __has_vector_type_v = false;

template <class _Tp>
inline constexpr bool __has_vector_type_v<_Tp, void_t<__type_to_vector_t<_Tp>>> = true;

template <class _Tp>
struct __abcd_results
{
  _Tp __ac;
  _Tp __bd;
  _Tp __ad;
  _Tp __bc;
};

template <class _Tp>
struct __ab_results
{
  _Tp __a;
  _Tp __b;
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__has_vector_type_v<_Tp>) )
_CCCL_API constexpr __abcd_results<_Tp> __complex_calculate_partials(_Tp __a, _Tp __b, _Tp __c, _Tp __d) noexcept
{
  return {__a * __c, __b * __d, __a * __d, __b * __c};
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__has_vector_type_v<_Tp>) )
_CCCL_API constexpr __ab_results<_Tp> __complex_piecewise_mul(_Tp __x1, _Tp __y1, _Tp __x2, _Tp __y2) noexcept
{
  return {__x1 * __x2, __y1 * __y2};
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__has_vector_type_v<_Tp>)
_CCCL_API constexpr __abcd_results<_Tp> __complex_calculate_partials(_Tp __a, _Tp __b, _Tp __c, _Tp __d) noexcept
{
  __abcd_results<_Tp> __ret;

  using _Vec = __type_to_vector_t<_Tp>;

  _Vec __first{__a, __b};
  _Vec __second{__c, __d};
  _Vec __second_flip{__d, __c};

  _Vec __ac_bd = __first * __second;
  _Vec __ad_bc = __first * __second_flip;

  __ret.__ac = __ac_bd.x;
  __ret.__bd = __ac_bd.y;
  __ret.__ad = __ad_bc.x;
  __ret.__bc = __ad_bc.y;

  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__has_vector_type_v<_Tp>)
_CCCL_API constexpr __ab_results<_Tp> __complex_piecewise_mul(_Tp __x1, _Tp __y1, _Tp __x2, _Tp __y2) noexcept
{
  __ab_results<_Tp> __ret;

  using _Vec = __type_to_vector_t<_Tp>;

  _Vec __v1{__x1, __y1};
  _Vec __v2{__x2, __y2};

  _Vec __result = __v1 * __v2;

  __ret.__a = __result.x;
  __ret.__b = __result.y;

  return __ret;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_TRAITS_H
