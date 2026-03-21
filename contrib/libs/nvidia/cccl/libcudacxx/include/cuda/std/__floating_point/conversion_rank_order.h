//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H
#define _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __fp_conv_rank_order
{
  __unordered,
  __greater,
  __equal,
  __less,
};

template <class _Lhs, class _Rhs>
[[nodiscard]] _CCCL_API constexpr __fp_conv_rank_order __fp_conv_rank_order_v_impl() noexcept
{
  if constexpr (__fp_is_subset_of_v<_Lhs, _Rhs> && __fp_is_subset_of_v<_Rhs, _Lhs>)
  {
#if _CCCL_HAS_LONG_DOUBLE()
    // If double and long double have the same properties, long double has the higher subrank
    if constexpr (__fp_is_subset_of_v<long double, double>)
    {
      if constexpr (_CCCL_TRAIT(is_same, _Lhs, long double) && !_CCCL_TRAIT(is_same, _Rhs, long double))
      {
        return __fp_conv_rank_order::__greater;
      }
      else if constexpr (!_CCCL_TRAIT(is_same, _Lhs, long double) && _CCCL_TRAIT(is_same, _Rhs, long double))
      {
        return __fp_conv_rank_order::__less;
      }
      else
      {
        return __fp_conv_rank_order::__equal;
      }
    }
    else
#endif // _CCCL_HAS_LONG_DOUBLE()
    {
      return __fp_conv_rank_order::__equal;
    }
  }
  else if constexpr (__fp_is_subset_of_v<_Rhs, _Lhs>)
  {
    return __fp_conv_rank_order::__greater;
  }
  else if constexpr (__fp_is_subset_of_v<_Lhs, _Rhs>)
  {
    return __fp_conv_rank_order::__less;
  }
  else
  {
    return __fp_conv_rank_order::__unordered;
  }
}

_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_fp, _Lhs) && _CCCL_TRAIT(__is_fp, _Rhs))
inline constexpr __fp_conv_rank_order __fp_conv_rank_order_v = __fp_conv_rank_order_v_impl<_Lhs, _Rhs>();

template <class _Lhs, class _Rhs>
inline constexpr __fp_conv_rank_order __fp_conv_rank_order_int_ext_v =
  __fp_conv_rank_order_v<conditional_t<_CCCL_TRAIT(is_integral, _Lhs), double, _Lhs>,
                         conditional_t<_CCCL_TRAIT(is_integral, _Rhs), double, _Rhs>>;

_CCCL_TEMPLATE(class _From, class _To)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_fp, _From) && _CCCL_TRAIT(__is_fp, _To))
inline constexpr bool __fp_is_implicit_conversion_v =
  __fp_conv_rank_order_v<_From, _To> == __fp_conv_rank_order::__less
  || __fp_conv_rank_order_v<_From, _To> == __fp_conv_rank_order::__equal;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H
