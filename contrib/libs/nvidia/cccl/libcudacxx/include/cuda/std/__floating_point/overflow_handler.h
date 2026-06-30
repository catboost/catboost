//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_OVERFLOW_HANDLER_H
#define _LIBCUDACXX___FLOATING_POINT_OVERFLOW_HANDLER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__type_traits/always_false.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __fp_overflow_handler_kind
{
  __no_sat,
  __sat_finite,
};

template <__fp_overflow_handler_kind _Kind>
struct __fp_overflow_handler;

// __fp_overflow_handler<__no_sat>

template <>
struct __fp_overflow_handler<__fp_overflow_handler_kind::__no_sat>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_API static constexpr _Tp __handle_overflow() noexcept
  {
    constexpr auto __fmt = __fp_format_of_v<_Tp>;

    if constexpr (__fp_has_inf_v<__fmt>)
    {
      return _CUDA_VSTD::__fp_inf<_Tp>();
    }
    else if constexpr (__fp_has_nan_v<__fmt>)
    {
      return _CUDA_VSTD::__fp_nan<_Tp>();
    }
    else if constexpr (__fmt == __fp_format::__fp6_nv_e2m3 || __fmt == __fp_format::__fp6_nv_e3m2
                       || __fmt == __fp_format::__fp4_nv_e2m1)
    {
      // NaN is converted to positive max value
      return _CUDA_VSTD::__fp_max<_Tp>();
    }
    else
    {
      static_assert(__always_false_v<_Tp>, "Unhandled floating-point format");
    }
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API static constexpr _Tp __handle_underflow() noexcept
  {
    constexpr auto __fmt = __fp_format_of_v<_Tp>;

    if constexpr (__fp_has_inf_v<__fmt>)
    {
      return _CUDA_VSTD::__fp_neg(_CUDA_VSTD::__fp_inf<_Tp>());
    }
    else if constexpr (__fp_has_nan_v<__fmt>)
    {
      return _CUDA_VSTD::__fp_neg(_CUDA_VSTD::__fp_nan<_Tp>());
    }
    else if constexpr (__fmt == __fp_format::__fp6_nv_e2m3 || __fmt == __fp_format::__fp6_nv_e3m2
                       || __fmt == __fp_format::__fp4_nv_e2m1)
    {
      // NaN is converted to positive max value
      return _CUDA_VSTD::__fp_max<_Tp>();
    }
    else
    {
      static_assert(__always_false_v<_Tp>, "Unhandled floating-point format");
    }
  }
};

// __fp_overflow_handler<__sat_finite>

template <>
struct __fp_overflow_handler<__fp_overflow_handler_kind::__sat_finite>
{
  template <class _Tp>
  [[nodiscard]] _CCCL_API static constexpr _Tp __handle_overflow() noexcept
  {
    return _CUDA_VSTD::__fp_max<_Tp>();
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API static constexpr _Tp __handle_underflow() noexcept
  {
    return _CUDA_VSTD::__fp_lowest<_Tp>();
  }
};

// __fp_is_overflow_handler_v

template <class _Tp>
inline constexpr bool __fp_is_overflow_handler_v = false;

template <class _Tp>
inline constexpr bool __fp_is_overflow_handler_v<const _Tp> = __fp_is_overflow_handler_v<_Tp>;

template <class _Tp>
inline constexpr bool __fp_is_overflow_handler_v<volatile _Tp> = __fp_is_overflow_handler_v<_Tp>;

template <class _Tp>
inline constexpr bool __fp_is_overflow_handler_v<const volatile _Tp> = __fp_is_overflow_handler_v<_Tp>;

template <__fp_overflow_handler_kind _Kind>
inline constexpr bool __fp_is_overflow_handler_v<__fp_overflow_handler<_Kind>> = true;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_OVERFLOW_HANDLER_H
