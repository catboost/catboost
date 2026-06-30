//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_CONSTANTS_H
#define _LIBCUDACXX___FLOATING_POINT_CONSTANTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __fp_inf

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_inf() noexcept
{
  static_assert(__fp_has_inf_v<_Fmt>, "The format does not support infinity");

  return __fp_exp_mask_v<_Fmt>;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_inf() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_inf<__fp_format_of_v<_Tp>>());
}

// __fp_nan

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_nan() noexcept
{
  static_assert(__fp_has_nan_v<_Fmt>, "The format does not support nan");

  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3)
  {
    return __fp_storage_t<_Fmt>(0x7fu);
  }
  else if constexpr (_Fmt == __fp_format::__fp8_nv_e8m0)
  {
    return __fp_storage_t<_Fmt>(0xffu);
  }
  else if constexpr (__fp_has_implicit_bit_v<_Fmt>)
  {
    return static_cast<__fp_storage_t<_Fmt>>(
      __fp_exp_mask_v<_Fmt> | (__fp_storage_t<_Fmt>(1) << (__fp_mant_nbits_v<_Fmt> - 1)));
  }
  else
  {
    return static_cast<__fp_storage_t<_Fmt>>(
      __fp_exp_mask_v<_Fmt> | (__fp_storage_t<_Fmt>(3) << (__fp_mant_nbits_v<_Fmt> - 2)));
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_nan() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_nan<__fp_format_of_v<_Tp>>());
}

// __fp_nans

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_nans() noexcept
{
  static_assert(__fp_has_nans_v<_Fmt>, "The format does not support nans");

  if constexpr (__fp_has_implicit_bit_v<_Fmt>)
  {
    return static_cast<__fp_storage_t<_Fmt>>(
      __fp_exp_mask_v<_Fmt> | (__fp_storage_t<_Fmt>(1) << (__fp_mant_nbits_v<_Fmt> - 2)));
  }
  else
  {
    return static_cast<__fp_storage_t<_Fmt>>(
      __fp_exp_mask_v<_Fmt> | (__fp_storage_t<_Fmt>(5) << (__fp_mant_nbits_v<_Fmt> - 3)));
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_nans() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_nans<__fp_format_of_v<_Tp>>());
}

// __fp_max

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_max() noexcept
{
  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3)
  {
    return __fp_storage_t<_Fmt>(0x7eu);
  }
  else
  {
    return static_cast<__fp_storage_t<_Fmt>>(
      (__fp_storage_t<_Fmt>(__fp_exp_max_v<_Fmt> + __fp_exp_bias_v<_Fmt>) << __fp_mant_nbits_v<_Fmt>)
      | __fp_mant_mask_v<_Fmt>);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_max() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_max<__fp_format_of_v<_Tp>>());
}

// __fp_min

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_min() noexcept
{
  return static_cast<__fp_storage_t<_Fmt>>(
    __fp_storage_t<_Fmt>(__fp_exp_min_v<_Fmt> + __fp_exp_bias_v<_Fmt>) << __fp_mant_nbits_v<_Fmt>);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_min() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_min<__fp_format_of_v<_Tp>>());
}

// __fp_lowest

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_lowest() noexcept
{
  if constexpr (_Fmt == __fp_format::__fp8_nv_e8m0)
  {
    return _CUDA_VSTD::__fp_min<_Fmt>();
  }
  else
  {
    return _CUDA_VSTD::__fp_neg<_Fmt>(_CUDA_VSTD::__fp_max<_Fmt>());
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_lowest() noexcept
{
  return _CUDA_VSTD::__fp_from_storage<_Tp>(_CUDA_VSTD::__fp_lowest<__fp_format_of_v<_Tp>>());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_CONSTANTS_H
