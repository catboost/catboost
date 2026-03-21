//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_PROPERTIES_H
#define _LIBCUDACXX___FLOATING_POINT_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/format.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __fp_is_signed_v

template <__fp_format _Fmt>
inline constexpr bool __fp_is_signed_v = true;

template <>
inline constexpr bool __fp_is_signed_v<__fp_format::__fp8_nv_e8m0> = false;

// __fp_exp_nbits_v

template <__fp_format _Fmt>
inline constexpr int __fp_exp_nbits_v = 0;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__binary16> = 5;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__binary32> = 8;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__binary64> = 11;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__binary128> = 15;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__bfloat16> = 8;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp80_x86> = 15;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp8_nv_e4m3> = 4;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp8_nv_e5m2> = 5;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp8_nv_e8m0> = 8;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp6_nv_e2m3> = 2;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp6_nv_e3m2> = 3;

template <>
inline constexpr int __fp_exp_nbits_v<__fp_format::__fp4_nv_e2m1> = 2;

// __fp_exp_bias_v

template <__fp_format _Fmt>
inline constexpr int __fp_exp_bias_v = (1 << (__fp_exp_nbits_v<_Fmt> - 1)) - 1;

// __fp_exp_min_v

template <__fp_format _Fmt>
inline constexpr int __fp_exp_min_v = 1 - __fp_exp_bias_v<_Fmt>;

template <>
inline constexpr int __fp_exp_min_v<__fp_format::__fp8_nv_e8m0> = -127;

// __fp_exp_max_v

template <__fp_format _Fmt>
inline constexpr int __fp_exp_max_v = (1 << __fp_exp_nbits_v<_Fmt>) -2 - __fp_exp_bias_v<_Fmt>;

template <>
inline constexpr int __fp_exp_max_v<__fp_format::__fp8_nv_e4m3> = 8;

template <>
inline constexpr int __fp_exp_max_v<__fp_format::__fp6_nv_e2m3> = 2;

template <>
inline constexpr int __fp_exp_max_v<__fp_format::__fp6_nv_e3m2> = 4;

template <>
inline constexpr int __fp_exp_max_v<__fp_format::__fp4_nv_e2m1> = 2;

// __fp_mant_nbits_v

template <__fp_format _Fmt>
inline constexpr int __fp_mant_nbits_v = 0;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__binary16> = 10;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__binary32> = 23;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__binary64> = 52;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__binary128> = 112;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__bfloat16> = 7;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp80_x86> = 64;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp8_nv_e4m3> = 3;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp8_nv_e5m2> = 2;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp8_nv_e8m0> = 0;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp6_nv_e2m3> = 3;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp6_nv_e3m2> = 2;

template <>
inline constexpr int __fp_mant_nbits_v<__fp_format::__fp4_nv_e2m1> = 1;

// __fp_has_implicit_bit_v

template <__fp_format _Fmt>
inline constexpr bool __fp_has_implicit_bit_v = true;

template <>
inline constexpr bool __fp_has_implicit_bit_v<__fp_format::__fp80_x86> = false;

// __fp_digits_v

template <__fp_format _Fmt>
inline constexpr int __fp_digits_v = __fp_mant_nbits_v<_Fmt> + static_cast<int>(__fp_has_implicit_bit_v<_Fmt>);

// __fp_has_denorm_v

template <__fp_format _Fmt>
inline constexpr bool __fp_has_denorm_v = true;

template <>
inline constexpr bool __fp_has_denorm_v<__fp_format::__fp8_nv_e8m0> = false;

// __fp_has_inf_v

template <__fp_format _Fmt>
inline constexpr bool __fp_has_inf_v = true;

template <>
inline constexpr bool __fp_has_inf_v<__fp_format::__fp8_nv_e4m3> = false;

template <>
inline constexpr bool __fp_has_inf_v<__fp_format::__fp8_nv_e8m0> = false;

template <>
inline constexpr bool __fp_has_inf_v<__fp_format::__fp6_nv_e2m3> = false;

template <>
inline constexpr bool __fp_has_inf_v<__fp_format::__fp6_nv_e3m2> = false;

template <>
inline constexpr bool __fp_has_inf_v<__fp_format::__fp4_nv_e2m1> = false;

// __fp_has_nan_v

template <__fp_format _Fmt>
inline constexpr bool __fp_has_nan_v = true;

template <>
inline constexpr bool __fp_has_nan_v<__fp_format::__fp6_nv_e2m3> = false;

template <>
inline constexpr bool __fp_has_nan_v<__fp_format::__fp6_nv_e3m2> = false;

template <>
inline constexpr bool __fp_has_nan_v<__fp_format::__fp4_nv_e2m1> = false;

// __fp_has_nans_v

template <__fp_format _Fmt>
inline constexpr bool __fp_has_nans_v = true;

template <>
inline constexpr bool __fp_has_nans_v<__fp_format::__fp8_nv_e4m3> = false;

template <>
inline constexpr bool __fp_has_nans_v<__fp_format::__fp8_nv_e8m0> = false;

template <>
inline constexpr bool __fp_has_nans_v<__fp_format::__fp6_nv_e2m3> = false;

template <>
inline constexpr bool __fp_has_nans_v<__fp_format::__fp6_nv_e3m2> = false;

template <>
inline constexpr bool __fp_has_nans_v<__fp_format::__fp4_nv_e2m1> = false;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_PROPERTIES_H
