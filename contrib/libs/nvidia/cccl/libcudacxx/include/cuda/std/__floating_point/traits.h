//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_TRAITS_H
#define _LIBCUDACXX___FLOATING_POINT_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__fwd/fp.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __is_std_fp_v

template <class _Tp>
inline constexpr bool __is_std_fp_v = false;

template <class _Tp>
inline constexpr bool __is_std_fp_v<const _Tp> = __is_std_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_std_fp_v<volatile _Tp> = __is_std_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_std_fp_v<const volatile _Tp> = __is_std_fp_v<_Tp>;

template <>
inline constexpr bool __is_std_fp_v<float> = true;

template <>
inline constexpr bool __is_std_fp_v<double> = true;

template <>
inline constexpr bool __is_std_fp_v<long double> = true;

// __is_ext_nv_fp_v

template <class _Tp>
inline constexpr bool __is_ext_nv_fp_v = false;

template <class _Tp>
inline constexpr bool __is_ext_nv_fp_v<const _Tp> = __is_ext_nv_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_nv_fp_v<volatile _Tp> = __is_ext_nv_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_nv_fp_v<const volatile _Tp> = __is_ext_nv_fp_v<_Tp>;

#if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool __is_ext_nv_fp_v<__half> = true;
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_bfloat16> = true;
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp8_e4m3> = true;
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp8_e5m2> = true;
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp8_e8m0> = true;
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp6_e2m3> = true;
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp6_e3m2> = true;
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
template <>
inline constexpr bool __is_ext_nv_fp_v<__nv_fp4_e2m1> = true;
#endif // _CCCL_HAS_NVFP4_E2M1()

// __is_ext_compiler_fp_v

template <class _Tp>
inline constexpr bool __is_ext_compiler_fp_v = false;

template <class _Tp>
inline constexpr bool __is_ext_compiler_fp_v<const _Tp> = __is_ext_compiler_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_compiler_fp_v<volatile _Tp> = __is_ext_compiler_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_compiler_fp_v<const volatile _Tp> = __is_ext_compiler_fp_v<_Tp>;

#if _CCCL_HAS_FLOAT128()
template <>
inline constexpr bool __is_ext_compiler_fp_v<__float128> = true;
#endif // _CCCL_HAS_FLOAT128()

// __is_ext_cccl_fp_v

template <class _Tp>
inline constexpr bool __is_ext_cccl_fp_v = false;

template <class _Tp>
inline constexpr bool __is_ext_cccl_fp_v<const _Tp> = __is_ext_cccl_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_cccl_fp_v<volatile _Tp> = __is_ext_cccl_fp_v<_Tp>;

template <class _Tp>
inline constexpr bool __is_ext_cccl_fp_v<const volatile _Tp> = __is_ext_cccl_fp_v<_Tp>;

template <__fp_format _Fmt>
inline constexpr bool __is_ext_cccl_fp_v<__cccl_fp<_Fmt>> = true;

// __is_ext_fp_v

template <class _Tp>
inline constexpr bool __is_ext_fp_v = __is_ext_nv_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp> || __is_ext_cccl_fp_v<_Tp>;

// __is_fp_v (todo: use cuda::std::is_floating_point_v instead in the future)

template <class _Tp>
inline constexpr bool __is_fp_v = __is_std_fp_v<_Tp> || __is_ext_fp_v<_Tp>;

// __fp_is_subset_v

template <__fp_format _LhsFmt, __fp_format _RhsFmt>
inline constexpr bool __fp_is_subset_v =
  (!__fp_is_signed_v<_LhsFmt> || __fp_is_signed_v<_RhsFmt>)
  && __fp_exp_min_v<_LhsFmt> >= __fp_exp_min_v<_RhsFmt> && __fp_exp_max_v<_LhsFmt> <= __fp_exp_max_v<_RhsFmt>
  && __fp_digits_v<_LhsFmt> <= __fp_digits_v<_RhsFmt> && (!__fp_has_denorm_v<_LhsFmt> || __fp_has_denorm_v<_RhsFmt>);

// __fp_is_subset_of_v

template <class _Lhs, class _Rhs>
inline constexpr bool __fp_is_subset_of_v = __fp_is_subset_v<__fp_format_of_v<_Lhs>, __fp_format_of_v<_Rhs>>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_TRAITS_H
