//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_IS_VALID_H
#define _LIBCUDACXX___RANDOM_IS_VALID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [rand.req.genl]/1.4:
// The effect of instantiating a template that has a template type parameter
// named RealType is undefined unless the corresponding template argument is
// cv-unqualified and is one of float, double, or long double.

template <class>
inline constexpr bool __libcpp_random_is_valid_realtype = false;

template <>
inline constexpr bool __libcpp_random_is_valid_realtype<float> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_realtype<double> = true;
#if _CCCL_HAS_LONG_DOUBLE()
template <>
inline constexpr bool __libcpp_random_is_valid_realtype<long double> = true;
#endif // _CCCL_HAS_LONG_DOUBLE()

// [rand.req.genl]/1.5:
// The effect of instantiating a template that has a template type parameter
// named IntType is undefined unless the corresponding template argument is
// cv-unqualified and is one of short, int, long, long long, unsigned short,
// unsigned int, unsigned long, or unsigned long long.

template <class>
inline constexpr bool __libcpp_random_is_valid_inttype = false;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<int8_t> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<short> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<int> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<long> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<long long> = true;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<uint8_t> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned short> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned int> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned long> = true;
template <>
inline constexpr bool __libcpp_random_is_valid_inttype<unsigned long long> = true;
#if _CCCL_HAS_INT128()
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<__int128_t> = true;
template <> // extension
inline constexpr bool __libcpp_random_is_valid_inttype<__uint128_t> = true;
#endif // _CCCL_HAS_INT128()

// [rand.req.urng]/3:
// A class G meets the uniform random bit generator requirements if G models
// uniform_random_bit_generator, invoke_result_t<G&> is an unsigned integer type,
// and G provides a nested typedef-name result_type that denotes the same type
// as invoke_result_t<G&>.
// (In particular, reject URNGs with signed result_types; our distributions cannot
// handle such generator types.)

template <class, class = void>
inline constexpr bool __libcpp_random_is_valid_urng = false;
template <class _Gp>
inline constexpr bool __libcpp_random_is_valid_urng<
  _Gp,
  enable_if_t<_CCCL_TRAIT(is_unsigned, typename _Gp::result_type)
              && is_same_v<decltype(_CUDA_VSTD::declval<_Gp&>()()), typename _Gp::result_type>>> = true;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANDOM_IS_VALID_H
