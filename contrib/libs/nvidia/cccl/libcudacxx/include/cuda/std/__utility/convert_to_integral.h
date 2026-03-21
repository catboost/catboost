//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H
#define _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_API constexpr int __convert_to_integral(int __val)
{
  return __val;
}

_CCCL_API constexpr unsigned __convert_to_integral(unsigned __val)
{
  return __val;
}

_CCCL_API constexpr long __convert_to_integral(long __val)
{
  return __val;
}

_CCCL_API constexpr unsigned long __convert_to_integral(unsigned long __val)
{
  return __val;
}

_CCCL_API constexpr long long __convert_to_integral(long long __val)
{
  return __val;
}

_CCCL_API constexpr unsigned long long __convert_to_integral(unsigned long long __val)
{
  return __val;
}

template <typename _Fp>
_CCCL_API constexpr enable_if_t<is_floating_point<_Fp>::value, long long> __convert_to_integral(_Fp __val)
{
  return __val;
}

#if _CCCL_HAS_INT128()
_CCCL_API constexpr __int128_t __convert_to_integral(__int128_t __val)
{
  return __val;
}

_CCCL_API constexpr __uint128_t __convert_to_integral(__uint128_t __val)
{
  return __val;
}
#endif // _CCCL_HAS_INT128()

template <class _Tp, bool = is_enum<_Tp>::value>
struct __sfinae_underlying_type
{
  using type            = typename underlying_type<_Tp>::type;
  using __promoted_type = decltype(((type) 1) + 0);
};

template <class _Tp>
struct __sfinae_underlying_type<_Tp, false>
{};

template <class _Tp>
_CCCL_API constexpr typename __sfinae_underlying_type<_Tp>::__promoted_type __convert_to_integral(_Tp __val)
{
  return __val;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H
