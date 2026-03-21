//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_NATIVE_TYPE_H
#define _LIBCUDACXX___FLOATING_POINT_NATIVE_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/cfloat>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <__fp_format _Fmt>
_CCCL_API constexpr auto __fp_native_type_impl()
{
  if constexpr (_Fmt == __fp_format::__binary32)
  {
    return float{};
  }
  else if constexpr (_Fmt == __fp_format::__binary64)
  {
    return double{};
  }
  else if constexpr (_Fmt == __fp_format::__binary128)
  {
#if _CCCL_HAS_FLOAT128()
    return __float128{0.0};
#elif _CCCL_HAS_LONG_DOUBLE() && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113
    return (long double) {};
#else // ^^^ has native binary128 ^^^ / vvv no native binary128 vvv
    return;
#endif // ^^^ no native binary128 ^^^
  }
  else if constexpr (_Fmt == __fp_format::__fp80_x86)
  {
#if _CCCL_HAS_LONG_DOUBLE() && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 64
    return (long double) {};
#else // ^^^ has native x86 fp80 ^^^ / vvv no native x86 fp80 vvv
    return;
#endif // ^^^ no native x86 fp80 ^^^
  }
  else
  {
    return;
  }
}

template <__fp_format _Fmt>
using __fp_native_type_t = decltype(__fp_native_type_impl<_Fmt>());

template <__fp_format _Fmt>
inline constexpr bool __fp_has_native_type_v = !_CCCL_TRAIT(is_void, __fp_native_type_t<_Fmt>);

template <class _Tp>
inline constexpr bool __fp_is_native_type_v = __is_std_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FLOATING_POINT_NATIVE_TYPE_H
