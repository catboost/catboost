// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_GCD_LCM_H
#define _LIBCUDACXX___NUMERIC_GCD_LCM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
constexpr _CCCL_API inline _Tp __gcd(_Tp __m, _Tp __n)
{
  static_assert((!_CCCL_TRAIT(is_signed, _Tp)), "");
  return __n == 0 ? __m : _CUDA_VSTD::__gcd<_Tp>(__n, __m % __n);
}

template <class _Tp, class _Up>
constexpr _CCCL_API inline common_type_t<_Tp, _Up> gcd(_Tp __m, _Up __n)
{
  static_assert((_CCCL_TRAIT(is_integral, _Tp) && _CCCL_TRAIT(is_integral, _Up)),
                "Arguments to gcd must be integer types");
  static_assert((!_CCCL_TRAIT(is_same, remove_cv_t<_Tp>, bool)), "First argument to gcd cannot be bool");
  static_assert((!_CCCL_TRAIT(is_same, remove_cv_t<_Up>, bool)), "Second argument to gcd cannot be bool");
  using _Rp = common_type_t<_Tp, _Up>;
  using _Wp = make_unsigned_t<_Rp>;
  return static_cast<_Rp>(_CUDA_VSTD::__gcd(static_cast<_Wp>(::cuda::uabs(__m)), static_cast<_Wp>(::cuda::uabs(__n))));
}

template <class _Tp, class _Up>
constexpr _CCCL_API inline common_type_t<_Tp, _Up> lcm(_Tp __m, _Up __n)
{
  static_assert((_CCCL_TRAIT(is_integral, _Tp) && _CCCL_TRAIT(is_integral, _Up)),
                "Arguments to lcm must be integer types");
  static_assert((!_CCCL_TRAIT(is_same, remove_cv_t<_Tp>, bool)), "First argument to lcm cannot be bool");
  static_assert((!_CCCL_TRAIT(is_same, remove_cv_t<_Up>, bool)), "Second argument to lcm cannot be bool");
  if (__m == 0 || __n == 0)
  {
    return 0;
  }

  using _Rp         = common_type_t<_Tp, _Up>;
  using _Wp         = make_unsigned_t<_Rp>;
  const auto __val1 = ::cuda::uabs(__m) / _CUDA_VSTD::gcd(__m, __n);
  const auto __val2 = ::cuda::uabs(__n);
  _CCCL_ASSERT((static_cast<_Wp>(numeric_limits<_Rp>::max()) / __val1 > __val2), "Overflow in lcm");
  return static_cast<_Rp>(__val1 * __val2);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_GCD_LCM_H
