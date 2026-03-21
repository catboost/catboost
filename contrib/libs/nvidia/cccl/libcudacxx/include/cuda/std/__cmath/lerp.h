// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_LERP_H
#define _LIBCUDACXX___CMATH_LERP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/promote.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Fp>
[[nodiscard]] _CCCL_API constexpr _Fp __lerp(_Fp __a, _Fp __b, _Fp __t) noexcept
{
  if ((__a <= 0 && __b >= 0) || (__a >= 0 && __b <= 0))
  {
    return __t * __b + (1 - __t) * __a;
  }

  if (__t == 1)
  {
    return __b;
  }
  const _Fp __x = __a + __t * (__b - __a);
  if ((__t > 1) == (__b > __a))
  {
    return __b < __x ? __x : __b;
  }
  else
  {
    return __x < __b ? __x : __b;
  }
}

[[nodiscard]] _CCCL_API constexpr float lerp(float __a, float __b, float __t) noexcept
{
  return _CUDA_VSTD::__lerp(__a, __b, __t);
}

[[nodiscard]] _CCCL_API constexpr double lerp(double __a, double __b, double __t) noexcept
{
  return _CUDA_VSTD::__lerp(__a, __b, __t);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double lerp(long double __a, long double __b, long double __t) noexcept
{
  return _CUDA_VSTD::__lerp(__a, __b, __t);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half lerp(__half __a, __half __b, __half __t) noexcept
{
  return __float2half(_CUDA_VSTD::__lerp(__half2float(__a), __half2float(__b), __half2float(__t)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 lerp(__nv_bfloat16 __a, __nv_bfloat16 __b, __nv_bfloat16 __t) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::__lerp(__bfloat162float(__a), __bfloat162float(__b), __bfloat162float(__t)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, class _A3>
[[nodiscard]] _CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2) && _CCCL_TRAIT(is_arithmetic, _A3),
  __promote_t<_A1, _A2, _A3>>
lerp(_A1 __a, _A2 __b, _A3 __t) noexcept
{
  using __result_type = __promote_t<_A1, _A2, _A3>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)
                  && _CCCL_TRAIT(is_same, _A3, __result_type)),
                "");
  return _CUDA_VSTD::__lerp((__result_type) __a, (__result_type) __b, (__result_type) __t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_LERP_H
