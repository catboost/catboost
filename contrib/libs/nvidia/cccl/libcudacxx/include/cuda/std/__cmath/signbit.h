//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_SIGNBIT_H
#define _LIBCUDACXX___CMATH_SIGNBIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/limits>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

[[nodiscard]] _CCCL_API inline bool signbit(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBIT
}

[[nodiscard]] _CCCL_API inline bool signbit(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBIT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline bool signbit(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#  else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#  endif // !_CCCL_BUILTIN_SIGNBIT
}
#endif // _CCCL_HAS_LONG_DOUBLE()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool __signbit_impl(_Tp __x) noexcept
{
  if constexpr (numeric_limits<_Tp>::is_signed)
  {
    return _CUDA_VSTD::__fp_get_storage(__x) & __fp_sign_mask_of_v<_Tp>;
  }
  else
  {
    return false;
  }
}

#if _CCCL_HAS_NVFP16()
[[nodiscard]] _CCCL_API constexpr bool signbit(__half __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp8_e4m3 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp8_e5m2 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp8_e8m0) noexcept
{
  return false;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp6_e2m3 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp6_e3m2 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
[[nodiscard]] _CCCL_API constexpr bool signbit(__nv_fp4_e2m1 __x) noexcept
{
  return _CUDA_VSTD::__signbit_impl(__x);
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
[[nodiscard]] _CCCL_API constexpr bool signbit([[maybe_unused]] _Tp __x) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    return __x < 0;
  }
  else
  {
    return false;
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_SIGNBIT_H
