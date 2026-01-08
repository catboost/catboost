//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ISNORMAL_H
#define _LIBCUDACXX___CMATH_ISNORMAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/fpclassify.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_CHECK_BUILTIN(builtin_isnormal) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISNORMAL(...) __builtin_isnormal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(isnormal)

// nvcc does not implement __builtin_isnormal
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_ISNORMAL
#endif // _CCCL_CUDA_COMPILER(NVCC)

[[nodiscard]] _CCCL_API constexpr bool isnormal(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
#endif // !_CCCL_BUILTIN_ISNORMAL
}

[[nodiscard]] _CCCL_API constexpr bool isnormal(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
#endif // !_CCCL_BUILTIN_ISNORMAL
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr bool isnormal(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNORMAL)
  return _CCCL_BUILTIN_ISNORMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_ISNORMAL ^^^ / vvv !_CCCL_BUILTIN_ISNORMAL vvv
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
#  endif // !_CCCL_BUILTIN_ISNORMAL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__half __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e4m3 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e5m2 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp8_e8m0 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp6_e2m3 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp6_e3m2 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
[[nodiscard]] _CCCL_API constexpr bool isnormal(__nv_fp4_e2m1 __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Tp))
[[nodiscard]] _CCCL_API constexpr bool isnormal(_Tp __x) noexcept
{
  return __x != 0;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_ISNORMAL_H
