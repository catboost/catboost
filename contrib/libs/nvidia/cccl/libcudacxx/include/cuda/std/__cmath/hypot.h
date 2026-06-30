// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_HYPOT_H
#define _LIBCUDACXX___CMATH_HYPOT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/limits>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// hypot

#if _CCCL_CHECK_BUILTIN(builtin_hypot) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_HYPOTF(...) __builtin_hypotf(__VA_ARGS__)
#  define _CCCL_BUILTIN_HYPOT(...)  __builtin_hypot(__VA_ARGS__)
#  define _CCCL_BUILTIN_HYPOTL(...) __builtin_hypotl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_hypot)

#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_HYPOTF
#  undef _CCCL_BUILTIN_HYPOT
#  undef _CCCL_BUILTIN_HYPOTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float hypot(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_HYPOTF)
  return _CCCL_BUILTIN_HYPOTF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_HYPOTF ^^^ // vvv !_CCCL_BUILTIN_HYPOTF vvv
  return ::hypotf(__x, __y);
#endif // !_CCCL_BUILTIN_HYPOTF
}

[[nodiscard]] _CCCL_API inline float hypotf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_HYPOTF)
  return _CCCL_BUILTIN_HYPOTF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_HYPOTF ^^^ // vvv !_CCCL_BUILTIN_HYPOTF vvv
  return ::hypotf(__x, __y);
#endif // !_CCCL_BUILTIN_HYPOTF
}

[[nodiscard]] _CCCL_API inline double hypot(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_HYPOT)
  return _CCCL_BUILTIN_HYPOT(__x, __y);
#else // ^^^ _CCCL_BUILTIN_HYPOT ^^^ // vvv !_CCCL_BUILTIN_HYPOT vvv
  return ::hypot(__x, __y);
#endif // !_CCCL_BUILTIN_HYPOT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double hypot(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_HYPOTL)
  return _CCCL_BUILTIN_HYPOTL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_HYPOTL ^^^ // vvv !_CCCL_BUILTIN_HYPOTL vvv
  return ::hypotl(__x, __y);
#  endif // !_CCCL_BUILTIN_HYPOTL
}

[[nodiscard]] _CCCL_API inline long double hypotl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_HYPOTL)
  return _CCCL_BUILTIN_HYPOTL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_HYPOTL ^^^ // vvv !_CCCL_BUILTIN_HYPOTL vvv
  return ::hypotl(__x, __y);
#  endif // !_CCCL_BUILTIN_HYPOTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half hypot(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::hypotf(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::hypotf(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2> hypot(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::hypot((__result_type) __x, (__result_type) __y);
}

// hypot 3-arg

// Computes the three-dimensional hypotenuse: `std::hypot(x,y,z)`.
// The naive implementation might over-/underflow which is why this implementation is more involved:
//    If the square of an argument might run into issues, we scale the arguments appropriately.
// See https://github.com/llvm/llvm-project/issues/92782 for a detailed discussion and summary.
template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp __hypot(_Tp __x, _Tp __y, _Tp __z)
{
  // Factors needed to determine if over-/underflow might happen
  constexpr int __exp            = _CUDA_VSTD::numeric_limits<_Tp>::max_exponent / 2;
  const _Tp __overflow_threshold = _CUDA_VSTD::ldexp(_Tp(1), __exp);
  const _Tp __overflow_scale     = _CUDA_VSTD::ldexp(_Tp(1), -(__exp + 20));

  // Scale arguments depending on their size
  const _Tp __max_abs =
    _CUDA_VSTD::fmax(_CUDA_VSTD::fabs(__x), _CUDA_VSTD::fmax(_CUDA_VSTD::fabs(__y), _CUDA_VSTD::fabs(__z)));
  _Tp __scale;
  if (__max_abs > __overflow_threshold)
  { // x*x + y*y + z*z might overflow
    __scale = __overflow_scale;
  }
  else if (__max_abs < 1 / __overflow_threshold)
  { // x*x + y*y + z*z might underflow
    __scale = 1 / __overflow_scale;
  }
  else
  {
    __scale = 1;
  }
  __x *= __scale;
  __y *= __scale;
  __z *= __scale;

  // Compute hypot of scaled arguments and undo scaling
  return _CUDA_VSTD::sqrt(__x * __x + __y * __y + __z * __z) / __scale;
}

[[nodiscard]] _CCCL_API inline float hypot(float __x, float __y, float __z) noexcept
{
  return _CUDA_VSTD::__hypot(__x, __y, __z);
}

[[nodiscard]] _CCCL_API inline float hypotf(float __x, float __y, float __z) noexcept
{
  return _CUDA_VSTD::__hypot(__x, __y, __z);
}

[[nodiscard]] _CCCL_API inline double hypot(double __x, double __y, double __z) noexcept
{
  return _CUDA_VSTD::__hypot(__x, __y, __z);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double hypot(long double __x, long double __y, long double __z) noexcept
{
  return _CUDA_VSTD::__hypot(__x, __y, __z);
}

[[nodiscard]] _CCCL_API inline long double hypotl(long double __x, long double __y, long double __z) noexcept
{
  return _CUDA_VSTD::__hypot(__x, __y, __z);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half hypot(__half __x, __half __y, __half __z) noexcept
{
  return __float2half(_CUDA_VSTD::__hypot(__half2float(__x), __half2float(__y), __half2float(__z)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y, __nv_bfloat16 __z) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::__hypot(__bfloat162float(__x), __bfloat162float(__y), __bfloat162float(__z)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_TEMPLATE(class _A1, class _A2, class _A3)
_CCCL_REQUIRES(_CCCL_TRAIT(is_arithmetic, _A1) _CCCL_AND _CCCL_TRAIT(is_arithmetic, _A2)
                 _CCCL_AND _CCCL_TRAIT(is_arithmetic, _A3))
[[nodiscard]] _CCCL_API inline __promote_t<_A1, _A2, _A3> hypot(_A1 __x, _A2 __y, _A3 __z) noexcept
{
  using __result_type = __promote_t<_A1, _A2, _A3>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)
                  && _CCCL_TRAIT(is_same, _A3, __result_type)),
                "");
  return _CUDA_VSTD::hypot((__result_type) __x, (__result_type) __y, (__result_type) __z);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_HYPOT_H
