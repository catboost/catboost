//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H
#define _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/exponential_functions.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/nvbf16.h>
#include <cuda/std/__complex/nvfp16.h>
#include <cuda/std/__complex/vector_support.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/limits>
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// exp

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> exp(const complex<_Tp>& __x)
{
  _Tp __i = __x.imag();
  if (__i == _Tp(0))
  {
    return complex<_Tp>(_CUDA_VSTD::exp(__x.real()), _CUDA_VSTD::copysign(_Tp(0), __x.imag()));
  }
  if (_CUDA_VSTD::isinf(__x.real()))
  {
    if (__x.real() < _Tp(0))
    {
      if (!_CUDA_VSTD::isfinite(__i))
      {
        __i = _Tp(1);
      }
    }
    else if (__i == _Tp(0) || !_CUDA_VSTD::isfinite(__i))
    {
      if (_CUDA_VSTD::isinf(__i))
      {
        __i = numeric_limits<_Tp>::quiet_NaN();
      }
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = _CUDA_VSTD::exp(__x.real());
  return complex<_Tp>(__e * _CUDA_VSTD::cos(__i), __e * _CUDA_VSTD::sin(__i));
}

// A real exp that doesn't combine the final polynomial estimate with the ldexp factor.
// Useful in cases where an extended-range exp is needed for intermediate calculations.
// fp32
[[nodiscard]] _CCCL_API inline float __internal_unsafe_exp_with_reduction(float __r, float* __ldexp_factor) noexcept
{
  // A slightly more efficient way of doing
  //    __j = round(__r * L2E)
  constexpr float __round_shift = 12582912.0f; // 1.5 * 2^23;
  constexpr float __log2_e      = _CUDA_VSTD::numbers::log2e_v<float>;
  float __j                     = _CUDA_VSTD::fmaf(__r, __log2_e, __round_shift);
  __j                           = __j - __round_shift;

  // exp() range reduction. Constants taken from:
  // https://arxiv.org/PS_cache/arxiv/pdf/0708/0708.3722v1.pdf
  float __r_reduced;
  __r_reduced = _CUDA_VSTD::fmaf(-__j, 0.693147182464599609375f, __r);
  __r_reduced = _CUDA_VSTD::fmaf(-__j, -1.904652435769094154e-9f, __r_reduced);

  // __r_reduced is in [log(sqrt(0.5)), log(sqrt(2))].
  float __exp_mant = 1.95693559362553060054779052734375e-4f;
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 1.39354146085679531097412109375e-3f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 8.333896286785602569580078125e-3f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 4.16664592921733856201171875e-2f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 0.16666664183139801025390625f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 0.5f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 1.0f);
  __exp_mant       = _CUDA_VSTD::fmaf(__exp_mant, __r_reduced, 1.0f);

  *__ldexp_factor = __j;
  return __exp_mant;
}

// fp64:
[[nodiscard]] _CCCL_API inline double __internal_unsafe_exp_with_reduction(double __r, double* __ldexp_factor) noexcept
{
  // A slightly more efficient way of doing
  //    __j = round(__r * L2E)
  constexpr double __round_shift = 6.755399441055744e15; // 1.5 * 2^52;
  constexpr double __log2_e      = _CUDA_VSTD::numbers::log2e_v<double>;
  double __j                     = _CUDA_VSTD::fma(__r, __log2_e, __round_shift);
  __j                            = __j - __round_shift;

  // exp() range reduction. Constants taken from:
  // https://arxiv.org/PS_cache/arxiv/pdf/0708/0708.3722v1.pdf
  double __r_reduced;
  __r_reduced = _CUDA_VSTD::fma(-__j, 0.6931471805599453972491, __r);
  __r_reduced = _CUDA_VSTD::fma(-__j, -8.78318343240526554e-17, __r_reduced);

  // __r_reduced is in [log(sqrt(0.5)), log(sqrt(2))].
  double __exp_mant = 2.5022322536502990e-8;
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 2.7630903488173108e-7);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 2.7557514545882439e-6);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 2.4801491039099165e-5);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 1.9841269589115497e-4);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 1.3888888945916380e-3);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 8.3333333334550432e-3);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 4.1666666666519754e-2);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 1.6666666666666477e-1);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 5.0000000000000122e-1);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 1.0);
  __exp_mant        = _CUDA_VSTD::fma(__exp_mant, __r_reduced, 1.0);

  *__ldexp_factor = __j;
  return __exp_mant;
}

// exp fp32 specialization
template <>
_CCCL_API inline complex<float> exp(const complex<float>& __x)
{
  const float __r = __x.real();
  const float __i = __x.imag();

  // Some Inf/NaN special cases that don't filter through:
  if (!_CUDA_VSTD::isfinite(__r))
  {
    if (_CUDA_VSTD::isinf(__r) && !_CUDA_VSTD::isfinite(__i))
    {
      // __r == +-INF
      return _CUDA_VSTD::signbit(__r)
             ? complex<float>{}
             : complex<float>{_CUDA_VSTD::numeric_limits<float>::infinity(),
                              _CUDA_VSTD::numeric_limits<float>::quiet_NaN()};
    }
    // __r NaN:
    if (_CUDA_VSTD::isnan(__r) && (__i == 0.0f))
    {
      return __x;
    }
  }

  // exp(__r), but without the range-reduced re-combination:
  float __exp_r_ldexp_factor;
  float __exp_r_reduced = __internal_unsafe_exp_with_reduction(__r, &__exp_r_ldexp_factor);

  // Get some close bounds when the answer is fully guaranteed to under/overflow.
  // Sometime close to but >= log(max_flt / min_denom_float):
  if (_CUDA_VSTD::fabsf(__r) >= 194.0f)
  {
    // real/imag return value must always underflow or overflow.
    // Clamp to a low value that still under/overflows.
    // This helps avoids multiple other branches earlier/later.
    __exp_r_reduced = (__r < 0.0f) ? 0.0f : 1e3f;
  }

  // Compile to sincos when possible:
  float __sin_i;
  float __cos_i;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::sincosf(__i, &__sin_i, &__cos_i);),
                    (__sin_i = _CUDA_VSTD::sinf(__i); __cos_i = _CUDA_VSTD::cosf(__i);))

  // Our answer now is: (ldexp(__exp_r_reduced * __sin_r, __j_int), ldexp(__exp_r_reduced * __sin_r, __j_int))
  // However we don't need a full ldexp here, and if __exp_r_reduced*__sin_r is denormal we can lose bits.
  // Instead, do an inlined/simplified ldexp.

  // With fabs(sin or cos) <= 1, the unbiased exponent of sin or cos is in [-149, 0].
  // So for an inlined ldexp, we can clamp __j to [-151, 277].
  // (Go as low as -151 for underflow to 0 denormal rounding.)
  // This allows to to do an ldexp via some faster integer ops.

  // __j converted to int32, clamp so nothing bad happens. Will not change any results.
  // 151 = 127 + 24
  // 277 = 23 + 2*127 + 1
  if (__exp_r_ldexp_factor > 278.0f)
  {
    __exp_r_ldexp_factor = 278.0f;
  }
  if (__exp_r_ldexp_factor < -151.0f)
  {
    __exp_r_ldexp_factor = -151.0f;
  }

  const int32_t __ans_ldexp_factor = static_cast<int32_t>(__exp_r_ldexp_factor);

  // Split this j up into four parts to fit it into four float exponents's.
  // (Splitting j in 4 better than in 3).
  uint32_t __ans_ldexp_factor_quarter   = (__ans_ldexp_factor / 4);
  uint32_t __ans_ldexp_factor_remainder = __ans_ldexp_factor - (3 * __ans_ldexp_factor_quarter);

  // Pack these into exponents:
  __ans_ldexp_factor_quarter   = (__ans_ldexp_factor_quarter + 127) << 23;
  __ans_ldexp_factor_remainder = (__ans_ldexp_factor_remainder + 127) << 23;

  const float __ldexp_factor_1 = _CUDA_VSTD::bit_cast<float>(__ans_ldexp_factor_quarter);
  const float __ldexp_factor_2 = _CUDA_VSTD::bit_cast<float>(__ans_ldexp_factor_remainder);

  // Need to order our multiplications to avoid intermediate under/overflow, including when __sin_r is denormal.
  // Experiment suggests this is (one of) the better ways to do it, there's not that many combinations that work for all
  // inputs:
  const float __ans_r =
    (((__cos_i * __ldexp_factor_1) * __ldexp_factor_1) * __ldexp_factor_1) * (__exp_r_reduced * __ldexp_factor_2);
  const float __ans_i =
    (((__sin_i * __ldexp_factor_1) * __ldexp_factor_1) * __ldexp_factor_1) * (__exp_r_reduced * __ldexp_factor_2);

  return complex<float>(__ans_r, __ans_i);
}

// exp fp64 specialization

template <>
_CCCL_API inline complex<double> exp<double>(const complex<double>& __x)
{
  const double __r = __x.real();
  const double __i = __x.imag();

  // Special cases that don't filter through:
  if (!_CUDA_VSTD::isfinite(__r))
  {
    // __r == +-INF
    if (_CUDA_VSTD::isinf(__r) && !_CUDA_VSTD::isfinite(__i))
    {
      return _CUDA_VSTD::signbit(__r)
             ? complex<double>{}
             : complex<double>{_CUDA_VSTD::numeric_limits<double>::infinity(),
                               _CUDA_VSTD::numeric_limits<double>::quiet_NaN()};
    }
    // __r NaN:
    if (_CUDA_VSTD::isnan(__r) && (__i == 0.0))
    {
      return __x;
    }
  }

  // exp(__r), but without the range-reduced re-combination:
  double __exp_r_ldexp_factor;
  double __exp_r_reduced = __internal_unsafe_exp_with_reduction(__r, &__exp_r_ldexp_factor);

  if (_CUDA_VSTD::fabs(__r) >= 1457.0)
  {
    // real/imag return value must always underflow or overflow.
    // This helps avoid other checks later.
    __exp_r_reduced = (__r < 0.0) ? 0.0 : 1e10;
  }

  // Compile to sincos when possible:
  double __sin_i;
  double __cos_i;
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (::sincos(__i, &__sin_i, &__cos_i);),
                    (__sin_i = _CUDA_VSTD::sin(__i); __cos_i = _CUDA_VSTD::cos(__i);))

  // Our answer now is: (ldexp(__exp_mant * __sin_r, __j_int), ldexp(__exp_mant * __sin_r, __j_int))
  // However we don't need a full ldexp here, and if __exp_mant*__sin_r is denormal we can lose bits.
  // Instead, do an inlined/simplified ldexp.

  // With fabs(sin or cos) <= 1, the unbiased exponent of sin or cos is in [-1074, 0].
  // So for an inlined ldexp, we can clamp __j to [-1076, 2098].
  // (Go as low as -1076 for underflow to 0 denormal rounding.)
  // This allows to to do an ldexp via some faster integer ops.

  // __j is converted to int32, clamp so nothing bad happens. Will not change any results.
  // Need to add 1 for rounding reasons
  // 1076 = 1023 + 53
  // 2099 = 52 + 2*1023 + 1
  if (__exp_r_ldexp_factor > 2099.0)
  {
    __exp_r_ldexp_factor = 2099.0;
  }
  if (__exp_r_ldexp_factor < -1076.0)
  {
    __exp_r_ldexp_factor = -1076.0;
  }

  const int64_t __ans_ldexp_factor = static_cast<int64_t>(__exp_r_ldexp_factor);

  // Split this j up into four parts to fit it into four float exponents's.
  // (Splitting j in 4 better than in 3).
  uint64_t __ans_ldexp_factor_quarter   = (__ans_ldexp_factor / 4);
  uint64_t __ans_ldexp_factor_remainder = __ans_ldexp_factor - (3 * __ans_ldexp_factor_quarter);

  // Pack these into exponents:
  __ans_ldexp_factor_quarter   = (__ans_ldexp_factor_quarter + 1023) << 52;
  __ans_ldexp_factor_remainder = (__ans_ldexp_factor_remainder + 1023) << 52;

  const double __ldexp_factor_1 = _CUDA_VSTD::bit_cast<double>(__ans_ldexp_factor_quarter);
  const double __ldexp_factor_2 = _CUDA_VSTD::bit_cast<double>(__ans_ldexp_factor_remainder);

  // Need to order our multiplications to avoid intermediate under/overflow, including when __sin_r is denormal.
  // Experiment suggests this is (one of) the better ways to do it, there's not that many combinations that work for all
  // inputs:
  const double __ans_r =
    (((__cos_i * __ldexp_factor_1) * __ldexp_factor_1) * __ldexp_factor_1) * (__exp_r_reduced * __ldexp_factor_2);
  const double __ans_i =
    (((__sin_i * __ldexp_factor_1) * __ldexp_factor_1) * __ldexp_factor_1) * (__exp_r_reduced * __ldexp_factor_2);

  return complex<double>(__ans_r, __ans_i);
}

#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> exp(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::exp(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> exp(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::exp(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

// pow

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  return _CUDA_VSTD::exp(__y * _CUDA_VSTD::log(__x));
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244)

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x), __result_type(__y));
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!__is_complex_v<_Up>) )
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const complex<_Tp>& __x, const _Up& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x), __result_type(__y));
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES((!__is_complex_v<_Tp>) )
[[nodiscard]] _CCCL_API inline complex<common_type_t<_Tp, _Up>> pow(const _Tp& __x, const complex<_Up>& __y)
{
  using __result_type = complex<common_type_t<_Tp, _Up>>;
  return _CUDA_VSTD::pow(__result_type(__x, 0), __result_type(__y));
}

_CCCL_DIAG_POP

// __sqr, computes pow(x, 2)

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> __sqr(const complex<_Tp>& __x)
{
  return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()), _Tp(2) * __x.real() * __x.imag());
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_EXPONENTIAL_FUNCTIONS_H
