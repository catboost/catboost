//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_COMPLEX_H
#define _LIBCUDACXX___COMPLEX_COMPLEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/vector_support.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

// Compatibility helpers for thrust to convert between `std::complex` and `cuda::std::complex`
#if !_CCCL_COMPILER(NVRTC)
#  include <complex>
#  include <sstream> // for std::basic_ostringstream

#  define _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__c) reinterpret_cast<const _Up(&)[2]>(__c)[0]
#  define _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__c) reinterpret_cast<const _Up(&)[2]>(__c)[1]
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

#ifdef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#  ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION
#    define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION
#  endif // LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION
#  ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION
#    define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION
#  endif // LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION
#endif // LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __get_complex_impl;

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_COMPLEX_ALIGNAS complex
{
  _Tp __re_;
  _Tp __im_;

  template <class _Up>
  friend class complex;

  template <class _Up>
  friend struct __get_complex_impl;

public:
  using value_type = _Tp;

  _CCCL_API constexpr complex(const value_type& __re = value_type(), const value_type& __im = value_type())
      : __re_(__re)
      , __im_(__im)
  {}

  template <class _Up, enable_if_t<__cccl_internal::__is_non_narrowing_convertible<_Tp, _Up>::value, int> = 0>
  _CCCL_API constexpr complex(const complex<_Up>& __c)
      : __re_(static_cast<_Tp>(__c.real()))
      , __im_(static_cast<_Tp>(__c.imag()))
  {}

  template <class _Up,
            enable_if_t<!__cccl_internal::__is_non_narrowing_convertible<_Tp, _Up>::value, int> = 0,
            enable_if_t<_CCCL_TRAIT(is_constructible, _Tp, _Up), int>                           = 0>
  _CCCL_API explicit constexpr complex(const complex<_Up>& __c)
      : __re_(static_cast<_Tp>(__c.real()))
      , __im_(static_cast<_Tp>(__c.imag()))
  {}

  _CCCL_API constexpr complex& operator=(const value_type& __re)
  {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }

  template <class _Up>
  _CCCL_API constexpr complex& operator=(const complex<_Up>& __c)
  {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }

#if !_CCCL_COMPILER(NVRTC)
  template <class _Up>
  _CCCL_API inline complex(const ::std::complex<_Up>& __other)
      : __re_(_LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other))
      , __im_(_LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other))
  {}

  template <class _Up>
  _CCCL_API inline complex& operator=(const ::std::complex<_Up>& __other)
  {
    __re_ = _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other);
    __im_ = _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other);
    return *this;
  }

  _CCCL_HOST constexpr operator ::std::complex<_Tp>() const
  {
    return {__re_, __im_};
  }
#endif // !_CCCL_COMPILER(NVRTC)

  [[nodiscard]] _CCCL_API constexpr value_type real() const
  {
    return __re_;
  }
  [[nodiscard]] _CCCL_API constexpr value_type imag() const
  {
    return __im_;
  }

  _CCCL_API constexpr void real(value_type __re)
  {
    __re_ = __re;
  }
  _CCCL_API constexpr void imag(value_type __im)
  {
    __im_ = __im;
  }

  // Those additional volatile overloads are meant to help with reductions in thrust
  [[nodiscard]] _CCCL_API inline value_type real() const volatile
  {
    return __re_;
  }
  [[nodiscard]] _CCCL_API inline value_type imag() const volatile
  {
    return __im_;
  }

  _CCCL_API inline void real(value_type __re) volatile
  {
    __re_ = __re;
  }
  _CCCL_API inline void imag(value_type __im) volatile
  {
    __im_ = __im;
  }

  _CCCL_API constexpr complex& operator+=(const value_type& __re)
  {
    __re_ += __re;
    return *this;
  }
  _CCCL_API constexpr complex& operator-=(const value_type& __re)
  {
    __re_ -= __re;
    return *this;
  }
  _CCCL_API constexpr complex& operator*=(const value_type& __re)
  {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _CCCL_API constexpr complex& operator/=(const value_type& __re)
  {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Up>
  _CCCL_API constexpr complex& operator+=(const complex<_Up>& __c)
  {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Up>
  _CCCL_API constexpr complex& operator-=(const complex<_Up>& __c)
  {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
};

template <class _Tp>
inline constexpr bool __is_complex_v = false;

template <class _Tp>
inline constexpr bool __is_complex_v<complex<_Tp>> = true;

template <class _Tp, class _Up>
_CCCL_API inline _CCCL_CONSTEXPR_CXX14_COMPLEX complex<_Tp>& operator*=(complex<_Tp>& __lhs, const complex<_Up>& __rhs)
{
  __lhs = __lhs * complex<_Tp>(__rhs.real(), __rhs.imag());
  return __lhs;
}
template <class _Tp, class _Up>
_CCCL_API inline _CCCL_CONSTEXPR_CXX14_COMPLEX complex<_Tp>& operator/=(complex<_Tp>& __lhs, const complex<_Up>& __rhs)
{
  __lhs = __lhs / complex<_Tp>(__rhs.real(), __rhs.imag());
  return __lhs;
}

// 26.3.6 operators:
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator+(const complex<_Tp>& __x, const _Tp& __y)
{
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator+(const _Tp& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(__y);
  __t += __x;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator-(const complex<_Tp>& __x, const _Tp& __y)
{
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator-(const _Tp& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(-__y);
  __t += __x;
  return __t;
}
template <class _Tp>
[[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX14_COMPLEX complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  // Avoid floating point operations that are invalid during constant evaluation
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    bool __z_zero = __a == _Tp(0) && __b == _Tp(0);
    bool __w_zero = __c == _Tp(0) && __d == _Tp(0);
    bool __z_inf  = _CUDA_VSTD::isinf(__a) || _CUDA_VSTD::isinf(__b);
    bool __w_inf  = _CUDA_VSTD::isinf(__c) || _CUDA_VSTD::isinf(__d);
    bool __z_nan  = !__z_inf
                && ((_CUDA_VSTD::isnan(__a) && _CUDA_VSTD::isnan(__b)) || (_CUDA_VSTD::isnan(__a) && __b == _Tp(0))
                    || (__a == _Tp(0) && _CUDA_VSTD::isnan(__b)));
    bool __w_nan = !__w_inf
                && ((_CUDA_VSTD::isnan(__c) && _CUDA_VSTD::isnan(__d)) || (_CUDA_VSTD::isnan(__c) && __d == _Tp(0))
                    || (__c == _Tp(0) && _CUDA_VSTD::isnan(__d)));
    if (__z_nan || __w_nan)
    {
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
    }
    if (__z_inf || __w_inf)
    {
      if (__z_zero || __w_zero)
      {
        return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
      }
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::infinity()), _Tp(numeric_limits<_Tp>::infinity()));
    }
    bool __z_nonzero_nan = !__z_inf && !__z_nan && (_CUDA_VSTD::isnan(__a) || _CUDA_VSTD::isnan(__b));
    bool __w_nonzero_nan = !__w_inf && !__w_nan && (_CUDA_VSTD::isnan(__c) || _CUDA_VSTD::isnan(__d));
    if (__z_nonzero_nan || __w_nonzero_nan)
    {
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
    }
  }
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  __abcd_results<_Tp> __partials = __complex_calculate_partials(__a, __b, __c, __d);

  _Tp __x = __partials.__ac - __partials.__bd;
  _Tp __y = __partials.__ad + __partials.__bc;
#ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION
  if (_CUDA_VSTD::isnan(__x) && _CUDA_VSTD::isnan(__y))
  {
    bool __recalc = false;
    if (_CUDA_VSTD::isinf(__a) || _CUDA_VSTD::isinf(__b))
    {
      __a = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__b) ? _Tp(1) : _Tp(0), __b);
      if (_CUDA_VSTD::isnan(__c))
      {
        __c = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __c);
      }
      if (_CUDA_VSTD::isnan(__d))
      {
        __d = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __d);
      }
      __recalc = true;
    }
    if (_CUDA_VSTD::isinf(__c) || _CUDA_VSTD::isinf(__d))
    {
      __c = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__d) ? _Tp(1) : _Tp(0), __d);
      if (_CUDA_VSTD::isnan(__a))
      {
        __a = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __a);
      }
      if (_CUDA_VSTD::isnan(__b))
      {
        __b = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __b);
      }
      __recalc = true;
    }
    if (!__recalc
        && (_CUDA_VSTD::isinf(__partials.__ac) || _CUDA_VSTD::isinf(__partials.__bd)
            || _CUDA_VSTD::isinf(__partials.__ad) || _CUDA_VSTD::isinf(__partials.__bc)))
    {
      if (_CUDA_VSTD::isnan(__a))
      {
        __a = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __a);
      }
      if (_CUDA_VSTD::isnan(__b))
      {
        __b = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __b);
      }
      if (_CUDA_VSTD::isnan(__c))
      {
        __c = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __c);
      }
      if (_CUDA_VSTD::isnan(__d))
      {
        __d = _CUDA_VSTD::__constexpr_copysign(_Tp(0), __d);
      }
      __recalc = true;
    }
    if (__recalc)
    {
      __partials = __complex_calculate_partials(__a, __b, __c, __d);

      __x = numeric_limits<_Tp>::infinity() * (__partials.__ac - __partials.__bd);
      __y = numeric_limits<_Tp>::infinity() * (__partials.__ad + __partials.__bc);
    }
  }
#endif // LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_MULTIPLICATION
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator*(const complex<_Tp>& __x, const _Tp& __y)
{
  complex<_Tp> __t(__x);
  __t *= __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator*(const _Tp& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(__y);
  __t *= __x;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API inline _CCCL_CONSTEXPR_CXX14_COMPLEX complex<_Tp>
operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
  int __ilogbw = 0;
  _Tp __a      = __z.real();
  _Tp __b      = __z.imag();
  _Tp __c      = __w.real();
  _Tp __d      = __w.imag();
  _Tp __logbw  = _CUDA_VSTD::__constexpr_logb(
    _CUDA_VSTD::__constexpr_fmax(_CUDA_VSTD::__constexpr_fabs(__c), _CUDA_VSTD::__constexpr_fabs(__d)));
  if (_CUDA_VSTD::isfinite(__logbw))
  {
    __ilogbw = static_cast<int>(__logbw);
    __c      = _CUDA_VSTD::__constexpr_scalbn(__c, -__ilogbw);
    __d      = _CUDA_VSTD::__constexpr_scalbn(__d, -__ilogbw);
  }

#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  // Avoid floating point operations that are invalid during constant evaluation
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    bool __z_zero = __a == _Tp(0) && __b == _Tp(0);
    bool __w_zero = __c == _Tp(0) && __d == _Tp(0);
    bool __z_inf  = _CUDA_VSTD::isinf(__a) || _CUDA_VSTD::isinf(__b);
    bool __w_inf  = _CUDA_VSTD::isinf(__c) || _CUDA_VSTD::isinf(__d);
    bool __z_nan  = !__z_inf
                && ((_CUDA_VSTD::isnan(__a) && _CUDA_VSTD::isnan(__b)) || (_CUDA_VSTD::isnan(__a) && __b == _Tp(0))
                    || (__a == _Tp(0) && _CUDA_VSTD::isnan(__b)));
    bool __w_nan = !__w_inf
                && ((_CUDA_VSTD::isnan(__c) && _CUDA_VSTD::isnan(__d)) || (_CUDA_VSTD::isnan(__c) && __d == _Tp(0))
                    || (__c == _Tp(0) && _CUDA_VSTD::isnan(__d)));
    if ((__z_nan || __w_nan) || (__z_inf && __w_inf))
    {
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
    }
    bool __z_nonzero_nan = !__z_inf && !__z_nan && (_CUDA_VSTD::isnan(__a) || _CUDA_VSTD::isnan(__b));
    bool __w_nonzero_nan = !__w_inf && !__w_nan && (_CUDA_VSTD::isnan(__c) || _CUDA_VSTD::isnan(__d));
    if (__z_nonzero_nan || __w_nonzero_nan)
    {
      if (__w_zero)
      {
        return complex<_Tp>(_Tp(numeric_limits<_Tp>::infinity()), _Tp(numeric_limits<_Tp>::infinity()));
      }
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
    }
    if (__w_inf)
    {
      return complex<_Tp>(_Tp(0), _Tp(0));
    }
    if (__z_inf)
    {
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::infinity()), _Tp(numeric_limits<_Tp>::infinity()));
    }
    if (__w_zero)
    {
      if (__z_zero)
      {
        return complex<_Tp>(_Tp(numeric_limits<_Tp>::quiet_NaN()), _Tp(0));
      }
      return complex<_Tp>(_Tp(numeric_limits<_Tp>::infinity()), _Tp(numeric_limits<_Tp>::infinity()));
    }
  }
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  __abcd_results<_Tp> __partials = __complex_calculate_partials(__a, __b, __c, __d);
  __ab_results<_Tp> __denom_vec  = __complex_piecewise_mul(__c, __d, __c, __d);

  _Tp __denom = __denom_vec.__a + __denom_vec.__b;
  _Tp __x     = _CUDA_VSTD::__constexpr_scalbn((__partials.__ac + __partials.__bd) / __denom, -__ilogbw);
  _Tp __y     = _CUDA_VSTD::__constexpr_scalbn((__partials.__bc - __partials.__ad) / __denom, -__ilogbw);
#ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION
  if (_CUDA_VSTD::isnan(__x) && _CUDA_VSTD::isnan(__y))
  {
    if ((__denom == _Tp(0)) && (!_CUDA_VSTD::isnan(__a) || !_CUDA_VSTD::isnan(__b)))
    {
      __x = _CUDA_VSTD::__constexpr_copysign(numeric_limits<_Tp>::infinity(), __c) * __a;
      __y = _CUDA_VSTD::__constexpr_copysign(numeric_limits<_Tp>::infinity(), __c) * __b;
    }
    else if ((_CUDA_VSTD::isinf(__a) || _CUDA_VSTD::isinf(__b)) && _CUDA_VSTD::isfinite(__c)
             && _CUDA_VSTD::isfinite(__d))
    {
      __a = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__b) ? _Tp(1) : _Tp(0), __b);
      __x = numeric_limits<_Tp>::infinity() * (__a * __c + __b * __d);
      __y = numeric_limits<_Tp>::infinity() * (__b * __c - __a * __d);
    }
    else if (_CUDA_VSTD::isinf(__logbw) && __logbw > _Tp(0) && _CUDA_VSTD::isfinite(__a) && _CUDA_VSTD::isfinite(__b))
    {
      __c = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = _CUDA_VSTD::__constexpr_copysign(_CUDA_VSTD::isinf(__d) ? _Tp(1) : _Tp(0), __d);
      __x = _Tp(0) * (__a * __c + __b * __d);
      __y = _Tp(0) * (__b * __c - __a * __d);
    }
  }
#endif // LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_DIVISION
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator/(const complex<_Tp>& __x, const _Tp& __y)
{
  return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator/(const _Tp& __x, const complex<_Tp>& __y)
{
  complex<_Tp> __t(__x);
  __t /= __y;
  return __t;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator+(const complex<_Tp>& __x)
{
  return __x;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr complex<_Tp> operator-(const complex<_Tp>& __x)
{
  return complex<_Tp>(-__x.real(), -__x.imag());
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator==(const complex<_Tp>& __x, const _Tp& __y)
{
  return __x.real() == __y && __x.imag() == _Tp(0);
}

#if _CCCL_STD_VER <= 2017
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator==(const _Tp& __x, const complex<_Tp>& __y)
{
  return __x == __y.real() && _Tp(0) == __y.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
  return !(__x == __y);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
  return !(__x == __y);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
  return !(__x == __y);
}
#endif // _CCCL_STD_VER <= 2017

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API constexpr bool operator==(const complex<_Tp>& __x, const ::std::complex<_Up>& __y)
{
  return __x.real() == _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__y)
      && __x.imag() == _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__y);
}

#  if _CCCL_STD_VER <= 2017
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API constexpr bool operator==(const ::std::complex<_Up>& __x, const complex<_Tp>& __y)
{
  return __y.real() == _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__x)
      && __y.imag() == _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__x);
}

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const complex<_Tp>& __x, const ::std::complex<_Up>& __y)
{
  return !(__x == __y);
}

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const ::std::complex<_Up>& __x, const complex<_Tp>& __y)
{
  return !(__x == __y);
}
#  endif // _CCCL_STD_VER <= 2017
#endif // !_CCCL_COMPILER(NVRTC)

// real

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp real(const complex<_Tp>& __c)
{
  return __c.real();
}

// 26.3.7 values:

template <class _Tp, bool = _CCCL_TRAIT(is_integral, _Tp), bool = _CCCL_TRAIT(is_floating_point, _Tp)>
struct __cccl_complex_overload_traits
{};

// Integral Types
template <class _Tp>
struct __cccl_complex_overload_traits<_Tp, true, false>
{
  using _ValueType   = double;
  using _ComplexType = complex<double>;
};

// Floating point types
template <class _Tp>
struct __cccl_complex_overload_traits<_Tp, false, true>
{
  using _ValueType   = _Tp;
  using _ComplexType = complex<_Tp>;
};

template <class _Tp>
using __cccl_complex_value_type = typename __cccl_complex_overload_traits<_Tp>::_ValueType;

template <class _Tp>
using __cccl_complex_complex_type = typename __cccl_complex_overload_traits<_Tp>::_ComplexType;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_value_type<_Tp> real(_Tp __re)
{
  return __re;
}

// imag

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp imag(const complex<_Tp>& __c)
{
  return __c.imag();
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __cccl_complex_value_type<_Tp> imag(_Tp)
{
  return 0;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp, class _CharT, class _Traits>
::std::basic_istream<_CharT, _Traits>& operator>>(::std::basic_istream<_CharT, _Traits>& __is, complex<_Tp>& __x)
{
  ::std::complex<_Tp> __temp;
  __is >> __temp;
  __x = __temp;
  return __is;
}

template <class _Tp, class _CharT, class _Traits>
::std::basic_ostream<_CharT, _Traits>& operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const complex<_Tp>& __x)
{
  return __os << static_cast<::std::complex<_Tp>>(__x);
}
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_COMPLEX_H
