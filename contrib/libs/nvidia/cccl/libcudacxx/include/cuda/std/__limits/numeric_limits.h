//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_H
#define _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cfloat>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum float_round_style
{
  round_indeterminate       = -1,
  round_toward_zero         = 0,
  round_to_nearest          = 1,
  round_toward_infinity     = 2,
  round_toward_neg_infinity = 3
};

enum _LIBCUDACXX_DEPRECATED_IN_CXX23 float_denorm_style
{
  denorm_indeterminate = -1,
  denorm_absent        = 0,
  denorm_present       = 1
};

enum class __numeric_limits_type
{
  __integral,
  __bool,
  __floating_point,
  __other,
};

template <class _Tp>
_CCCL_API constexpr __numeric_limits_type __make_numeric_limits_type()
{
  if constexpr (_CCCL_TRAIT(is_same, _Tp, bool))
  {
    return __numeric_limits_type::__bool;
  }
  else if constexpr (_CCCL_TRAIT(is_integral, _Tp))
  {
    return __numeric_limits_type::__integral;
  }
  else if constexpr (_CCCL_TRAIT(is_floating_point, _Tp) || _CCCL_TRAIT(__is_extended_floating_point, _Tp))
  {
    return __numeric_limits_type::__floating_point;
  }
  else
  {
    return __numeric_limits_type::__other;
  }
}

template <class _Tp, __numeric_limits_type = __make_numeric_limits_type<_Tp>()>
class __numeric_limits_impl
{
public:
  using type = _Tp;

  static constexpr bool is_specialized = false;
  _CCCL_API static constexpr type min() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return type();
  }

  static constexpr int digits       = 0;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 0;
  static constexpr bool is_signed   = false;
  static constexpr bool is_integer  = false;
  static constexpr bool is_exact    = false;
  static constexpr int radix        = 0;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type();
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                             = false;
  static constexpr bool has_quiet_NaN                                            = false;
  static constexpr bool has_signaling_NaN                                        = false;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type();
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = false;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

// MSVC warns about overflowing left shift
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4309)
template <class _Tp, int __digits, bool _IsSigned>
struct __int_min
{
  static constexpr _Tp value = static_cast<_Tp>(_Tp(1) << __digits);
};
_CCCL_DIAG_POP

template <class _Tp, int __digits>
struct __int_min<_Tp, __digits, false>
{
  static constexpr _Tp value = _Tp(0);
};

template <class _Tp>
class __numeric_limits_impl<_Tp, __numeric_limits_type::__integral>
{
public:
  using type = _Tp;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = type(-1) < type(0);
  static constexpr int digits       = static_cast<int>(sizeof(type) * CHAR_BIT - is_signed);
  static constexpr int digits10     = digits * 3 / 10;
  static constexpr int max_digits10 = 0;
  _CCCL_API static constexpr type min() noexcept
  {
    return __int_min<type, digits, is_signed>::value;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return is_signed ? type(type(~0) ^ min()) : type(~0);
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return min();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_exact   = true;
  static constexpr int radix       = 2;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type(0);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                             = false;
  static constexpr bool has_quiet_NaN                                            = false;
  static constexpr bool has_signaling_NaN                                        = false;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type(0);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = !is_signed;

#if (_CCCL_ARCH(X86_64) && _CCCL_OS(LINUX)) || defined(__pnacl__) || defined(__wasm__)
  static constexpr bool traps = true;
#else
  static constexpr bool traps = false;
#endif
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

template <>
class __numeric_limits_impl<bool, __numeric_limits_type::__bool>
{
public:
  using type = bool;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = false;
  static constexpr int digits       = 1;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 0;

  _CCCL_API static constexpr type min() noexcept
  {
    return false;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return true;
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return min();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_exact   = true;
  static constexpr int radix       = 2;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type(0);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                             = false;
  static constexpr bool has_quiet_NaN                                            = false;
  static constexpr bool has_signaling_NaN                                        = false;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type(0);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

template <>
class __numeric_limits_impl<float, __numeric_limits_type::__floating_point>
{
public:
  using type = float;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = FLT_MANT_DIG;
  static constexpr int digits10     = FLT_DIG;
  static constexpr int max_digits10 = 2 + (digits * 30103l) / 100000l;
  _CCCL_API static constexpr type min() noexcept
  {
    return FLT_MIN;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return FLT_MAX;
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return -max();
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = FLT_RADIX;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return FLT_EPSILON;
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return 0.5F;
  }

  static constexpr int min_exponent   = FLT_MIN_EXP;
  static constexpr int min_exponent10 = FLT_MIN_10_EXP;
  static constexpr int max_exponent   = FLT_MAX_EXP;
  static constexpr int max_exponent10 = FLT_MAX_10_EXP;

  static constexpr bool has_infinity                                             = true;
  static constexpr bool has_quiet_NaN                                            = true;
  static constexpr bool has_signaling_NaN                                        = true;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_present;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;

#if defined(_CCCL_BUILTIN_HUGE_VALF)
  _CCCL_API static constexpr type infinity() noexcept
  {
    return _CCCL_BUILTIN_HUGE_VALF();
  }
#else // ^^^ _CCCL_BUILTIN_HUGE_VALF ^^^ // vvv !_CCCL_BUILTIN_HUGE_VALF vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type infinity() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7f800000);
  }
#endif // !_CCCL_BUILTIN_HUGE_VALF
#if defined(_CCCL_BUILTIN_NANF)
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return _CCCL_BUILTIN_NANF("");
  }
#else // ^^^ _CCCL_BUILTIN_NANF ^^^ // vvv !_CCCL_BUILTIN_NANF vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7fc00000);
  }
#endif // !_CCCL_BUILTIN_NANF
#if defined(_CCCL_BUILTIN_NANSF)
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return _CCCL_BUILTIN_NANSF("");
  }
#else // ^^^ _CCCL_BUILTIN_NANSF ^^^ // vvv !_CCCL_BUILTIN_NANSF vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type signaling_NaN() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7fa00000);
  }
#endif // !_CCCL_BUILTIN_NANSF
  _CCCL_API static constexpr type denorm_min() noexcept
  {
#if defined(FLT_TRUE_MIN)
    return FLT_TRUE_MIN;
#else // ^^^ FLT_TRUE_MIN ^^^ // vvv !FLT_TRUE_MIN vvv
    return __FLT_DENORM_MIN__;
#endif // ^^^ !FLT_TRUE_MIN ^^^
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

template <>
class __numeric_limits_impl<double, __numeric_limits_type::__floating_point>
{
public:
  using type = double;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = DBL_MANT_DIG;
  static constexpr int digits10     = DBL_DIG;
  static constexpr int max_digits10 = 2 + (digits * 30103l) / 100000l;
  _CCCL_API static constexpr type min() noexcept
  {
    return DBL_MIN;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return DBL_MAX;
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return -max();
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = FLT_RADIX;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return DBL_EPSILON;
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return 0.5;
  }

  static constexpr int min_exponent   = DBL_MIN_EXP;
  static constexpr int min_exponent10 = DBL_MIN_10_EXP;
  static constexpr int max_exponent   = DBL_MAX_EXP;
  static constexpr int max_exponent10 = DBL_MAX_10_EXP;

  static constexpr bool has_infinity                                             = true;
  static constexpr bool has_quiet_NaN                                            = true;
  static constexpr bool has_signaling_NaN                                        = true;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_present;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;

#if defined(_CCCL_BUILTIN_HUGE_VAL)
  _CCCL_API static constexpr type infinity() noexcept
  {
    return _CCCL_BUILTIN_HUGE_VAL();
  }
#else // ^^^ _CCCL_BUILTIN_HUGE_VAL ^^^ // vvv !_CCCL_BUILTIN_HUGE_VAL vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type infinity() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7ff0000000000000);
  }
#endif // !_CCCL_BUILTIN_HUGE_VAL
#if defined(_CCCL_BUILTIN_NAN)
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return _CCCL_BUILTIN_NAN("");
  }
#else // ^^^ _CCCL_BUILTIN_NAN ^^^ // vvv !_CCCL_BUILTIN_NAN vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7ff8000000000000);
  }
#endif // !_CCCL_BUILTIN_NAN
#if defined(_CCCL_BUILTIN_NANS)
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return _CCCL_BUILTIN_NANS("");
  }
#else // ^^^ _CCCL_BUILTIN_NANS ^^^ // vvv !_CCCL_BUILTIN_NANS vvv
  _CCCL_API inline static _LIBCUDACXX_CONSTEXPR_BIT_CAST type signaling_NaN() noexcept
  {
    return _CUDA_VSTD::bit_cast<type>(0x7ff4000000000000);
  }
#endif // !_CCCL_BUILTIN_NANS
  _CCCL_API static constexpr type denorm_min() noexcept
  {
#if defined(DBL_TRUE_MIN)
    return DBL_TRUE_MIN;
#else // ^^^ DBL_TRUE_MIN ^^^ // vvv !DBL_TRUE_MIN vvv
    return __DBL_DENORM_MIN__;
#endif // ^^^ !DBL_TRUE_MIN ^^^
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

template <>
class __numeric_limits_impl<long double, __numeric_limits_type::__floating_point>
{
#if _CCCL_HAS_LONG_DOUBLE()

public:
  using type = long double;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = LDBL_MANT_DIG;
  static constexpr int digits10     = LDBL_DIG;
  static constexpr int max_digits10 = 2 + (digits * 30103l) / 100000l;
  _CCCL_API static constexpr type min() noexcept
  {
    return LDBL_MIN;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return LDBL_MAX;
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return -max();
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = FLT_RADIX;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return LDBL_EPSILON;
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return 0.5L;
  }

  static constexpr int min_exponent   = LDBL_MIN_EXP;
  static constexpr int min_exponent10 = LDBL_MIN_10_EXP;
  static constexpr int max_exponent   = LDBL_MAX_EXP;
  static constexpr int max_exponent10 = LDBL_MAX_10_EXP;

  static constexpr bool has_infinity                                             = true;
  static constexpr bool has_quiet_NaN                                            = true;
  static constexpr bool has_signaling_NaN                                        = true;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_present;
  _LIBCUDACXX_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return _CCCL_BUILTIN_HUGE_VALL();
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return _CCCL_BUILTIN_NANL("");
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return _CCCL_BUILTIN_NANSL("");
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
#  if defined(LDBL_TRUE_MIN)
    return LDBL_TRUE_MIN;
#  else // ^^^ LDBL_TRUE_MIN ^^^ // vvv !LDBL_TRUE_MIN vvv
    return __LDBL_DENORM_MIN__;
#  endif // ^^^ !LDBL_TRUE_MIN ^^^
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
#endif // _CCCL_HAS_LONG_DOUBLE()
};

template <class _Tp>
class numeric_limits : public __numeric_limits_impl<_Tp>
{};

template <class _Tp>
class numeric_limits<const _Tp> : public numeric_limits<_Tp>
{};

template <class _Tp>
class numeric_limits<volatile _Tp> : public numeric_limits<_Tp>
{};

template <class _Tp>
class numeric_limits<const volatile _Tp> : public numeric_limits<_Tp>
{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_H
