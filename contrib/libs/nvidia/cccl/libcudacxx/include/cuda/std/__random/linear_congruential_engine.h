//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
#define _LIBCUDACXX___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/iosfwd>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <uint64_t __a,
          uint64_t __c,
          uint64_t __m,
          uint64_t _Mp,
          bool _MightOverflow = (__a != 0 && __m != 0 && __m - 1 > (_Mp - __c) / __a),
          bool _OverflowOK    = ((__m | (__m - 1)) > __m), // m = 2^n
          bool _SchrageOK     = (__a != 0 && __m != 0 && __m % __a <= __m / __a)> // r <= q
struct __lce_alg_picker
{
  static_assert(__a != 0 || __m != 0 || !_MightOverflow || _OverflowOK || _SchrageOK,
                "The current values of a, c, and m cannot generate a number "
                "within bounds of linear_congruential_engine.");

  static constexpr const bool __use_schrage = _MightOverflow && !_OverflowOK && _SchrageOK;
};

template <uint64_t __a,
          uint64_t __c,
          uint64_t __m,
          uint64_t _Mp,
          bool _UseSchrage = __lce_alg_picker<__a, __c, __m, _Mp>::__use_schrage>
struct __lce_ta;

// 64

template <uint64_t __a, uint64_t __c, uint64_t __m>
struct __lce_ta<__a, __c, __m, ~uint64_t{0}, true>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    // Schrage's algorithm
    constexpr result_type __q = __m / __a;
    constexpr result_type __r = __m % __a;
    const result_type __t0    = __a * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __m - __t1;
    __x += __c - (__x >= __m - __c) * __m;
    return __x;
  }
};

template <uint64_t __a, uint64_t __m>
struct __lce_ta<__a, 0, __m, ~uint64_t{0}, true>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    // Schrage's algorithm
    constexpr result_type __q = __m / __a;
    constexpr result_type __r = __m % __a;
    const result_type __t0    = __a * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __m - __t1;
    return __x;
  }
};

template <uint64_t __a, uint64_t __c, uint64_t __m>
struct __lce_ta<__a, __c, __m, ~uint64_t{0}, false>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    return (__a * __x + __c) % __m;
  }
};

template <uint64_t __a, uint64_t __c>
struct __lce_ta<__a, __c, 0, ~uint64_t{0}, false>
{
  using result_type = uint64_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    return __a * __x + __c;
  }
};

// 32

template <uint64_t _Ap, uint64_t _Cp, uint64_t _Mp>
struct __lce_ta<_Ap, _Cp, _Mp, ~uint32_t{0}, true>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    constexpr auto __a = static_cast<result_type>(_Ap);
    constexpr auto __c = static_cast<result_type>(_Cp);
    constexpr auto __m = static_cast<result_type>(_Mp);
    // Schrage's algorithm
    constexpr result_type __q = __m / __a;
    constexpr result_type __r = __m % __a;
    const result_type __t0    = __a * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __m - __t1;
    __x += __c - (__x >= __m - __c) * __m;
    return __x;
  }
};

template <uint64_t _Ap, uint64_t _Mp>
struct __lce_ta<_Ap, 0, _Mp, ~uint32_t{0}, true>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    constexpr result_type __a = static_cast<result_type>(_Ap);
    constexpr result_type __m = static_cast<result_type>(_Mp);
    // Schrage's algorithm
    constexpr result_type __q = __m / __a;
    constexpr result_type __r = __m % __a;
    const result_type __t0    = __a * (__x % __q);
    const result_type __t1    = __r * (__x / __q);
    __x                       = __t0 + (__t0 < __t1) * __m - __t1;
    return __x;
  }
};

template <uint64_t _Ap, uint64_t _Cp, uint64_t _Mp>
struct __lce_ta<_Ap, _Cp, _Mp, ~uint32_t{0}, false>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    constexpr result_type __a = static_cast<result_type>(_Ap);
    constexpr result_type __c = static_cast<result_type>(_Cp);
    constexpr result_type __m = static_cast<result_type>(_Mp);
    return (__a * __x + __c) % __m;
  }
};

template <uint64_t _Ap, uint64_t _Cp>
struct __lce_ta<_Ap, _Cp, 0, ~uint32_t{0}, false>
{
  using result_type = uint32_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    constexpr result_type __a = static_cast<result_type>(_Ap);
    constexpr result_type __c = static_cast<result_type>(_Cp);
    return __a * __x + __c;
  }
};

// 16

template <uint64_t __a, uint64_t __c, uint64_t __m, bool __b>
struct __lce_ta<__a, __c, __m, static_cast<uint16_t>(~0), __b>
{
  using result_type = uint16_t;
  [[nodiscard]] _CCCL_API static result_type next(result_type __x) noexcept
  {
    return static_cast<result_type>(__lce_ta<__a, __c, __m, ~uint32_t{0}>::next(__x));
  }
};

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
class _CCCL_TYPE_VISIBILITY_DEFAULT linear_congruential_engine;

#if 0 // Not Implemented
template <class _CharT, class _Traits, class _Up, _Up _Ap, _Up _Cp, _Up _Np>
_CCCL_API basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const linear_congruential_engine<_Up, _Ap, _Cp, _Np>&);

template <class _CharT, class _Traits, class _Up, _Up _Ap, _Up _Cp, _Up _Np>
_CCCL_API basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, linear_congruential_engine<_Up, _Ap, _Cp, _Np>& __x);
#endif //

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
class _CCCL_TYPE_VISIBILITY_DEFAULT linear_congruential_engine
{
public:
  // types
  using result_type = _UIntType;

private:
  result_type __x_;

  static constexpr const result_type _Mp = result_type(~0);

  static_assert(__m == 0 || __a < __m, "linear_congruential_engine invalid parameters");
  static_assert(__m == 0 || __c < __m, "linear_congruential_engine invalid parameters");
  static_assert(is_unsigned_v<_UIntType>, "_UIntType must be uint32_t type");

public:
  static constexpr const result_type _Min = __c == 0u ? 1u : 0u;
  static constexpr const result_type _Max = __m - _UIntType(1u);
  static_assert(_Min < _Max, "linear_congruential_engine invalid parameters");

  // engine characteristics
  static constexpr const result_type multiplier = __a;
  static constexpr const result_type increment  = __c;
  static constexpr const result_type modulus    = __m;
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return _Min;
  }
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return _Max;
  }
  static constexpr const result_type default_seed = 1u;

  // constructors and seeding functions
  _CCCL_API linear_congruential_engine() noexcept
      : linear_congruential_engine(default_seed)
  {}
  _CCCL_API explicit linear_congruential_engine(result_type __s) noexcept
  {
    seed(__s);
  }

  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, linear_congruential_engine>, int> = 0>
  _CCCL_API explicit linear_congruential_engine(_Sseq& __q) noexcept
  {
    seed(__q);
  }
  _CCCL_API void seed(result_type __s = default_seed)
  {
    seed(integral_constant<bool, __m == 0>(), integral_constant<bool, __c == 0>(), __s);
  }
  template <class _Sseq, enable_if_t<__is_seed_sequence<_Sseq, linear_congruential_engine>, int> = 0>
  _CCCL_API void seed(_Sseq& __q) noexcept
  {
    __seed(__q,
           integral_constant<uint32_t,
                             1 + (__m == 0 ? (sizeof(result_type) * CHAR_BIT - 1) / 32 : (__m > 0x100000000ull))>());
  }

  // generating functions
  [[nodiscard]] _CCCL_API result_type operator()() noexcept
  {
    return __x_ = static_cast<result_type>(__lce_ta<__a, __c, __m, _Mp>::next(__x_));
  }
  _CCCL_API void discard(uint64_t __z) noexcept
  {
    for (; __z; --__z)
    {
      (void) operator()();
    }
  }

  [[nodiscard]] _CCCL_API friend bool
  operator==(const linear_congruential_engine& __x, const linear_congruential_engine& __y) noexcept
  {
    return __x.__x_ == __y.__x_;
  }
  [[nodiscard]] _CCCL_API friend bool
  operator!=(const linear_congruential_engine& __x, const linear_congruential_engine& __y) noexcept
  {
    return !(__x == __y);
  }

private:
  _CCCL_API void seed(true_type, true_type, result_type __s) noexcept
  {
    __x_ = __s == 0 ? 1 : __s;
  }
  _CCCL_API void seed(true_type, false_type, result_type __s) noexcept
  {
    __x_ = __s;
  }
  _CCCL_API void seed(false_type, true_type, result_type __s) noexcept
  {
    __x_ = __s % __m == 0 ? 1 : __s % __m;
  }
  _CCCL_API void seed(false_type, false_type, result_type __s) noexcept
  {
    __x_ = __s % __m;
  }

  template <class _Sseq>
  _CCCL_API void __seed(_Sseq& __q, integral_constant<uint32_t, 1>) noexcept;
  template <class _Sseq>
  _CCCL_API void __seed(_Sseq& __q, integral_constant<uint32_t, 2>) noexcept;

#if 0 // Not Implemented
  template <class _CharT, class _Traits, class _Up, _Up _Ap, _Up _Cp, _Up _Np>
  friend basic_ostream<_CharT, _Traits>&
  operator<<(basic_ostream<_CharT, _Traits>& __os, const linear_congruential_engine<_Up, _Ap, _Cp, _Np>&);

  template <class _CharT, class _Traits, class _Up, _Up _Ap, _Up _Cp, _Up _Np>
  friend basic_istream<_CharT, _Traits>&
  operator>>(basic_istream<_CharT, _Traits>& __is, linear_congruential_engine<_Up, _Ap, _Cp, _Np>& __x);
#endif // Not Implemented
};

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
constexpr const typename linear_congruential_engine<_UIntType, __a, __c, __m>::result_type
  linear_congruential_engine<_UIntType, __a, __c, __m>::multiplier;

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
constexpr const typename linear_congruential_engine<_UIntType, __a, __c, __m>::result_type
  linear_congruential_engine<_UIntType, __a, __c, __m>::increment;

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
constexpr const typename linear_congruential_engine<_UIntType, __a, __c, __m>::result_type
  linear_congruential_engine<_UIntType, __a, __c, __m>::modulus;

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
constexpr const typename linear_congruential_engine<_UIntType, __a, __c, __m>::result_type
  linear_congruential_engine<_UIntType, __a, __c, __m>::default_seed;

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
template <class _Sseq>
_CCCL_API void
linear_congruential_engine<_UIntType, __a, __c, __m>::__seed(_Sseq& __q, integral_constant<uint32_t, 1>) noexcept
{
  constexpr uint32_t __k = 1;
  uint32_t __ar[__k + 3];
  __q.generate(__ar, __ar + __k + 3);
  result_type __s = static_cast<result_type>(__ar[3] % __m);
  __x_            = __c == 0 && __s == 0 ? result_type(1) : __s;
}

template <class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
template <class _Sseq>
_CCCL_API void
linear_congruential_engine<_UIntType, __a, __c, __m>::__seed(_Sseq& __q, integral_constant<uint32_t, 2>) noexcept
{
  constexpr uint32_t __k = 2;
  uint32_t __ar[__k + 3];
  __q.generate(__ar, __ar + __k + 3);
  result_type __s = static_cast<result_type>((__ar[3] + ((uint64_t) __ar[4] << 32)) % __m);
  __x_            = __c == 0 && __s == 0 ? result_type(1) : __s;
}

#if 0 // Not Implemented
template <class _CharT, class _Traits, class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
_CCCL_API basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const linear_congruential_engine<_UIntType, __a, __c, __m>& __x)
{
  __save_flags<_CharT, _Traits> __lx(__os);
  using _Ostream = basic_ostream<_CharT, _Traits>;
  __os.flags(_Ostream::dec | _Ostream::left);
  __os.fill(__os.widen(' '));
  return __os << __x.__x_;
}

template <class _CharT, class _Traits, class _UIntType, _UIntType __a, _UIntType __c, _UIntType __m>
_CCCL_API basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, linear_congruential_engine<_UIntType, __a, __c, __m>& __x)
{
  __save_flags<_CharT, _Traits> __lx(__is);
  using _Istream = basic_istream<_CharT, _Traits>;
  __is.flags(_Istream::dec | _Istream::skipws);
  _UIntType __t;
  __is >> __t;
  if (!__is.fail())
  {
    __x.__x_ = __t;
  }
  return __is;
}
#endif // Not Implemented

using minstd_rand0 = linear_congruential_engine<uint_fast32_t, 16807, 0, 2147483647>;
using minstd_rand  = linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANDOM_LINEAR_CONGRUENTIAL_ENGINE_H
