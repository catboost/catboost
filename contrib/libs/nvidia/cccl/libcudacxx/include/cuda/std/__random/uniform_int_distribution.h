//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_UNIFORM_INT_DISTRIBUTION_H
#define _LIBCUDACXX___RANDOM_UNIFORM_INT_DISTRIBUTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__random/is_valid.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Engine, class _UIntType>
class __independent_bits_engine
{
public:
  // types
  using result_type = _UIntType;

private:
  using _Engine_result_type = typename _Engine::result_type;
  using _Working_result_type =
    conditional_t<sizeof(_Engine_result_type) <= sizeof(result_type), result_type, _Engine_result_type>;

  _Engine& __e_;
  size_t __w_;
  size_t __w0_;
  size_t __n_;
  size_t __n0_;
  _Working_result_type __y0_;
  _Working_result_type __y1_;
  _Engine_result_type __mask0_;
  _Engine_result_type __mask1_;

  static constexpr const _Working_result_type _Rp = _Engine::max() - _Engine::min() + _Working_result_type(1);
  static constexpr const size_t __m               = _CUDA_VSTD::__bit_log2<_Working_result_type>(_Rp);
  static constexpr const size_t _WDt              = numeric_limits<_Working_result_type>::digits;
  static constexpr const size_t _EDt              = numeric_limits<_Engine_result_type>::digits;

public:
  // constructors and seeding functions
  _CCCL_API __independent_bits_engine(_Engine& __e, size_t __w) noexcept
      : __e_(__e)
      , __w_(__w)
  {
    __n_  = __w_ / __m + (__w_ % __m != 0);
    __w0_ = __w_ / __n_;
    if constexpr (_Rp == 0)
    {
      __y0_ = _Rp;
    }
    else if (__w0_ < _WDt)
    {
      __y0_ = (_Rp >> __w0_) << __w0_;
    }
    else
    {
      __y0_ = 0;
    }
    if (_Rp - __y0_ > __y0_ / __n_)
    {
      ++__n_;
      __w0_ = __w_ / __n_;
      if (__w0_ < _WDt)
      {
        __y0_ = (_Rp >> __w0_) << __w0_;
      }
      else
      {
        __y0_ = 0;
      }
    }
    __n0_ = __n_ - __w_ % __n_;
    if (__w0_ < _WDt - 1)
    {
      __y1_ = (_Rp >> (__w0_ + 1)) << (__w0_ + 1);
    }
    else
    {
      __y1_ = 0;
    }
    __mask0_ = __w0_ > 0 ? _Engine_result_type(~0) >> (_EDt - __w0_) : _Engine_result_type(0);
    __mask1_ = __w0_ < _EDt - 1 ? _Engine_result_type(~0) >> (_EDt - (__w0_ + 1)) : _Engine_result_type(~0);
  }

  // generating functions
  [[nodiscard]] _CCCL_API result_type operator()() noexcept
  {
    if constexpr (_Rp == 0)
    {
      return static_cast<result_type>(__e_() & __mask0_);
    }

    constexpr size_t __w_rt = numeric_limits<result_type>::digits;
    result_type __sp        = 0;
    for (size_t __k = 0; __k < __n0_; ++__k)
    {
      _Engine_result_type __u;
      do
      {
        __u = __e_() - _Engine::min();
      } while (__u >= __y0_);
      if (__w0_ < __w_rt)
      {
        __sp <<= __w0_;
      }
      else
      {
        __sp = 0;
      }
      __sp += __u & __mask0_;
    }

    for (size_t __k = __n0_; __k < __n_; ++__k)
    {
      _Engine_result_type __u;
      do
      {
        __u = __e_() - _Engine::min();
      } while (__u >= __y1_);
      if (__w0_ < __w_rt - 1)
      {
        __sp <<= __w0_ + 1;
      }
      else
      {
        __sp = 0;
      }
      __sp += __u & __mask1_;
    }
    return __sp;
  }
};

template <class _IntType = int>
class uniform_int_distribution
{
  static_assert(__libcpp_random_is_valid_inttype<_IntType>, "IntType must be a supported integer type");

public:
  // types
  using result_type = _IntType;

  class param_type
  {
    result_type __a_;
    result_type __b_;

  public:
    using distribution_type = uniform_int_distribution;

    _CCCL_API explicit param_type(result_type __a = 0, result_type __b = (numeric_limits<result_type>::max)()) noexcept
        : __a_(__a)
        , __b_(__b)
    {}

    [[nodiscard]] _CCCL_API result_type a() const noexcept
    {
      return __a_;
    }
    [[nodiscard]] _CCCL_API result_type b() const noexcept
    {
      return __b_;
    }

    [[nodiscard]] _CCCL_API friend bool operator==(const param_type& __x, const param_type& __y) noexcept
    {
      return __x.__a_ == __y.__a_ && __x.__b_ == __y.__b_;
    }
    [[nodiscard]] _CCCL_API friend bool operator!=(const param_type& __x, const param_type& __y) noexcept
    {
      return !(__x == __y);
    }
  };

private:
  param_type __p_;

public:
  // constructors and reset functions
  _CCCL_API uniform_int_distribution() noexcept
      : uniform_int_distribution(0)
  {}
  _CCCL_API explicit uniform_int_distribution(result_type __a,
                                              result_type __b = (numeric_limits<result_type>::max)()) noexcept
      : __p_(param_type(__a, __b))
  {}
  _CCCL_API explicit uniform_int_distribution(const param_type& __p) noexcept
      : __p_(__p)
  {}
  _CCCL_API void reset() noexcept {}

  // generating functions
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g) noexcept
  {
    return (*this)(__g, __p_);
  }
  _CCCL_EXEC_CHECK_DISABLE
  template <class _URng>
  [[nodiscard]] _CCCL_API result_type operator()(_URng& __g, const param_type& __p) noexcept
  {
    static_assert(__libcpp_random_is_valid_urng<_URng>, "");
    using _UIntType = conditional_t<sizeof(result_type) <= sizeof(uint32_t), uint32_t, make_unsigned_t<result_type>>;
    const _UIntType __rp = _UIntType(__p.b()) - _UIntType(__p.a()) + _UIntType(1);
    if (__rp == 1)
    {
      return __p.a();
    }
    constexpr size_t __dt = numeric_limits<_UIntType>::digits;

    using _Eng = __independent_bits_engine<_URng, _UIntType>;
    if (__rp == 0)
    {
      return static_cast<result_type>(_Eng(__g, __dt)());
    }

    size_t __w = __dt - _CUDA_VSTD::countl_zero(__rp) - 1;
    if ((__rp & ((numeric_limits<_UIntType>::max)() >> (__dt - __w))) != 0)
    {
      ++__w;
    }

    _Eng __e(__g, __w);
    _UIntType __u;
    do
    {
      __u = __e();
    } while (__u >= __rp);

    return static_cast<result_type>(__u + __p.a());
  }

  // property functions
  [[nodiscard]] _CCCL_API result_type a() const noexcept
  {
    return __p_.a();
  }
  [[nodiscard]] _CCCL_API result_type b() const noexcept
  {
    return __p_.b();
  }

  [[nodiscard]] _CCCL_API param_type param() const noexcept
  {
    return __p_;
  }
  _CCCL_API void param(const param_type& __p) noexcept
  {
    __p_ = __p;
  }

  [[nodiscard]] _CCCL_API result_type min() const noexcept
  {
    return a();
  }
  [[nodiscard]] _CCCL_API result_type max() const noexcept
  {
    return b();
  }

  [[nodiscard]] _CCCL_API friend bool
  operator==(const uniform_int_distribution& __x, const uniform_int_distribution& __y) noexcept
  {
    return __x.__p_ == __y.__p_;
  }
  [[nodiscard]] _CCCL_API friend bool
  operator!=(const uniform_int_distribution& __x, const uniform_int_distribution& __y) noexcept
  {
    return !(__x == __y);
  }
};

#if 0 // Implement stream operators
template <class _CharT, class _Traits, class _IT>
_CCCL_API basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const uniform_int_distribution<_IT>& __x)
{
  __save_flags<_CharT, _Traits> __lx(__os);
  using _Ostream = basic_ostream<_CharT, _Traits>;
  __os.flags(_Ostream::dec | _Ostream::left);
  _CharT __sp = __os.widen(' ');
  __os.fill(__sp);
  return __os << __x.a() << __sp << __x.b();
}

template <class _CharT, class _Traits, class _IT>
_CCCL_API basic_istream<_CharT, _Traits>&
operator>>(basic_istream<_CharT, _Traits>& __is, uniform_int_distribution<_IT>& __x)
{
  using _Eng = uniform_int_distribution<_IT>;
  using result_type = typename _Eng::result_type;
  using param_type = typename _Eng::param_type;
  __save_flags<_CharT, _Traits> __lx(__is);
  using _Istream = basic_istream<_CharT, _Traits>;
  __is.flags(_Istream::dec | _Istream::skipws);
  result_type __a;
  result_type __b;
  __is >> __a >> __b;
  if (!__is.fail())
  {
    __x.param(param_type(__a, __b));
  }
  return __is;
}
#endif // Implement stream operators

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANDOM_UNIFORM_INT_DISTRIBUTION_H
