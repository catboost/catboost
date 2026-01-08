//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_NVBF16_H
#define _LIBCUDACXX___COMPLEX_NVBF16_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_NVBF16()

#  include <cuda/std/__cmath/nvbf16.h>
#  include <cuda/std/__complex/complex.h>
#  include <cuda/std/__complex/tuple.h>
#  include <cuda/std/__complex/vector_support.h>
#  include <cuda/std/__floating_point/nvfp_types.h>
#  include <cuda/std/__fwd/get.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_constructible.h>

#  if !_CCCL_COMPILER(NVRTC)
#    include <sstream> // for std::basic_ostringstream
#  endif // !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cccl/prologue.h>

// This is a workaround against the user defining macros __CUDA_NO_HALF_CONVERSIONS__ __CUDA_NO_HALF_OPERATORS__
namespace __cccl_internal
{
template <>
struct __is_non_narrowing_convertible<__nv_bfloat16, float>
{
  static constexpr bool value = true;
};

template <>
struct __is_non_narrowing_convertible<__nv_bfloat16, double>
{
  static constexpr bool value = true;
};

template <>
struct __is_non_narrowing_convertible<float, __nv_bfloat16>
{
  static constexpr bool value = true;
};

template <>
struct __is_non_narrowing_convertible<double, __nv_bfloat16>
{
  static constexpr bool value = true;
};
} // namespace __cccl_internal

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <>
inline constexpr size_t __complex_alignment_v<__nv_bfloat16> = alignof(__nv_bfloat162);

template <>
struct __type_to_vector<__nv_bfloat16>
{
  using __type = __nv_bfloat162;
};

template <>
struct __cccl_complex_overload_traits<__nv_bfloat16, false, false>
{
  using _ValueType   = __nv_bfloat16;
  using _ComplexType = complex<__nv_bfloat16>;
};

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_ALIGNAS(alignof(__nv_bfloat162)) complex<__nv_bfloat16>
{
  __nv_bfloat162 __repr_;

  template <class _Up>
  friend class complex;

  template <class _Up>
  friend struct __get_complex_impl;

  template <class _Tp>
  [[nodiscard]] _CCCL_API inline static __nv_bfloat16 __convert_to_bfloat16(const _Tp& __value) noexcept
  {
    return __value;
  }

  [[nodiscard]] _CCCL_API inline static __nv_bfloat16 __convert_to_bfloat16(const float& __value) noexcept
  {
    return ::__float2bfloat16(__value);
  }

  [[nodiscard]] _CCCL_API inline static __nv_bfloat16 __convert_to_bfloat16(const double& __value) noexcept
  {
    return ::__double2bfloat16(__value);
  }

public:
  using value_type = __nv_bfloat16;

  _CCCL_API inline complex(const value_type& __re = value_type(), const value_type& __im = value_type())
      : __repr_(__re, __im)
  {}

  template <class _Up, enable_if_t<__cccl_internal::__is_non_narrowing_convertible<value_type, _Up>::value, int> = 0>
  _CCCL_API inline complex(const complex<_Up>& __c)
      : __repr_(__convert_to_bfloat16(__c.real()), __convert_to_bfloat16(__c.imag()))
  {}

  template <class _Up,
            enable_if_t<!__cccl_internal::__is_non_narrowing_convertible<value_type, _Up>::value, int> = 0,
            enable_if_t<_CCCL_TRAIT(is_constructible, value_type, _Up), int>                           = 0>
  _CCCL_API inline explicit complex(const complex<_Up>& __c)
      : __repr_(__convert_to_bfloat16(__c.real()), __convert_to_bfloat16(__c.imag()))
  {}

  _CCCL_API inline complex& operator=(const value_type& __re)
  {
    __repr_.x = __re;
    __repr_.y = value_type();
    return *this;
  }

  template <class _Up>
  _CCCL_API inline complex& operator=(const complex<_Up>& __c)
  {
    __repr_.x = __convert_to_bfloat16(__c.real());
    __repr_.y = __convert_to_bfloat16(__c.imag());
    return *this;
  }

#  if !_CCCL_COMPILER(NVRTC)
  template <class _Up>
  _CCCL_API inline complex(const ::std::complex<_Up>& __other)
      : __repr_(_LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other), _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other))
  {}

  template <class _Up>
  _CCCL_API inline complex& operator=(const ::std::complex<_Up>& __other)
  {
    __repr_.x = _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other);
    __repr_.y = _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other);
    return *this;
  }

  _CCCL_HOST operator ::std::complex<value_type>() const
  {
    return {__repr_.x, __repr_.y};
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  [[nodiscard]] _CCCL_API inline value_type real() const
  {
    return __repr_.x;
  }
  [[nodiscard]] _CCCL_API inline value_type imag() const
  {
    return __repr_.y;
  }

  _CCCL_API inline void real(value_type __re)
  {
    __repr_.x = __re;
  }
  _CCCL_API inline void imag(value_type __im)
  {
    __repr_.y = __im;
  }

  // Those additional volatile overloads are meant to help with reductions in thrust
  [[nodiscard]] _CCCL_API inline value_type real() const volatile
  {
    return __repr_.x;
  }
  [[nodiscard]] _CCCL_API inline value_type imag() const volatile
  {
    return __repr_.y;
  }

  _CCCL_API inline complex& operator+=(const value_type& __re)
  {
    __repr_.x = ::__hadd(__repr_.x, __re);
    return *this;
  }
  _CCCL_API inline complex& operator-=(const value_type& __re)
  {
    __repr_.x = ::__hsub(__repr_.x, __re);
    return *this;
  }
  _CCCL_API inline complex& operator*=(const value_type& __re)
  {
    __repr_.x = ::__hmul(__repr_.x, __re);
    __repr_.y = ::__hmul(__repr_.y, __re);
    return *this;
  }
  _CCCL_API inline complex& operator/=(const value_type& __re)
  {
    __repr_.x = ::__hdiv(__repr_.x, __re);
    __repr_.y = ::__hdiv(__repr_.y, __re);
    return *this;
  }

  // We can utilize vectorized operations for those operators
  _CCCL_API inline friend complex& operator+=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = ::__hadd2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  _CCCL_API inline friend complex& operator-=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = ::__hsub2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  [[nodiscard]] _CCCL_API inline friend bool operator==(const complex& __lhs, const complex& __rhs) noexcept
  {
    return ::__hbeq2(__lhs.__repr_, __rhs.__repr_);
  }
};

template <> // complex<float>
template <> // complex<__half>
_CCCL_API inline complex<float>::complex(const complex<__nv_bfloat16>& __c)
    : __re_(::__bfloat162float(__c.real()))
    , __im_(::__bfloat162float(__c.imag()))
{}

template <> // complex<double>
template <> // complex<__half>
_CCCL_API inline complex<double>::complex(const complex<__nv_bfloat16>& __c)
    : __re_(::__bfloat162float(__c.real()))
    , __im_(::__bfloat162float(__c.imag()))
{}

template <> // complex<float>
template <> // complex<__nv_bfloat16>
_CCCL_API inline complex<float>& complex<float>::operator=(const complex<__nv_bfloat16>& __c)
{
  __re_ = ::__bfloat162float(__c.real());
  __im_ = ::__bfloat162float(__c.imag());
  return *this;
}

template <> // complex<double>
template <> // complex<__nv_bfloat16>
_CCCL_API inline complex<double>& complex<double>::operator=(const complex<__nv_bfloat16>& __c)
{
  __re_ = ::__bfloat162float(__c.real());
  __im_ = ::__bfloat162float(__c.imag());
  return *this;
}

template <>
struct __get_complex_impl<__nv_bfloat16>
{
  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr __nv_bfloat16& get(complex<__nv_bfloat16>& __z) noexcept
  {
    return (_Index == 0) ? __z.__repr_.x : __z.__repr_.y;
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr __nv_bfloat16&& get(complex<__nv_bfloat16>&& __z) noexcept
  {
    return _CUDA_VSTD::move((_Index == 0) ? __z.__repr_.x : __z.__repr_.y);
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr const __nv_bfloat16& get(const complex<__nv_bfloat16>& __z) noexcept
  {
    return (_Index == 0) ? __z.__repr_.x : __z.__repr_.y;
  }

  template <size_t _Index>
  [[nodiscard]] static _CCCL_API constexpr const __nv_bfloat16&& get(const complex<__nv_bfloat16>&& __z) noexcept
  {
    return _CUDA_VSTD::move((_Index == 0) ? __z.__repr_.x : __z.__repr_.y);
  }
};

#  if !_CCCL_COMPILER(NVRTC)
template <class _CharT, class _Traits>
::std::basic_istream<_CharT, _Traits>&
operator>>(::std::basic_istream<_CharT, _Traits>& __is, complex<__nv_bfloat16>& __x)
{
  ::std::complex<float> __temp;
  __is >> __temp;
  __x = __temp;
  return __is;
}

template <class _CharT, class _Traits>
::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const complex<__nv_bfloat16>& __x)
{
  return __os << complex<float>{__x};
}
#  endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_HAS_NVBF16()

#endif // _LIBCUDACXX___COMPLEX_NVBF16_H
