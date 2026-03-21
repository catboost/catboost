//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
#define _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename _Tp, typename _Lhs>
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _Lhs>
__multiply_extract_higher_bits_fallback(_Tp __x, _Lhs __y)
{
  using __ret_t         = _CUDA_VSTD::common_type_t<_Tp, _Lhs>;
  constexpr int __shift = _CUDA_VSTD::__num_bits_v<__ret_t> / 2;
  using __half_bits_t   = _CUDA_VSTD::__make_nbit_uint_t<_CUDA_VSTD::__num_bits_v<__ret_t>>;
  auto __x_high         = static_cast<__half_bits_t>(__x >> __shift);
  auto __x_low          = static_cast<__half_bits_t>(__x);
  auto __y_high         = static_cast<__half_bits_t>(__y >> __shift);
  auto __y_low          = static_cast<__half_bits_t>(__y);
  auto __p0             = __x_low * __y_low;
  auto __p1             = __x_low * __y_high;
  auto __p2             = __x_high * __y_low;
  auto __p3             = __x_high * __y_high;
  auto __mid            = __p1 + __p2;
  __half_bits_t __carry = (__mid < __p1);
  auto __po_half        = __p0 >> __shift;
  __mid                 = __mid + __po_half;
  __carry += (__mid < __po_half);
  return __p3 + (__mid >> __shift) + (__carry << __shift);
}

template <typename _Tp, typename _Lhs>
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _Lhs> __multiply_extract_higher_bits(_Tp __x, _Lhs __y)
{
  using _CUDA_VSTD::__cccl_is_integer_v;
  using _CUDA_VSTD::__num_bits_v;
  using _CUDA_VSTD::is_signed_v;
  static_assert(__cccl_is_integer_v<_Tp>, "__multiply_extract_higher_bits: T is required to be an integer type");
  static_assert(__cccl_is_integer_v<_Lhs>, "__multiply_extract_higher_bits: T is required to be an integer type");
  if constexpr (is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__x >= 0, "__x must be non-negative");
    _CCCL_ASSUME(__x >= 0);
  }
  if constexpr (is_signed_v<_Lhs>)
  {
    _CCCL_ASSERT(__y >= 0, "__y must be non-negative");
    _CCCL_ASSUME(__y >= 0);
  }
  using __ret_t = _CUDA_VSTD::common_type_t<_Tp, _Lhs>;
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) == sizeof(uint32_t) && sizeof(_Lhs) == sizeof(uint32_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__umulhi(static_cast<uint32_t>(__x), static_cast<uint32_t>(__y));));
    }
#if !_CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t) && sizeof(_Lhs) == sizeof(uint64_t))
    {
      NV_DISPATCH_TARGET(NV_IS_DEVICE, (return ::__umul64hi(static_cast<uint64_t>(__x), static_cast<uint64_t>(__y));));
    }
#endif // !_CCCL_HAS_INT128()
  }
  if constexpr (sizeof(__ret_t) < sizeof(uint64_t) || (sizeof(__ret_t) == sizeof(uint64_t) && _CCCL_HAS_INT128()))
  {
    constexpr auto __mul_bits = ::cuda::next_power_of_two(__num_bits_v<_Tp> + __num_bits_v<_Lhs>);
    using __larger_t          = _CUDA_VSTD::__make_nbit_uint_t<__mul_bits>;
    auto __ret                = (static_cast<__larger_t>(__x) * __y) >> (__mul_bits / 2);
    return static_cast<__ret_t>(__ret);
  }
  else
  {
    return ::cuda::__multiply_extract_higher_bits_fallback(__x, __y);
  }
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

// The implementation is based on the following references depending on the data type:
// - Hacker's Delight, Second Edition, Chapter 10
// - Labor of Division (Episode III): Faster Unsigned Division by Constants (libdivide)
//   https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
// - Classic Round-Up Variant of Fast Unsigned Division by Constants
//   https://arxiv.org/pdf/2412.03680
template <typename _Tp, bool _DivisorIsNeverOne = false>
class fast_mod_div
{
#if !_CCCL_HAS_INT128()
  using __max_supported_t = uint32_t;
#else
  using __max_supported_t = uint64_t;
#endif

  static_assert(_CUDA_VSTD::__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= sizeof(__max_supported_t),
                "fast_mod_div: T is required to be an integer type");

  using __unsigned_t = _CUDA_VSTD::make_unsigned_t<_Tp>;

public:
  fast_mod_div() = delete;

  _CCCL_API explicit fast_mod_div(_Tp __divisor1) noexcept
      : __divisor{__divisor1}
  {
    using _CUDA_VSTD::__num_bits_v;
    using __larger_t = _CUDA_VSTD::__make_nbit_uint_t<__num_bits_v<_Tp> * 2>;
    _CCCL_ASSERT(__divisor > 0, "divisor must be positive");
    _CCCL_ASSERT(!_DivisorIsNeverOne || __divisor1 != 1, "cuda::fast_mod_div: divisor must not be one");
    if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
    {
      __shift            = ::cuda::ceil_ilog2(__divisor) - 1; // is_pow2(x) ? log2(x) - 1 : log2(x)
      auto __k           = __num_bits_v<_Tp> + __shift; // k: [N, 2*N-2]
      auto __multiplier1 = ::cuda::ceil_div(__larger_t{1} << __k, __divisor); // ceil(2^k / divisor)
      __multiplier       = static_cast<__unsigned_t>(__multiplier1);
    }
    else
    {
      __shift = ::cuda::ilog2(__divisor);
      if (::cuda::is_power_of_two(__divisor))
      {
        __multiplier = 0;
        return;
      }
      const auto __k        = __num_bits_v<_Tp> + __shift;
      __multiplier          = ((__larger_t{1} << __k) + (__larger_t{1} << __shift)) / __divisor;
      auto __multiplier_low = (__larger_t{1} << __k) / __divisor;
      __add                 = (__multiplier_low == __multiplier);
    }
  }

  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend _CUDA_VSTD::common_type_t<_Tp, _Lhs>
  operator/(_Lhs __dividend, fast_mod_div<_Tp> __divisor1) noexcept
  {
    using _CUDA_VSTD::is_same_v;
    using _CUDA_VSTD::is_signed_v;
    using _CUDA_VSTD::is_unsigned_v;
    static_assert(_CUDA_VSTD::__cccl_is_integer_v<_Lhs> && sizeof(_Tp) <= sizeof(__max_supported_t),
                  "cuda::fast_mod_div: T is required to be an integer type");
    static_assert(sizeof(_Lhs) < sizeof(_Tp) || is_same_v<_Lhs, _Tp> || (is_signed_v<_Lhs> && is_unsigned_v<_Tp>),
                  "cuda::fast_mod_div: if dividend and divisor have the same size, dividend must be signed and divisor "
                  "must be unsigned");
    if constexpr (is_signed_v<_Lhs>)
    {
      _CCCL_ASSERT(__dividend >= 0, "dividend must be non-negative");
    }
    using __common_t    = _CUDA_VSTD::common_type_t<_Tp, _Lhs>;
    using _Up           = _CUDA_VSTD::make_unsigned_t<_Lhs>;
    const auto __div    = __divisor1.__divisor; // cannot use structure binding because of clang-14
    const auto __mul    = __divisor1.__multiplier;
    const auto __shift_ = __divisor1.__shift;
    auto __udividend    = static_cast<_Up>(__dividend);
    if constexpr (is_unsigned_v<_Tp>)
    {
      if (__mul == 0) // divisor is a power of two
      {
        return static_cast<__common_t>(__udividend >> __shift_);
      }
      // if dividend is a signed type, overflow is not possible
      if (is_signed_v<_Lhs> || __udividend != _CUDA_VSTD::numeric_limits<_Up>::max()) // avoid overflow
      {
        __udividend += static_cast<_Up>(__divisor1.__add);
      }
    }
    else if (!_DivisorIsNeverOne && __div == 1)
    {
      return static_cast<__common_t>(__dividend);
    }
    auto __higher_bits = ::cuda::__multiply_extract_higher_bits(__udividend, __mul);
    auto __quotient    = static_cast<__common_t>(__higher_bits >> __shift_);
    _CCCL_ASSERT(__quotient == static_cast<__common_t>(__dividend / __div), "wrong __quotient");
    return __quotient;
  }

  template <typename _Lhs>
  [[nodiscard]] _CCCL_API friend _CUDA_VSTD::common_type_t<_Tp, _Lhs>
  operator%(_Lhs __dividend, fast_mod_div<_Tp> __divisor1) noexcept
  {
    return __dividend - (__dividend / __divisor1) * __divisor1.__divisor;
  }

  [[nodiscard]] _CCCL_API operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__divisor);
  }

private:
  _Tp __divisor             = 1;
  __unsigned_t __multiplier = 0;
  unsigned __add            = 0;
  int __shift               = 0;
};

/***********************************************************************************************************************
 * Non-member functions
 **********************************************************************************************************************/

template <typename _Tp, typename _Lhs>
[[nodiscard]] _CCCL_API _CUDA_VSTD::pair<_Tp, _Lhs> div(_Tp __dividend, fast_mod_div<_Lhs> __divisor) noexcept
{
  auto __quotient  = __dividend / __divisor;
  auto __remainder = __dividend - __quotient * __divisor;
  return {__quotient, __remainder};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
