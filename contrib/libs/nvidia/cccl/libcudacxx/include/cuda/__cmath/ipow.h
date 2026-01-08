//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_IPOW
#define _LIBCUDACXX___CMATH_IPOW

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp, class _Ep>
[[nodiscard]] _CCCL_API constexpr _Tp __cccl_ipow_impl_base_pow2(_Tp __b, _Ep __e) noexcept
{
  const auto __shift = static_cast<int>(__e - 1) * ::cuda::ilog2(__b);
  const auto __lz    = _CUDA_VSTD::countl_zero(__b);
  return (__shift >= __lz) ? _Tp{0} : (_Tp{__b} << __shift);
}

template <class _Tp, class _Ep>
[[nodiscard]] _CCCL_API constexpr _Tp __cccl_ipow_impl(_Tp __b, _Ep __e) noexcept
{
  static_assert(_CUDA_VSTD::is_unsigned_v<_Tp>);

  if (_CUDA_VSTD::has_single_bit(__b))
  {
    return ::cuda::__cccl_ipow_impl_base_pow2(__b, __e);
  }

  auto __x = __b;
  auto __y = _Tp{1};

  while (__e > 1)
  {
    if (__e % 2 == 1)
    {
      __y *= __x;
      --__e;
    }
    __x *= __x;
    __e /= 2;
  }
  return __x * __y;
}

//! @brief Computes the integer power of a base to an exponent.
//! @param __b The base
//! @param __e The exponent
//! @pre \p __b must be an integer type
//! @pre \p __e must be an integer type
//! @return The result of raising \p __b to the power of \p __e
//! @note The result is undefined if \p __b is 0 and \p __e is negative.
_CCCL_TEMPLATE(class _Tp, class _Ep)
_CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_integer_v<_Tp> _CCCL_AND _CUDA_VSTD::__cccl_is_integer_v<_Ep>)
[[nodiscard]] _CCCL_API constexpr _Tp ipow(_Tp __b, _Ep __e) noexcept
{
  _CCCL_ASSERT(__b != _Tp{0} || _CUDA_VSTD::cmp_greater_equal(__e, _Ep{0}),
               "cuda::ipow() requires non-negative exponent for base 0");

  if (__e == _Ep{0} || __b == _Tp{1})
  {
    return _Tp{1};
  }
  else if (_CUDA_VSTD::cmp_less(__e, _Ep{0}) || __b == _Tp{0})
  {
    return _Tp{0};
  }
  auto __res = ::cuda::__cccl_ipow_impl(::cuda::uabs(__b), _CUDA_VSTD::__to_unsigned_like(__e));
  if (_CUDA_VSTD::cmp_less(__b, _Tp{0}) && (__e % 2u == 1))
  {
    // todo: replace with ::cuda::__neg(__res) when available
    __res = (~__res + 1);
  }
  return static_cast<_Tp>(__res);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_IPOW
