// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_MIDPOINT_H
#define _LIBCUDACXX___NUMERIC_MIDPOINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_integral, _Tp) && !_CCCL_TRAIT(is_same, bool, _Tp) && !_CCCL_TRAIT(is_null_pointer, _Tp),
  _Tp>
midpoint(_Tp __a, _Tp __b) noexcept
{
  using _Up = make_unsigned_t<_Tp>;

  if (__a > __b)
  {
    const _Up __diff = _Up(__a) - _Up(__b);
    return static_cast<_Tp>(__a - static_cast<_Tp>(__diff / 2));
  }
  else
  {
    const _Up __diff = _Up(__b) - _Up(__a);
    return static_cast<_Tp>(__a + static_cast<_Tp>(__diff / 2));
  }
}

template <class _Tp,
          enable_if_t<_CCCL_TRAIT(is_object, _Tp) && !_CCCL_TRAIT(is_void, _Tp) && (sizeof(_Tp) > 0), int> = 0>
[[nodiscard]] _CCCL_API constexpr _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
{
  return __a + _CUDA_VSTD::midpoint(ptrdiff_t(0), __b - __a);
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __sign(_Tp __val)
{
  return (_Tp(0) < __val) - (__val < _Tp(0));
}

template <typename _Fp>
[[nodiscard]] _CCCL_API constexpr _Fp __fp_abs(_Fp __f)
{
  return __f >= 0 ? __f : -__f;
}

template <class _Fp>
[[nodiscard]] _CCCL_API constexpr enable_if_t<_CCCL_TRAIT(is_floating_point, _Fp), _Fp>
midpoint(_Fp __a, _Fp __b) noexcept
{
  constexpr _Fp __lo = numeric_limits<_Fp>::min() * 2;
  constexpr _Fp __hi = numeric_limits<_Fp>::max() / 2;
  return _CUDA_VSTD::__fp_abs(__a) <= __hi && _CUDA_VSTD::__fp_abs(__b) <= __hi
         ? // typical case: overflow is impossible
           (__a + __b) / 2
         : // always correctly rounded
           _CUDA_VSTD::__fp_abs(__a) < __lo ? __a + __b / 2 : // not safe to halve a
             _CUDA_VSTD::__fp_abs(__b) < __lo
             ? __a / 2 + __b
             : // not safe to halve b
             __a / 2 + __b / 2; // otherwise correctly rounded
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___NUMERIC_MIDPOINT_H
