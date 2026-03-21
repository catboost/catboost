//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_POW2
#define _CUDA___CMATH_POW2

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr bool is_power_of_two(_Tp __t) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    _CCCL_ASSERT(__t >= _Tp{0}, "cuda::is_power_of_two requires non-negative input");
  }
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  return _CUDA_VSTD::has_single_bit(static_cast<_Up>(__t));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp next_power_of_two(_Tp __t) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    _CCCL_ASSERT(__t >= _Tp{0}, "cuda::is_power_of_two requires non-negative input");
  }
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  return _CUDA_VSTD::bit_ceil(static_cast<_Up>(__t));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp prev_power_of_two(_Tp __t) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    _CCCL_ASSERT(__t >= _Tp{0}, "cuda::is_power_of_two requires non-negative input");
  }
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  return _CUDA_VSTD::bit_floor(static_cast<_Up>(__t));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_POW2
