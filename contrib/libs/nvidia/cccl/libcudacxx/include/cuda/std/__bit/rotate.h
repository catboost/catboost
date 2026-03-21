//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ROTATE_H
#define _LIBCUDACXX___BIT_ROTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/neg.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __cccl_rotr_impl(_Tp __v, int __cnt) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_r(__v, __v, __cnt);))
    }
  }
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  auto __cnt_mod          = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __v : (__v >> __cnt_mod) | (__v << (__digits - __cnt_mod));
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __cccl_rotl_impl(_Tp __v, int __cnt) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_l(__v, __v, __cnt);))
    }
  }
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  auto __cnt_mod          = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __v : (__v << __cnt_mod) | (__v >> (__digits - __cnt_mod));
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp rotl(_Tp __v, int __cnt) noexcept
{
  if (__cnt < 0)
  {
    __cnt = static_cast<int>(static_cast<unsigned>(::cuda::neg(__cnt)) % numeric_limits<_Tp>::digits);
    return _CUDA_VSTD::__cccl_rotr_impl(__v, __cnt);
  }
  return _CUDA_VSTD::__cccl_rotl_impl(__v, __cnt);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp rotr(_Tp __v, int __cnt) noexcept
{
  if (__cnt < 0)
  {
    __cnt = static_cast<int>(static_cast<unsigned>(::cuda::neg(__cnt)) % numeric_limits<_Tp>::digits);
    return _CUDA_VSTD::__cccl_rotl_impl(__v, __cnt);
  }
  return _CUDA_VSTD::__cccl_rotr_impl(__v, __cnt);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_ROTATE_H
