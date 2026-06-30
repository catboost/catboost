//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_REVERSE_H
#define _CUDA___BIT_BIT_REVERSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#if defined(_CCCL_BUILTIN_BITREVERSE32)

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_reverse_builtin(_Tp __value) noexcept
{
#  if _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    auto __high = static_cast<__uint128_t>(_CCCL_BUILTIN_BITREVERSE64(static_cast<uint64_t>(__value))) << 64;
    auto __low  = static_cast<__uint128_t>(_CCCL_BUILTIN_BITREVERSE64(static_cast<uint64_t>(__value >> 64)));
    return __high | __low;
  }
#  endif // _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    return _CCCL_BUILTIN_BITREVERSE64(__value);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return _CCCL_BUILTIN_BITREVERSE32(__value);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return _CCCL_BUILTIN_BITREVERSE16(__value);
  }
  else
  {
    return _CCCL_BUILTIN_BITREVERSE8(__value);
  }
}

#endif // defined(_CCCL_BUILTIN_BITREVERSE32)

#if _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr _Tp __bit_reverse_device(_Tp __value) noexcept
{
#  if _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    auto __high = static_cast<__uint128_t>(::cuda::__bit_reverse_device(static_cast<uint64_t>(__value))) << 64;
    auto __low  = static_cast<__uint128_t>(::cuda::__bit_reverse_device(static_cast<uint64_t>(__value >> 64)));
    return __high | __low;
  }
#  endif // _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::__brevll(__value);))
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::__brev(__value);))
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::__brev(static_cast<uint32_t>(__value) << 16);))
  }
  else
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::__brev(static_cast<uint32_t>(__value) << 24);))
  }
  _CCCL_UNREACHABLE();
}

#endif // _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_reverse_generic(_Tp __value) noexcept
{
#if _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    constexpr auto __c1 = __uint128_t{0x5555555555555555} << 64 | uint64_t{0x5555555555555555};
    constexpr auto __c2 = __uint128_t{0x3333333333333333} << 64 | uint64_t{0x3333333333333333};
    constexpr auto __c3 = __uint128_t{0x0F0F0F0F0F0F0F0F} << 64 | uint64_t{0x0F0F0F0F0F0F0F0F};
    constexpr auto __c4 = __uint128_t{0x00FF00FF00FF00FF} << 64 | uint64_t{0x00FF00FF00FF00FF};
    constexpr auto __c5 = __uint128_t{0x0000FFFF0000FFFF} << 64 | uint64_t{0x0000FFFF0000FFFF};
    constexpr auto __c6 = __uint128_t{0x00000000FFFFFFFF} << 64 | uint64_t{0x00000000FFFFFFFF};
    __value             = ((__value >> 1) & __c1) | ((__value & __c1) << 1);
    __value             = ((__value >> 2) & __c2) | ((__value & __c2) << 2);
    __value             = ((__value >> 4) & __c3) | ((__value & __c3) << 4);
    __value             = ((__value >> 8) & __c4) | ((__value & __c4) << 8);
    __value             = ((__value >> 16) & __c5) | ((__value & __c5) << 16);
    __value             = ((__value >> 32) & __c6) | ((__value & __c6) << 32);
    return (__value >> 64) | (__value << 64);
  }
#endif // _CCCL_HAS_INT128()
  if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    __value = ((__value >> 1) & 0x5555555555555555) | ((__value & 0x5555555555555555) << 1);
    __value = ((__value >> 2) & 0x3333333333333333) | ((__value & 0x3333333333333333) << 2);
    __value = ((__value >> 4) & 0x0F0F0F0F0F0F0F0F) | ((__value & 0x0F0F0F0F0F0F0F0F) << 4);
    __value = ((__value >> 8) & 0x00FF00FF00FF00FF) | ((__value & 0x00FF00FF00FF00FF) << 8);
    __value = ((__value >> 16) & 0x0000FFFF0000FFFF) | ((__value & 0x0000FFFF0000FFFF) << 16);
    return (__value >> 32) | (__value << 32);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    __value = ((__value >> 1) & 0x55555555) | ((__value & 0x55555555) << 1);
    __value = ((__value >> 2) & 0x33333333) | ((__value & 0x33333333) << 2);
    __value = ((__value >> 4) & 0x0F0F0F0F) | ((__value & 0x0F0F0F0F) << 4);
    __value = ((__value >> 8) & 0x00FF00FF) | ((__value & 0x00FF00FF) << 8);
    return (__value >> 16) | (__value << 16);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    __value = ((__value >> 1) & 0x5555) | ((__value & 0x5555) << 1);
    __value = ((__value >> 2) & 0x3333) | ((__value & 0x3333) << 2);
    __value = ((__value >> 4) & 0x0F0F) | ((__value & 0x0F0F) << 4);
    return (__value >> 8) | (__value << 8);
  }
  else
  {
    __value = ((__value >> 1) & 0x55) | ((__value & 0x55) << 1);
    __value = ((__value >> 2) & 0x33) | ((__value & 0x33) << 2);
    return (__value >> 4) | (__value << 4);
  }
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp bit_reverse(_Tp __value) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_cv_unsigned_integer_v<_Tp>, "bit_reverse() requires unsigned integer types");
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::__bit_reverse_device(__value);))
  }
#if defined(_CCCL_BUILTIN_BITREVERSE32)
  return ::cuda::__bit_reverse_builtin(__value);
#else
  return ::cuda::__bit_reverse_generic(__value);
#endif
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_REVERSE_H
