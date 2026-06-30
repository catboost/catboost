//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_POPCOUNT_H
#define _LIBCUDACXX___BIT_POPCOUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_popcount_impl_constexpr(_Tp __v) noexcept
{
  if constexpr (is_same_v<_Tp, uint32_t>)
  {
#if defined(_CCCL_BUILTIN_POPCOUNT)
    return _CCCL_BUILTIN_POPCOUNT(__v);
#else // ^^^ _CCCL_BUILTIN_POPCOUNT ^^^ / vvv !_CCCL_BUILTIN_POPCOUNT vvv
    __v = __v - ((__v >> 1) & 0x55555555);
    __v = (__v & 0x33333333) + ((__v >> 2) & 0x33333333);
    return (((__v + (__v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif // ^^^ !_CCCL_BUILTIN_POPCOUNT ^^^
  }
  else
  {
#if defined(_CCCL_BUILTIN_POPCOUNTLL)
    return _CCCL_BUILTIN_POPCOUNTLL(__v);
#else // ^^^ _CCCL_BUILTIN_POPCOUNTLL ^^^ / vvv !_CCCL_BUILTIN_POPCOUNTLL vvv
    return _CUDA_VSTD::__cccl_popcount_impl_constexpr(static_cast<uint32_t>(__v))
         + _CUDA_VSTD::__cccl_popcount_impl_constexpr(static_cast<uint32_t>(__v >> 32));
#endif // ^^^ !_CCCL_BUILTIN_POPCOUNTLL ^^^
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int __cccl_popcount_impl_host(_Tp __v) noexcept
{
#  if _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return static_cast<int>(::__popcnt(__v));
  }
  else
  {
    return static_cast<int>(::__popcnt64(__v));
  }
  // _CountOneBits exists after MSVC 1931
#  elif _CCCL_COMPILER(MSVC, >, 19, 30) && _CCCL_ARCH(ARM64)
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return static_cast<int>(::_CountOneBits(__v));
  }
  else
  {
    return static_cast<int>(::_CountOneBits64(__v));
  }
#  else // ^^^ msvc intrinsics ^^^ / vvv other vvv
  return _CUDA_VSTD::__cccl_popcount_impl_constexpr(__v);
#  endif // ^^^ other ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE int __cccl_popcount_impl_device(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return static_cast<int>(::__popc(__v));
  }
  else
  {
    return static_cast<int>(::__popcll(__v));
  }
}
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_popcount_impl(_Tp __v) noexcept
{
  static_assert(_CCCL_TRAIT(is_same, _Tp, uint32_t) || _CCCL_TRAIT(is_same, _Tp, uint64_t));

  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return _CUDA_VSTD::__cccl_popcount_impl_host(__v);),
                      (return _CUDA_VSTD::__cccl_popcount_impl_device(__v);))
  }
  return _CUDA_VSTD::__cccl_popcount_impl_constexpr(__v);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr int popcount(_Tp __v) noexcept
{
  int __count{};

#if defined(_CCCL_BUILTIN_POPCOUNTG)
  __count = _CCCL_BUILTIN_POPCOUNTG(__v);
#else // ^^^ _CCCL_BUILTIN_POPCOUNTG ^^^ / vvv !_CCCL_BUILTIN_POPCOUNTG vvv
  if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
  {
    using _Sp = _If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
    __count   = _CUDA_VSTD::__cccl_popcount_impl(static_cast<_Sp>(__v));
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t __i = 0; __i < sizeof(_Tp) / sizeof(uint64_t); ++__i)
    {
      __count += _CUDA_VSTD::__cccl_popcount_impl(static_cast<uint64_t>(__v));
      __v >>= numeric_limits<uint64_t>::digits;
    }
  }
#endif // ^^^ !_CCCL_BUILTIN_POPCOUNTG ^^^

  _CCCL_ASSUME(__count >= 0 && __count <= numeric_limits<_Tp>::digits);
  return __count;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_POPCOUNT_H
