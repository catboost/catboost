//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_COUNTL_H
#define _LIBCUDACXX___BIT_COUNTL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_countl_zero_impl_constexpr(_Tp __v) noexcept
{
  constexpr auto __digits = numeric_limits<_Tp>::digits;

  if (__v == 0)
  {
    return __digits;
  }

  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
#if defined(_CCCL_BUILTIN_CLZ)
    return _CCCL_BUILTIN_CLZ(__v);
#else // ^^^ _CCCL_BUILTIN_CLZ ^^^ // vvv !_CCCL_BUILTIN_CLZ vvv
    uint32_t __res = 0;
    for (uint32_t __i = __digits / 2; __i >= 1; __i /= 2)
    {
      const auto __mark = (~uint32_t{0} >> (__digits - __i)) << __i;
      if (__v & __mark)
      {
        __v >>= __i;
        __res |= __i;
      }
    }
    return __digits - 1 - __res;
#endif // ^^^ !_CCCL_BUILTIN_CLZ ^^^
  }
  else
  {
#if defined(_CCCL_BUILTIN_CLZLL)
    return _CCCL_BUILTIN_CLZLL(__v);
#else // ^^^ _CCCL_BUILTIN_CLZLL ^^^ // vvv !_CCCL_BUILTIN_CLZLL vvv
    const auto __hi = static_cast<uint32_t>(__v >> 32);
    const auto __lo = static_cast<uint32_t>(__v);
    return (__hi != 0) ? _CUDA_VSTD::__cccl_countl_zero_impl_constexpr(__hi)
                       : (numeric_limits<uint32_t>::digits + _CUDA_VSTD::__cccl_countl_zero_impl_constexpr(__lo));
#endif // ^^^ !_CCCL_BUILTIN_CLZLL ^^^
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int __cccl_countl_zero_impl_host(_Tp __v) noexcept
{
#  if _CCCL_COMPILER(MSVC)
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  unsigned long __where{};
  const auto __res = sizeof(_Tp) == sizeof(uint32_t)
                     ? ::_BitScanReverse(&__where, static_cast<uint32_t>(__v))
                     : ::_BitScanReverse64(&__where, static_cast<uint64_t>(__v));
  return (__res) ? (__digits - 1 - static_cast<int>(__where)) : __digits;
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ // vvv !_CCCL_COMPILER(MSVC) vvv
  return _CUDA_VSTD::__cccl_countl_zero_impl_constexpr(__v);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE int __cccl_countl_zero_impl_device(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return static_cast<int>(::__clz(static_cast<int>(__v)));
  }
  else
  {
    return static_cast<int>(::__clzll(static_cast<long long>(__v)));
  }
}
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_countl_zero_impl(_Tp __v) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return _CUDA_VSTD::__cccl_countl_zero_impl_host(__v);),
                      (return _CUDA_VSTD::__cccl_countl_zero_impl_device(__v);));
  }
  return _CUDA_VSTD::__cccl_countl_zero_impl_constexpr(__v);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr int countl_zero(_Tp __v) noexcept
{
  int __count{};
#if defined(_CCCL_BUILTIN_CLZG)
  __count = _CCCL_BUILTIN_CLZG(__v, numeric_limits<_Tp>::digits);
#else // ^^^ _CCCL_BUILTIN_CLZG ^^^ // vvv !_CCCL_BUILTIN_CLZG vvv
  if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
  {
    using _Sp                    = _If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
    constexpr auto __digits_diff = numeric_limits<_Sp>::digits - numeric_limits<_Tp>::digits;
    __count                      = _CUDA_VSTD::__cccl_countl_zero_impl(static_cast<_Sp>(__v)) - __digits_diff;
  }
  else
  {
    constexpr int _Ratio = sizeof(_Tp) / sizeof(uint64_t);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = _Ratio - 1; __i >= 0; --__i)
    {
      const auto __value64 = static_cast<uint64_t>(__v >> (__i * numeric_limits<uint64_t>::digits));
      if (__value64 != 0)
      {
        __count += _CUDA_VSTD::countl_zero(__value64);
        break;
      }
      __count += numeric_limits<uint64_t>::digits;
    }
  }
#endif // ^^^ !_CCCL_BUILTIN_CLZG ^^^

  _CCCL_ASSUME(__count >= 0 && __count <= numeric_limits<_Tp>::digits);
  return __count;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr int countl_one(_Tp __v) noexcept
{
  return _CUDA_VSTD::countl_zero(static_cast<_Tp>(~__v));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_COUNTL_H
