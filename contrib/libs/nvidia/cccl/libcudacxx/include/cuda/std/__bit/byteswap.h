//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_BYTESWAP_H
#define _LIBCUDACXX___BIT_BYTESWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/prmt.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __byteswap_impl(_Tp __val) noexcept;

template <class _Full>
[[nodiscard]] _CCCL_API constexpr _Full __byteswap_impl_recursive(_Full __val) noexcept
{
  using _Half            = __make_nbit_uint_t<numeric_limits<_Full>::digits / 2>;
  constexpr auto __shift = numeric_limits<_Half>::digits;

  if constexpr (sizeof(_Full) > 2)
  {
    return static_cast<_Full>(_CUDA_VSTD::__byteswap_impl(static_cast<_Half>(__val >> __shift)))
         | (static_cast<_Full>(_CUDA_VSTD::__byteswap_impl(static_cast<_Half>(__val))) << __shift);
  }
  else
  {
    return static_cast<_Full>((__val << __shift) | (__val >> __shift));
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __byteswap_impl_device(_Tp __val) noexcept
{
#if __cccl_ptx_isa >= 200
  if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return static_cast<uint16_t>(_CUDA_VPTX::prmt(static_cast<uint32_t>(__val), uint32_t{0}, uint32_t{0x3201}));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return _CUDA_VPTX::prmt(__val, uint32_t{0}, uint32_t{0x0123});
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __hi     = static_cast<uint32_t>(__val >> 32);
    const auto __lo     = static_cast<uint32_t>(__val);
    const auto __new_lo = _CUDA_VPTX::prmt(__hi, uint32_t{0}, uint32_t{0x0123});
    const auto __new_hi = _CUDA_VPTX::prmt(__lo, uint32_t{0}, uint32_t{0x0123});

    return static_cast<uint64_t>(__new_hi) << 32 | static_cast<uint64_t>(__new_lo);
  }
  else
#endif // __cccl_ptx_isa >= 200
  {
    return _CUDA_VSTD::__byteswap_impl_recursive(__val);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __byteswap_impl(_Tp __val) noexcept
{
  constexpr auto __shift = numeric_limits<uint8_t>::digits;

  _Tp __result{};

  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 0; __i < sizeof(_Tp); ++__i)
  {
    __result <<= __shift;
    __result |= _Tp(__val & _Tp(numeric_limits<uint8_t>::max()));
    __val >>= __shift;
  }
  return __result;
}

[[nodiscard]] _CCCL_API constexpr uint16_t __byteswap_impl(uint16_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP16)
  return _CCCL_BUILTIN_BSWAP16(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP16 ^^^ / vvv !_CCCL_BUILTIN_BSWAP16 vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_ushort(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return _CUDA_VSTD::__byteswap_impl_device(__val);)
  }
  return _CUDA_VSTD::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP16
}

[[nodiscard]] _CCCL_API constexpr uint32_t __byteswap_impl(uint32_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP32)
  return _CCCL_BUILTIN_BSWAP32(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP32 ^^^ / vvv !_CCCL_BUILTIN_BSWAP32 vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_ulong(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return _CUDA_VSTD::__byteswap_impl_device(__val);)
  }
  return _CUDA_VSTD::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP32
}

[[nodiscard]] _CCCL_API constexpr uint64_t __byteswap_impl(uint64_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP64)
  return _CCCL_BUILTIN_BSWAP64(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP64 ^^^ / vvv !_CCCL_BUILTIN_BSWAP64 vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_uint64(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return _CUDA_VSTD::__byteswap_impl_device(__val);)
  }
  return _CUDA_VSTD::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP64
}

#if _CCCL_HAS_INT128()
[[nodiscard]] _CCCL_API constexpr __uint128_t __byteswap_impl(__uint128_t __val) noexcept
{
#  if defined(_CCCL_BUILTIN_BSWAP128)
  return _CCCL_BUILTIN_BSWAP128(__val);
#  else // ^^^ _CCCL_BUILTIN_BSWAP128 ^^^ / vvv !_CCCL_BUILTIN_BSWAP128 vvv
  return _CUDA_VSTD::__byteswap_impl_recursive(__val);
#  endif // !_CCCL_BUILTIN_BSWAP128
}
#endif // _CCCL_HAS_INT128()

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
[[nodiscard]] _CCCL_API constexpr _Integer byteswap(_Integer __val) noexcept
{
  if constexpr (sizeof(_Integer) > 1)
  {
    return static_cast<_Integer>(_CUDA_VSTD::__byteswap_impl(_CUDA_VSTD::__to_unsigned_like(__val)));
  }
  else
  {
    return __val;
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_BYTESWAP_H
