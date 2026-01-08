//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_INTEGRAL_H
#define _LIBCUDACXX___BIT_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/bfind.h>
#  include <cuda/__ptx/instructions/shl.h>
#  include <cuda/__ptx/instructions/shr.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_API constexpr uint32_t __bit_log2(_Tp __t) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) <= 8)
    {
      using _Up [[maybe_unused]] = _If<sizeof(_Tp) <= 4, uint32_t, uint64_t>;
      NV_IF_TARGET(NV_IS_DEVICE, (return _CUDA_VPTX::bfind(static_cast<_Up>(__t));))
    }
    else
    {
      NV_IF_TARGET(NV_IS_DEVICE,
                   (auto __high = _CUDA_VPTX::bfind(static_cast<uint64_t>(__t >> 64));
                    return __high == ~uint32_t{0} ? _CUDA_VPTX::bfind(static_cast<uint64_t>(__t)) : __high + 64;))
    }
  }
  return numeric_limits<_Tp>::digits - 1 - _CUDA_VSTD::countl_zero(__t);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr int bit_width(_Tp __t) noexcept
{
  // if __t == 0, __bit_log2(0) returns 0xFFFFFFFF. Since unsigned overflow is well-defined, the result is -1 + 1 = 0
  auto __ret = _CUDA_VSTD::__bit_log2(__t) + 1;
  _CCCL_ASSUME(__ret <= numeric_limits<_Tp>::digits);
  return static_cast<int>(__ret);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp bit_ceil(_Tp __t) noexcept
{
  using _Up = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  _CCCL_ASSERT(__t <= numeric_limits<_Tp>::max() / 2, "bit_ceil overflow");
  // if __t == 0, __t - 1 == 0xFFFFFFFF, bit_width(0xFFFFFFFF) returns 32
  auto __width = _CUDA_VSTD::bit_width(static_cast<_Up>(__t) - 1);
  if constexpr (sizeof(_Tp) <= 8)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // CUDA right shift (ptx::shr) returns 0 if the right operand is larger than the number of bits of the type
      // The result is computed as max(1, bit_width(__t - 1)) because it is more efficient than the ternary operator
      NV_IF_TARGET(NV_IS_DEVICE, //
                   (auto __shift = _CUDA_VPTX::shl(_Up{1}, __width); // 2^(ceil(log2(__t - 1)))
                    auto __ret   = static_cast<_Tp>(_CUDA_VSTD::max(_Up{1}, __shift)); //
                    _CCCL_ASSUME(__ret >= __t);
                    return __ret;))
    }
  }
  auto __ret = static_cast<_Tp>(__t <= 1 ? _Up{1} : _Up{1} << __width);
  _CCCL_ASSUME(__ret >= __t);
  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _Tp bit_floor(_Tp __t) noexcept
{
  using _Up   = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  auto __log2 = _CUDA_VSTD::__bit_log2(static_cast<_Up>(__t));
  // __bit_log2 returns 0xFFFFFFFF if __t == 0
  if constexpr (sizeof(_Tp) <= 8)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // CUDA left shift (ptx::shl) returns 0 if the right operand is larger than the number of bits of the type
      // -> the result is 0 if __t == 0
      NV_IF_TARGET(NV_IS_DEVICE, //
                   (auto __ret = static_cast<_Tp>(_CUDA_VPTX::shl(_Up{1}, __log2)); // 2^(log2(t))
                    _CCCL_ASSUME(__ret >= __t / 2 && __ret <= __t);
                    return __ret;))
    }
  }
  auto __ret = static_cast<_Tp>(__t == 0 ? _Up{0} : _Up{1} << __log2);
  _CCCL_ASSUME(__ret >= __t / 2 && __ret <= __t);
  return __ret;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___BIT_INTEGRAL_H
