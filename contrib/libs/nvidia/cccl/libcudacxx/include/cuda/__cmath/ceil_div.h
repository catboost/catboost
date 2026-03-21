//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_CEIL_DIV_H
#define _CUDA___CMATH_CEIL_DIV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/to_underlying.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _Up> ceil_div(const _Tp __a, const _Up __b) noexcept
{
  _CCCL_ASSERT(__b > _Up{0}, "cuda::ceil_div: 'b' must be positive");
  if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__a >= _Tp{0}, "cuda::ceil_div: 'a' must be non negative");
  }
  using _Common = _CUDA_VSTD::common_type_t<_Tp, _Up>;
  using _Prom   = decltype(_Tp{} / _Up{});
  using _UProm  = _CUDA_VSTD::make_unsigned_t<_Prom>;
  auto __a1     = static_cast<_UProm>(__a);
  auto __b1     = static_cast<_UProm>(__b);
  if constexpr (_CUDA_VSTD::is_signed_v<_Prom>)
  {
    return static_cast<_Common>((__a1 + __b1 - 1) / __b1);
  }
  else
  {
    if (_CUDA_VSTD::is_constant_evaluated())
    {
      const auto __res = __a1 / __b1;
      return static_cast<_Common>(__res + (__res * __b1 != __a1));
    }
    else
    {
      // the ::min method is faster even if __b is a compile-time constant
      NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                        (return static_cast<_Common>(_CUDA_VSTD::min(__a1, 1 + ((__a1 - 1) / __b1)));),
                        (const auto __res = __a1 / __b1; //
                         return static_cast<_Common>(__res + (__res * __b1 != __a1));))
    }
  }
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _CUDA_VSTD::underlying_type_t<_Up>>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(__a, _CUDA_VSTD::to_underlying(__b));
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_CUDA_VSTD::underlying_type_t<_Tp>, _Up>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(_CUDA_VSTD::to_underlying(__a), __b);
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
[[nodiscard]]
_CCCL_API constexpr _CUDA_VSTD::common_type_t<_CUDA_VSTD::underlying_type_t<_Tp>, _CUDA_VSTD::underlying_type_t<_Up>>
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(_CUDA_VSTD::to_underlying(__a), _CUDA_VSTD::to_underlying(__b));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_CEIL_DIV_H
