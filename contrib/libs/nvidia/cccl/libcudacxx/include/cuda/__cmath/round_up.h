//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ROUND_UP_H
#define _CUDA___CMATH_ROUND_UP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _Up> round_up(const _Tp __a, const _Up __b) noexcept
{
  _CCCL_ASSERT(__b > _Up{0}, "cuda::round_up: 'b' must be positive");
  if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__a >= _Tp{0}, "cuda::round_up: 'a' must be non negative");
  }
  using _Common = _CUDA_VSTD::common_type_t<_Tp, _Up>;
  using _Prom   = decltype(_Tp{} / _Up{});
  auto __c      = ::cuda::ceil_div(static_cast<_Prom>(__a), static_cast<_Prom>(__b));
  _CCCL_ASSERT(static_cast<_Common>(__c) <= _CUDA_VSTD::numeric_limits<_Common>::max() / static_cast<_Common>(__b),
               "cuda::round_up: result overflow");
  return static_cast<_Common>(static_cast<_Prom>(__c) * static_cast<_Prom>(__b));
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_Tp, _CUDA_VSTD::underlying_type_t<_Up>>
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(__a, _CUDA_VSTD::to_underlying(__b));
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_CUDA_VSTD::underlying_type_t<_Tp>, _Up>
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(_CUDA_VSTD::to_underlying(__a), __b);
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
[[nodiscard]]
_CCCL_API constexpr _CUDA_VSTD::common_type_t<_CUDA_VSTD::underlying_type_t<_Tp>, _CUDA_VSTD::underlying_type_t<_Up>>
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(_CUDA_VSTD::to_underlying(__a), _CUDA_VSTD::to_underlying(__b));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_ROUND_UP_H
