//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_OVERFLOW_CAST_H
#define _CUDA___NUMERIC_OVERFLOW_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Casts a number \p __from to a number of type \p _To with overflow detection
//! @param __from The number to cast
//! @return An overflow_result object containing the casted number and a boolean indicating whether an overflow
//! occurred
_CCCL_TEMPLATE(class _To, class _From)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _To)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _From))
[[nodiscard]] _CCCL_API constexpr overflow_result<_To> overflow_cast(const _From& __from) noexcept
{
  bool __overflow = false;
  if constexpr (_CUDA_VSTD::cmp_greater(_CUDA_VSTD::numeric_limits<_From>::max(), _CUDA_VSTD::numeric_limits<_To>::max())
                || _CUDA_VSTD::cmp_less(_CUDA_VSTD::numeric_limits<_From>::min(),
                                        _CUDA_VSTD::numeric_limits<_To>::min()))
  {
    __overflow = !_CUDA_VSTD::in_range<_To>(__from);
  }
  return overflow_result<_To>{static_cast<_To>(__from), __overflow};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_OVERFLOW_CAST_H
