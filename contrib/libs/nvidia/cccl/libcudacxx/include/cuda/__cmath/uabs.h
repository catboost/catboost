//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_UABS_H
#define _CUDA___CMATH_UABS_H

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
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Returns the *unsigned* absolute value of the given number.
//! @param __v The input number
//! @pre \p __v must be an integer type
//! @return The unsigned absolute value of \p __v
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _Tp))
[[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::make_unsigned_t<_Tp> uabs(_Tp __v) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
    return (__v < _Tp(0)) ? static_cast<_Up>(::cuda::neg(__v)) : static_cast<_Up>(__v);
  }
  else
  {
    return __v;
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_UABS_H
