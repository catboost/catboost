//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_HALF_POSITIVE_H
#define _LIBCUDACXX___ALGORITHM_HALF_POSITIVE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Perform division by two quickly for positive integers (llvm.org/PR39129)

template <class _Integral, enable_if_t<_CCCL_TRAIT(is_integral, _Integral), int> = 0>
[[nodiscard]] _CCCL_API constexpr _Integral __half_positive(_Integral __value)
{
  return static_cast<_Integral>(static_cast<make_unsigned_t<_Integral>>(__value) / 2);
}

template <class _Tp, enable_if_t<!_CCCL_TRAIT(is_integral, _Tp), int> = 0>
[[nodiscard]] _CCCL_API constexpr _Tp __half_positive(_Tp __value)
{
  return __value / 2;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_HALF_POSITIVE_H
