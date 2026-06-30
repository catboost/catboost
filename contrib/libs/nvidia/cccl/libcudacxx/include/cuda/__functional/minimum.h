//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_MINIMUM_H
#define _CUDA_FUNCTIONAL_MINIMUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/common_type.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minimum
{
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
    noexcept(noexcept((__lhs < __rhs) ? __lhs : __rhs))
  {
    return (__lhs < __rhs) ? __lhs : __rhs;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(minimum);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minimum<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_T1, _T2>
  operator()(const _T1& __lhs, const _T2& __rhs) const noexcept(noexcept((__lhs < __rhs) ? __lhs : __rhs))
  {
    return (__lhs < __rhs) ? __lhs : __rhs;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MINIMUM_H
