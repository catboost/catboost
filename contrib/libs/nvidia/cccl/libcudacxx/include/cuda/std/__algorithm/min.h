//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MIN_H
#define _LIBCUDACXX___ALGORITHM_MIN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr const _Tp& min(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
  return __comp(__b, __a) ? __b : __a;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& min(const _Tp& __a, const _Tp& __b)
{
  return __b < __a ? __b : __a;
}

template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr _Tp min(initializer_list<_Tp> __t, _Compare __comp)
{
  return *_CUDA_VSTD::__min_element<__comp_ref_type<_Compare>>(__t.begin(), __t.end(), __comp);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp min(initializer_list<_Tp> __t)
{
  return *_CUDA_VSTD::min_element(__t.begin(), __t.end(), __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MIN_H
