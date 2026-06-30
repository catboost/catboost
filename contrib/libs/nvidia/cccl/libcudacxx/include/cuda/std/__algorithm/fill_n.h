//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FILL_N_H
#define _LIBCUDACXX___ALGORITHM_FILL_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/convert_to_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _OutputIterator, class _Size, class _Tp>
_CCCL_API constexpr _OutputIterator __fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
  for (; __n > 0; ++__first, (void) --__n)
  {
    *__first = __value_;
  }
  return __first;
}

template <class _OutputIterator, class _Size, class _Tp>
_CCCL_API constexpr _OutputIterator fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
  return _CUDA_VSTD::__fill_n(__first, __convert_to_integral(__n), __value_);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_FILL_N_H
