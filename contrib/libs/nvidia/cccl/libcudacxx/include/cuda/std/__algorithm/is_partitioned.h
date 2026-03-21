//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IS_PARTITIONED_H
#define _LIBCUDACXX___ALGORITHM_IS_PARTITIONED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _Predicate>
[[nodiscard]] _CCCL_API constexpr bool is_partitioned(_InputIterator __first, _InputIterator __last, _Predicate __pred)
{
  for (; __first != __last; ++__first)
  {
    if (!__pred(*__first))
    {
      break;
    }
  }
  if (__first == __last)
  {
    return true;
  }
  ++__first;
  for (; __first != __last; ++__first)
  {
    if (__pred(*__first))
    {
      return false;
    }
  }
  return true;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_CLAMP_H
