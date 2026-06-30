//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REMOVE_IF_H
#define _LIBCUDACXX___ALGORITHM_REMOVE_IF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/find_if.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Predicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
remove_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
  __first = _CUDA_VSTD::find_if<_ForwardIterator, add_lvalue_reference_t<_Predicate>>(__first, __last, __pred);
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    while (++__i != __last)
    {
      if (!__pred(*__i))
      {
        *__first = _CUDA_VSTD::move(*__i);
        ++__first;
      }
    }
  }
  return __first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_REMOVE_IF_H
