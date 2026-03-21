// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H
#define _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H

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

template <class _Container, class _Predicate>
_CCCL_API inline typename _Container::size_type __cccl_erase_if_container(_Container& __c, _Predicate& __pred)
{
  typename _Container::size_type __old_size = __c.size();

  const typename _Container::iterator __last = __c.end();
  for (typename _Container::iterator __iter = __c.begin(); __iter != __last;)
  {
    if (__pred(*__iter))
    {
      __iter = __c.erase(__iter);
    }
    else
    {
      ++__iter;
    }
  }

  return __old_size - __c.size();
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_ERASE_IF_CONTAINER_H
