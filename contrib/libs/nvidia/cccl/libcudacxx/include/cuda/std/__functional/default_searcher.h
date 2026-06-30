// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
#define _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/search.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef __cuda_std__

// default searcher
template <class _ForwardIterator, class _BinaryPredicate = equal_to<>>
class _CCCL_TYPE_VISIBILITY_DEFAULT default_searcher
{
public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20
  default_searcher(_ForwardIterator __f, _ForwardIterator __l, _BinaryPredicate __p = _BinaryPredicate())
      : __first_(__f)
      , __last_(__l)
      , __pred_(__p)
  {}

  template <typename _ForwardIterator2>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 pair<_ForwardIterator2, _ForwardIterator2>
  operator()(_ForwardIterator2 __f, _ForwardIterator2 __l) const
  {
    return _CUDA_VSTD::__search(
      __f,
      __l,
      __first_,
      __last_,
      __pred_,
      typename _CUDA_VSTD::iterator_traits<_ForwardIterator>::iterator_category(),
      typename _CUDA_VSTD::iterator_traits<_ForwardIterator2>::iterator_category());
  }

private:
  _ForwardIterator __first_;
  _ForwardIterator __last_;
  _BinaryPredicate __pred_;
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(default_searcher);

#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
