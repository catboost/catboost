// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_FRONT_INSERT_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_FRONT_INSERT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Container>
class _CCCL_TYPE_VISIBILITY_DEFAULT front_insert_iterator
#if !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif // !_LIBCUDACXX_ABI_NO_ITERATOR_BASES
{
protected:
  _Container* container;

public:
  using iterator_category = output_iterator_tag;
  using value_type        = void;
#if _CCCL_STD_VER > 2017
  using difference_type = ptrdiff_t;
#else
  using difference_type = void;
#endif
  using pointer        = void;
  using reference      = void;
  using container_type = _Container;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit front_insert_iterator(_Container& __x)
      : container(_CUDA_VSTD::addressof(__x))
  {}
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator& operator=(const typename _Container::value_type& __value)
  {
    container->push_front(__value);
    return *this;
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator& operator=(typename _Container::value_type&& __value)
  {
    container->push_front(_CUDA_VSTD::move(__value));
    return *this;
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator& operator*()
  {
    return *this;
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator& operator++()
  {
    return *this;
  }
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator operator++(int)
  {
    return *this;
  }
};
_CCCL_SUPPRESS_DEPRECATED_POP
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(front_insert_iterator);

template <class _Container>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 front_insert_iterator<_Container> front_inserter(_Container& __x)
{
  return front_insert_iterator<_Container>(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_FRONT_INSERT_ITERATOR_H
