// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_PERMUTABLE_H
#define _LIBCUDACXX___ITERATOR_PERMUTABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_swap.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Iterator>
concept permutable = forward_iterator<_Iterator> && indirectly_movable_storable<_Iterator, _Iterator>
                  && indirectly_swappable<_Iterator, _Iterator>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Iterator>
_CCCL_CONCEPT_FRAGMENT(__permutable_,
                       requires()(requires(forward_iterator<_Iterator>),
                                  requires(indirectly_movable_storable<_Iterator, _Iterator>),
                                  requires(indirectly_swappable<_Iterator, _Iterator>)));

template <class _Iterator>
_CCCL_CONCEPT permutable = _CCCL_FRAGMENT(__permutable_, _Iterator);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_PERMUTABLE_H
