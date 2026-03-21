// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_MERGEABLE_H
#define _LIBCUDACXX___ITERATOR_MERGEABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Input1,
          class _Input2,
          class _Output,
          class _Comp  = _CUDA_VRANGES::less,
          class _Proj1 = identity,
          class _Proj2 = identity>
concept mergeable =
  input_iterator<_Input1> && input_iterator<_Input2> && weakly_incrementable<_Output>
  && indirectly_copyable<_Input1, _Output> && indirectly_copyable<_Input2, _Output>
  && indirect_strict_weak_order<_Comp, projected<_Input1, _Proj1>, projected<_Input2, _Proj2>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Input1, class _Input2, class _Output, class _Comp, class _Proj1, class _Proj2>
_CCCL_CONCEPT_FRAGMENT(
  __mergeable_,
  requires()(requires(input_iterator<_Input1>),
             requires(input_iterator<_Input2>),
             requires(weakly_incrementable<_Output>),
             requires(indirectly_copyable<_Input1, _Output>),
             requires(indirectly_copyable<_Input2, _Output>),
             requires(indirect_strict_weak_order<_Comp, projected<_Input1, _Proj1>, projected<_Input2, _Proj2>>)));

template <class _Input1,
          class _Input2,
          class _Output,
          class _Comp  = _CUDA_VRANGES::less,
          class _Proj1 = identity,
          class _Proj2 = identity>
_CCCL_CONCEPT mergeable = _CCCL_FRAGMENT(__mergeable_, _Input1, _Input2, _Output, _Comp, _Proj1, _Proj2);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_MERGEABLE_H
