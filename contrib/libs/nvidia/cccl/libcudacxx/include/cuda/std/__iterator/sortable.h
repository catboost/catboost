// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_SORTABLE_H
#define _LIBCUDACXX___ITERATOR_SORTABLE_H

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
#include <cuda/std/__iterator/permutable.h>
#include <cuda/std/__iterator/projected.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Iter, class _Comp = _CUDA_VRANGES::less, class _Proj = identity>
concept sortable = permutable<_Iter> && indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Iter, class _Comp, class _Proj>
_CCCL_CONCEPT_FRAGMENT(
  __sortable_,
  requires()(requires(permutable<_Iter>), requires(indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>)));

template <class _Iter, class _Comp = _CUDA_VRANGES::less, class _Proj = identity>
_CCCL_CONCEPT sortable = _CCCL_FRAGMENT(__sortable_, _Iter, _Comp, _Proj);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_SORTABLE_H
