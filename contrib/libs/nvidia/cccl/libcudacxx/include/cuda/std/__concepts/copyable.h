//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_COPYABLE_H
#define _LIBCUDACXX___CONCEPTS_COPYABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/movable.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concepts.object]

template <class _Tp>
concept copyable = copy_constructible<_Tp> && movable<_Tp> && assignable_from<_Tp&, _Tp&>
                && assignable_from<_Tp&, const _Tp&> && assignable_from<_Tp&, const _Tp>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __copyable_,
  requires()(requires(copy_constructible<_Tp>),
             requires(movable<_Tp>),
             requires(assignable_from<_Tp&, _Tp&>),
             requires(assignable_from<_Tp&, const _Tp&>),
             requires(assignable_from<_Tp&, const _Tp>)));

template <class _Tp>
_CCCL_CONCEPT copyable = _CCCL_FRAGMENT(__copyable_, _Tp);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_COPYABLE_H
