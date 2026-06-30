//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_PREDICATE_H
#define _LIBCUDACXX___CONCEPTS_PREDICATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/boolean_testable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__functional/invoke.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

template <class _Fn, class... _Args>
concept predicate = regular_invocable<_Fn, _Args...> && __boolean_testable<invoke_result_t<_Fn, _Args...>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

// [concept.predicate]
template <class _Fn, class... _Args>
_CCCL_CONCEPT_FRAGMENT(
  _Predicate_,
  requires()(requires(regular_invocable<_Fn, _Args...>), requires(__boolean_testable<invoke_result_t<_Fn, _Args...>>)));

template <class _Fn, class... _Args>
_CCCL_CONCEPT predicate = _CCCL_FRAGMENT(_Predicate_, _Fn, _Args...);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_PREDICATE_H
