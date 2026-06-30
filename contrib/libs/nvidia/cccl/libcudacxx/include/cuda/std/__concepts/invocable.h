//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_INVOCABLE_H
#define _LIBCUDACXX___CONCEPTS_INVOCABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.invocable]

template <class _Fn, class... _Args>
concept invocable = requires(_Fn&& __fn, _Args&&... __args) {
  _CUDA_VSTD::__invoke(_CUDA_VSTD::forward<_Fn>(__fn), _CUDA_VSTD::forward<_Args>(__args)...); // not required to be
                                                                                               // equality preserving
};

// [concept.regular.invocable]

template <class _Fn, class... _Args>
concept regular_invocable = invocable<_Fn, _Args...>;

template <class _Fun, class... _Args>
concept __invoke_constructible = requires(_Fun&& __fun, _Args&&... __args) {
  static_cast<remove_cvref_t<invoke_result_t<_Fun, _Args...>>>(
    _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...));
};

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Fn, class... _Args>
_CCCL_CONCEPT_FRAGMENT(_Invocable_,
                       requires(_Fn&& __fn, _Args&&... __args)(
                         (_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn>(__fn), _CUDA_VSTD::forward<_Args>(__args)...))));

template <class _Fn, class... _Args>
_CCCL_CONCEPT invocable = _CCCL_FRAGMENT(_Invocable_, _Fn, _Args...);

template <class _Fn, class... _Args>
_CCCL_CONCEPT regular_invocable = invocable<_Fn, _Args...>;

template <class _Fun, class... _Args>
_CCCL_CONCEPT_FRAGMENT(
  __invoke_constructible_,
  requires(_Fun&& __fun, _Args&&... __args)((static_cast<remove_cvref_t<invoke_result_t<_Fun, _Args...>>>(
    _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)))));
template <class _Fun, class... _Args>
_CCCL_CONCEPT __invoke_constructible = _CCCL_FRAGMENT(__invoke_constructible_, _Fun, _Args...);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_INVOCABLE_H
