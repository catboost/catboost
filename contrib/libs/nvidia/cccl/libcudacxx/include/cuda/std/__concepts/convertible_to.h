//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H
#define _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [concept.convertible]

#if _CCCL_HAS_CONCEPTS()

template <class _From, class _To>
concept convertible_to = is_convertible_v<_From, _To> && requires { static_cast<_To>(_CUDA_VSTD::declval<_From>()); };

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

#  if _CCCL_COMPILER(MSVC)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(1211) // nonstandard cast to array type ignored
#  endif // _CCCL_COMPILER(MSVC)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(171) // invalid type conversion, e.g. [with _From=int **, _To=const int *const *]

// We cannot put this conversion check with the other constraint, as types with deleted operator will break here
template <class _From, class _To>
_CCCL_CONCEPT_FRAGMENT(__test_conversion_, requires()(static_cast<_To>(_CUDA_VSTD::declval<_From>())));

template <class _From, class _To>
_CCCL_CONCEPT __test_conversion = _CCCL_FRAGMENT(__test_conversion_, _From, _To);

template <class _From, class _To>
_CCCL_CONCEPT_FRAGMENT(
  __convertible_to_,
  requires()(requires(_CCCL_TRAIT(is_convertible, _From, _To)), requires(__test_conversion<_From, _To>)));

template <class _From, class _To>
_CCCL_CONCEPT convertible_to = _CCCL_FRAGMENT(__convertible_to_, _From, _To);

#  if _CCCL_COMPILER(MSVC)
_CCCL_END_NV_DIAG_SUPPRESS() // nonstandard cast to array type ignored
#  endif // _CCCL_COMPILER(MSVC)
_CCCL_END_NV_DIAG_SUPPRESS() // invalid type conversion, e.g. [with _From=int **, _To=const int *const *]

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H
