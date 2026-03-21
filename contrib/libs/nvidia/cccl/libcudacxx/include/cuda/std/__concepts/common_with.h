//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_COMMON_WITH_H
#define _LIBCUDACXX___CONCEPTS_COMMON_WITH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/common_reference_with.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.common]

template <class _Tp, class _Up>
concept common_with = same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>> && requires {
  static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Tp>());
  static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Up>());
} && common_reference_with<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>> && common_reference_with<add_lvalue_reference_t<common_type_t<_Tp, _Up>>, common_reference_t<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(__common_type_exists_,
                       requires()(typename(common_type_t<_Tp, _Up>), typename(common_type_t<_Up, _Tp>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT _Common_type_exists = _CCCL_FRAGMENT(__common_type_exists_, _Tp, _Up);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(__common_type_constructible_,
                       requires()(requires(_Common_type_exists<_Tp, _Up>),
                                  (static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Tp>())),
                                  (static_cast<common_type_t<_Tp, _Up>>(_CUDA_VSTD::declval<_Up>()))));

template <class _Tp, class _Up>
_CCCL_CONCEPT _Common_type_constructible = _CCCL_FRAGMENT(__common_type_constructible_, _Tp, _Up);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __common_with_,
  requires()(
    requires(_Common_type_constructible<_Tp, _Up>),
    requires(same_as<common_type_t<_Tp, _Up>, common_type_t<_Up, _Tp>>),
    requires(common_reference_with<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>),
    requires(
      common_reference_with<add_lvalue_reference_t<common_type_t<_Tp, _Up>>,
                            common_reference_t<add_lvalue_reference_t<const _Tp>, add_lvalue_reference_t<const _Up>>>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT common_with = _CCCL_FRAGMENT(__common_with_, _Tp, _Up);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_COMMON_WITH_H
