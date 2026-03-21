//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H
#define _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/copy_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.commonref]

template <class _Tp, class _Up>
concept common_reference_with =
  same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>>
  && convertible_to<_Tp, common_reference_t<_Tp, _Up>> && convertible_to<_Up, common_reference_t<_Tp, _Up>>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(__common_reference_exists_,
                       requires()(typename(common_reference_t<_Tp, _Up>), typename(common_reference_t<_Up, _Tp>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT _Common_reference_exists = _CCCL_FRAGMENT(__common_reference_exists_, _Tp, _Up);

template <class _Tp, class _Up>
_CCCL_CONCEPT_FRAGMENT(
  __common_reference_with_,
  requires()(requires(_Common_reference_exists<_Tp, _Up>),
             requires(same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>>),
             requires(convertible_to<_Tp, common_reference_t<_Tp, _Up>>),
             requires(convertible_to<_Up, common_reference_t<_Tp, _Up>>)));

template <class _Tp, class _Up>
_CCCL_CONCEPT common_reference_with = _CCCL_FRAGMENT(__common_reference_with_, _Tp, _Up);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_COMMON_REFERENCE_WITH_H
