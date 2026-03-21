//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H
#define _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H

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
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/make_const_lvalue_ref.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.assignable]

template <class _Lhs, class _Rhs>
concept assignable_from =
  is_lvalue_reference_v<_Lhs> && common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>>
  && requires(_Lhs __lhs, _Rhs&& __rhs) {
       { __lhs = _CUDA_VSTD::forward<_Rhs>(__rhs) } -> same_as<_Lhs>;
     };

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Lhs, class _Rhs>
_CCCL_CONCEPT_FRAGMENT(
  __assignable_from_,
  requires(_Lhs __lhs,
           _Rhs&& __rhs)(requires(_CCCL_TRAIT(is_lvalue_reference, _Lhs)),
                         requires(common_reference_with<__make_const_lvalue_ref<_Lhs>, __make_const_lvalue_ref<_Rhs>>),
                         requires(same_as<_Lhs, decltype(__lhs = _CUDA_VSTD::forward<_Rhs>(__rhs))>)));

template <class _Lhs, class _Rhs>
_CCCL_CONCEPT assignable_from = _CCCL_FRAGMENT(__assignable_from_, _Lhs, _Rhs);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_ASSIGNABLE_H
