//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H

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
#include <cuda/std/__concepts/destructible.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/is_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.constructible]
template <class _Tp, class... _Args>
concept constructible_from = destructible<_Tp> && is_constructible_v<_Tp, _Args...>;

// [concept.default.init]
template <class _Tp>
concept __default_initializable = requires { ::new _Tp; };

template <class _Tp>
concept default_initializable = constructible_from<_Tp> && requires { _Tp{}; } && __default_initializable<_Tp>;

// [concept.moveconstructible]
template <class _Tp>
concept move_constructible = constructible_from<_Tp, _Tp> && convertible_to<_Tp, _Tp>;

// [concept.copyconstructible]
template <class _Tp>
concept copy_constructible =
  move_constructible<_Tp> && constructible_from<_Tp, _Tp&> && convertible_to<_Tp&, _Tp>
  && constructible_from<_Tp, const _Tp&> && convertible_to<const _Tp&, _Tp> && constructible_from<_Tp, const _Tp>
  && convertible_to<const _Tp, _Tp>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class... _Args>
_CCCL_CONCEPT_FRAGMENT(__constructible_from_,
                       requires()(requires(destructible<_Tp>), requires(_CCCL_TRAIT(is_constructible, _Tp, _Args...))));

template <class _Tp, class... _Args>
_CCCL_CONCEPT constructible_from = _CCCL_FRAGMENT(__constructible_from_, _Tp, _Args...);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__default_initializable_, requires()((::new _Tp)));

template <class _Tp>
_CCCL_CONCEPT __default_initializable = _CCCL_FRAGMENT(__default_initializable_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(_Default_initializable_,
                       requires(_Tp = _Tp{})(requires(constructible_from<_Tp>), requires(__default_initializable<_Tp>)));

template <class _Tp>
_CCCL_CONCEPT default_initializable = _CCCL_FRAGMENT(_Default_initializable_, _Tp);

// [concept.moveconstructible]
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__move_constructible_,
                       requires()(requires(constructible_from<_Tp, _Tp>), requires(convertible_to<_Tp, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT move_constructible = _CCCL_FRAGMENT(__move_constructible_, _Tp);

// [concept.copyconstructible]
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __copy_constructible_,
  requires()(
    requires(move_constructible<_Tp>),
    requires(constructible_from<_Tp, add_lvalue_reference_t<_Tp>>&& convertible_to<add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const add_lvalue_reference_t<_Tp>>&&
               convertible_to<const add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const _Tp>&& convertible_to<const _Tp, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT copy_constructible = _CCCL_FRAGMENT(__copy_constructible_, _Tp);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_CONSTRUCTIBLE_H
