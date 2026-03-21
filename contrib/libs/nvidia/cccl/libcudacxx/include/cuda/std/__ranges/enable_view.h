// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_ENABLE_VIEW_H
#define _LIBCUDACXX___RANGES_ENABLE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

struct view_base
{};

#if _CCCL_HAS_CONCEPTS()

template <class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
class view_interface;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Derived, enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int> = 0>
class view_interface;

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_TEMPLATE(class _Op, class _Yp)
_CCCL_REQUIRES(is_convertible_v<_Op*, view_interface<_Yp>*>)
_CCCL_API inline void __is_derived_from_view_interface(const _Op*, const view_interface<_Yp>*);

#if _CCCL_HAS_CONCEPTS()

template <class _Tp>
inline constexpr bool enable_view = derived_from<_Tp, view_base> || requires {
  _CUDA_VRANGES::__is_derived_from_view_interface((_Tp*) nullptr, (_Tp*) nullptr);
};

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class = void>
inline constexpr bool enable_view = derived_from<_Tp, view_base>;

template <class _Tp>
inline constexpr bool
  enable_view<_Tp, void_t<decltype(_CUDA_VRANGES::__is_derived_from_view_interface((_Tp*) nullptr, (_Tp*) nullptr))>> =
    true;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_ENABLE_VIEW_H
