// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___ITERATOR_PROJECTED_H
#define _LIBCUDACXX___ITERATOR_PROJECTED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _It, class _Proj, class = void>
struct __projected_impl
{
  struct __type
  {
    using value_type = remove_cvref_t<indirect_result_t<_Proj, _It>>;
    _CCCL_API inline indirect_result_t<_Proj, _It> operator*() const; // not defined
  };
};

template <class _It, class _Proj>
struct __projected_impl<_It, _Proj, enable_if_t<weakly_incrementable<_It>>>
{
  struct __type
  {
    using value_type      = remove_cvref_t<indirect_result_t<_Proj, _It>>;
    using difference_type = iter_difference_t<_It>;
    _CCCL_API inline indirect_result_t<_Proj, _It> operator*() const; // not defined
  };
};

_CCCL_TEMPLATE(class _It, class _Proj)
_CCCL_REQUIRES(indirectly_readable<_It> _CCCL_AND indirectly_regular_unary_invocable<_Proj, _It>)
using projected = typename __projected_impl<_It, _Proj>::__type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_PROJECTED_H
