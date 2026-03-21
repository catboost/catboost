//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_default_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// First of all, we can't implement this check in C++03 mode because the {}
// default initialization syntax isn't valid.
// Second, we implement the trait in a funny manner with two defaulted template
// arguments to workaround Clang's PR43454.
template <class _Tp>
_CCCL_API inline void __test_implicit_default_constructible(_Tp);

template <class _Tp, class = void, class = typename is_default_constructible<_Tp>::type>
struct __is_implicitly_default_constructible : false_type
{};

template <class _Tp>
struct __is_implicitly_default_constructible<_Tp,
                                             decltype(__test_implicit_default_constructible<_Tp const&>({})),
                                             true_type> : true_type
{};

template <class _Tp>
struct __is_implicitly_default_constructible<_Tp,
                                             decltype(__test_implicit_default_constructible<_Tp const&>({})),
                                             false_type> : false_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_IMPLICITLY_DEFAULT_CONSTRUCTIBLE_H
