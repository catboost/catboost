//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// result_of

#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)
template <class _Callable>
class result_of;

template <class _Fp, class... _Args>
class _LIBCUDACXX_DEPRECATED _CCCL_TYPE_VISIBILITY_DEFAULT result_of<_Fp(_Args...)> : public __invoke_of<_Fp, _Args...>
{};

template <class _Tp>
using result_of_t _LIBCUDACXX_DEPRECATED = typename result_of<_Tp>::type;
#endif // _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H
