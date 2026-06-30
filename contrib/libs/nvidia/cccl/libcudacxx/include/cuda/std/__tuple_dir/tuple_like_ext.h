//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_LIKE_EXT_H
#define _LIBCUDACXX___TUPLE_TUPLE_LIKE_EXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __tuple_like_ext : false_type
{};

template <class _Tp>
struct __tuple_like_ext<const _Tp> : public __tuple_like_ext<_Tp>
{};
template <class _Tp>
struct __tuple_like_ext<volatile _Tp> : public __tuple_like_ext<_Tp>
{};
template <class _Tp>
struct __tuple_like_ext<const volatile _Tp> : public __tuple_like_ext<_Tp>
{};

template <class... _Tp>
struct __tuple_like_ext<tuple<_Tp...>> : true_type
{};

template <class _T1, class _T2>
struct __tuple_like_ext<pair<_T1, _T2>> : true_type
{};

template <class _Tp, size_t _Size>
struct __tuple_like_ext<array<_Tp, _Size>> : true_type
{};

template <class _Tp>
struct __tuple_like_ext<complex<_Tp>> : true_type
{};

template <class... _Tp>
struct __tuple_like_ext<__tuple_types<_Tp...>> : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_TUPLE_LIKE_EXT_H
