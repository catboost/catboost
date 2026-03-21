//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
#define _LIBCUDACXX___TUPLE_TUPLE_LIKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __tuple_like_impl : false_type
{};

template <class _Tp>
struct __tuple_like_impl<const _Tp> : public __tuple_like_impl<_Tp>
{};
template <class _Tp>
struct __tuple_like_impl<volatile _Tp> : public __tuple_like_impl<_Tp>
{};
template <class _Tp>
struct __tuple_like_impl<const volatile _Tp> : public __tuple_like_impl<_Tp>
{};

template <class... _Tp>
struct __tuple_like_impl<tuple<_Tp...>> : true_type
{};

template <class _T1, class _T2>
struct __tuple_like_impl<pair<_T1, _T2>> : true_type
{};

template <class _Tp, size_t _Size>
struct __tuple_like_impl<array<_Tp, _Size>> : true_type
{};

template <class _Tp>
struct __tuple_like_impl<complex<_Tp>> : true_type
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct __tuple_like_impl<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>> : true_type
{};

template <class... _Tp>
struct __tuple_like_impl<__tuple_types<_Tp...>> : true_type
{};

#if _CCCL_STD_VER >= 2014
template <class _Tp>
_CCCL_CONCEPT __tuple_like = __tuple_like_impl<remove_cvref_t<_Tp>>::value;

template <class _Tp>
_CCCL_CONCEPT __pair_like = _CCCL_REQUIRES_EXPR((_Tp)) //
  (requires(__tuple_like_impl<remove_cvref_t<_Tp>>::value), requires(tuple_size<remove_cvref_t<_Tp>>::value == 2));
#endif // _CCCL_STD_VER >= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_TUPLE_LIKE_H
