//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_LVALUE_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_LVALUE_REFERENCE_FALLBACK)  \
  && defined(_CCCL_BUILTIN_IS_RVALUE_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_RVALUE_REFERENCE_FALLBACK) \
  && defined(_CCCL_BUILTIN_IS_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_REFERENCE_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_lvalue_reference : public integral_constant<bool, _CCCL_BUILTIN_IS_LVALUE_REFERENCE(_Tp)>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_rvalue_reference : public integral_constant<bool, _CCCL_BUILTIN_IS_RVALUE_REFERENCE(_Tp)>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference : public integral_constant<bool, _CCCL_BUILTIN_IS_REFERENCE(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_lvalue_reference_v = _CCCL_BUILTIN_IS_LVALUE_REFERENCE(_Tp);
template <class _Tp>
inline constexpr bool is_rvalue_reference_v = _CCCL_BUILTIN_IS_RVALUE_REFERENCE(_Tp);
template <class _Tp>
inline constexpr bool is_reference_v = _CCCL_BUILTIN_IS_REFERENCE(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_lvalue_reference : public false_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_lvalue_reference<_Tp&> : public true_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_rvalue_reference : public false_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_rvalue_reference<_Tp&&> : public true_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference : public false_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference<_Tp&> : public true_type
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference<_Tp&&> : public true_type
{};

template <class _Tp>
inline constexpr bool is_lvalue_reference_v = is_lvalue_reference<_Tp>::value;

template <class _Tp>
inline constexpr bool is_rvalue_reference_v = is_rvalue_reference<_Tp>::value;

template <class _Tp>
inline constexpr bool is_reference_v = is_reference<_Tp>::value;

#endif // !_CCCL_BUILTIN_IS_LVALUE_REFERENCE

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_H
