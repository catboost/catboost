//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_EXTENT_H
#define _LIBCUDACXX___TYPE_TRAITS_EXTENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_ARRAY_EXTENT) && !defined(_LIBCUDACXX_USE_ARRAY_EXTENT_FALLBACK)

template <class _Tp, size_t _Dim = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : integral_constant<size_t, _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Dim)>
{};

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Ip);

#else // ^^^ _CCCL_BUILTIN_ARRAY_EXTENT ^^^ / vvv !_CCCL_BUILTIN_ARRAY_EXTENT vvv

template <class _Tp, unsigned _Ip = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : public integral_constant<size_t, 0>
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[], 0> : public integral_constant<size_t, 0>
{};
template <class _Tp, unsigned _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[], _Ip> : public integral_constant<size_t, extent<_Tp, _Ip - 1>::value>
{};
template <class _Tp, size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent<_Tp[_Np], 0> : public integral_constant<size_t, _Np>
{};
template <class _Tp, size_t _Np, unsigned _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
extent<_Tp[_Np], _Ip> : public integral_constant<size_t, extent<_Tp, _Ip - 1>::value>
{};

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = extent<_Tp, _Ip>::value;

#endif // !_CCCL_BUILTIN_ARRAY_EXTENT

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_EXTENT_H
