//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FUNCTIONAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FUNCTIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_reference.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4180) // qualifier applied to function type has no meaning; ignored

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_FUNCTION) && !defined(_LIBCUDACXX_USE_IS_FUNCTION_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_function : integral_constant<bool, _CCCL_BUILTIN_IS_FUNCTION(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_function_v = _CCCL_BUILTIN_IS_FUNCTION(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_function : public integral_constant<bool, !(is_reference<_Tp>::value || is_const<const _Tp>::value)>
{};

template <class _Tp>
inline constexpr bool is_function_v = is_function<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_FUNCTION) && !defined(_LIBCUDACXX_USE_IS_FUNCTION_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FUNCTIONAL_H
