//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
#define _LIBCUDACXX___CONCEPTS_ARITHMETIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_signed_integer.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// [concepts.arithmetic], arithmetic concepts

template <class _Tp>
_CCCL_CONCEPT integral = _CCCL_TRAIT(is_integral, _Tp);

template <class _Tp>
_CCCL_CONCEPT signed_integral = integral<_Tp> && _CCCL_TRAIT(is_signed, _Tp);

template <class _Tp>
_CCCL_CONCEPT unsigned_integral = integral<_Tp> && !signed_integral<_Tp>;

template <class _Tp>
_CCCL_CONCEPT floating_point = _CCCL_TRAIT(is_floating_point, _Tp);

template <class _Tp>
_CCCL_CONCEPT __cccl_signed_integer = __cccl_is_signed_integer_v<_Tp>;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
