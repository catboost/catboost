//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_nothrow_destructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_COMPILER(MSVC)

template <class _Tp>
_CCCL_CONCEPT destructible = __is_nothrow_destructible(_Tp);

#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv

template <class _Tp, class = void, class = void>
inline constexpr bool __destructible_impl = false;

template <class _Tp>
inline constexpr bool __destructible_impl<_Tp,
                                          enable_if_t<_CCCL_TRAIT(is_object, _Tp)>,
#  if _CCCL_COMPILER(GCC)
                                          enable_if_t<_CCCL_TRAIT(is_destructible, _Tp)>>
#  else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
                                          void_t<decltype(_CUDA_VSTD::declval<_Tp>().~_Tp())>>
#  endif // !_CCCL_COMPILER(GCC)
  = noexcept(_CUDA_VSTD::declval<_Tp>().~_Tp());

template <class _Tp>
inline constexpr bool __destructible = __destructible_impl<_Tp>;

template <class _Tp>
inline constexpr bool __destructible<_Tp&> = true;

template <class _Tp>
inline constexpr bool __destructible<_Tp&&> = true;

template <class _Tp, size_t _Nm>
inline constexpr bool __destructible<_Tp[_Nm]> = __destructible<_Tp>;

template <class _Tp>
_CCCL_CONCEPT destructible = __destructible<_Tp>;

#endif // !_CCCL_COMPILER(MSVC)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
