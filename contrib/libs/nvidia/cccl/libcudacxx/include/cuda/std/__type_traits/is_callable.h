//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Func, class... _Args>
using __call_result_t _CCCL_NODEBUG_ALIAS = decltype(_CUDA_VSTD::declval<_Func>()(_CUDA_VSTD::declval<_Args>()...));

template <class _Func, class... _Args>
struct __is_callable : _IsValidExpansion<__call_result_t, _Func, _Args...>
{};

template <class _Func, class... _Args>
inline constexpr bool __is_callable_v = _IsValidExpansion<__call_result_t, _Func, _Args...>::value;

namespace detail
{
template <class _Func, class... _Args>
using __if_nothrow_callable_t _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::enable_if_t<noexcept(_CUDA_VSTD::declval<_Func>()(_CUDA_VSTD::declval<_Args>()...))>;
} // namespace detail

template <class _Func, class... _Args>
struct __is_nothrow_callable : _IsValidExpansion<detail::__if_nothrow_callable_t, _Func, _Args...>
{};

template <class _Func, class... _Args>
inline constexpr bool __is_nothrow_callable_v =
  _IsValidExpansion<detail::__if_nothrow_callable_t, _Func, _Args...>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_CALLABLE_H
