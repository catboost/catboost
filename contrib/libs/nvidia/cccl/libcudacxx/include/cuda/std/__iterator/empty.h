// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_EMPTY_H
#define _LIBCUDACXX___ITERATOR_EMPTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Cont>
[[nodiscard]] _CCCL_API constexpr auto empty(const _Cont& __c) noexcept(noexcept(__c.empty())) -> decltype(__c.empty())
{
  return __c.empty();
}

template <class _Tp, size_t _Sz>
[[nodiscard]] _CCCL_API constexpr bool empty(const _Tp (&)[_Sz]) noexcept
{
  return false;
}

template <class _Ep>
[[nodiscard]] _CCCL_API constexpr bool empty(initializer_list<_Ep> __il) noexcept
{
  return __il.size() == 0;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_EMPTY_H
