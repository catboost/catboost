// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_DATA_H
#define _LIBCUDACXX___ITERATOR_DATA_H

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

_CCCL_EXEC_CHECK_DISABLE
template <class _Cont>
[[nodiscard]] _CCCL_API constexpr auto data(_Cont& __c) noexcept(noexcept(__c.data())) -> decltype(__c.data())
{
  return __c.data();
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Cont>
[[nodiscard]] _CCCL_API constexpr auto data(const _Cont& __c) noexcept(noexcept(__c.data())) -> decltype(__c.data())
{
  return __c.data();
}

template <class _Tp, size_t _Sz>
_CCCL_API constexpr _Tp* data(_Tp (&__array)[_Sz]) noexcept
{
  return __array;
}

template <class _Ep>
_CCCL_API constexpr const _Ep* data(initializer_list<_Ep> __il) noexcept
{
  return __il.begin();
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_DATA_H
