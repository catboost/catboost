// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_SIZE_H
#define _LIBCUDACXX___ITERATOR_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Cont>
_CCCL_API constexpr auto size(const _Cont& __c) noexcept(noexcept(__c.size())) -> decltype(__c.size())
{
  return __c.size();
}

template <class _Tp, size_t _Sz>
_CCCL_API constexpr size_t size(const _Tp (&)[_Sz]) noexcept
{
  return _Sz;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Cont>
_CCCL_API constexpr auto ssize(const _Cont& __c) noexcept(
  noexcept(static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size())))
  -> common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>
{
  return static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size());
}

// GCC complains about the implicit conversion from ptrdiff_t to size_t in
// the array bound.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wsign-conversion")
template <class _Tp, ptrdiff_t _Sz>
_CCCL_API constexpr ptrdiff_t ssize(const _Tp (&)[_Sz]) noexcept
{
  return _Sz;
}
_CCCL_DIAG_POP

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_SIZE_H
