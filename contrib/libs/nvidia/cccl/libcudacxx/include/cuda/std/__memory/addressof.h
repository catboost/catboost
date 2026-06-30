// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ADDRESSOF_H
#define _LIBCUDACXX___MEMORY_ADDRESSOF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4312) // warning C4312: 'type cast': conversion from '_Tp' to '_Tp *' of greater size

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_BUILTIN_STD_ADDRESSOF()

// The compiler treats ::std::addressof as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::addressof;

#elif defined(_CCCL_BUILTIN_ADDRESSOF)

template <class _Tp>
[[nodiscard]] _CCCL_API inline _CCCL_NO_CFI constexpr _Tp* addressof(_Tp& __x) noexcept
{
  return _CCCL_BUILTIN_ADDRESSOF(__x);
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#else

template <class _Tp>
[[nodiscard]] _CCCL_API inline _CCCL_NO_CFI _Tp* addressof(_Tp& __x) noexcept
{
  return reinterpret_cast<_Tp*>(const_cast<char*>(&reinterpret_cast<const volatile char&>(__x)));
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#endif // defined(_CCCL_BUILTIN_ADDRESSOF)

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_ADDRESSOF_H
