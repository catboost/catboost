// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2023 Microsoft Corporation.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H
#define _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// MSVC requires an interesting workaround for a /permissive- bug
// We cannot simply define unreachable_sentinel_t with it friendfunctions,
// but we must derive from a base class in a different namespace so that they
// are only ever found through ADL

struct unreachable_sentinel_t
#if _CCCL_COMPILER(MSVC)
  ;
namespace __unreachable_sentinel_detail
{
struct __unreachable_base
#endif // _CCCL_COMPILER(MSVC)
{
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(weakly_incrementable<_Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const unreachable_sentinel_t&, const _Iter&) noexcept
  {
    return false;
  }
#if _CCCL_STD_VER < 2020
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(weakly_incrementable<_Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const _Iter&, const unreachable_sentinel_t&) noexcept
  {
    return false;
  }
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(weakly_incrementable<_Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const unreachable_sentinel_t&, const _Iter&) noexcept
  {
    return true;
  }
  _CCCL_TEMPLATE(class _Iter)
  _CCCL_REQUIRES(weakly_incrementable<_Iter>)
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const _Iter&, const unreachable_sentinel_t&) noexcept
  {
    return true;
  }
#endif // _CCCL_STD_VER < 2020
};

#if _CCCL_COMPILER(MSVC)
} // namespace __unreachable_sentinel_detail
struct unreachable_sentinel_t : __unreachable_sentinel_detail::__unreachable_base
{};
#endif // _CCCL_COMPILER(MSVC)

_CCCL_GLOBAL_CONSTANT unreachable_sentinel_t unreachable_sentinel{};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_UNREACHABLE_SENTINEL_H
