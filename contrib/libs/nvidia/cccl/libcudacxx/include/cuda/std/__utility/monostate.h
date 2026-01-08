// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_MONOSTATE_H
#define _LIBCUDACXX___UTILITY_MONOSTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__functional/hash.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT monostate
{};

_CCCL_API constexpr bool operator==(monostate, monostate) noexcept
{
  return true;
}

#if _CCCL_STD_VER < 2020

_CCCL_API constexpr bool operator!=(monostate, monostate) noexcept
{
  return false;
}

#endif // _CCCL_STD_VER < 2020

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_API constexpr strong_ordering operator<=>(monostate, monostate) noexcept
{
  return strong_ordering::equal;
}

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_API constexpr bool operator<(monostate, monostate) noexcept
{
  return false;
}

_CCCL_API constexpr bool operator>(monostate, monostate) noexcept
{
  return false;
}

_CCCL_API constexpr bool operator<=(monostate, monostate) noexcept
{
  return true;
}

_CCCL_API constexpr bool operator>=(monostate, monostate) noexcept
{
  return true;
}

#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#ifndef __cuda_std__
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<monostate>
{
  using argument_type = monostate;
  using result_type   = size_t;

  _CCCL_API inline result_type operator()(const argument_type&) const noexcept
  {
    return 66740831; // return a fundamentally attractive random value.
  }
};
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_MONOSTATE_H
