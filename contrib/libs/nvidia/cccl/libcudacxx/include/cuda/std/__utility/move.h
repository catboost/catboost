// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_MOVE_H
#define _LIBCUDACXX___UTILITY_MOVE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_reference.h>

// When _CCCL_HAS_BUILTIN_STD_MOVE() is 1, the compiler treats ::std::move as a builtin
// function so it never needs to be instantiated and will be compiled away even at -O0. We
// would prefer to bring the ::std:: function into the ::cuda::std:: namespace with a using
// declaration, like this:
//
//   _LIBCUDACXX_BEGIN_NAMESPACE_STD
//   using ::std::move;
//   _LIBCUDACXX_END_NAMESPACE_STD
//
// But "using ::std::move;" would also drag in the algorithm ::std::move(In, In, Out),
// which would conflict with ::cuda::std::move algorithm in <cuda/std/algorithm.h>.
//
// So instead, we define a _CCCL_MOVE macro that can be used in place of _CUDA_VSTD::move

#if _CCCL_HAS_BUILTIN_STD_MOVE()
#  define _CCCL_MOVE(...) ::std::move(__VA_ARGS__)
#else // ^^^ _CCCL_HAS_BUILTIN_STD_MOVE() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_MOVE() vvv
#  define _CCCL_MOVE(...) _CUDA_VSTD::move(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN_STD_MOVE()

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr remove_reference_t<_Tp>&& move(_Tp&& __t) noexcept
{
  using _Up _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tp>;
  return static_cast<_Up&&>(__t);
}

template <class _Tp>
using __move_if_noexcept_result_t =
  conditional_t<!is_nothrow_move_constructible<_Tp>::value && is_copy_constructible<_Tp>::value, const _Tp&, _Tp&&>;

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr __move_if_noexcept_result_t<_Tp> move_if_noexcept(_Tp& __x) noexcept
{
  return _CUDA_VSTD::move(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_MOVE_H
