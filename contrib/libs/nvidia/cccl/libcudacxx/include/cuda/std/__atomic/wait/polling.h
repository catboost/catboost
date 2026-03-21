//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_WAIT_POLLING_H
#define _LIBCUDACXX___ATOMIC_WAIT_POLLING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/types.h>
#include <cuda/std/__thread/threading_support.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp, typename _Sco>
struct __atomic_poll_tester
{
  using __underlying_t = __atomic_underlying_remove_cv_t<_Tp>;

  _Tp const volatile* __atom;
  __underlying_t __val;
  memory_order __order;

  _CCCL_HOST_DEVICE __atomic_poll_tester(_Tp const volatile* __a, __underlying_t __v, memory_order __o)
      : __atom(__a)
      , __val(__v)
      , __order(__o)
  {}

  _CCCL_HOST_DEVICE bool operator()() const
  {
    return !(__atomic_load_dispatch(__atom, __order, _Sco{}) == __val);
  }
};

template <typename _Tp, typename _Sco>
_CCCL_HOST_DEVICE void __atomic_try_wait_slow_fallback(
  _Tp const volatile* __a, __atomic_underlying_remove_cv_t<_Tp> __val, memory_order __order, _Sco)
{
  _CUDA_VSTD::__cccl_thread_poll_with_backoff(__atomic_poll_tester<_Tp, _Sco>(__a, __val, __order));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_WAIT_POLLING_H
