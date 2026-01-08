//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H
#define _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H

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
#include <cuda/std/__atomic/wait/polling.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

extern "C" _CCCL_DEVICE void __atomic_try_wait_unsupported_before_SM_70__();

template <typename _Tp, typename _Sco>
_CCCL_API inline void
__atomic_try_wait_slow(_Tp const volatile* __a, __atomic_underlying_remove_cv_t<_Tp> __val, memory_order __order, _Sco)
{
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});
                     , NV_IS_HOST, __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});
                     , NV_ANY_TARGET, __atomic_try_wait_unsupported_before_SM_70__(););
}

template <typename _Tp, typename _Sco>
_CCCL_API inline void __atomic_notify_one(_Tp const volatile*, _Sco)
{
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, , NV_IS_HOST, , NV_ANY_TARGET, __atomic_try_wait_unsupported_before_SM_70__(););
}

template <typename _Tp, typename _Sco>
_CCCL_API inline void __atomic_notify_all(_Tp const volatile*, _Sco)
{
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, , NV_IS_HOST, , NV_ANY_TARGET, __atomic_try_wait_unsupported_before_SM_70__(););
}

template <typename _Tp>
_CCCL_API inline bool __nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs)
{
#if _CCCL_CUDA_COMPILATION()
  return __lhs == __rhs;
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
  return _CUDA_VSTD::memcmp(&__lhs, &__rhs, sizeof(_Tp)) == 0;
#endif // ^^^ !_CCCL_CUDA_COMPILATION() ^^^
}

template <typename _Tp, typename _Sco>
_CCCL_API inline void __atomic_wait(
  _Tp const volatile* __a, __atomic_underlying_remove_cv_t<_Tp> const __val, memory_order __order, _Sco = {})
{
  for (int __i = 0; __i < _LIBCUDACXX_POLLING_COUNT; ++__i)
  {
    if (!__nonatomic_compare_equal(__atomic_load_dispatch(__a, __order, _Sco{}), __val))
    {
      return;
    }
    if (__i < 12)
    {
      _CUDA_VSTD::__cccl_thread_yield_processor();
    }
    else
    {
      _CUDA_VSTD::__cccl_thread_yield();
    }
  }
  while (__nonatomic_compare_equal(__atomic_load_dispatch(__a, __order, _Sco{}), __val))
  {
    __atomic_try_wait_slow(__a, __val, __order, _Sco{});
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H
