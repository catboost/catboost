//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_ORDER_H
#define __LIBCUDACXX___ATOMIC_ORDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#define _LIBCUDACXX_CHECK_STORE_MEMORY_ORDER(__m)                                              \
  _LIBCUDACXX_DIAGNOSE_WARNING(                                                                \
    __m == memory_order_consume || __m == memory_order_acquire || __m == memory_order_acq_rel, \
    "memory order argument to atomic operation is invalid")

#define _LIBCUDACXX_CHECK_LOAD_MEMORY_ORDER(__m)                                           \
  _LIBCUDACXX_DIAGNOSE_WARNING(__m == memory_order_release || __m == memory_order_acq_rel, \
                               "memory order argument to atomic operation is invalid")

#define _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__m, __f)                                  \
  _LIBCUDACXX_DIAGNOSE_WARNING(__f == memory_order_release || __f == memory_order_acq_rel, \
                               "memory order argument to atomic operation is invalid")

#ifndef __ATOMIC_RELAXED
#  define __ATOMIC_RELAXED 0
#  define __ATOMIC_CONSUME 1
#  define __ATOMIC_ACQUIRE 2
#  define __ATOMIC_RELEASE 3
#  define __ATOMIC_ACQ_REL 4
#  define __ATOMIC_SEQ_CST 5
#endif //__ATOMIC_RELAXED

// Figure out what the underlying type for `memory_order` would be if it were
// declared as an unscoped enum (accounting for -fshort-enums). Use this result
// to pin the underlying type in C++20.
enum __legacy_memory_order
{
  __mo_relaxed,
  __mo_consume,
  __mo_acquire,
  __mo_release,
  __mo_acq_rel,
  __mo_seq_cst
};

using __memory_order_underlying_t = underlying_type<__legacy_memory_order>::type;

#if _CCCL_STD_VER >= 2020

enum class memory_order : __memory_order_underlying_t
{
  relaxed = __mo_relaxed,
  consume = __mo_consume,
  acquire = __mo_acquire,
  release = __mo_release,
  acq_rel = __mo_acq_rel,
  seq_cst = __mo_seq_cst
};

inline constexpr auto memory_order_relaxed = memory_order::relaxed;
inline constexpr auto memory_order_consume = memory_order::consume;
inline constexpr auto memory_order_acquire = memory_order::acquire;
inline constexpr auto memory_order_release = memory_order::release;
inline constexpr auto memory_order_acq_rel = memory_order::acq_rel;
inline constexpr auto memory_order_seq_cst = memory_order::seq_cst;

#else // ^^^ C++20 ^^^ / vvv C++17 vvv

using memory_order = enum memory_order {
  memory_order_relaxed = __mo_relaxed,
  memory_order_consume = __mo_consume,
  memory_order_acquire = __mo_acquire,
  memory_order_release = __mo_release,
  memory_order_acq_rel = __mo_acq_rel,
  memory_order_seq_cst = __mo_seq_cst,
};

#endif // _CCCL_STD_VER >= 2020

_CCCL_HOST_DEVICE inline int __stronger_order_cuda(int __a, int __b)
{
  int const __max = __a > __b ? __a : __b;
  if (__max != __ATOMIC_RELEASE)
  {
    return __max;
  }
  constexpr int __xform[] = {__ATOMIC_RELEASE, __ATOMIC_ACQ_REL, __ATOMIC_ACQ_REL, __ATOMIC_RELEASE};
  return __xform[__a < __b ? __a : __b];
}

_CCCL_HOST_DEVICE inline constexpr int __atomic_order_to_int(memory_order __order)
{
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_relaxed
         ? __ATOMIC_RELAXED
         : (__order == memory_order_acquire
              ? __ATOMIC_ACQUIRE
              : (__order == memory_order_release
                   ? __ATOMIC_RELEASE
                   : (__order == memory_order_seq_cst
                        ? __ATOMIC_SEQ_CST
                        : (__order == memory_order_acq_rel ? __ATOMIC_ACQ_REL : __ATOMIC_CONSUME))));
}

_CCCL_HOST_DEVICE inline constexpr int __atomic_failure_order_to_int(memory_order __order)
{
  // Avoid switch statement to make this a constexpr.
  return __order == memory_order_relaxed
         ? __ATOMIC_RELAXED
         : (__order == memory_order_acquire
              ? __ATOMIC_ACQUIRE
              : (__order == memory_order_release
                   ? __ATOMIC_RELAXED
                   : (__order == memory_order_seq_cst
                        ? __ATOMIC_SEQ_CST
                        : (__order == memory_order_acq_rel ? __ATOMIC_ACQUIRE : __ATOMIC_CONSUME))));
}

static_assert((is_same<underlying_type<memory_order>::type, __memory_order_underlying_t>::value),
              "unexpected underlying type for std::memory_order");

_LIBCUDACXX_END_NAMESPACE_STD

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

using memory_order = _CUDA_VSTD::memory_order;

inline constexpr memory_order memory_order_relaxed = _CUDA_VSTD::memory_order_relaxed;
inline constexpr memory_order memory_order_consume = _CUDA_VSTD::memory_order_consume;
inline constexpr memory_order memory_order_acquire = _CUDA_VSTD::memory_order_acquire;
inline constexpr memory_order memory_order_release = _CUDA_VSTD::memory_order_release;
inline constexpr memory_order memory_order_acq_rel = _CUDA_VSTD::memory_order_acq_rel;
inline constexpr memory_order memory_order_seq_cst = _CUDA_VSTD::memory_order_seq_cst;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __LIBCUDACXX___ATOMIC_ORDER_H
