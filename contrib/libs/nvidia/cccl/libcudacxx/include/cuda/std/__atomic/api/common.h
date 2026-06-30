//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_API_COMMON_H
#define __LIBCUDACXX___ATOMIC_API_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/types/base.h>

// API definitions for the base atomic implementation
#define _LIBCUDACXX_ATOMIC_COMMON_IMPL(_CONST, _VOLATILE)                                                           \
  _CCCL_HOST_DEVICE inline bool is_lock_free() const _VOLATILE noexcept                                             \
  {                                                                                                                 \
    return _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(sizeof(_Tp));                                                            \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline void store(_Tp __d, memory_order __m = memory_order_seq_cst)                             \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_STORE_MEMORY_ORDER(__m)                                             \
  {                                                                                                                 \
    __atomic_store_dispatch(&__a, __d, __m, _Sco{});                                                                \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline _Tp load(memory_order __m = memory_order_seq_cst)                                        \
    const _VOLATILE noexcept _LIBCUDACXX_CHECK_LOAD_MEMORY_ORDER(__m)                                               \
  {                                                                                                                 \
    return __atomic_load_dispatch(&__a, __m, _Sco{});                                                               \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline operator _Tp() const _VOLATILE noexcept                                                  \
  {                                                                                                                 \
    return load();                                                                                                  \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline _Tp exchange(_Tp __d, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept \
  {                                                                                                                 \
    return __atomic_exchange_dispatch(&__a, __d, __m, _Sco{});                                                      \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline bool compare_exchange_weak(_Tp& __e, _Tp __d, memory_order __s, memory_order __f)        \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)                                     \
  {                                                                                                                 \
    return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __s, __f, _Sco{});                              \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline bool compare_exchange_strong(_Tp& __e, _Tp __d, memory_order __s, memory_order __f)      \
    _CONST _VOLATILE noexcept _LIBCUDACXX_CHECK_EXCHANGE_MEMORY_ORDER(__s, __f)                                     \
  {                                                                                                                 \
    return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __s, __f, _Sco{});                            \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline bool compare_exchange_weak(_Tp& __e, _Tp __d, memory_order __m = memory_order_seq_cst)   \
    _CONST _VOLATILE noexcept                                                                                       \
  {                                                                                                                 \
    if (memory_order_acq_rel == __m)                                                                                \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, memory_order_acquire, _Sco{});           \
    else if (memory_order_release == __m)                                                                           \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, memory_order_relaxed, _Sco{});           \
    else                                                                                                            \
      return __atomic_compare_exchange_weak_dispatch(&__a, &__e, __d, __m, __m, _Sco{});                            \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline bool compare_exchange_strong(_Tp& __e, _Tp __d, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                                       \
  {                                                                                                                 \
    if (memory_order_acq_rel == __m)                                                                                \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, memory_order_acquire, _Sco{});         \
    else if (memory_order_release == __m)                                                                           \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, memory_order_relaxed, _Sco{});         \
    else                                                                                                            \
      return __atomic_compare_exchange_strong_dispatch(&__a, &__e, __d, __m, __m, _Sco{});                          \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline void wait(_Tp __v, memory_order __m = memory_order_seq_cst) const _VOLATILE noexcept     \
  {                                                                                                                 \
    __atomic_wait(&__a, __v, __m, _Sco{});                                                                          \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline void notify_one() _CONST _VOLATILE noexcept                                              \
  {                                                                                                                 \
    __atomic_notify_one(&__a, _Sco{});                                                                              \
  }                                                                                                                 \
  _CCCL_HOST_DEVICE inline void notify_all() _CONST _VOLATILE noexcept                                              \
  {                                                                                                                 \
    __atomic_notify_all(&__a, _Sco{});                                                                              \
  }

// API definitions for arithmetic atomics
#define _LIBCUDACXX_ATOMIC_ARITHMETIC_IMPL(_CONST, _VOLATILE)                                                         \
  _CCCL_HOST_DEVICE inline _Tp fetch_add(_Tp __op, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept \
  {                                                                                                                   \
    return __atomic_fetch_add_dispatch(&__a, __op, __m, _Sco{});                                                      \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp fetch_sub(_Tp __op, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept \
  {                                                                                                                   \
    return __atomic_fetch_sub_dispatch(&__a, __op, __m, _Sco{});                                                      \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator++(int) _CONST _VOLATILE noexcept                                              \
  {                                                                                                                   \
    return fetch_add(_Tp(1));                                                                                         \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator--(int) _CONST _VOLATILE noexcept                                              \
  {                                                                                                                   \
    return fetch_sub(_Tp(1));                                                                                         \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator++() _CONST _VOLATILE noexcept                                                 \
  {                                                                                                                   \
    return fetch_add(_Tp(1)) + _Tp(1);                                                                                \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator--() _CONST _VOLATILE noexcept                                                 \
  {                                                                                                                   \
    return fetch_sub(_Tp(1)) - _Tp(1);                                                                                \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator+=(_Tp __op) _CONST _VOLATILE noexcept                                         \
  {                                                                                                                   \
    return fetch_add(__op) + __op;                                                                                    \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator-=(_Tp __op) _CONST _VOLATILE noexcept                                         \
  {                                                                                                                   \
    return fetch_sub(__op) - __op;                                                                                    \
  }

// API definitions for bitwise atomics
#define _LIBCUDACXX_ATOMIC_BITWISE_IMPL(_CONST, _VOLATILE)                                                            \
  _CCCL_HOST_DEVICE inline _Tp fetch_and(_Tp __op, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept \
  {                                                                                                                   \
    return __atomic_fetch_and_dispatch(&__a, __op, __m, _Sco{});                                                      \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp fetch_or(_Tp __op, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept  \
  {                                                                                                                   \
    return __atomic_fetch_or_dispatch(&__a, __op, __m, _Sco{});                                                       \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp fetch_xor(_Tp __op, memory_order __m = memory_order_seq_cst) _CONST _VOLATILE noexcept \
  {                                                                                                                   \
    return __atomic_fetch_xor_dispatch(&__a, __op, __m, _Sco{});                                                      \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator&=(_Tp __op) _CONST _VOLATILE noexcept                                         \
  {                                                                                                                   \
    return fetch_and(__op) & __op;                                                                                    \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator|=(_Tp __op) _CONST _VOLATILE noexcept                                         \
  {                                                                                                                   \
    return fetch_or(__op) | __op;                                                                                     \
  }                                                                                                                   \
  _CCCL_HOST_DEVICE inline _Tp operator^=(_Tp __op) _CONST _VOLATILE noexcept                                         \
  {                                                                                                                   \
    return fetch_xor(__op) ^ __op;                                                                                    \
  }

// API definitions for atomics with pointers
#define _LIBCUDACXX_ATOMIC_POINTER_IMPL(_CONST, _VOLATILE)                                        \
  _CCCL_HOST_DEVICE inline _Tp fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                     \
  {                                                                                               \
    return __atomic_fetch_add_dispatch(&__a, __op, __m, __thread_scope_system_tag{});             \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) \
    _CONST _VOLATILE noexcept                                                                     \
  {                                                                                               \
    return __atomic_fetch_sub_dispatch(&__a, __op, __m, __thread_scope_system_tag{});             \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator++(int) _CONST _VOLATILE noexcept                          \
  {                                                                                               \
    return fetch_add(1);                                                                          \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator--(int) _CONST _VOLATILE noexcept                          \
  {                                                                                               \
    return fetch_sub(1);                                                                          \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator++() _CONST _VOLATILE noexcept                             \
  {                                                                                               \
    return fetch_add(1) + 1;                                                                      \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator--() _CONST _VOLATILE noexcept                             \
  {                                                                                               \
    return fetch_sub(1) - 1;                                                                      \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator+=(ptrdiff_t __op) _CONST _VOLATILE noexcept               \
  {                                                                                               \
    return fetch_add(__op) + __op;                                                                \
  }                                                                                               \
  _CCCL_HOST_DEVICE inline _Tp operator-=(ptrdiff_t __op) _CONST _VOLATILE noexcept               \
  {                                                                                               \
    return fetch_sub(__op) - __op;                                                                \
  }

#endif // __LIBCUDACXX___ATOMIC_API_COMMON_H
