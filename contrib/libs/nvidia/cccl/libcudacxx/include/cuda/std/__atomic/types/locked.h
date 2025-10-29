//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_TYPES_LOCKED_H
#define _LIBCUDACXX___ATOMIC_TYPES_LOCKED_H

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
#include <cuda/std/__atomic/types/base.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Locked atomics must override the dispatch to be able to implement RMW primitives around the embedded lock.
template <typename _Tp>
struct __atomic_locked_storage
{
  using __underlying_t                = _Tp;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_locked_tag;

  _Tp __a_value;
  mutable __atomic_storage<_LIBCUDACXX_ATOMIC_FLAG_TYPE> __a_lock;

  _CCCL_HIDE_FROM_ABI explicit constexpr __atomic_locked_storage() noexcept = default;

  _CCCL_HOST_DEVICE constexpr explicit inline __atomic_locked_storage(_Tp value) noexcept
      : __a_value(value)
      , __a_lock{}
  {}

  template <typename _Sco>
  _CCCL_HOST_DEVICE inline void __lock(_Sco) const volatile noexcept
  {
    while (1 == __atomic_exchange_dispatch(&__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(true), memory_order_acquire, _Sco{}))
      /*spin*/;
  }
  template <typename _Sco>
  _CCCL_HOST_DEVICE inline void __lock(_Sco) const noexcept
  {
    while (1 == __atomic_exchange_dispatch(&__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(true), memory_order_acquire, _Sco{}))
      /*spin*/;
  }
  template <typename _Sco>
  _CCCL_HOST_DEVICE inline void __unlock(_Sco) const volatile noexcept
  {
    __atomic_store_dispatch(&__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(false), memory_order_release, _Sco{});
  }
  template <typename _Sco>
  _CCCL_HOST_DEVICE inline void __unlock(_Sco) const noexcept
  {
    __atomic_store_dispatch(&__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(false), memory_order_release, _Sco{});
  }
};

template <typename _Sto, typename _Up, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_init_dispatch(_Sto* __a, _Up __val)
{
  __atomic_assign_volatile(&__a->__a_value, __val);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_store_dispatch(_Sto* __a, _Up __val, memory_order, _Sco = {})
{
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__a->__a_value, __val);
  __a->__unlock(_Sco{});
}

template <typename _Sto, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_load_dispatch(const _Sto* __a, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_exchange_dispatch(_Sto* __a, _Up __value, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, __value);
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_strong_dispatch(
  _Sto* __a, _Up* __expected, _Up __value, memory_order, memory_order, _Sco = {})
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __temp;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__temp, __a->__a_value);
  bool __ret = __temp == *__expected;
  if (__ret)
  {
    __atomic_assign_volatile(&__a->__a_value, __value);
  }
  else
  {
    __atomic_assign_volatile(__expected, __a->__a_value);
  }
  __a->__unlock(_Sco{});
  return __ret;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool
__atomic_compare_exchange_weak_dispatch(_Sto* __a, _Up* __expected, _Up __value, memory_order, memory_order, _Sco = {})
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __temp;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__temp, __a->__a_value);
  bool __ret = __temp == *__expected;
  if (__ret)
  {
    __atomic_assign_volatile(&__a->__a_value, __value);
  }
  else
  {
    __atomic_assign_volatile(__expected, __a->__a_value);
  }
  __a->__unlock(_Sco{});
  return __ret;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_add_dispatch(_Sto* __a, _Up __delta, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, _Tp(__old + __delta));
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_sub_dispatch(_Sto* __a, _Up __delta, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, _Tp(__old - __delta));
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_and_dispatch(_Sto* __a, _Up __pattern, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, _Tp(__old & __pattern));
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_or_dispatch(_Sto* __a, _Up __pattern, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, _Tp(__old | __pattern));
  __a->__unlock(_Sco{});
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_locked<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_xor_dispatch(_Sto* __a, _Up __pattern, memory_order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  _Tp __old;
  __a->__lock(_Sco{});
  __atomic_assign_volatile(&__old, __a->__a_value);
  __atomic_assign_volatile(&__a->__a_value, _Tp(__old ^ __pattern));
  __a->__unlock(_Sco{});
  return __old;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_TYPES_LOCKED_H
