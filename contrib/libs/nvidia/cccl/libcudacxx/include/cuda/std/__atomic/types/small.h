//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_TYPES_SMALL_H
#define _LIBCUDACXX___ATOMIC_TYPES_SMALL_H

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
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// manipulated by PTX without any performance overhead
template <typename _Tp>
using __atomic_small_proxy_t = _If<_CCCL_TRAIT(is_signed, _Tp), int32_t, uint32_t>;

// Arithmetic conversions to/from proxy types
template <class _Tp, enable_if_t<_CCCL_TRAIT(is_arithmetic, _Tp), int> = 0>
_CCCL_HOST_DEVICE constexpr __atomic_small_proxy_t<_Tp> __atomic_small_to_32(_Tp __val)
{
  return static_cast<__atomic_small_proxy_t<_Tp>>(__val);
}

template <class _Tp, enable_if_t<_CCCL_TRAIT(is_arithmetic, _Tp), int> = 0>
_CCCL_HOST_DEVICE constexpr _Tp __atomic_small_from_32(__atomic_small_proxy_t<_Tp> __val)
{
  return static_cast<_Tp>(__val);
}

// Non-arithmetic conversion to/from proxy types
template <class _Tp, enable_if_t<!_CCCL_TRAIT(is_arithmetic, _Tp), int> = 0>
_CCCL_HOST_DEVICE inline __atomic_small_proxy_t<_Tp> __atomic_small_to_32(_Tp __val)
{
  __atomic_small_proxy_t<_Tp> __temp{};
  _CUDA_VSTD::memcpy(&__temp, &__val, sizeof(_Tp));
  return __temp;
}

template <class _Tp, enable_if_t<!_CCCL_TRAIT(is_arithmetic, _Tp), int> = 0>
_CCCL_HOST_DEVICE inline _Tp __atomic_small_from_32(__atomic_small_proxy_t<_Tp> __val)
{
  _Tp __temp{};
  _CUDA_VSTD::memcpy(&__temp, &__val, sizeof(_Tp));
  return __temp;
}

template <typename _Tp>
struct __atomic_small_storage
{
  using __underlying_t                = _Tp;
  using __proxy_t                     = __atomic_small_proxy_t<_Tp>;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_small_tag;

  _CCCL_HOST_DEVICE constexpr explicit __atomic_small_storage() noexcept
      : __a_value{__proxy_t{}}
  {}

  _CCCL_HOST_DEVICE constexpr explicit __atomic_small_storage(_Tp __value) noexcept
      : __a_value{__atomic_small_to_32(__value)}
  {}

  __atomic_storage<__proxy_t> __a_value;
};

template <typename _Sto, typename _Up, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_init_dispatch(_Sto* __a, _Up __val)
{
  __atomic_init_dispatch(&__a->__a_value, __atomic_small_to_32(__val));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_store_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
{
  __atomic_store_dispatch(&__a->__a_value, __atomic_small_to_32(__val), __order, _Sco{});
}

template <typename _Sto, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_load_dispatch(const _Sto* __a, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(__atomic_load_dispatch(&__a->__a_value, __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_exchange_dispatch(_Sto* __a, _Up __value, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_exchange_dispatch(&__a->__a_value, __atomic_small_to_32(__value), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_weak_dispatch(
  _Sto* __a, _Up* __expected, _Up __value, memory_order __success, memory_order __failure, _Sco = {})
{
  using _Tp            = __atomic_underlying_t<_Sto>;
  auto __temp_expected = __atomic_small_to_32(*__expected);
  auto const __ret     = __atomic_compare_exchange_weak_dispatch(
    &__a->__a_value, &__temp_expected, __atomic_small_to_32(__value), __success, __failure, _Sco{});
  auto const __actual   = __atomic_small_from_32<_Tp>(__temp_expected);
  constexpr auto __mask = static_cast<decltype(__temp_expected)>((1u << (8 * sizeof(_Tp))) - 1);
  if (!__ret)
  {
    if (0 == __atomic_memcmp(&__actual, __expected, sizeof(_Tp)))
    {
      __atomic_fetch_and_dispatch(&__a->__a_value, __mask, memory_order_relaxed, _Sco{});
    }
    else
    {
      *__expected = __actual;
    }
  }
  return __ret;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_strong_dispatch(
  _Sto* __a, _Up* __expected, _Up __value, memory_order __success, memory_order __failure, _Sco = {})
{
  using _Tp        = __atomic_underlying_t<_Sto>;
  auto const __old = *__expected;
  while (1)
  {
    if (__atomic_compare_exchange_weak_dispatch(__a, __expected, __value, __success, __failure, _Sco{}))
    {
      return true;
    }
    if (0 != __atomic_memcmp(&__old, __expected, sizeof(_Tp)))
    {
      return false;
    }
  }
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_add_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_add_dispatch(&__a->__a_value, __atomic_small_to_32(__delta), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_sub_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_sub_dispatch(&__a->__a_value, __atomic_small_to_32(__delta), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_and_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_and_dispatch(&__a->__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_or_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_or_dispatch(&__a->__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_xor_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_xor_dispatch(&__a->__a_value, __atomic_small_to_32(__pattern), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_max_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_max_dispatch(&__a->__a_value, __atomic_small_to_32(__val), __order, _Sco{}));
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_min_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  return __atomic_small_from_32<_Tp>(
    __atomic_fetch_min_dispatch(&__a->__a_value, __atomic_small_to_32(__val), __order, _Sco{}));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_TYPES_SMALL_H
