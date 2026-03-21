//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_TYPES_BASE_H
#define _LIBCUDACXX___ATOMIC_TYPES_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __atomic_storage
{
  using __underlying_t                = _Tp;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_base_tag;

#if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 5)
  static_assert(_CCCL_TRAIT(is_trivially_copyable, _Tp),
                "std::atomic<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _CCCL_ALIGNAS(sizeof(_Tp)) _Tp __a_value;

  _CCCL_HIDE_FROM_ABI explicit constexpr __atomic_storage() noexcept = default;

  _CCCL_HOST_DEVICE constexpr explicit inline __atomic_storage(_Tp value) noexcept
      : __a_value(value)
  {}

  _CCCL_HOST_DEVICE inline auto get() noexcept -> __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() const noexcept -> const __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() volatile noexcept -> volatile __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() const volatile noexcept -> const volatile __underlying_t*
  {
    return &__a_value;
  }
};

_CCCL_HOST_DEVICE inline void __atomic_thread_fence_dispatch(memory_order __order)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (__atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(__order), __thread_scope_system_tag());),
    NV_IS_HOST,
    (__atomic_thread_fence_host(__order);))
}

_CCCL_HOST_DEVICE inline void __atomic_signal_fence_dispatch(memory_order __order)
{
  NV_DISPATCH_TARGET(NV_IS_DEVICE,
                     (__atomic_signal_fence_cuda(static_cast<__memory_order_underlying_t>(__order));),
                     NV_IS_HOST,
                     (__atomic_signal_fence_host(__order);))
}

template <typename _Sto, typename _Up, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_init_dispatch(_Sto* __a, _Up __val)
{
  __atomic_assign_volatile(__a->get(), __val);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_store_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (__atomic_store_n_cuda(__a->get(), __val, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (__atomic_store_host(__a->get(), __val, __order);))
}

template <typename _Sto, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_load_dispatch(const _Sto* __a, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_load_n_cuda(__a->get(), static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_load_host(__a->get(), __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_exchange_dispatch(_Sto* __a, _Up __value, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_exchange_n_cuda(__a->get(), __value, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_exchange_host(__a->get(), __value, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_strong_dispatch(
  _Sto* __a, _Up* __expected, _Up __val, memory_order __success, memory_order __failure, _Sco = {})
{
  bool __result = false;
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (__result = __atomic_compare_exchange_cuda(
       __a->get(),
       __expected,
       __val,
       false,
       static_cast<__memory_order_underlying_t>(__success),
       static_cast<__memory_order_underlying_t>(__failure),
       _Sco{});),
    NV_IS_HOST,
    (__result = __atomic_compare_exchange_strong_host(__a->get(), __expected, __val, __success, __failure);))
  return __result;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_weak_dispatch(
  _Sto* __a, _Up* __expected, _Up __val, memory_order __success, memory_order __failure, _Sco = {})
{
  bool __result = false;
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (__result = __atomic_compare_exchange_cuda(
       __a->get(),
       __expected,
       __val,
       true,
       static_cast<__memory_order_underlying_t>(__success),
       static_cast<__memory_order_underlying_t>(__failure),
       _Sco{});),
    NV_IS_HOST,
    (__result = __atomic_compare_exchange_weak_host(__a->get(), __expected, __val, __success, __failure);))
  return __result;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_add_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_add_cuda(__a->get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_fetch_add_host(__a->get(), __delta, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_sub_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_sub_cuda(__a->get(), __delta, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_fetch_sub_host(__a->get(), __delta, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_and_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_and_cuda(__a->get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_fetch_and_host(__a->get(), __pattern, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_or_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_or_cuda(__a->get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_fetch_or_host(__a->get(), __pattern, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_xor_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_xor_cuda(__a->get(), __pattern, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    NV_IS_HOST,
    (return __atomic_fetch_xor_host(__a->get(), __pattern, __order);))
  _CCCL_UNREACHABLE();
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_max_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_max_cuda(__a->get(), __val, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    (return __atomic_fetch_max_host(__a->get(), __val, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_min_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_fetch_min_cuda(__a->get(), __val, static_cast<__memory_order_underlying_t>(__order), _Sco{});),
    (return __atomic_fetch_min_host(__a->get(), __val, __order);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_TYPES_BASE_H
