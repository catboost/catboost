//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
#define __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx_generated.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_CUDA_COMPILATION()

template <class _Operand>
using __cuda_atomic_enable_non_native_arithmetic =
  enable_if_t<_Operand::__size <= 16 || _Operand::__op == __atomic_cuda_operand::_f, bool>;

template <class _Operand>
using __cuda_atomic_enable_non_native_bitwise = enable_if_t<_Operand::__size <= 16, bool>;

template <class _Operand>
using __cuda_atomic_enable_native_bitwise = enable_if_t<_Operand::__size >= 32, bool>;

template <class _Operand>
using __cuda_atomic_enable_non_native_ld_st = enable_if_t<_Operand::__size <= 8, bool>;

template <class _Operand>
using __cuda_atomic_enable_native_ld_st = enable_if_t<_Operand::__size >= 16, bool>;

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_ld_st<_Operand> = 0>
static inline _CCCL_DEVICE void
__cuda_atomic_load(const _Type* __ptr, _Type& __dst, _Order, _Operand, _Sco, __atomic_cuda_mmio_disable)
{
  constexpr uint64_t __alignmask = (sizeof(uint16_t) - 1);
  uint16_t* __aligned            = (uint16_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint16_t((intptr_t) __ptr & __alignmask) * 8;

  uint16_t __value = 0;
  __cuda_atomic_load(__aligned, __value, _Order{}, __atomic_cuda_operand_b16{}, _Sco{}, __atomic_cuda_mmio_disable{});

  __dst = static_cast<_Type>(__value >> __offset);
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE bool
__cuda_atomic_compare_exchange(_Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, _Order, _Operand, _Sco)
{
  constexpr uint64_t __alignmask = (sizeof(uint32_t) - 1);
  constexpr uint32_t __sizemask  = (1 << (sizeof(_Type) * 8)) - 1;
  uint32_t* __aligned            = (uint32_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint32_t((intptr_t) __ptr & __alignmask) * 8;
  const uint32_t __valueMask     = __sizemask << __offset;
  const uint32_t __windowMask    = ~__valueMask;
  const uint32_t __cmpOffset     = __cmp << __offset;
  const uint32_t __opOffset      = __op << __offset;

  // Algorithm for 8b CAS with 32b intrinsics
  // __old = __window[0:32] where [__cmp] resides within some offset.
  uint32_t __old;
  // Start by loading __old with the current value, this optimizes for early return when __cmp is wrong
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_relaxed{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});),
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_volatile{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});))
  // Reemit CAS instructions until we succeed or the old value is a mismatch
  while (__cmpOffset == (__old & __valueMask))
  {
    // Combine the desired value and most recently fetched expected masked portion of the window
    const uint32_t __attempt = (__old & __windowMask) | __opOffset;

    if (__cuda_atomic_compare_exchange(
          __aligned, __old, __old, __attempt, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}))
    {
      // CAS was successful
      return true;
    }
  }
  __dst = static_cast<_Type>(__old >> __offset);
  return false;
}

// Optimized fetch_update CAS loop with op determined after first load reducing waste.
template <class _Type,
          class _Fn,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
_CCCL_DEVICE _Type __cuda_atomic_fetch_update(_Type* __ptr, const _Fn& __op, _Order, _Operand, _Sco)
{
  constexpr uint64_t __alignmask = (sizeof(uint32_t) - 1);
  constexpr uint32_t __sizemask  = (1 << (sizeof(_Type) * 8)) - 1;
  uint32_t* __aligned            = (uint32_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint8_t((intptr_t) __ptr & __alignmask) * 8;
  const uint32_t __valueMask     = __sizemask << __offset;
  const uint32_t __windowMask    = ~__valueMask;

  // 8/16b fetch update is similar to CAS implementation, but compresses the logic for recalculating the operand
  // __old = __window[0:32] where [__cmp] resides within some offset.
  uint32_t __old;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_relaxed{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});),
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_volatile{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});))

  // Reemit CAS instructions until we succeed
  while (1)
  {
    // Calculate new desired value from last fetched __old
    // Use of the value mask is required due to the possibility of overflow when ops are widened. Possible compiler bug?
    const uint32_t __attempt =
      ((static_cast<uint32_t>(__op(static_cast<_Type>(__old >> __offset))) << __offset) & __valueMask)
      | (__old & __windowMask);

    if (__cuda_atomic_compare_exchange(
          __aligned, __old, __old, __attempt, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}))
    {
      // CAS was successful
      return static_cast<_Type>(__old >> __offset);
    }
  }
}

// Optimized fetch_update CAS loop with op determined after first load reducing waste.
template <class _Type,
          class _Fn,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_native_bitwise<_Operand> = 0>
_CCCL_DEVICE _Type __cuda_atomic_fetch_update(_Type* __ptr, const _Fn& __op, _Order, _Operand, _Sco)
{
  _Type __expected = 0;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (__cuda_atomic_load(
       __ptr, __expected, __atomic_cuda_relaxed{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});),
    (__cuda_atomic_load(
       __ptr, __expected, __atomic_cuda_volatile{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});))

  _Type __desired = __op(__expected);
  while (!__cuda_atomic_compare_exchange(__ptr, __expected, __expected, __desired, _Order{}, _Operand{}, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_ld_st<_Operand> = 0>
static inline _CCCL_DEVICE void
__cuda_atomic_store(_Type* __ptr, _Type __val, _Order, _Operand, _Sco, __atomic_cuda_mmio_disable)
{
  // Store requires cas on 8/16b types
  __cuda_atomic_fetch_update(
    __ptr,
    [__val](_Type __old) {
      return __val;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_arithmetic<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_add(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old + __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_and(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old & __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_xor(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old ^ __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_or(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old | __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_arithmetic<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_min(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __op < __old ? __op : __old;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}
template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_arithmetic<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_max(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __old < __op ? __op : __old;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE void __cuda_atomic_exchange(_Type* __ptr, _Type& __dst, _Type __op, _Order, _Operand, _Sco)
{
  __dst = __cuda_atomic_fetch_update(
    __ptr,
    [__op](_Type __old) {
      return __op;
    },
    _Order{},
    __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, _Operand::__size>{},
    _Sco{});
}

template <typename _Tp, typename _Fn, typename _Sco>
_CCCL_DEVICE _Tp __atomic_fetch_update_cuda(_Tp* __ptr, const _Fn& __op, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __op(__expected);
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}
template <typename _Tp, typename _Fn, typename _Sco>
_CCCL_DEVICE _Tp __atomic_fetch_update_cuda(_Tp volatile* __ptr, const _Fn& __op, int __memorder, _Sco)
{
  _Tp __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, _Sco{});
  _Tp __desired  = __op(__expected);
  while (!__atomic_compare_exchange_cuda(__ptr, &__expected, __desired, true, __memorder, __memorder, _Sco{}))
  {
    __desired = __op(__expected);
  }
  return __expected;
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_load_n_cuda(const _Tp* __ptr, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_load_cuda(__ptr, __ret, __memorder, _Sco{});
  return __ret;
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_load_n_cuda(const _Tp volatile* __ptr, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_load_cuda(__ptr, __ret, __memorder, _Sco{});
  return __ret;
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE void __atomic_store_n_cuda(_Tp* __ptr, _Tp __val, int __memorder, _Sco)
{
  __atomic_store_cuda(__ptr, __val, __memorder, _Sco{});
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE void __atomic_store_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  __atomic_store_cuda(__ptr, __val, __memorder, _Sco{});
}

template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_exchange_n_cuda(_Tp* __ptr, _Tp __val, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, _Sco{});
  return __ret;
}
template <typename _Tp, typename _Sco>
_CCCL_DEVICE _Tp __atomic_exchange_n_cuda(_Tp volatile* __ptr, _Tp __val, int __memorder, _Sco)
{
  _Tp __ret;
  __atomic_exchange_cuda(__ptr, __ret, __val, __memorder, _Sco{});
  return __ret;
}

template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_min_cuda(_Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __val < __old ? __val : __old;
    },
    __memorder,
    _Sco{});
}
template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_min_cuda(volatile _Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __val < __old ? __val : __old;
    },
    __memorder,
    _Sco{});
}

template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_max_cuda(_Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old < __val ? __val : __old;
    },
    __memorder,
    _Sco{});
}
template <typename _Tp, typename _Up, typename _Sco, __atomic_enable_if_not_native_minmax<_Tp> = 0>
_CCCL_DEVICE _Tp __atomic_fetch_max_cuda(volatile _Tp* __ptr, _Up __val, int __memorder, _Sco)
{
  return __atomic_fetch_update_cuda(
    __ptr,
    [__val](_Tp __old) {
      return __old < __val ? __val : __old;
    },
    __memorder,
    _Sco{});
}

_CCCL_DEVICE static inline void __atomic_signal_fence_cuda(int)
{
  asm volatile("" ::: "memory");
}

#endif // _CCCL_CUDA_COMPILATION()

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __LIBCUDACXX___ATOMIC_FUNCTIONS_DERIVED_H
