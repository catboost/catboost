//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCUDACXX___ATOMIC_PLATFORM_MSVC_H
#define __LIBCUDACXX___ATOMIC_PLATFORM_MSVC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_COMPILER(MSVC)

#  include <cuda/std/__atomic/order.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/cassert>

#  include <intrin.h>

// MSVC exposed __iso_volatile intrinsics beginning on 1924 for x86
#  if _CCCL_COMPILER(MSVC, <, 19, 24)
#    define _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
#  endif // _CCCL_COMPILER(MSVC, <, 19, 24)

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#  define _LIBCUDACXX_COMPILER_BARRIER() _ReadWriteBarrier()

#  if _CCCL_ARCH(ARM64)
#    define _LIBCUDACXX_MEMORY_BARRIER()             __dmb(0xB) // inner shared data memory barrier
#    define _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER() _LIBCUDACXX_MEMORY_BARRIER()
#  elif _CCCL_ARCH(X86_64)
#    define _LIBCUDACXX_MEMORY_BARRIER()             __faststorefence()
// x86/x64 hardware only emits memory barriers inside _Interlocked intrinsics
#    define _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER() _LIBCUDACXX_COMPILER_BARRIER()
#  else // ^^^ x86/x64 / unsupported hardware vvv
#    error Unsupported hardware
#  endif // hardware

// MSVC Does not have compiler intrinsics for lock-free checking
inline int __stronger_order_msvc(int __a, int __b)
{
  int const __max = __a > __b ? __a : __b;
  if (__max != __ATOMIC_RELEASE)
  {
    return __max;
  }
  static int const __xform[] = {__ATOMIC_RELEASE, __ATOMIC_ACQ_REL, __ATOMIC_ACQ_REL, __ATOMIC_RELEASE};
  return __xform[__a < __b ? __a : __b];
}

static inline void __atomic_signal_fence(int __memorder)
{
  if (__memorder != __ATOMIC_RELAXED)
  {
    _LIBCUDACXX_COMPILER_BARRIER();
  }
}

static inline void __atomic_thread_fence(int __memorder)
{
  if (__memorder != __ATOMIC_RELAXED)
  {
    _LIBCUDACXX_MEMORY_BARRIER();
  }
}

template <typename _Type, size_t _Size>
using __enable_if_sized_as = enable_if_t<sizeof(_Type) == _Size, int>;

template <class _Type, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_load_relaxed(const volatile _Type* __ptr, _Type* __ret)
{
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  __int8 __tmp = *(const volatile __int8*) __ptr;
#  else
  __int8 __tmp = __iso_volatile_load8((const volatile __int8*) __ptr);
#  endif
  *__ret = reinterpret_cast<_Type&>(__tmp);
}
template <class _Type, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_load_relaxed(const volatile _Type* __ptr, _Type* __ret)
{
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  __int16 __tmp = *(const volatile __int16*) __ptr;
#  else
  __int16 __tmp = __iso_volatile_load16((const volatile __int16*) __ptr);
#  endif
  *__ret = reinterpret_cast<_Type&>(__tmp);
}
template <class _Type, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_load_relaxed(const volatile _Type* __ptr, _Type* __ret)
{
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  __int32 __tmp = *(const volatile __int32*) __ptr;
#  else
  __int32 __tmp = __iso_volatile_load32((const volatile __int32*) __ptr);
#  endif
  *__ret = reinterpret_cast<_Type&>(__tmp);
}
template <class _Type, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_load_relaxed(const volatile _Type* __ptr, _Type* __ret)
{
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  __int64 __tmp = *(const volatile __int64*) __ptr;
#  else
  __int64 __tmp = __iso_volatile_load64((const volatile __int64*) __ptr);
#  endif
  *__ret = reinterpret_cast<_Type&>(__tmp);
}

template <class _Type>
void __atomic_load(const volatile _Type* __ptr, _Type* __ret, int __memorder)
{
  switch (__memorder)
  {
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_load_relaxed(__ptr, __ret);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_load_relaxed(__ptr, __ret);
      break;
    default:
      assert(0);
  }
}

template <class _Type, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_store_relaxed(volatile _Type* __ptr, _Type* __val)
{
  auto __t = reinterpret_cast<__int8*>(__val);
  auto __d = reinterpret_cast<volatile __int8*>(__ptr);
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  (void) _InterlockedExchange8(__d, *__t);
#  else
  __iso_volatile_store8(__d, *__t);
#  endif
}
template <class _Type, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_store_relaxed(volatile _Type* __ptr, _Type* __val)
{
  auto __t = reinterpret_cast<__int16*>(__val);
  auto __d = reinterpret_cast<volatile __int16*>(__ptr);
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  (void) _InterlockedExchange16(__d, *__t);
#  else
  __iso_volatile_store16(__d, *__t);
#  endif
}
template <class _Type, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_store_relaxed(volatile _Type* __ptr, _Type* __val)
{
  auto __t = reinterpret_cast<__int32*>(__val);
  auto __d = reinterpret_cast<volatile __int32*>(__ptr);
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  // int cannot be converted to long?...
  (void) _InterlockedExchange(reinterpret_cast<volatile long*>(__d), *__t);
#  else
  __iso_volatile_store32(__d, *__t);
#  endif
}
template <class _Type, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_store_relaxed(volatile _Type* __ptr, _Type* __val)
{
  auto __t = reinterpret_cast<__int64*>(__val);
  auto __d = reinterpret_cast<volatile __int64*>(__ptr);
#  ifdef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN
  (void) _InterlockedExchange64(__d, *__t);
#  else
  __iso_volatile_store64(__d, *__t);
#  endif
}

template <class _Type>
void __atomic_store(volatile _Type* __ptr, _Type* __val, int __memorder)
{
  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_store_relaxed(__ptr, __val);
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_RELAXED:
      __atomic_store_relaxed(__ptr, __val);
      break;
    default:
      assert(0);
  }
}

template <class _Type, __enable_if_sized_as<_Type, 1> = 0>
bool __atomic_compare_exchange_relaxed(const volatile _Type* __ptr, _Type* __expected, const _Type* __desired)
{
  auto __tmp_desired  = reinterpret_cast<const char&>(*__desired);
  auto __tmp_expected = reinterpret_cast<char&>(*__expected);
  auto const __old    = _InterlockedCompareExchange8((volatile char*) __ptr, __tmp_desired, __tmp_expected);
  if (__old == __tmp_expected)
  {
    return true;
  }
  *__expected = reinterpret_cast<const _Type&>(__old);
  return false;
}
template <class _Type, __enable_if_sized_as<_Type, 2> = 0>
bool __atomic_compare_exchange_relaxed(const volatile _Type* __ptr, _Type* __expected, const _Type* __desired)
{
  auto __tmp_desired  = reinterpret_cast<const short&>(*__desired);
  auto __tmp_expected = reinterpret_cast<short&>(*__expected);
  auto const __old    = _InterlockedCompareExchange16((volatile short*) __ptr, __tmp_desired, __tmp_expected);
  if (__old == __tmp_expected)
  {
    return true;
  }
  *__expected = reinterpret_cast<const _Type&>(__old);
  return false;
}
template <class _Type, __enable_if_sized_as<_Type, 4> = 0>
bool __atomic_compare_exchange_relaxed(const volatile _Type* __ptr, _Type* __expected, const _Type* __desired)
{
  auto __tmp_desired  = reinterpret_cast<const long&>(*__desired);
  auto __tmp_expected = reinterpret_cast<long&>(*__expected);
  auto const __old    = _InterlockedCompareExchange((volatile long*) __ptr, __tmp_desired, __tmp_expected);
  if (__old == __tmp_expected)
  {
    return true;
  }
  *__expected = reinterpret_cast<const _Type&>(__old);
  return false;
}
template <class _Type, __enable_if_sized_as<_Type, 8> = 0>
bool __atomic_compare_exchange_relaxed(const volatile _Type* __ptr, _Type* __expected, const _Type* __desired)
{
  auto __tmp_desired  = reinterpret_cast<const __int64&>(*__desired);
  auto __tmp_expected = reinterpret_cast<__int64&>(*__expected);
  auto const __old    = _InterlockedCompareExchange64((volatile __int64*) __ptr, __tmp_desired, __tmp_expected);
  if (__old == __tmp_expected)
  {
    return true;
  }
  *__expected = reinterpret_cast<const _Type&>(__old);
  return false;
}
template <class _Type>
bool __atomic_compare_exchange(
  _Type volatile* __ptr, _Type* __expected, const _Type* __desired, bool, int __success_memorder, int __failure_memorder)
{
  bool success = false;
  switch (__stronger_order_msvc(__success_memorder, __failure_memorder))
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      success = __atomic_compare_exchange_relaxed(__ptr, __expected, __desired);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      success = __atomic_compare_exchange_relaxed(__ptr, __expected, __desired);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      success = __atomic_compare_exchange_relaxed(__ptr, __expected, __desired);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      success = __atomic_compare_exchange_relaxed(__ptr, __expected, __desired);
      break;
    default:
      assert(0);
  }
  return success;
}

template <class _Type, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_exchange_relaxed(const volatile _Type* __ptr, const _Type* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchange8((volatile char*) __ptr, reinterpret_cast<char const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_exchange_relaxed(const volatile _Type* __ptr, const _Type* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchange16((volatile short*) __ptr, reinterpret_cast<short const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_exchange_relaxed(const volatile _Type* __ptr, const _Type* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchange((volatile long*) __ptr, reinterpret_cast<long const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_exchange_relaxed(const volatile _Type* __ptr, const _Type* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchange64((volatile __int64*) __ptr, reinterpret_cast<__int64 const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type>
void __atomic_exchange(_Type volatile* __ptr, const _Type* __val, _Type* __ret, int __memorder)
{
  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_exchange_relaxed(__ptr, __val, __ret);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_exchange_relaxed(__ptr, __val, __ret);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      __atomic_exchange_relaxed(__ptr, __val, __ret);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_exchange_relaxed(__ptr, __val, __ret);
      break;
    default:
      assert(0);
  }
}

template <class _Type, class _Delta, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_fetch_add_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchangeAdd8((volatile char*) __ptr, reinterpret_cast<char const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_fetch_add_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchangeAdd16((volatile short*) __ptr, reinterpret_cast<short const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_fetch_add_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchangeAdd((volatile long*) __ptr, reinterpret_cast<long const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_fetch_add_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedExchangeAdd64((volatile __int64*) __ptr, reinterpret_cast<__int64 const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta>
_Type __atomic_fetch_add(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_fetch_add_relaxed(__ptr, &__val, __dest);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_fetch_add_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      __atomic_fetch_add_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_fetch_add_relaxed(__ptr, &__val, __dest);
      break;
    default:
      assert(0);
  }
  return *__dest;
}
template <class _Type, class _Delta>
_Type __atomic_fetch_sub(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  return __atomic_fetch_add(__ptr, 0 - __val, __memorder);
}

template <class _Type, class _Delta, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_fetch_and_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedAnd8((volatile char*) __ptr, reinterpret_cast<char const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_fetch_and_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedAnd16((volatile short*) __ptr, reinterpret_cast<short const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_fetch_and_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedAnd((volatile long*) __ptr, reinterpret_cast<long const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_fetch_and_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedAnd64((volatile __int64*) __ptr, reinterpret_cast<__int64 const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta>
_Type __atomic_fetch_and(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_fetch_and_relaxed(__ptr, &__val, __dest);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_fetch_and_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      __atomic_fetch_and_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_fetch_and_relaxed(__ptr, &__val, __dest);
      break;
    default:
      assert(0);
  }
  return *__dest;
}

template <class _Type, class _Delta, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_fetch_xor_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedXor8((volatile char*) __ptr, reinterpret_cast<char const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_fetch_xor_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedXor16((volatile short*) __ptr, reinterpret_cast<short const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_fetch_xor_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedXor((volatile long*) __ptr, reinterpret_cast<long const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_fetch_xor_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedXor64((volatile __int64*) __ptr, reinterpret_cast<__int64 const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta>
_Type __atomic_fetch_xor(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_fetch_xor_relaxed(__ptr, &__val, __dest);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_fetch_xor_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      __atomic_fetch_xor_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_fetch_xor_relaxed(__ptr, &__val, __dest);
      break;
    default:
      assert(0);
  }
  return *__dest;
}

template <class _Type, class _Delta, __enable_if_sized_as<_Type, 1> = 0>
void __atomic_fetch_or_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedOr8((volatile char*) __ptr, reinterpret_cast<char const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 2> = 0>
void __atomic_fetch_or_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedOr16((volatile short*) __ptr, reinterpret_cast<short const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 4> = 0>
void __atomic_fetch_or_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedOr((volatile long*) __ptr, reinterpret_cast<long const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta, __enable_if_sized_as<_Type, 8> = 0>
void __atomic_fetch_or_relaxed(const volatile _Type* __ptr, const _Delta* __val, _Type* __ret)
{
  auto const __old = _InterlockedOr64((volatile __int64*) __ptr, reinterpret_cast<__int64 const&>(*__val));
  *__ret           = reinterpret_cast<_Type const&>(__old);
}
template <class _Type, class _Delta>
_Type __atomic_fetch_or(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  switch (__memorder)
  {
    case __ATOMIC_RELEASE:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      __atomic_fetch_or_relaxed(__ptr, &__val, __dest);
      break;
    case __ATOMIC_ACQ_REL:
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      [[fallthrough]];
    case __ATOMIC_CONSUME:
    case __ATOMIC_ACQUIRE:
      __atomic_fetch_or_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_SEQ_CST:
      _LIBCUDACXX_MEMORY_BARRIER();
      __atomic_fetch_or_relaxed(__ptr, &__val, __dest);
      _LIBCUDACXX_COMPILER_OR_MEMORY_BARRIER();
      break;
    case __ATOMIC_RELAXED:
      __atomic_fetch_or_relaxed(__ptr, &__val, __dest);
      break;
    default:
      assert(0);
  }
  return *__dest;
}

template <class _Type>
_Type __atomic_load_n(const _Type volatile* __ptr, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  __atomic_load(__ptr, __dest, __memorder);
  return *__dest;
}

template <class _Type>
void __atomic_store_n(_Type volatile* __ptr, _Type __val, int __memorder)
{
  __atomic_store(__ptr, &__val, __memorder);
}

template <class _Type>
bool __atomic_compare_exchange_n(
  _Type volatile* __ptr, _Type* __expected, _Type __desired, bool __weak, int __success_memorder, int __failure_memorder)
{
  return __atomic_compare_exchange(__ptr, __expected, &__desired, __weak, __success_memorder, __failure_memorder);
}

template <class _Type>
_Type __atomic_exchange_n(_Type volatile* __ptr, _Type __val, int __memorder)
{
  alignas(_Type) unsigned char __buf[sizeof(_Type)] = {};
  auto* __dest                                      = reinterpret_cast<_Type*>(__buf);

  __atomic_exchange(__ptr, &__val, __dest, __memorder);
  return *__dest;
}

template <class _Type, class _Delta>
_Type __atomic_fetch_max(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  _Type __expected = __atomic_load_n(__ptr, __ATOMIC_RELAXED);
  _Type __desired  = __expected < __val ? __expected : __val;
  while (__desired == __val && !__atomic_compare_exchange_n(__ptr, &__expected, __desired, __memorder, __memorder))
  {
    __desired = __expected > __val ? __expected : __val;
  }
  return __expected;
}

template <class _Type, class _Delta>
_Type __atomic_fetch_min(_Type volatile* __ptr, _Delta __val, int __memorder)
{
  _Type __expected = __atomic_load_n(__ptr, __ATOMIC_RELAXED);
  _Type __desired  = __expected < __val ? __expected : __val;
  while (__desired != __val && !__atomic_compare_exchange_n(__ptr, &__expected, __desired, __memorder, __memorder))
  {
    __desired = __expected < __val ? __expected : __val;
  }
  return __expected;
}

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#  undef _LIBCUDACXX_MSVC_HAS_NO_ISO_INTRIN

#endif // _CCCL_COMPILER(MSVC)

#endif // __LIBCUDACXX___ATOMIC_PLATFORM_MSVC_H
