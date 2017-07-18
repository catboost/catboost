// -*- C++ -*-
//===--------------------- support/win32/sync_win32.h ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_SYNC_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_SYNC_WIN32_H

#if !defined(_LIBCPP_MSVC)
#error "This header complements Microsoft's Visual C++, and should not be included otherwise."
#else

#include <intrin.h>

template <bool, class _Tp> struct __sync_win32_enable_if {};
template <class _Tp> struct __sync_win32_enable_if<true, _Tp> {typedef _Tp type;};

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_add_and_fetch(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(long), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchangeAdd((long*)(__ptr), (long)(__val))) + __val;
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_add_and_fetch(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(__int64), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchangeAdd64((__int64*)(__ptr), (__int64)(__val))) + __val;
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_fetch_and_add(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(long), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchangeAdd((long*)(__ptr), (long)(__val)));
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_fetch_and_add(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(__int64), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchangeAdd64((__int64*)(__ptr), (__int64)(__val)));
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_lock_test_and_set(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(long), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchange((long*)(__ptr), (long)(__val)));
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE _Tp __sync_lock_test_and_set(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(__int64), _Tp>::type __val)
{
    return (_Tp)(_InterlockedExchange64((__int64*)(__ptr), (__int64)(__val)));
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE bool __sync_bool_compare_and_swap(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(long), _Tp>::type __old_val,
    typename __sync_win32_enable_if<true, _Tp>::type __new_val)
{
    return _InterlockedCompareExchange((long*)(__ptr), (long)(__new_val),
                                       (long)(__old_val)) == (long)(__old_val);
}

template <class _Tp>
_LIBCPP_ALWAYS_INLINE bool __sync_bool_compare_and_swap(
    _Tp *__ptr,
    typename __sync_win32_enable_if<sizeof(_Tp) == sizeof(__int64), _Tp>::type __old_val,
    typename __sync_win32_enable_if<true, _Tp>::type __new_val)
{
    return _InterlockedCompareExchange64((__int64*)(__ptr), (__int64)(__new_val),
                                         (__int64)(__old_val)) == (__int64)(__old_val);
}

#endif // _LIBCPP_MSVCRT

#endif // _LIBCPP_SUPPORT_WIN32_SYNC_WIN32_H
