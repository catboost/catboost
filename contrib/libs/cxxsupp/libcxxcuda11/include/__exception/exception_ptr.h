//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___EXCEPTION_EXCEPTION_PTR_H
#define _LIBCPP___EXCEPTION_EXCEPTION_PTR_H

#include <__config>
#include <__exception/operations.h>
#include <__memory/addressof.h>
#include <cstddef>
#include <cstdlib>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

namespace std { // purposefully not using versioning namespace


class _LIBCPP_EXPORTED_FROM_ABI exception_ptr {
  void* __ptr_;

public:
  _LIBCPP_HIDE_FROM_ABI exception_ptr() _NOEXCEPT : __ptr_() {}
  _LIBCPP_HIDE_FROM_ABI exception_ptr(nullptr_t) _NOEXCEPT : __ptr_() {}

  exception_ptr(const exception_ptr&) _NOEXCEPT;
  exception_ptr& operator=(const exception_ptr&) _NOEXCEPT;
  ~exception_ptr() _NOEXCEPT;

  _LIBCPP_HIDE_FROM_ABI explicit operator bool() const _NOEXCEPT { return __ptr_ != nullptr; }

  friend _LIBCPP_HIDE_FROM_ABI bool operator==(const exception_ptr& __x, const exception_ptr& __y) _NOEXCEPT {
    return __x.__ptr_ == __y.__ptr_;
  }

  friend _LIBCPP_HIDE_FROM_ABI bool operator!=(const exception_ptr& __x, const exception_ptr& __y) _NOEXCEPT {
    return !(__x == __y);
  }

  friend _LIBCPP_HIDE_FROM_ABI void swap(exception_ptr& __x, exception_ptr& __y) _NOEXCEPT
  {
      void* __tmp = __x.__ptr_;
      __x.__ptr_ = __y.__ptr_;
      __y.__ptr_ = __tmp;
  }

  friend _LIBCPP_EXPORTED_FROM_ABI exception_ptr current_exception() _NOEXCEPT;
  friend _LIBCPP_EXPORTED_FROM_ABI void rethrow_exception(exception_ptr);
};

#ifndef _LIBCPP_ABI_MICROSOFT

template <class _Ep>
_LIBCPP_HIDE_FROM_ABI exception_ptr make_exception_ptr(_Ep __e) _NOEXCEPT {
#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
  try {
    throw __e;
  } catch (...) {
    return current_exception();
  }
#  else
  ((void)__e);
  std::abort();
#  endif
}

#else // _LIBCPP_ABI_MICROSOFT

_LIBCPP_HIDE_FROM_ABI exception_ptr __copy_exception_ptr(void *__exception, const void* __ptr);

// This is a built-in template function which automagically extracts the required
// information.
template <class _E>
void* __GetExceptionInfo(_E);

template <class _Ep>
exception_ptr make_exception_ptr(_Ep __e) _NOEXCEPT {
  return __copy_exception_ptr(std::addressof(__e), __GetExceptionInfo(__e));
}

#endif // _LIBCPP_ABI_MICROSOFT
} // namespace std

#endif // _LIBCPP___EXCEPTION_EXCEPTION_PTR_H
