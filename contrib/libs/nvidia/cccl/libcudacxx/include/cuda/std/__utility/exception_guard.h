//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_EXCEPTION_GUARD_H
#define _LIBCUDACXX___UTILITY_EXCEPTION_GUARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __exception_guard is a helper class for writing code with the strong exception guarantee.
//
// When writing code that can throw an exception, one can store rollback instructions in an
// exception guard so that if an exception is thrown at any point during the lifetime of the
// exception guard, it will be rolled back automatically. When the exception guard is done, one
// must mark it as being complete so it isn't rolled back when the exception guard is destroyed.
//
// Exception guards are not default constructible, they can't be copied or assigned to, but
// they can be moved around for convenience.
//
// __exception_guard is a no-op in -fno-exceptions mode to produce better code-gen. This means
// that we don't provide the strong exception guarantees. However, Clang doesn't generate cleanup
// code with exceptions disabled, so even if we wanted to provide the strong exception guarantees
// we couldn't. This is also only relevant for constructs with a stack of
// -fexceptions > -fno-exceptions > -fexceptions code, since the exception can't be caught where
// exceptions are disabled. While -fexceptions > -fno-exceptions is quite common
// (e.g. libc++.dylib > -fno-exceptions), having another layer with exceptions enabled seems a lot
// less common, especially one that tries to catch an exception through -fno-exceptions code.
//
// __exception_guard can help greatly simplify code that would normally be cluttered by
// `#if !_CCCL_HAS_EXCEPTIONS()`. For example:
//
//    template <class Iterator, class Size, class OutputIterator>
//    Iterator uninitialized_copy_n(Iterator iter, Size n, OutputIterator out) {
//        typedef typename iterator_traits<Iterator>::value_type value_type;
//        __exception_guard guard([start=out, &out] {
//            _CUDA_VSTD::destroy(start, out);
//        });
//
//        for (; n > 0; ++iter, ++out, --n) {
//            ::new ((void*)_CUDA_VSTD::addressof(*out)) value_type(*iter);
//        }
//        guard.__complete();
//        return out;
//    }
//

template <class _Rollback>
struct __exception_guard_exceptions
{
  __exception_guard_exceptions() = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 explicit __exception_guard_exceptions(_Rollback __rollback)
      : __rollback_(_CUDA_VSTD::move(__rollback))
      , __completed_(false)
  {}

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __exception_guard_exceptions(__exception_guard_exceptions&& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_move_constructible, _Rollback))
      : __rollback_(_CUDA_VSTD::move(__other.__rollback_))
      , __completed_(__other.__completed_)
  {
    __other.__completed_ = true;
  }

  __exception_guard_exceptions(__exception_guard_exceptions const&)            = delete;
  __exception_guard_exceptions& operator=(__exception_guard_exceptions const&) = delete;
  __exception_guard_exceptions& operator=(__exception_guard_exceptions&&)      = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __complete() noexcept
  {
    __completed_ = true;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__exception_guard_exceptions()
  {
    if (!__completed_)
    {
      __rollback_();
    }
  }

private:
  _Rollback __rollback_;
  bool __completed_;
};

_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(__exception_guard_exceptions);

template <class _Rollback>
struct __exception_guard_noexceptions
{
  __exception_guard_noexceptions() = delete;
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 _CCCL_NODEBUG_ALIAS explicit __exception_guard_noexceptions(_Rollback) {}

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 _CCCL_NODEBUG_ALIAS __exception_guard_noexceptions(
    __exception_guard_noexceptions&& __other) noexcept(_CCCL_TRAIT(is_nothrow_move_constructible, _Rollback))
      : __completed_(__other.__completed_)
  {
    __other.__completed_ = true;
  }

  __exception_guard_noexceptions(__exception_guard_noexceptions const&)            = delete;
  __exception_guard_noexceptions& operator=(__exception_guard_noexceptions const&) = delete;
  __exception_guard_noexceptions& operator=(__exception_guard_noexceptions&&)      = delete;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 _CCCL_NODEBUG_ALIAS void __complete() noexcept
  {
    __completed_ = true;
  }

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 _CCCL_NODEBUG_ALIAS ~__exception_guard_noexceptions()
  {
    _CCCL_ASSERT(__completed_, "__exception_guard not completed with exceptions disabled");
  }

private:
  bool __completed_ = false;
};

_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(__exception_guard_noexceptions);

#if !_CCCL_HAS_EXCEPTIONS()
template <class _Rollback>
using __exception_guard = __exception_guard_noexceptions<_Rollback>;
#else // ^^^ !_CCCL_HAS_EXCEPTIONS() ^^^ / vvv _CCCL_HAS_EXCEPTIONS() vvv
template <class _Rollback>
using __exception_guard = __exception_guard_exceptions<_Rollback>;
#endif // _CCCL_HAS_EXCEPTIONS()

template <class _Rollback>
_CCCL_API constexpr __exception_guard<_Rollback> __make_exception_guard(_Rollback __rollback)
{
  return __exception_guard<_Rollback>(_CUDA_VSTD::move(__rollback));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_EXCEPTION_GUARD_H
