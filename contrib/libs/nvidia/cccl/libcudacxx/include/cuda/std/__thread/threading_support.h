// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/chrono>

#if defined(_LIBCUDACXX_HAS_THREAD_API_EXTERNAL)
#  include <cuda/std/__thread/threading_support_external.h>
#endif // _LIBCUDACXX_HAS_THREAD_API_EXTERNAL

#if defined(_LIBCUDACXX_HAS_THREAD_API_CUDA)
#  include <cuda/std/__thread/threading_support_cuda.h>
#elif defined(_LIBCUDACXX_HAS_THREAD_API_PTHREAD)
#  include <cuda/std/__thread/threading_support_pthread.h>
#elif defined(_LIBCUDACXX_HAS_THREAD_API_WIN32)
#  include <cuda/std/__thread/threading_support_win32.h>
#else // ^^^ _LIBCUDACXX_HAS_THREAD_API_WIN32 ^^^ / vvv Unknown Thread API vvv
#  error "Unknown Thread API"
#endif // Unknown Thread API

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#define _LIBCUDACXX_POLLING_COUNT 16

#if _CCCL_ARCH(ARM64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("yield" :: :);)
#elif _CCCL_ARCH(X86_64) && _CCCL_OS(LINUX)
#  define __LIBCUDACXX_ASM_THREAD_YIELD (asm volatile("pause" :: :);)
#else // ^^^  _CCCL_ARCH(X86_64) ^^^ / vvv ! _CCCL_ARCH(X86_64) vvv
#  define __LIBCUDACXX_ASM_THREAD_YIELD (;)
#endif // ! _CCCL_ARCH(X86_64)

_CCCL_API inline void __cccl_thread_yield_processor()
{
  NV_IF_TARGET(NV_IS_HOST, __LIBCUDACXX_ASM_THREAD_YIELD)
}

template <class _Fn>
_CCCL_API inline bool __cccl_thread_poll_with_backoff(
  _Fn&& __f, _CUDA_VSTD::chrono::nanoseconds __max = _CUDA_VSTD::chrono::nanoseconds::zero())
{
  _CUDA_VSTD::chrono::high_resolution_clock::time_point const __start =
    _CUDA_VSTD::chrono::high_resolution_clock::now();
  for (int __count = 0;;)
  {
    if (__f())
    {
      return true;
    }
    if (__count < _LIBCUDACXX_POLLING_COUNT)
    {
      if (__count > (_LIBCUDACXX_POLLING_COUNT >> 1))
      {
        _CUDA_VSTD::__cccl_thread_yield_processor();
      }
      __count += 1;
      continue;
    }
    _CUDA_VSTD::chrono::high_resolution_clock::duration const __elapsed =
      _CUDA_VSTD::chrono::high_resolution_clock::now() - __start;
    if (__max != _CUDA_VSTD::chrono::nanoseconds::zero() && __max < __elapsed)
    {
      return false;
    }
    _CUDA_VSTD::chrono::nanoseconds const __step = __elapsed / 4;
    if (__step >= _CUDA_VSTD::chrono::milliseconds(1))
    {
      _CUDA_VSTD::__cccl_thread_sleep_for(_CUDA_VSTD::chrono::milliseconds(1));
    }
    else if (__step >= _CUDA_VSTD::chrono::microseconds(10))
    {
      _CUDA_VSTD::__cccl_thread_sleep_for(__step);
    }
    else
    {
      _CUDA_VSTD::__cccl_thread_yield();
    }
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_H
