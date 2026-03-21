// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_THREAD_API_CUDA)

#  include <cuda/std/chrono>
#  include <cuda/std/climits>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_API inline void __cccl_thread_yield() {}

_CCCL_API inline void __cccl_thread_sleep_for(_CUDA_VSTD::chrono::nanoseconds __ns)
{
  NV_IF_TARGET(NV_IS_DEVICE,
               (auto const __step = __ns.count(); assert(__step < numeric_limits<unsigned>::max());
                asm volatile("nanosleep.u32 %0;" ::"r"((unsigned) __step) :);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_HAS_THREAD_API_CUDA

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H
