// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_THREAD_API_PTHREAD)

#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/chrono>
#  include <cuda/std/climits>
#  include <cuda/std/ctime>

#  include <errno.h>
#  include <pthread.h>
#  include <sched.h>
#  include <semaphore.h>
#  if defined(__linux__)
#    include <unistd.h>

#    include <linux/futex.h>
#    include <sys/syscall.h>
#  endif // __linux__

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Mutex
using __cccl_mutex_t = pthread_mutex_t;
#  define _LIBCUDACXX_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

using __cccl_recursive_mutex_t = pthread_mutex_t;

// Condition Variable
using __cccl_condvar_t = pthread_cond_t;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER PTHREAD_COND_INITIALIZER

// Semaphore
using __cccl_semaphore_t = sem_t;
#  define _LIBCUDACXX_SEMAPHORE_MAX SEM_VALUE_MAX

// Execute once
using __cccl_exec_once_flag = pthread_once_t;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER PTHREAD_ONCE_INIT

// Thread id
using __cccl_thread_id = pthread_t;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

using __cccl_thread_t = pthread_t;

// Thread Local Storage
using __cccl_tls_key = pthread_key_t;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC

[[nodiscard]] _CCCL_API constexpr timespec __cccl_to_timespec(const _CUDA_VSTD::chrono::nanoseconds& __ns)
{
  constexpr auto __ts_sec_max = numeric_limits<time_t>::max();

  timespec __ts{};
  const auto __s = _CUDA_VSTD::chrono::duration_cast<chrono::seconds>(__ns);

  if (_CUDA_VSTD::cmp_less(__s.count(), __ts_sec_max))
  {
    __ts.tv_sec  = static_cast<time_t>(__s.count());
    __ts.tv_nsec = static_cast<decltype(__ts.tv_nsec)>((__ns - __s).count());
  }
  else
  {
    __ts.tv_sec  = __ts_sec_max;
    __ts.tv_nsec = 999'999'999;
  }
  return __ts;
}

// Semaphore

_CCCL_API inline bool __cccl_semaphore_init(__cccl_semaphore_t* __sem, int __init)
{
  return sem_init(__sem, 0, __init) == 0;
}

_CCCL_API inline bool __cccl_semaphore_destroy(__cccl_semaphore_t* __sem)
{
  return sem_destroy(__sem) == 0;
}

_CCCL_API inline bool __cccl_semaphore_post(__cccl_semaphore_t* __sem)
{
  return sem_post(__sem) == 0;
}

_CCCL_API inline bool __cccl_semaphore_wait(__cccl_semaphore_t* __sem)
{
  return sem_wait(__sem) == 0;
}

_CCCL_API inline bool __cccl_semaphore_wait_timed(__cccl_semaphore_t* __sem, _CUDA_VSTD::chrono::nanoseconds const& __ns)
{
  const auto __ts = __cccl_to_timespec(__ns);
  return sem_timedwait(__sem, &__ts) == 0;
}

_CCCL_API inline void __cccl_thread_yield()
{
  sched_yield();
}

_CCCL_API inline void __cccl_thread_sleep_for(_CUDA_VSTD::chrono::nanoseconds __ns)
{
  auto __ts = __cccl_to_timespec(__ns);
  while (nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
    ;
}

_LIBCUDACXX_END_NAMESPACE_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_LIBCUDACXX_HAS_THREAD_API_PTHREAD

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H
