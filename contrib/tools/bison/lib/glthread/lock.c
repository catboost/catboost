/* Locking in multithreaded situations.
   Copyright (C) 2005-2020 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2005.
   Based on GCC's gthr-posix.h, gthr-posix95.h.  */

#include <config.h>

#include "glthread/lock.h"

/* ========================================================================= */

#if USE_ISOC_THREADS || USE_ISOC_AND_POSIX_THREADS

/* -------------------------- gl_lock_t datatype -------------------------- */

int
glthread_lock_init (gl_lock_t *lock)
{
  if (mtx_init (&lock->mutex, mtx_plain) != thrd_success)
    return ENOMEM;
  lock->init_needed = 0;
  return 0;
}

int
glthread_lock_lock (gl_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_lock (&lock->mutex) != thrd_success)
    return EAGAIN;
  return 0;
}

int
glthread_lock_unlock (gl_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_unlock (&lock->mutex) != thrd_success)
    return EINVAL;
  return 0;
}

int
glthread_lock_destroy (gl_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  mtx_destroy (&lock->mutex);
  return 0;
}

/* ------------------------- gl_rwlock_t datatype ------------------------- */

int
glthread_rwlock_init (gl_rwlock_t *lock)
{
  if (mtx_init (&lock->lock, mtx_plain) != thrd_success
      || cnd_init (&lock->waiting_readers) != thrd_success
      || cnd_init (&lock->waiting_writers) != thrd_success)
    return ENOMEM;
  lock->waiting_writers_count = 0;
  lock->runcount = 0;
  lock->init_needed = 0;
  return 0;
}

int
glthread_rwlock_rdlock (gl_rwlock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_lock (&lock->lock) != thrd_success)
    return EAGAIN;
  /* Test whether only readers are currently running, and whether the runcount
     field will not overflow, and whether no writer is waiting.  The latter
     condition is because POSIX recommends that "write locks shall take
     precedence over read locks", to avoid "writer starvation".  */
  while (!(lock->runcount + 1 > 0 && lock->waiting_writers_count == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_readers.  */
      if (cnd_wait (&lock->waiting_readers, &lock->lock) != thrd_success)
        {
          mtx_unlock (&lock->lock);
          return EINVAL;
        }
    }
  lock->runcount++;
  if (mtx_unlock (&lock->lock) != thrd_success)
    return EINVAL;
  return 0;
}

int
glthread_rwlock_wrlock (gl_rwlock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_lock (&lock->lock) != thrd_success)
    return EAGAIN;
  /* Test whether no readers or writers are currently running.  */
  while (!(lock->runcount == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_writers.  */
      lock->waiting_writers_count++;
      if (cnd_wait (&lock->waiting_writers, &lock->lock) != thrd_success)
        {
          lock->waiting_writers_count--;
          mtx_unlock (&lock->lock);
          return EINVAL;
        }
      lock->waiting_writers_count--;
    }
  lock->runcount--; /* runcount becomes -1 */
  if (mtx_unlock (&lock->lock) != thrd_success)
    return EINVAL;
  return 0;
}

int
glthread_rwlock_unlock (gl_rwlock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_lock (&lock->lock) != thrd_success)
    return EAGAIN;
  if (lock->runcount < 0)
    {
      /* Drop a writer lock.  */
      if (!(lock->runcount == -1))
        {
          mtx_unlock (&lock->lock);
          return EINVAL;
        }
      lock->runcount = 0;
    }
  else
    {
      /* Drop a reader lock.  */
      if (!(lock->runcount > 0))
        {
          mtx_unlock (&lock->lock);
          return EINVAL;
        }
      lock->runcount--;
    }
  if (lock->runcount == 0)
    {
      /* POSIX recommends that "write locks shall take precedence over read
         locks", to avoid "writer starvation".  */
      if (lock->waiting_writers_count > 0)
        {
          /* Wake up one of the waiting writers.  */
          if (cnd_signal (&lock->waiting_writers) != thrd_success)
            {
              mtx_unlock (&lock->lock);
              return EINVAL;
            }
        }
      else
        {
          /* Wake up all waiting readers.  */
          if (cnd_broadcast (&lock->waiting_readers) != thrd_success)
            {
              mtx_unlock (&lock->lock);
              return EINVAL;
            }
        }
    }
  if (mtx_unlock (&lock->lock) != thrd_success)
    return EINVAL;
  return 0;
}

int
glthread_rwlock_destroy (gl_rwlock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  mtx_destroy (&lock->lock);
  cnd_destroy (&lock->waiting_readers);
  cnd_destroy (&lock->waiting_writers);
  return 0;
}

/* --------------------- gl_recursive_lock_t datatype --------------------- */

int
glthread_recursive_lock_init (gl_recursive_lock_t *lock)
{
  if (mtx_init (&lock->mutex, mtx_plain | mtx_recursive) != thrd_success)
    return ENOMEM;
  lock->init_needed = 0;
  return 0;
}

int
glthread_recursive_lock_lock (gl_recursive_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_lock (&lock->mutex) != thrd_success)
    return EAGAIN;
  return 0;
}

int
glthread_recursive_lock_unlock (gl_recursive_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  if (mtx_unlock (&lock->mutex) != thrd_success)
    return EINVAL;
  return 0;
}

int
glthread_recursive_lock_destroy (gl_recursive_lock_t *lock)
{
  if (lock->init_needed)
    call_once (&lock->init_once, lock->init_func);
  mtx_destroy (&lock->mutex);
  return 0;
}

/* -------------------------- gl_once_t datatype -------------------------- */

#endif

/* ========================================================================= */

#if USE_POSIX_THREADS

/* -------------------------- gl_lock_t datatype -------------------------- */

/* ------------------------- gl_rwlock_t datatype ------------------------- */

# if HAVE_PTHREAD_RWLOCK && (HAVE_PTHREAD_RWLOCK_RDLOCK_PREFER_WRITER || (defined PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP && (__GNU_LIBRARY__ > 1)))

#  if defined PTHREAD_RWLOCK_INITIALIZER || defined PTHREAD_RWLOCK_INITIALIZER_NP

#   if !HAVE_PTHREAD_RWLOCK_RDLOCK_PREFER_WRITER
     /* glibc with bug https://sourceware.org/bugzilla/show_bug.cgi?id=13701 */

int
glthread_rwlock_init_for_glibc (pthread_rwlock_t *lock)
{
  pthread_rwlockattr_t attributes;
  int err;

  err = pthread_rwlockattr_init (&attributes);
  if (err != 0)
    return err;
  /* Note: PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP is the only value that
     causes the writer to be preferred. PTHREAD_RWLOCK_PREFER_WRITER_NP does not
     do this; see
     http://man7.org/linux/man-pages/man3/pthread_rwlockattr_setkind_np.3.html */
  err = pthread_rwlockattr_setkind_np (&attributes,
                                       PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
  if (err == 0)
    err = pthread_rwlock_init(lock, &attributes);
  /* pthread_rwlockattr_destroy always returns 0.  It cannot influence the
     return value.  */
  pthread_rwlockattr_destroy (&attributes);
  return err;
}

#   endif
#  else

int
glthread_rwlock_init_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_rwlock_init (&lock->rwlock, NULL);
  if (err != 0)
    return err;
  lock->initialized = 1;
  return 0;
}

int
glthread_rwlock_rdlock_multithreaded (gl_rwlock_t *lock)
{
  if (!lock->initialized)
    {
      int err;

      err = pthread_mutex_lock (&lock->guard);
      if (err != 0)
        return err;
      if (!lock->initialized)
        {
          err = glthread_rwlock_init_multithreaded (lock);
          if (err != 0)
            {
              pthread_mutex_unlock (&lock->guard);
              return err;
            }
        }
      err = pthread_mutex_unlock (&lock->guard);
      if (err != 0)
        return err;
    }
  return pthread_rwlock_rdlock (&lock->rwlock);
}

int
glthread_rwlock_wrlock_multithreaded (gl_rwlock_t *lock)
{
  if (!lock->initialized)
    {
      int err;

      err = pthread_mutex_lock (&lock->guard);
      if (err != 0)
        return err;
      if (!lock->initialized)
        {
          err = glthread_rwlock_init_multithreaded (lock);
          if (err != 0)
            {
              pthread_mutex_unlock (&lock->guard);
              return err;
            }
        }
      err = pthread_mutex_unlock (&lock->guard);
      if (err != 0)
        return err;
    }
  return pthread_rwlock_wrlock (&lock->rwlock);
}

int
glthread_rwlock_unlock_multithreaded (gl_rwlock_t *lock)
{
  if (!lock->initialized)
    return EINVAL;
  return pthread_rwlock_unlock (&lock->rwlock);
}

int
glthread_rwlock_destroy_multithreaded (gl_rwlock_t *lock)
{
  int err;

  if (!lock->initialized)
    return EINVAL;
  err = pthread_rwlock_destroy (&lock->rwlock);
  if (err != 0)
    return err;
  lock->initialized = 0;
  return 0;
}

#  endif

# else

int
glthread_rwlock_init_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_mutex_init (&lock->lock, NULL);
  if (err != 0)
    return err;
  err = pthread_cond_init (&lock->waiting_readers, NULL);
  if (err != 0)
    return err;
  err = pthread_cond_init (&lock->waiting_writers, NULL);
  if (err != 0)
    return err;
  lock->waiting_writers_count = 0;
  lock->runcount = 0;
  return 0;
}

int
glthread_rwlock_rdlock_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_mutex_lock (&lock->lock);
  if (err != 0)
    return err;
  /* Test whether only readers are currently running, and whether the runcount
     field will not overflow, and whether no writer is waiting.  The latter
     condition is because POSIX recommends that "write locks shall take
     precedence over read locks", to avoid "writer starvation".  */
  while (!(lock->runcount + 1 > 0 && lock->waiting_writers_count == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_readers.  */
      err = pthread_cond_wait (&lock->waiting_readers, &lock->lock);
      if (err != 0)
        {
          pthread_mutex_unlock (&lock->lock);
          return err;
        }
    }
  lock->runcount++;
  return pthread_mutex_unlock (&lock->lock);
}

int
glthread_rwlock_wrlock_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_mutex_lock (&lock->lock);
  if (err != 0)
    return err;
  /* Test whether no readers or writers are currently running.  */
  while (!(lock->runcount == 0))
    {
      /* This thread has to wait for a while.  Enqueue it among the
         waiting_writers.  */
      lock->waiting_writers_count++;
      err = pthread_cond_wait (&lock->waiting_writers, &lock->lock);
      if (err != 0)
        {
          lock->waiting_writers_count--;
          pthread_mutex_unlock (&lock->lock);
          return err;
        }
      lock->waiting_writers_count--;
    }
  lock->runcount--; /* runcount becomes -1 */
  return pthread_mutex_unlock (&lock->lock);
}

int
glthread_rwlock_unlock_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_mutex_lock (&lock->lock);
  if (err != 0)
    return err;
  if (lock->runcount < 0)
    {
      /* Drop a writer lock.  */
      if (!(lock->runcount == -1))
        {
          pthread_mutex_unlock (&lock->lock);
          return EINVAL;
        }
      lock->runcount = 0;
    }
  else
    {
      /* Drop a reader lock.  */
      if (!(lock->runcount > 0))
        {
          pthread_mutex_unlock (&lock->lock);
          return EINVAL;
        }
      lock->runcount--;
    }
  if (lock->runcount == 0)
    {
      /* POSIX recommends that "write locks shall take precedence over read
         locks", to avoid "writer starvation".  */
      if (lock->waiting_writers_count > 0)
        {
          /* Wake up one of the waiting writers.  */
          err = pthread_cond_signal (&lock->waiting_writers);
          if (err != 0)
            {
              pthread_mutex_unlock (&lock->lock);
              return err;
            }
        }
      else
        {
          /* Wake up all waiting readers.  */
          err = pthread_cond_broadcast (&lock->waiting_readers);
          if (err != 0)
            {
              pthread_mutex_unlock (&lock->lock);
              return err;
            }
        }
    }
  return pthread_mutex_unlock (&lock->lock);
}

int
glthread_rwlock_destroy_multithreaded (gl_rwlock_t *lock)
{
  int err;

  err = pthread_mutex_destroy (&lock->lock);
  if (err != 0)
    return err;
  err = pthread_cond_destroy (&lock->waiting_readers);
  if (err != 0)
    return err;
  err = pthread_cond_destroy (&lock->waiting_writers);
  if (err != 0)
    return err;
  return 0;
}

# endif

/* --------------------- gl_recursive_lock_t datatype --------------------- */

# if HAVE_PTHREAD_MUTEX_RECURSIVE

#  if defined PTHREAD_RECURSIVE_MUTEX_INITIALIZER || defined PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP

int
glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock)
{
  pthread_mutexattr_t attributes;
  int err;

  err = pthread_mutexattr_init (&attributes);
  if (err != 0)
    return err;
  err = pthread_mutexattr_settype (&attributes, PTHREAD_MUTEX_RECURSIVE);
  if (err != 0)
    {
      pthread_mutexattr_destroy (&attributes);
      return err;
    }
  err = pthread_mutex_init (lock, &attributes);
  if (err != 0)
    {
      pthread_mutexattr_destroy (&attributes);
      return err;
    }
  err = pthread_mutexattr_destroy (&attributes);
  if (err != 0)
    return err;
  return 0;
}

#  else

int
glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock)
{
  pthread_mutexattr_t attributes;
  int err;

  err = pthread_mutexattr_init (&attributes);
  if (err != 0)
    return err;
  err = pthread_mutexattr_settype (&attributes, PTHREAD_MUTEX_RECURSIVE);
  if (err != 0)
    {
      pthread_mutexattr_destroy (&attributes);
      return err;
    }
  err = pthread_mutex_init (&lock->recmutex, &attributes);
  if (err != 0)
    {
      pthread_mutexattr_destroy (&attributes);
      return err;
    }
  err = pthread_mutexattr_destroy (&attributes);
  if (err != 0)
    return err;
  lock->initialized = 1;
  return 0;
}

int
glthread_recursive_lock_lock_multithreaded (gl_recursive_lock_t *lock)
{
  if (!lock->initialized)
    {
      int err;

      err = pthread_mutex_lock (&lock->guard);
      if (err != 0)
        return err;
      if (!lock->initialized)
        {
          err = glthread_recursive_lock_init_multithreaded (lock);
          if (err != 0)
            {
              pthread_mutex_unlock (&lock->guard);
              return err;
            }
        }
      err = pthread_mutex_unlock (&lock->guard);
      if (err != 0)
        return err;
    }
  return pthread_mutex_lock (&lock->recmutex);
}

int
glthread_recursive_lock_unlock_multithreaded (gl_recursive_lock_t *lock)
{
  if (!lock->initialized)
    return EINVAL;
  return pthread_mutex_unlock (&lock->recmutex);
}

int
glthread_recursive_lock_destroy_multithreaded (gl_recursive_lock_t *lock)
{
  int err;

  if (!lock->initialized)
    return EINVAL;
  err = pthread_mutex_destroy (&lock->recmutex);
  if (err != 0)
    return err;
  lock->initialized = 0;
  return 0;
}

#  endif

# else

int
glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock)
{
  int err;

  err = pthread_mutex_init (&lock->mutex, NULL);
  if (err != 0)
    return err;
  lock->owner = (pthread_t) 0;
  lock->depth = 0;
  return 0;
}

int
glthread_recursive_lock_lock_multithreaded (gl_recursive_lock_t *lock)
{
  pthread_t self = pthread_self ();
  if (lock->owner != self)
    {
      int err;

      err = pthread_mutex_lock (&lock->mutex);
      if (err != 0)
        return err;
      lock->owner = self;
    }
  if (++(lock->depth) == 0) /* wraparound? */
    {
      lock->depth--;
      return EAGAIN;
    }
  return 0;
}

int
glthread_recursive_lock_unlock_multithreaded (gl_recursive_lock_t *lock)
{
  if (lock->owner != pthread_self ())
    return EPERM;
  if (lock->depth == 0)
    return EINVAL;
  if (--(lock->depth) == 0)
    {
      lock->owner = (pthread_t) 0;
      return pthread_mutex_unlock (&lock->mutex);
    }
  else
    return 0;
}

int
glthread_recursive_lock_destroy_multithreaded (gl_recursive_lock_t *lock)
{
  if (lock->owner != (pthread_t) 0)
    return EBUSY;
  return pthread_mutex_destroy (&lock->mutex);
}

# endif

/* -------------------------- gl_once_t datatype -------------------------- */

static const pthread_once_t fresh_once = PTHREAD_ONCE_INIT;

int
glthread_once_singlethreaded (pthread_once_t *once_control)
{
  /* We don't know whether pthread_once_t is an integer type, a floating-point
     type, a pointer type, or a structure type.  */
  char *firstbyte = (char *)once_control;
  if (*firstbyte == *(const char *)&fresh_once)
    {
      /* First time use of once_control.  Invert the first byte.  */
      *firstbyte = ~ *(const char *)&fresh_once;
      return 1;
    }
  else
    return 0;
}

# if !(PTHREAD_IN_USE_DETECTION_HARD || USE_POSIX_THREADS_WEAK)

int
glthread_once_multithreaded (pthread_once_t *once_control,
                             void (*init_function) (void))
{
  int err = pthread_once (once_control, init_function);
  if (err == ENOSYS)
    {
      /* This happens on FreeBSD 11: The pthread_once function in libc returns
         ENOSYS.  */
      if (glthread_once_singlethreaded (once_control))
        init_function ();
      return 0;
    }
  return err;
}

# endif

#endif

/* ========================================================================= */

#if USE_WINDOWS_THREADS

#endif

/* ========================================================================= */
