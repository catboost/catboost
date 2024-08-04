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
   Based on GCC's gthr-posix.h, gthr-posix95.h, gthr-win32.h.  */

/* This file contains locking primitives for use with a given thread library.
   It does not contain primitives for creating threads or for other
   synchronization primitives.

   Normal (non-recursive) locks:
     Type:                gl_lock_t
     Declaration:         gl_lock_define(extern, name)
     Initializer:         gl_lock_define_initialized(, name)
     Initialization:      gl_lock_init (name);
     Taking the lock:     gl_lock_lock (name);
     Releasing the lock:  gl_lock_unlock (name);
     De-initialization:   gl_lock_destroy (name);
   Equivalent functions with control of error handling:
     Initialization:      err = glthread_lock_init (&name);
     Taking the lock:     err = glthread_lock_lock (&name);
     Releasing the lock:  err = glthread_lock_unlock (&name);
     De-initialization:   err = glthread_lock_destroy (&name);

   Read-Write (non-recursive) locks:
     Type:                gl_rwlock_t
     Declaration:         gl_rwlock_define(extern, name)
     Initializer:         gl_rwlock_define_initialized(, name)
     Initialization:      gl_rwlock_init (name);
     Taking the lock:     gl_rwlock_rdlock (name);
                          gl_rwlock_wrlock (name);
     Releasing the lock:  gl_rwlock_unlock (name);
     De-initialization:   gl_rwlock_destroy (name);
   Equivalent functions with control of error handling:
     Initialization:      err = glthread_rwlock_init (&name);
     Taking the lock:     err = glthread_rwlock_rdlock (&name);
                          err = glthread_rwlock_wrlock (&name);
     Releasing the lock:  err = glthread_rwlock_unlock (&name);
     De-initialization:   err = glthread_rwlock_destroy (&name);

   Recursive locks:
     Type:                gl_recursive_lock_t
     Declaration:         gl_recursive_lock_define(extern, name)
     Initializer:         gl_recursive_lock_define_initialized(, name)
     Initialization:      gl_recursive_lock_init (name);
     Taking the lock:     gl_recursive_lock_lock (name);
     Releasing the lock:  gl_recursive_lock_unlock (name);
     De-initialization:   gl_recursive_lock_destroy (name);
   Equivalent functions with control of error handling:
     Initialization:      err = glthread_recursive_lock_init (&name);
     Taking the lock:     err = glthread_recursive_lock_lock (&name);
     Releasing the lock:  err = glthread_recursive_lock_unlock (&name);
     De-initialization:   err = glthread_recursive_lock_destroy (&name);

  Once-only execution:
     Type:                gl_once_t
     Initializer:         gl_once_define(extern, name)
     Execution:           gl_once (name, initfunction);
   Equivalent functions with control of error handling:
     Execution:           err = glthread_once (&name, initfunction);
*/


#ifndef _LOCK_H
#define _LOCK_H

#include <errno.h>
#include <stdlib.h>

#if !defined c11_threads_in_use
# if HAVE_THREADS_H && USE_POSIX_THREADS_WEAK
#  include <threads.h>
#  pragma weak thrd_exit
#  define c11_threads_in_use() (thrd_exit != NULL)
# else
#  define c11_threads_in_use() 0
# endif
#endif

/* ========================================================================= */

#if USE_ISOC_THREADS || USE_ISOC_AND_POSIX_THREADS

/* Use the ISO C threads library.  */

# include <threads.h>

# ifdef __cplusplus
extern "C" {
# endif

/* -------------------------- gl_lock_t datatype -------------------------- */

typedef struct
        {
          int volatile init_needed;
          once_flag init_once;
          void (*init_func) (void);
          mtx_t mutex;
        }
        gl_lock_t;
# define gl_lock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_lock_t NAME;
# define gl_lock_define_initialized(STORAGECLASS, NAME) \
    static void _atomic_init_##NAME (void);       \
    STORAGECLASS gl_lock_t NAME =                 \
      { 1, ONCE_FLAG_INIT, _atomic_init_##NAME }; \
    static void _atomic_init_##NAME (void)        \
    {                                             \
      if (glthread_lock_init (&(NAME)))           \
        abort ();                                 \
    }
extern int glthread_lock_init (gl_lock_t *lock);
extern int glthread_lock_lock (gl_lock_t *lock);
extern int glthread_lock_unlock (gl_lock_t *lock);
extern int glthread_lock_destroy (gl_lock_t *lock);

/* ------------------------- gl_rwlock_t datatype ------------------------- */

typedef struct
        {
          int volatile init_needed;
          once_flag init_once;
          void (*init_func) (void);
          mtx_t lock; /* protects the remaining fields */
          cnd_t waiting_readers; /* waiting readers */
          cnd_t waiting_writers; /* waiting writers */
          unsigned int waiting_writers_count; /* number of waiting writers */
          int runcount; /* number of readers running, or -1 when a writer runs */
        }
        gl_rwlock_t;
# define gl_rwlock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_rwlock_t NAME;
# define gl_rwlock_define_initialized(STORAGECLASS, NAME) \
    static void _atomic_init_##NAME (void);       \
    STORAGECLASS gl_rwlock_t NAME =               \
      { 1, ONCE_FLAG_INIT, _atomic_init_##NAME }; \
    static void _atomic_init_##NAME (void)        \
    {                                             \
      if (glthread_rwlock_init (&(NAME)))         \
        abort ();                                 \
    }
extern int glthread_rwlock_init (gl_rwlock_t *lock);
extern int glthread_rwlock_rdlock (gl_rwlock_t *lock);
extern int glthread_rwlock_wrlock (gl_rwlock_t *lock);
extern int glthread_rwlock_unlock (gl_rwlock_t *lock);
extern int glthread_rwlock_destroy (gl_rwlock_t *lock);

/* --------------------- gl_recursive_lock_t datatype --------------------- */

typedef struct
        {
          int volatile init_needed;
          once_flag init_once;
          void (*init_func) (void);
          mtx_t mutex;
        }
        gl_recursive_lock_t;
# define gl_recursive_lock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_recursive_lock_t NAME;
# define gl_recursive_lock_define_initialized(STORAGECLASS, NAME) \
    static void _atomic_init_##NAME (void);       \
    STORAGECLASS gl_recursive_lock_t NAME =       \
      { 1, ONCE_FLAG_INIT, _atomic_init_##NAME }; \
    static void _atomic_init_##NAME (void)        \
    {                                             \
      if (glthread_recursive_lock_init (&(NAME))) \
        abort ();                                 \
    }
extern int glthread_recursive_lock_init (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_lock (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_unlock (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_destroy (gl_recursive_lock_t *lock);

/* -------------------------- gl_once_t datatype -------------------------- */

typedef once_flag gl_once_t;
# define gl_once_define(STORAGECLASS, NAME) \
    STORAGECLASS once_flag NAME = ONCE_FLAG_INIT;
# define glthread_once(ONCE_CONTROL, INITFUNCTION) \
    (call_once (ONCE_CONTROL, INITFUNCTION), 0)

# ifdef __cplusplus
}
# endif

#endif

/* ========================================================================= */

#if USE_POSIX_THREADS

/* Use the POSIX threads library.  */

# include <pthread.h>

# ifdef __cplusplus
extern "C" {
# endif

# if PTHREAD_IN_USE_DETECTION_HARD

/* The pthread_in_use() detection needs to be done at runtime.  */
#  define pthread_in_use() \
     glthread_in_use ()
extern int glthread_in_use (void);

# endif

# if USE_POSIX_THREADS_WEAK

/* Use weak references to the POSIX threads library.  */

/* Weak references avoid dragging in external libraries if the other parts
   of the program don't use them.  Here we use them, because we don't want
   every program that uses libintl to depend on libpthread.  This assumes
   that libpthread would not be loaded after libintl; i.e. if libintl is
   loaded first, by an executable that does not depend on libpthread, and
   then a module is dynamically loaded that depends on libpthread, libintl
   will not be multithread-safe.  */

/* The way to test at runtime whether libpthread is present is to test
   whether a function pointer's value, such as &pthread_mutex_init, is
   non-NULL.  However, some versions of GCC have a bug through which, in
   PIC mode, &foo != NULL always evaluates to true if there is a direct
   call to foo(...) in the same function.  To avoid this, we test the
   address of a function in libpthread that we don't use.  */

#  pragma weak pthread_mutex_init
#  pragma weak pthread_mutex_lock
#  pragma weak pthread_mutex_unlock
#  pragma weak pthread_mutex_destroy
#  pragma weak pthread_rwlock_init
#  pragma weak pthread_rwlock_rdlock
#  pragma weak pthread_rwlock_wrlock
#  pragma weak pthread_rwlock_unlock
#  pragma weak pthread_rwlock_destroy
#  pragma weak pthread_once
#  pragma weak pthread_cond_init
#  pragma weak pthread_cond_wait
#  pragma weak pthread_cond_signal
#  pragma weak pthread_cond_broadcast
#  pragma weak pthread_cond_destroy
#  pragma weak pthread_mutexattr_init
#  pragma weak pthread_mutexattr_settype
#  pragma weak pthread_mutexattr_destroy
#  pragma weak pthread_rwlockattr_init
#  if __GNU_LIBRARY__ > 1
#   pragma weak pthread_rwlockattr_setkind_np
#  endif
#  pragma weak pthread_rwlockattr_destroy
#  ifndef pthread_self
#   pragma weak pthread_self
#  endif

#  if !PTHREAD_IN_USE_DETECTION_HARD
    /* Considering all platforms with USE_POSIX_THREADS_WEAK, only few symbols
       can be used to determine whether libpthread is in use.  These are:
         pthread_mutexattr_gettype
         pthread_rwlockattr_destroy
         pthread_rwlockattr_init
     */
#   pragma weak pthread_mutexattr_gettype
#   define pthread_in_use() \
      (pthread_mutexattr_gettype != NULL || c11_threads_in_use ())
#  endif

# else

#  if !PTHREAD_IN_USE_DETECTION_HARD
#   define pthread_in_use() 1
#  endif

# endif

/* -------------------------- gl_lock_t datatype -------------------------- */

typedef pthread_mutex_t gl_lock_t;
# define gl_lock_define(STORAGECLASS, NAME) \
    STORAGECLASS pthread_mutex_t NAME;
# define gl_lock_define_initialized(STORAGECLASS, NAME) \
    STORAGECLASS pthread_mutex_t NAME = gl_lock_initializer;
# define gl_lock_initializer \
    PTHREAD_MUTEX_INITIALIZER
# define glthread_lock_init(LOCK) \
    (pthread_in_use () ? pthread_mutex_init (LOCK, NULL) : 0)
# define glthread_lock_lock(LOCK) \
    (pthread_in_use () ? pthread_mutex_lock (LOCK) : 0)
# define glthread_lock_unlock(LOCK) \
    (pthread_in_use () ? pthread_mutex_unlock (LOCK) : 0)
# define glthread_lock_destroy(LOCK) \
    (pthread_in_use () ? pthread_mutex_destroy (LOCK) : 0)

/* ------------------------- gl_rwlock_t datatype ------------------------- */

# if HAVE_PTHREAD_RWLOCK && (HAVE_PTHREAD_RWLOCK_RDLOCK_PREFER_WRITER || (defined PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP && (__GNU_LIBRARY__ > 1)))

#  if defined PTHREAD_RWLOCK_INITIALIZER || defined PTHREAD_RWLOCK_INITIALIZER_NP

typedef pthread_rwlock_t gl_rwlock_t;
#   define gl_rwlock_define(STORAGECLASS, NAME) \
      STORAGECLASS pthread_rwlock_t NAME;
#   define gl_rwlock_define_initialized(STORAGECLASS, NAME) \
      STORAGECLASS pthread_rwlock_t NAME = gl_rwlock_initializer;
#   if HAVE_PTHREAD_RWLOCK_RDLOCK_PREFER_WRITER
#    if defined PTHREAD_RWLOCK_INITIALIZER
#     define gl_rwlock_initializer \
        PTHREAD_RWLOCK_INITIALIZER
#    else
#     define gl_rwlock_initializer \
        PTHREAD_RWLOCK_INITIALIZER_NP
#    endif
#    define glthread_rwlock_init(LOCK) \
       (pthread_in_use () ? pthread_rwlock_init (LOCK, NULL) : 0)
#   else /* glibc with bug https://sourceware.org/bugzilla/show_bug.cgi?id=13701 */
#    define gl_rwlock_initializer \
       PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP
#    define glthread_rwlock_init(LOCK) \
       (pthread_in_use () ? glthread_rwlock_init_for_glibc (LOCK) : 0)
extern int glthread_rwlock_init_for_glibc (pthread_rwlock_t *lock);
#   endif
#   define glthread_rwlock_rdlock(LOCK) \
      (pthread_in_use () ? pthread_rwlock_rdlock (LOCK) : 0)
#   define glthread_rwlock_wrlock(LOCK) \
      (pthread_in_use () ? pthread_rwlock_wrlock (LOCK) : 0)
#   define glthread_rwlock_unlock(LOCK) \
      (pthread_in_use () ? pthread_rwlock_unlock (LOCK) : 0)
#   define glthread_rwlock_destroy(LOCK) \
      (pthread_in_use () ? pthread_rwlock_destroy (LOCK) : 0)

#  else

typedef struct
        {
          int initialized;
          pthread_mutex_t guard;   /* protects the initialization */
          pthread_rwlock_t rwlock; /* read-write lock */
        }
        gl_rwlock_t;
#   define gl_rwlock_define(STORAGECLASS, NAME) \
      STORAGECLASS gl_rwlock_t NAME;
#   define gl_rwlock_define_initialized(STORAGECLASS, NAME) \
      STORAGECLASS gl_rwlock_t NAME = gl_rwlock_initializer;
#   define gl_rwlock_initializer \
      { 0, PTHREAD_MUTEX_INITIALIZER }
#   define glthread_rwlock_init(LOCK) \
      (pthread_in_use () ? glthread_rwlock_init_multithreaded (LOCK) : 0)
#   define glthread_rwlock_rdlock(LOCK) \
      (pthread_in_use () ? glthread_rwlock_rdlock_multithreaded (LOCK) : 0)
#   define glthread_rwlock_wrlock(LOCK) \
      (pthread_in_use () ? glthread_rwlock_wrlock_multithreaded (LOCK) : 0)
#   define glthread_rwlock_unlock(LOCK) \
      (pthread_in_use () ? glthread_rwlock_unlock_multithreaded (LOCK) : 0)
#   define glthread_rwlock_destroy(LOCK) \
      (pthread_in_use () ? glthread_rwlock_destroy_multithreaded (LOCK) : 0)
extern int glthread_rwlock_init_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_rdlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_wrlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_unlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_destroy_multithreaded (gl_rwlock_t *lock);

#  endif

# else

typedef struct
        {
          pthread_mutex_t lock; /* protects the remaining fields */
          pthread_cond_t waiting_readers; /* waiting readers */
          pthread_cond_t waiting_writers; /* waiting writers */
          unsigned int waiting_writers_count; /* number of waiting writers */
          int runcount; /* number of readers running, or -1 when a writer runs */
        }
        gl_rwlock_t;
# define gl_rwlock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_rwlock_t NAME;
# define gl_rwlock_define_initialized(STORAGECLASS, NAME) \
    STORAGECLASS gl_rwlock_t NAME = gl_rwlock_initializer;
# define gl_rwlock_initializer \
    { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0 }
# define glthread_rwlock_init(LOCK) \
    (pthread_in_use () ? glthread_rwlock_init_multithreaded (LOCK) : 0)
# define glthread_rwlock_rdlock(LOCK) \
    (pthread_in_use () ? glthread_rwlock_rdlock_multithreaded (LOCK) : 0)
# define glthread_rwlock_wrlock(LOCK) \
    (pthread_in_use () ? glthread_rwlock_wrlock_multithreaded (LOCK) : 0)
# define glthread_rwlock_unlock(LOCK) \
    (pthread_in_use () ? glthread_rwlock_unlock_multithreaded (LOCK) : 0)
# define glthread_rwlock_destroy(LOCK) \
    (pthread_in_use () ? glthread_rwlock_destroy_multithreaded (LOCK) : 0)
extern int glthread_rwlock_init_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_rdlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_wrlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_unlock_multithreaded (gl_rwlock_t *lock);
extern int glthread_rwlock_destroy_multithreaded (gl_rwlock_t *lock);

# endif

/* --------------------- gl_recursive_lock_t datatype --------------------- */

# if HAVE_PTHREAD_MUTEX_RECURSIVE

#  if defined PTHREAD_RECURSIVE_MUTEX_INITIALIZER || defined PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP

typedef pthread_mutex_t gl_recursive_lock_t;
#   define gl_recursive_lock_define(STORAGECLASS, NAME) \
      STORAGECLASS pthread_mutex_t NAME;
#   define gl_recursive_lock_define_initialized(STORAGECLASS, NAME) \
      STORAGECLASS pthread_mutex_t NAME = gl_recursive_lock_initializer;
#   ifdef PTHREAD_RECURSIVE_MUTEX_INITIALIZER
#    define gl_recursive_lock_initializer \
       PTHREAD_RECURSIVE_MUTEX_INITIALIZER
#   else
#    define gl_recursive_lock_initializer \
       PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
#   endif
#   define glthread_recursive_lock_init(LOCK) \
      (pthread_in_use () ? glthread_recursive_lock_init_multithreaded (LOCK) : 0)
#   define glthread_recursive_lock_lock(LOCK) \
      (pthread_in_use () ? pthread_mutex_lock (LOCK) : 0)
#   define glthread_recursive_lock_unlock(LOCK) \
      (pthread_in_use () ? pthread_mutex_unlock (LOCK) : 0)
#   define glthread_recursive_lock_destroy(LOCK) \
      (pthread_in_use () ? pthread_mutex_destroy (LOCK) : 0)
extern int glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock);

#  else

typedef struct
        {
          pthread_mutex_t recmutex; /* recursive mutex */
          pthread_mutex_t guard;    /* protects the initialization */
          int initialized;
        }
        gl_recursive_lock_t;
#   define gl_recursive_lock_define(STORAGECLASS, NAME) \
      STORAGECLASS gl_recursive_lock_t NAME;
#   define gl_recursive_lock_define_initialized(STORAGECLASS, NAME) \
      STORAGECLASS gl_recursive_lock_t NAME = gl_recursive_lock_initializer;
#   define gl_recursive_lock_initializer \
      { PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, 0 }
#   define glthread_recursive_lock_init(LOCK) \
      (pthread_in_use () ? glthread_recursive_lock_init_multithreaded (LOCK) : 0)
#   define glthread_recursive_lock_lock(LOCK) \
      (pthread_in_use () ? glthread_recursive_lock_lock_multithreaded (LOCK) : 0)
#   define glthread_recursive_lock_unlock(LOCK) \
      (pthread_in_use () ? glthread_recursive_lock_unlock_multithreaded (LOCK) : 0)
#   define glthread_recursive_lock_destroy(LOCK) \
      (pthread_in_use () ? glthread_recursive_lock_destroy_multithreaded (LOCK) : 0)
extern int glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_lock_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_unlock_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_destroy_multithreaded (gl_recursive_lock_t *lock);

#  endif

# else

/* Old versions of POSIX threads on Solaris did not have recursive locks.
   We have to implement them ourselves.  */

typedef struct
        {
          pthread_mutex_t mutex;
          pthread_t owner;
          unsigned long depth;
        }
        gl_recursive_lock_t;
#  define gl_recursive_lock_define(STORAGECLASS, NAME) \
     STORAGECLASS gl_recursive_lock_t NAME;
#  define gl_recursive_lock_define_initialized(STORAGECLASS, NAME) \
     STORAGECLASS gl_recursive_lock_t NAME = gl_recursive_lock_initializer;
#  define gl_recursive_lock_initializer \
     { PTHREAD_MUTEX_INITIALIZER, (pthread_t) 0, 0 }
#  define glthread_recursive_lock_init(LOCK) \
     (pthread_in_use () ? glthread_recursive_lock_init_multithreaded (LOCK) : 0)
#  define glthread_recursive_lock_lock(LOCK) \
     (pthread_in_use () ? glthread_recursive_lock_lock_multithreaded (LOCK) : 0)
#  define glthread_recursive_lock_unlock(LOCK) \
     (pthread_in_use () ? glthread_recursive_lock_unlock_multithreaded (LOCK) : 0)
#  define glthread_recursive_lock_destroy(LOCK) \
     (pthread_in_use () ? glthread_recursive_lock_destroy_multithreaded (LOCK) : 0)
extern int glthread_recursive_lock_init_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_lock_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_unlock_multithreaded (gl_recursive_lock_t *lock);
extern int glthread_recursive_lock_destroy_multithreaded (gl_recursive_lock_t *lock);

# endif

/* -------------------------- gl_once_t datatype -------------------------- */

typedef pthread_once_t gl_once_t;
# define gl_once_define(STORAGECLASS, NAME) \
    STORAGECLASS pthread_once_t NAME = PTHREAD_ONCE_INIT;
# if PTHREAD_IN_USE_DETECTION_HARD || USE_POSIX_THREADS_WEAK
#  define glthread_once(ONCE_CONTROL, INITFUNCTION) \
     (pthread_in_use ()                                                        \
      ? pthread_once (ONCE_CONTROL, INITFUNCTION)                              \
      : (glthread_once_singlethreaded (ONCE_CONTROL) ? (INITFUNCTION (), 0) : 0))
# else
#  define glthread_once(ONCE_CONTROL, INITFUNCTION) \
     (pthread_in_use ()                                                        \
      ? glthread_once_multithreaded (ONCE_CONTROL, INITFUNCTION)               \
      : (glthread_once_singlethreaded (ONCE_CONTROL) ? (INITFUNCTION (), 0) : 0))
extern int glthread_once_multithreaded (pthread_once_t *once_control,
                                        void (*init_function) (void));
# endif
extern int glthread_once_singlethreaded (pthread_once_t *once_control);

# ifdef __cplusplus
}
# endif

#endif

/* ========================================================================= */

#if USE_WINDOWS_THREADS

# define WIN32_LEAN_AND_MEAN  /* avoid including junk */
# include <windows.h>

# include "windows-mutex.h"
# include "windows-rwlock.h"
# include "windows-recmutex.h"
# include "windows-once.h"

# ifdef __cplusplus
extern "C" {
# endif

/* We can use CRITICAL_SECTION directly, rather than the native Windows Event,
   Mutex, Semaphore types, because
     - we need only to synchronize inside a single process (address space),
       not inter-process locking,
     - we don't need to support trylock operations.  (TryEnterCriticalSection
       does not work on Windows 95/98/ME.  Packages that need trylock usually
       define their own mutex type.)  */

/* There is no way to statically initialize a CRITICAL_SECTION.  It needs
   to be done lazily, once only.  For this we need spinlocks.  */

/* -------------------------- gl_lock_t datatype -------------------------- */

typedef glwthread_mutex_t gl_lock_t;
# define gl_lock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_lock_t NAME;
# define gl_lock_define_initialized(STORAGECLASS, NAME) \
    STORAGECLASS gl_lock_t NAME = gl_lock_initializer;
# define gl_lock_initializer \
    GLWTHREAD_MUTEX_INIT
# define glthread_lock_init(LOCK) \
    (glwthread_mutex_init (LOCK), 0)
# define glthread_lock_lock(LOCK) \
    glwthread_mutex_lock (LOCK)
# define glthread_lock_unlock(LOCK) \
    glwthread_mutex_unlock (LOCK)
# define glthread_lock_destroy(LOCK) \
    glwthread_mutex_destroy (LOCK)

/* ------------------------- gl_rwlock_t datatype ------------------------- */

typedef glwthread_rwlock_t gl_rwlock_t;
# define gl_rwlock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_rwlock_t NAME;
# define gl_rwlock_define_initialized(STORAGECLASS, NAME) \
    STORAGECLASS gl_rwlock_t NAME = gl_rwlock_initializer;
# define gl_rwlock_initializer \
    GLWTHREAD_RWLOCK_INIT
# define glthread_rwlock_init(LOCK) \
    (glwthread_rwlock_init (LOCK), 0)
# define glthread_rwlock_rdlock(LOCK) \
    glwthread_rwlock_rdlock (LOCK)
# define glthread_rwlock_wrlock(LOCK) \
    glwthread_rwlock_wrlock (LOCK)
# define glthread_rwlock_unlock(LOCK) \
    glwthread_rwlock_unlock (LOCK)
# define glthread_rwlock_destroy(LOCK) \
    glwthread_rwlock_destroy (LOCK)

/* --------------------- gl_recursive_lock_t datatype --------------------- */

typedef glwthread_recmutex_t gl_recursive_lock_t;
# define gl_recursive_lock_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_recursive_lock_t NAME;
# define gl_recursive_lock_define_initialized(STORAGECLASS, NAME) \
    STORAGECLASS gl_recursive_lock_t NAME = gl_recursive_lock_initializer;
# define gl_recursive_lock_initializer \
    GLWTHREAD_RECMUTEX_INIT
# define glthread_recursive_lock_init(LOCK) \
    (glwthread_recmutex_init (LOCK), 0)
# define glthread_recursive_lock_lock(LOCK) \
    glwthread_recmutex_lock (LOCK)
# define glthread_recursive_lock_unlock(LOCK) \
    glwthread_recmutex_unlock (LOCK)
# define glthread_recursive_lock_destroy(LOCK) \
    glwthread_recmutex_destroy (LOCK)

/* -------------------------- gl_once_t datatype -------------------------- */

typedef glwthread_once_t gl_once_t;
# define gl_once_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_once_t NAME = GLWTHREAD_ONCE_INIT;
# define glthread_once(ONCE_CONTROL, INITFUNCTION) \
    (glwthread_once (ONCE_CONTROL, INITFUNCTION), 0)

# ifdef __cplusplus
}
# endif

#endif

/* ========================================================================= */

#if !(USE_ISOC_THREADS || USE_POSIX_THREADS || USE_ISOC_AND_POSIX_THREADS || USE_WINDOWS_THREADS)

/* Provide dummy implementation if threads are not supported.  */

/* -------------------------- gl_lock_t datatype -------------------------- */

typedef int gl_lock_t;
# define gl_lock_define(STORAGECLASS, NAME)
# define gl_lock_define_initialized(STORAGECLASS, NAME)
# define glthread_lock_init(NAME) 0
# define glthread_lock_lock(NAME) 0
# define glthread_lock_unlock(NAME) 0
# define glthread_lock_destroy(NAME) 0

/* ------------------------- gl_rwlock_t datatype ------------------------- */

typedef int gl_rwlock_t;
# define gl_rwlock_define(STORAGECLASS, NAME)
# define gl_rwlock_define_initialized(STORAGECLASS, NAME)
# define glthread_rwlock_init(NAME) 0
# define glthread_rwlock_rdlock(NAME) 0
# define glthread_rwlock_wrlock(NAME) 0
# define glthread_rwlock_unlock(NAME) 0
# define glthread_rwlock_destroy(NAME) 0

/* --------------------- gl_recursive_lock_t datatype --------------------- */

typedef int gl_recursive_lock_t;
# define gl_recursive_lock_define(STORAGECLASS, NAME)
# define gl_recursive_lock_define_initialized(STORAGECLASS, NAME)
# define glthread_recursive_lock_init(NAME) 0
# define glthread_recursive_lock_lock(NAME) 0
# define glthread_recursive_lock_unlock(NAME) 0
# define glthread_recursive_lock_destroy(NAME) 0

/* -------------------------- gl_once_t datatype -------------------------- */

typedef int gl_once_t;
# define gl_once_define(STORAGECLASS, NAME) \
    STORAGECLASS gl_once_t NAME = 0;
# define glthread_once(ONCE_CONTROL, INITFUNCTION) \
    (*(ONCE_CONTROL) == 0 ? (*(ONCE_CONTROL) = ~ 0, INITFUNCTION (), 0) : 0)

#endif

/* ========================================================================= */

/* Macros with built-in error handling.  */

/* -------------------------- gl_lock_t datatype -------------------------- */

#define gl_lock_init(NAME) \
   do                                  \
     {                                 \
       if (glthread_lock_init (&NAME)) \
         abort ();                     \
     }                                 \
   while (0)
#define gl_lock_lock(NAME) \
   do                                  \
     {                                 \
       if (glthread_lock_lock (&NAME)) \
         abort ();                     \
     }                                 \
   while (0)
#define gl_lock_unlock(NAME) \
   do                                    \
     {                                   \
       if (glthread_lock_unlock (&NAME)) \
         abort ();                       \
     }                                   \
   while (0)
#define gl_lock_destroy(NAME) \
   do                                     \
     {                                    \
       if (glthread_lock_destroy (&NAME)) \
         abort ();                        \
     }                                    \
   while (0)

/* ------------------------- gl_rwlock_t datatype ------------------------- */

#define gl_rwlock_init(NAME) \
   do                                    \
     {                                   \
       if (glthread_rwlock_init (&NAME)) \
         abort ();                       \
     }                                   \
   while (0)
#define gl_rwlock_rdlock(NAME) \
   do                                      \
     {                                     \
       if (glthread_rwlock_rdlock (&NAME)) \
         abort ();                         \
     }                                     \
   while (0)
#define gl_rwlock_wrlock(NAME) \
   do                                      \
     {                                     \
       if (glthread_rwlock_wrlock (&NAME)) \
         abort ();                         \
     }                                     \
   while (0)
#define gl_rwlock_unlock(NAME) \
   do                                      \
     {                                     \
       if (glthread_rwlock_unlock (&NAME)) \
         abort ();                         \
     }                                     \
   while (0)
#define gl_rwlock_destroy(NAME) \
   do                                       \
     {                                      \
       if (glthread_rwlock_destroy (&NAME)) \
         abort ();                          \
     }                                      \
   while (0)

/* --------------------- gl_recursive_lock_t datatype --------------------- */

#define gl_recursive_lock_init(NAME) \
   do                                            \
     {                                           \
       if (glthread_recursive_lock_init (&NAME)) \
         abort ();                               \
     }                                           \
   while (0)
#define gl_recursive_lock_lock(NAME) \
   do                                            \
     {                                           \
       if (glthread_recursive_lock_lock (&NAME)) \
         abort ();                               \
     }                                           \
   while (0)
#define gl_recursive_lock_unlock(NAME) \
   do                                              \
     {                                             \
       if (glthread_recursive_lock_unlock (&NAME)) \
         abort ();                                 \
     }                                             \
   while (0)
#define gl_recursive_lock_destroy(NAME) \
   do                                               \
     {                                              \
       if (glthread_recursive_lock_destroy (&NAME)) \
         abort ();                                  \
     }                                              \
   while (0)

/* -------------------------- gl_once_t datatype -------------------------- */

#define gl_once(NAME, INITFUNCTION) \
   do                                           \
     {                                          \
       if (glthread_once (&NAME, INITFUNCTION)) \
         abort ();                              \
     }                                          \
   while (0)

/* ========================================================================= */

#endif /* _LOCK_H */
