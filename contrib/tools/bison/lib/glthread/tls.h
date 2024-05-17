/* Thread-local storage in multithreaded situations.
   Copyright (C) 2005, 2007-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2005.  */

/* This file contains thread-local storage primitives for use with a given
   thread library.  It does not contain primitives for creating threads or
   for other multithreading primitives.

     Type:                      gl_tls_key_t
     Initialization:            gl_tls_key_init (name, destructor);
     Getting per-thread value:  gl_tls_get (name)
     Setting per-thread value:  gl_tls_set (name, pointer);
     De-initialization:         gl_tls_key_destroy (name);
   Equivalent functions with control of error handling:
     Initialization:            err = glthread_tls_key_init (&name, destructor);
     Setting per-thread value:  err = glthread_tls_set (&name, pointer);
     De-initialization:         err = glthread_tls_key_destroy (&name);

   A per-thread value is of type 'void *'.

   A destructor is a function pointer of type 'void (*) (void *)', called
   when a thread exits, and taking the last per-thread value as argument.  It
   is unspecified whether the destructor function is called when the last
   per-thread value is NULL.  On some platforms, the destructor function is
   not called at all.
*/


#ifndef _TLS_H
#define _TLS_H

#include <errno.h>
#include <stdlib.h>

/* ========================================================================= */

#if USE_POSIX_THREADS

/* Use the POSIX threads library.  */

# include <pthread.h>

# if PTHREAD_IN_USE_DETECTION_HARD

/* The pthread_in_use() detection needs to be done at runtime.  */
#  define pthread_in_use() \
     glthread_in_use ()
extern int glthread_in_use (void);

# endif

# if USE_POSIX_THREADS_WEAK

/* Use weak references to the POSIX threads library.  */

#  pragma weak pthread_key_create
#  pragma weak pthread_getspecific
#  pragma weak pthread_setspecific
#  pragma weak pthread_key_delete
#  ifndef pthread_self
#   pragma weak pthread_self
#  endif

#  if !PTHREAD_IN_USE_DETECTION_HARD
#   pragma weak pthread_cancel
#   define pthread_in_use() (pthread_cancel != NULL)
#  endif

# else

#  if !PTHREAD_IN_USE_DETECTION_HARD
#   define pthread_in_use() 1
#  endif

# endif

/* ------------------------- gl_tls_key_t datatype ------------------------- */

typedef union
        {
          void *singlethread_value;
          pthread_key_t key;
        }
        gl_tls_key_t;
# define glthread_tls_key_init(KEY, DESTRUCTOR) \
    (pthread_in_use ()                              \
     ? pthread_key_create (&(KEY)->key, DESTRUCTOR) \
     : ((KEY)->singlethread_value = NULL, 0))
# define gl_tls_get(NAME) \
    (pthread_in_use ()                  \
     ? pthread_getspecific ((NAME).key) \
     : (NAME).singlethread_value)
# define glthread_tls_set(KEY, POINTER) \
    (pthread_in_use ()                             \
     ? pthread_setspecific ((KEY)->key, (POINTER)) \
     : ((KEY)->singlethread_value = (POINTER), 0))
# define glthread_tls_key_destroy(KEY) \
    (pthread_in_use () ? pthread_key_delete ((KEY)->key) : 0)

#endif

/* ========================================================================= */

#if USE_PTH_THREADS

/* Use the GNU Pth threads library.  */

# include <pth.h>

# if USE_PTH_THREADS_WEAK

/* Use weak references to the GNU Pth threads library.  */

#  pragma weak pth_key_create
#  pragma weak pth_key_getdata
#  pragma weak pth_key_setdata
#  pragma weak pth_key_delete

#  pragma weak pth_cancel
#  define pth_in_use() (pth_cancel != NULL)

# else

#  define pth_in_use() 1

# endif

/* ------------------------- gl_tls_key_t datatype ------------------------- */

typedef union
        {
          void *singlethread_value;
          pth_key_t key;
        }
        gl_tls_key_t;
# define glthread_tls_key_init(KEY, DESTRUCTOR) \
    (pth_in_use ()                                             \
     ? (!pth_key_create (&(KEY)->key, DESTRUCTOR) ? errno : 0) \
     : ((KEY)->singlethread_value = NULL, 0))
# define gl_tls_get(NAME) \
    (pth_in_use ()                  \
     ? pth_key_getdata ((NAME).key) \
     : (NAME).singlethread_value)
# define glthread_tls_set(KEY, POINTER) \
    (pth_in_use ()                                            \
     ? (!pth_key_setdata ((KEY)->key, (POINTER)) ? errno : 0) \
     : ((KEY)->singlethread_value = (POINTER), 0))
# define glthread_tls_key_destroy(KEY) \
    (pth_in_use ()                                \
     ? (!pth_key_delete ((KEY)->key) ? errno : 0) \
     : 0)

#endif

/* ========================================================================= */

#if USE_SOLARIS_THREADS

/* Use the old Solaris threads library.  */

# include <thread.h>

# if USE_SOLARIS_THREADS_WEAK

/* Use weak references to the old Solaris threads library.  */

#  pragma weak thr_keycreate
#  pragma weak thr_getspecific
#  pragma weak thr_setspecific

#  pragma weak thr_suspend
#  define thread_in_use() (thr_suspend != NULL)

# else

#  define thread_in_use() 1

# endif

/* ------------------------- gl_tls_key_t datatype ------------------------- */

typedef union
        {
          void *singlethread_value;
          thread_key_t key;
        }
        gl_tls_key_t;
# define glthread_tls_key_init(KEY, DESTRUCTOR) \
    (thread_in_use ()                          \
     ? thr_keycreate (&(KEY)->key, DESTRUCTOR) \
     : ((KEY)->singlethread_value = NULL, 0))
# define gl_tls_get(NAME) \
    (thread_in_use ()                \
     ? glthread_tls_get_multithreaded ((NAME).key) \
     : (NAME).singlethread_value)
extern void *glthread_tls_get_multithreaded (thread_key_t key);
# define glthread_tls_set(KEY, POINTER) \
    (thread_in_use ()                              \
     ? thr_setspecific ((KEY)->key, (POINTER))     \
     : ((KEY)->singlethread_value = (POINTER), 0))
# define glthread_tls_key_destroy(KEY) \
    /* Unsupported.  */ \
    0

#endif

/* ========================================================================= */

#if USE_WINDOWS_THREADS

# define WIN32_LEAN_AND_MEAN  /* avoid including junk */
# include <windows.h>

/* ------------------------- gl_tls_key_t datatype ------------------------- */

typedef DWORD gl_tls_key_t;
# define glthread_tls_key_init(KEY, DESTRUCTOR) \
    /* The destructor is unsupported.  */    \
    ((*(KEY) = TlsAlloc ()) == (DWORD)-1 ? EAGAIN : ((void) (DESTRUCTOR), 0))
# define gl_tls_get(NAME) \
    TlsGetValue (NAME)
# define glthread_tls_set(KEY, POINTER) \
    (!TlsSetValue (*(KEY), POINTER) ? EINVAL : 0)
# define glthread_tls_key_destroy(KEY) \
    (!TlsFree (*(KEY)) ? EINVAL : 0)

#endif

/* ========================================================================= */

#if !(USE_POSIX_THREADS || USE_PTH_THREADS || USE_SOLARIS_THREADS || USE_WINDOWS_THREADS)

/* Provide dummy implementation if threads are not supported.  */

/* ------------------------- gl_tls_key_t datatype ------------------------- */

typedef struct
        {
          void *singlethread_value;
        }
        gl_tls_key_t;
# define glthread_tls_key_init(KEY, DESTRUCTOR) \
    ((KEY)->singlethread_value = NULL, \
     (void) (DESTRUCTOR),              \
     0)
# define gl_tls_get(NAME) \
    (NAME).singlethread_value
# define glthread_tls_set(KEY, POINTER) \
    ((KEY)->singlethread_value = (POINTER), 0)
# define glthread_tls_key_destroy(KEY) \
    0

#endif

/* ========================================================================= */

/* Macros with built-in error handling.  */

/* ------------------------- gl_tls_key_t datatype ------------------------- */

#define gl_tls_key_init(NAME, DESTRUCTOR) \
   do                                                 \
     {                                                \
       if (glthread_tls_key_init (&NAME, DESTRUCTOR)) \
         abort ();                                    \
     }                                                \
   while (0)
#define gl_tls_set(NAME, POINTER) \
   do                                         \
     {                                        \
       if (glthread_tls_set (&NAME, POINTER)) \
         abort ();                            \
     }                                        \
   while (0)
#define gl_tls_key_destroy(NAME) \
   do                                        \
     {                                       \
       if (glthread_tls_key_destroy (&NAME)) \
         abort ();                           \
     }                                       \
   while (0)

/* ========================================================================= */

#endif /* _TLS_H */
