/* Read-write locks (native Windows implementation).
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
   Based on GCC's gthr-win32.h.  */

#ifndef _WINDOWS_RWLOCK_H
#define _WINDOWS_RWLOCK_H

#define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#include <windows.h>

#include "windows-initguard.h"

/* It is impossible to implement read-write locks using plain locks, without
   introducing an extra thread dedicated to managing read-write locks.
   Therefore here we need to use the low-level Event type.  */

typedef struct
        {
          HANDLE *array; /* array of waiting threads, each represented by an event */
          unsigned int count; /* number of waiting threads */
          unsigned int alloc; /* length of allocated array */
          unsigned int offset; /* index of first waiting thread in array */
        }
        glwthread_carray_waitqueue_t;
typedef struct
        {
          glwthread_initguard_t guard; /* protects the initialization */
          CRITICAL_SECTION lock; /* protects the remaining fields */
          glwthread_carray_waitqueue_t waiting_readers; /* waiting readers */
          glwthread_carray_waitqueue_t waiting_writers; /* waiting writers */
          int runcount; /* number of readers running, or -1 when a writer runs */
        }
        glwthread_rwlock_t;

#define GLWTHREAD_RWLOCK_INIT { GLWTHREAD_INITGUARD_INIT }

#ifdef __cplusplus
extern "C" {
#endif

extern void glwthread_rwlock_init (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_rdlock (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_wrlock (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_tryrdlock (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_trywrlock (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_unlock (glwthread_rwlock_t *lock);
extern int glwthread_rwlock_destroy (glwthread_rwlock_t *lock);

#ifdef __cplusplus
}
#endif

#endif /* _WINDOWS_RWLOCK_H */
