/* Plain mutexes (native Windows implementation).
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

#ifndef _WINDOWS_MUTEX_H
#define _WINDOWS_MUTEX_H

#define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#include <windows.h>

#include "windows-initguard.h"

typedef struct
        {
          glwthread_initguard_t guard; /* protects the initialization */
          CRITICAL_SECTION lock;
        }
        glwthread_mutex_t;

#define GLWTHREAD_MUTEX_INIT { GLWTHREAD_INITGUARD_INIT }

#ifdef __cplusplus
extern "C" {
#endif

extern void glwthread_mutex_init (glwthread_mutex_t *mutex);
extern int glwthread_mutex_lock (glwthread_mutex_t *mutex);
extern int glwthread_mutex_trylock (glwthread_mutex_t *mutex);
extern int glwthread_mutex_unlock (glwthread_mutex_t *mutex);
extern int glwthread_mutex_destroy (glwthread_mutex_t *mutex);

#ifdef __cplusplus
}
#endif

#endif /* _WINDOWS_MUTEX_H */
