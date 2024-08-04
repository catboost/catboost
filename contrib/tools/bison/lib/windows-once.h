/* Once-only control (native Windows implementation).
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

#ifndef _WINDOWS_ONCE_H
#define _WINDOWS_ONCE_H

#define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#include <windows.h>

typedef struct
        {
          volatile int inited;
          volatile LONG started;
          CRITICAL_SECTION lock;
        }
        glwthread_once_t;

#define GLWTHREAD_ONCE_INIT { -1, -1 }

#ifdef __cplusplus
extern "C" {
#endif

extern void glwthread_once (glwthread_once_t *once_control,
                            void (*initfunction) (void));

#ifdef __cplusplus
}
#endif

#endif /* _WINDOWS_ONCE_H */
