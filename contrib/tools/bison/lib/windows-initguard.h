/* Init guards, somewhat like spinlocks (native Windows implementation).
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

#ifndef _WINDOWS_INITGUARD_H
#define _WINDOWS_INITGUARD_H

#define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#include <windows.h>

typedef struct
        {
          volatile int done;
          volatile LONG started;
        }
        glwthread_initguard_t;

#define GLWTHREAD_INITGUARD_INIT { 0, -1 }

#endif /* _WINDOWS_INITGUARD_H */
