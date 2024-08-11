/* Optimization of multithreaded code.

   Copyright (C) 2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2020.  */

#ifndef _THREAD_OPTIM_H
#define _THREAD_OPTIM_H

/* This file defines a way to optimize multithreaded code for the single-thread
   case, based on the variable '__libc_single_threaded', defined in
   glibc >= 2.32.  */

/* Typical use: In a block or function, use

     bool mt = gl_multithreaded ();
     ...
     if (mt)
       if (pthread_mutex_lock (&lock)) abort ();
     ...
     if (mt)
       if (pthread_mutex_unlock (&lock)) abort ();

   The gl_multithreaded () invocation determines whether the program currently
   is multithreaded.

   if (mt) STATEMENT executes STATEMENT in the multithreaded case, and skips
   it in the single-threaded case.

   The code between the gl_multithreaded () invocation and any use of the
   variable 'mt' must not create threads or invoke functions that may
   indirectly create threads (e.g. 'dlopen' may, indirectly through C++
   initializers of global variables in the shared library being opened,
   create threads).

   The lock here is meant to synchronize threads in the same process.  The
   same optimization cannot be applied to locks that synchronize different
   processes (e.g. through shared memory mappings).  */

#if HAVE_SYS_SINGLE_THREADED_H /* glibc >= 2.32 */
# error #include <sys/single_threaded.h>
# define gl_multithreaded()  !__libc_single_threaded
#else
# define gl_multithreaded()  1
#endif

#endif /* _THREAD_OPTIM_H */
