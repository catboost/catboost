/* Thread-local storage in multithreaded situations.
   Copyright (C) 2005-2013 Free Software Foundation, Inc.

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

#include <config.h>

#include "glthread/tls.h"

/* ========================================================================= */

#if USE_POSIX_THREADS

#endif

/* ========================================================================= */

#if USE_PTH_THREADS

#endif

/* ========================================================================= */

#if USE_SOLARIS_THREADS

/* Use the old Solaris threads library.  */

/* ------------------------- gl_tls_key_t datatype ------------------------- */

void *
glthread_tls_get_multithreaded (thread_key_t key)
{
  void *value;

  if (thr_getspecific (key, &value) != 0)
    abort ();
  return value;
}

#endif

/* ========================================================================= */

#if USE_WINDOWS_THREADS

#endif

/* ========================================================================= */
