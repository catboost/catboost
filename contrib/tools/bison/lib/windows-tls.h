/* Thread-local storage (native Windows implementation).
   Copyright (C) 2005-2020 Free Software Foundation, Inc.

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

/* Written by Bruno Haible <bruno@clisp.org>, 2005.  */

#ifndef _WINDOWS_TLS_H
#define _WINDOWS_TLS_H

#define WIN32_LEAN_AND_MEAN  /* avoid including junk */
#include <windows.h>

typedef DWORD glwthread_tls_key_t;

#ifdef __cplusplus
extern "C" {
#endif

extern int glwthread_tls_key_create (glwthread_tls_key_t *keyp, void (*destructor) (void *));
extern void *glwthread_tls_get (glwthread_tls_key_t key);
extern int glwthread_tls_set (glwthread_tls_key_t key, void *value);
extern int glwthread_tls_key_delete (glwthread_tls_key_t key);
extern void glwthread_tls_process_destructors (void);
#define GLWTHREAD_DESTRUCTOR_ITERATIONS 4

#ifdef __cplusplus
}
#endif

#endif /* _WINDOWS_TLS_H */
