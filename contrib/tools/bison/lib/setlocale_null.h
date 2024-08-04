/* Query the name of the current global locale.
   Copyright (C) 2019-2020 Free Software Foundation, Inc.

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

/* Written by Bruno Haible <bruno@clisp.org>, 2019.  */

#ifndef _SETLOCALE_NULL_H
#define _SETLOCALE_NULL_H

#include <stddef.h>

#include "arg-nonnull.h"


#ifdef __cplusplus
extern "C" {
#endif


/* Recommended size of a buffer for a locale name for a single category.
   On glibc systems, you can have locale names that are relative file names;
   assume a maximum length 256.
   In native Windows, in 2018 the longest locale name was of length 58
   ("FYRO Macedonian_Former Yugoslav Republic of Macedonia.1251").  */
#define SETLOCALE_NULL_MAX (256+1)

/* Recommended size of a buffer for a locale name with all categories.
   On glibc systems, you can have locale names that are relative file names;
   assume maximum length 256 for each.  There are 12 categories; so, the
   maximum total length is 148+12*256.
   In native Windows, there are 5 categories, and the maximum total length is
   55+5*58.  */
#define SETLOCALE_NULL_ALL_MAX (148+12*256+1)

/* setlocale_null_r (CATEGORY, BUF, BUFSIZE) is like setlocale (CATEGORY, NULL),
   except that
     - it is guaranteed to be multithread-safe,
     - it returns the resulting locale category name or locale name in the
       user-supplied buffer BUF, which must be BUFSIZE bytes long.
   The recommended minimum buffer size is
     - SETLOCALE_NULL_MAX for CATEGORY != LC_ALL, and
     - SETLOCALE_NULL_ALL_MAX for CATEGORY == LC_ALL.
   The return value is an error code: 0 if the call is successful, EINVAL if
   CATEGORY is invalid, or ERANGE if BUFSIZE is smaller than the length needed
   size (including the trailing NUL byte).  In the latter case, a truncated
   result is returned in BUF, but still NUL-terminated if BUFSIZE > 0.
   For this call to be multithread-safe, *all* calls to
   setlocale (CATEGORY, NULL) in all other threads must have been converted
   to use setlocale_null_r or setlocale_null as well, and the other threads
   must not make other setlocale invocations (since changing the global locale
   has side effects on all threads).  */
extern int setlocale_null_r (int category, char *buf, size_t bufsize)
  _GL_ARG_NONNULL ((2));

/* setlocale_null (CATEGORY) is like setlocale (CATEGORY, NULL), except that
   it is guaranteed to be multithread-safe.
   The return value is NULL if CATEGORY is invalid.
   For this call to be multithread-safe, *all* calls to
   setlocale (CATEGORY, NULL) in all other threads must have been converted
   to use setlocale_null_r or setlocale_null as well, and the other threads
   must not make other setlocale invocations (since changing the global locale
   has side effects on all threads).  */
extern const char *setlocale_null (int category);


#ifdef __cplusplus
}
#endif

#endif /* _SETLOCALE_NULL_H */
