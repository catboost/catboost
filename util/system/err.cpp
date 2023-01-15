#include "defaults.h"

#if defined(_MSC_VER) || defined(_sun_)
/* err.c --- 4.4BSD utility functions for error messages.
   Copyright (C) 1995, 1996 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */
//Modified for MSVC

#include <cstdarg>
//#include <err.h> -> compat.h
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstdio>

#include "compat.h"

#define VA(call)              \
    {                         \
        va_list ap;           \
        va_start(ap, format); \
        call;                 \
        va_end(ap);           \
    }

#define __set_errno(error) \
    { errno = error; }

void vwarnx(const char* format, va_list ap) {
    const char* progname = getprogname();
    if (progname)
        fprintf(stderr, "%s: ", progname);
    if (format)
        vfprintf(stderr, format, ap);
    putc('\n', stderr);
}

void vwarn(const char* format, va_list ap) {
    int error = errno;

    const char* progname = getprogname();
    if (progname)
        fprintf(stderr, "%s: ", progname);
    if (format) {
        vfprintf(stderr, format, ap);
        fputs(": ", stderr);
    }
    __set_errno(error);
    perror("");
}

void warn(const char* format, ...) {
    VA(vwarn(format, ap))
}

void warnx(const char* format, ...) {
    VA(vwarnx(format, ap))
}

void verr(int status, const char* format, va_list ap) {
    vwarn(format, ap);
    exit(status);
}

void verrx(int status, const char* format, va_list ap) {
    vwarnx(format, ap);
    exit(status);
}

void err(int status, const char* format, ...) {
    VA(verr(status, format, ap))
}

void errx(int status, const char* format, ...) {
    VA(verrx(status, format, ap))
}
#endif
