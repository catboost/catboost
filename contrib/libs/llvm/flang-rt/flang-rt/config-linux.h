/*===-- cmake/config.cmake.in ---------------------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/

#ifndef FORTRAN_RUNTIME_CONFIG_H
#define FORTRAN_RUNTIME_CONFIG_H

/* Define to 1 if you have the `strerror_r' function. */
#define HAVE_STRERROR_R 1

/* Define to 1 if you have the declaration of `strerror_s', and to 0 if you
   don't. */
#define HAVE_DECL_STRERROR_S 0

/* Define to 1 if you have the `backtrace' function. */
#define HAVE_BACKTRACE TRUE

#define BACKTRACE_HEADER <execinfo.h>

#endif
