/* Unicode character output to streams with locale dependent encoding.

   Copyright (C) 2000-2003, 2005, 2008-2020 Free Software Foundation, Inc.

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

#ifndef UNICODEIO_H
# define UNICODEIO_H

# include <stddef.h>
# include <stdio.h>

/* Converts the Unicode character CODE to its multibyte representation
   in the current locale and calls the SUCCESS callback on the resulting
   byte sequence.  If an error occurs, invokes the FAILURE callback instead,
   passing it CODE and an English error string.
   Returns whatever the callback returned.
   Assumes that the locale doesn't change between two calls.  */
extern long unicode_to_mb (unsigned int code,
                           long (*success) (const char *buf, size_t buflen,
                                            void *callback_arg),
                           long (*failure) (unsigned int code, const char *msg,
                                            void *callback_arg),
                           void *callback_arg);

/* Outputs the Unicode character CODE to the output stream STREAM.
   Upon failure, exit if exit_on_error is true, otherwise output a fallback
   notation.  */
extern void print_unicode_char (FILE *stream, unsigned int code,
                                int exit_on_error);

/* Simple success callback that outputs the converted string.
   The STREAM is passed as callback_arg.  */
extern long fwrite_success_callback (const char *buf, size_t buflen,
                                     void *callback_arg);

#endif
