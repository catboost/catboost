/* Print --version and bug-reporting information in a consistent format.
   Copyright (C) 1999, 2003, 2005, 2009-2013 Free Software Foundation, Inc.

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

/* Written by Jim Meyering. */

#ifndef VERSION_ETC_H
# define VERSION_ETC_H 1

# include <stdarg.h>
# include <stdio.h>

/* The 'sentinel' attribute was added in gcc 4.0.  */
#ifndef _GL_ATTRIBUTE_SENTINEL
# if 4 <= __GNUC__
#  define _GL_ATTRIBUTE_SENTINEL __attribute__ ((__sentinel__))
# else
#  define _GL_ATTRIBUTE_SENTINEL /* empty */
# endif
#endif

extern const char version_etc_copyright[];

/* The three functions below display the --version information in the
   standard way: command and package names, package version, followed
   by a short GPLv3+ notice and a list of up to 10 author names.

   If COMMAND_NAME is NULL, the PACKAGE is assumed to be the name of
   the program.  The formats are therefore:

   PACKAGE VERSION

   or

   COMMAND_NAME (PACKAGE) VERSION.

   The functions differ in the way they are passed author names: */

/* N_AUTHORS names are supplied in array AUTHORS.  */
extern void version_etc_arn (FILE *stream,
                             const char *command_name, const char *package,
                             const char *version,
                             const char * const * authors, size_t n_authors);

/* Names are passed in the NULL-terminated array AUTHORS.  */
extern void version_etc_ar (FILE *stream,
                            const char *command_name, const char *package,
                            const char *version, const char * const * authors);

/* Names are passed in the NULL-terminated va_list.  */
extern void version_etc_va (FILE *stream,
                            const char *command_name, const char *package,
                            const char *version, va_list authors);

/* Names are passed as separate arguments, with an additional
   NULL argument at the end.  */
extern void version_etc (FILE *stream,
                         const char *command_name, const char *package,
                         const char *version,
                         /* const char *author1, ..., NULL */ ...)
  _GL_ATTRIBUTE_SENTINEL;

/* Display the usual "Report bugs to" stanza.  */
extern void emit_bug_reporting_address (void);

#endif /* VERSION_ETC_H */
