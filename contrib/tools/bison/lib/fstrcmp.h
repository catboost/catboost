/* Fuzzy string comparison.
   Copyright (C) 1995, 2000, 2002-2003, 2006, 2008-2020 Free Software
   Foundation, Inc.

   This file was written by Peter Miller <pmiller@agso.gov.au>

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

#ifndef _FSTRCMP_H
#define _FSTRCMP_H

#ifdef __cplusplus
extern "C" {
#endif

/* Fuzzy compare of S1 and S2.  Return a measure for the similarity of S1
   and S1.  The higher the result, the more similar the strings are.
   The result is bounded between 0 (meaning very dissimilar strings) and
   1 (meaning identical strings).  */
extern double fstrcmp (const char *s1, const char *s2);

/* Like fstrcmp (S1, S2), except that if the result is < LOWER_BOUND, an
   arbitrary other value < LOWER_BOUND can be returned.  */
extern double fstrcmp_bounded (const char *s1, const char *s2,
                               double lower_bound);

/* A shortcut for fstrcmp.  Avoids a function call.  */
#define fstrcmp(s1,s2) fstrcmp_bounded (s1, s2, 0.0)

/* Frees the per-thread resources allocated by this module for the current
   thread.
   You don't need to call this function in threads other than the main thread,
   because per-thread resources are reclaimed automatically when the thread
   exits.  However, per-thread resources allocated by the main thread are
   comparable to static allocations; calling this function can be useful to
   avoid an error report from valgrind.  */
extern void fstrcmp_free_resources (void);

#ifdef __cplusplus
}
#endif

#endif
