/****************************************************************
Copyright 1990, 2000 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/

/* This is for the benefit of people whose systems don't provide
 * memset, memcpy, and memcmp.  If yours is such a system, adjust
 * the makefile by adding memset.o to the "OBJECTS =" assignment.
 * WARNING: the memcpy below is adequate for f2c, but is not a
 * general memcpy routine (which must correctly handle overlapping
 * fields).
 */

 int
#ifdef KR_headers
memcmp(s1, s2, n) char *s1, *s2; int n;
#else
memcmp(char *s1, char *s2, int n)
#endif
{
	char *se;

	for(se = s1 + n; s1 < se; s1++, s2++)
		if (*s1 != *s2)
			return *s1 - *s2;
	return 0;
	}

 char *
#ifdef KR_headers
memcpy(s1, s2, n) char *s1, *s2; int n;
#else
memcpy(char *s1, char *s2, int n)
#endif
{
	char *s0 = s1, *se = s1 + n;

	while(s1 < se)
		*s1++ = *s2++;
	return s0;
	}

 void
#ifdef KR_headers
memset(s, c, n) char *s; int c, n;
#else
memset(char *s, int c, int n)
#endif
{
	char *se = s + n;

	while(s < se)
		*s++ = c;
	}
