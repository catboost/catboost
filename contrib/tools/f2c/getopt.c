/****************************************************************
Copyright 1996 by Lucent Technologies.

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of Bell Laboratories or Lucent
Technologies or any of their entities not be used in advertising
or publicity pertaining to distribution of the software without
specific, written prior permission.

Lucent disclaims all warranties with regard to this software,
including all implied warranties of merchantability and fitness.
In no event shall Lucent be liable for any special, indirect or
consequential damages or any damages whatsoever resulting from
loss of use, data or profits, whether in an action of contract,
negligence or other tortious action, arising out of or in
connection with the use or performance of this software.
****************************************************************/

/* Source for a "getopt" command, as invoked by the "fc" script. */

#include <stdio.h>

static char opts[256];	/* assume 8-bit bytes */

 int
#ifdef KR_headers
main(argc, argv) int argc; char **argv;
#else
main(int argc, char **argv)
#endif
{
	char **av, *fmt, *s, *s0;
	int i;

	if (argc < 2) {
		fprintf(stderr, "Usage: getopt optstring arg1 arg2...\n");
		return 1;
		}
	for(s = argv[1]; *s; ) {
		i = *(unsigned char *)s++;
		if (!opts[i])
			opts[i] = 1;
		if (*s == ':') {
			s++;
			opts[i] = 2;
			}
		}
	/* scan for legal args */
	av = argv + 2;
 nextarg:
	while(s = *av++) {
		if (*s++ != '-' || s[0] == '-' && s[1] == 0)
			break;
		while(i = *(unsigned char *)s++) {
			switch(opts[i]) {
			  case 0:
				fprintf(stderr,
					"getopt: Illegal option -- %c\n", s[-1]);
				return 1;
			  case 2:
				s0 = s - 1;
				if (*s || *av++)
					goto nextarg;
				fprintf(stderr,
				 "getopt: Option requires an argument -- %c\n",
					*s0);
				return 1;
			  }
			}
		}
	/* output modified args */
	av = argv + 2;
	fmt = "-%c";
 nextarg1:
	while(s = *av++) {
		if (s[0] != '-')
			break;
		if (*++s == '-' && !s[1]) {
			s = *av++;
			break;
			}
		while(*s) {
			printf(fmt, *s);
			fmt = " -%c";
			if (opts[*(unsigned char *)s++] == 2) {
				if (!*s)
					s = *av++;
				printf(" %s", s);
				goto nextarg1;
				}
			}
		}
	printf(*fmt == ' ' ? " --" : "--");
	for(; s; s = *av++)
		printf(" %s", s);
	printf("\n");
	return 0;
	}
