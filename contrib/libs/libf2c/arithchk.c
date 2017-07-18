/****************************************************************
Copyright (C) 1997, 1998, 2000 Lucent Technologies
All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the name of Lucent or any of its entities
not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

LUCENT DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL LUCENT OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
****************************************************************/

/* Try to deduce arith.h from arithmetic properties. */

#include <stdio.h>
#include <math.h>
#include <errno.h>

#ifdef NO_FPINIT
#define fpinit_ASL()
#else
#ifndef KR_headers
extern
#ifdef __cplusplus
	"C"
#endif
	void fpinit_ASL(void);
#endif /*KR_headers*/
#endif /*NO_FPINIT*/

 static int dalign;
 typedef struct
Akind {
	char *name;
	int   kind;
	} Akind;

 static Akind
IEEE_8087	= { "IEEE_8087", 1 },
IEEE_MC68k	= { "IEEE_MC68k", 2 },
IBM		= { "IBM", 3 },
VAX		= { "VAX", 4 },
CRAY		= { "CRAY", 5};

 static double t_nan;

 static Akind *
Lcheck(void)
{
	union {
		double d;
		long L[2];
		} u;
	struct {
		double d;
		long L;
		} x[2];

	if (sizeof(x) > 2*(sizeof(double) + sizeof(long)))
		dalign = 1;
	u.L[0] = u.L[1] = 0;
	u.d = 1e13;
	if (u.L[0] == 1117925532 && u.L[1] == -448790528)
		return &IEEE_MC68k;
	if (u.L[1] == 1117925532 && u.L[0] == -448790528)
		return &IEEE_8087;
	if (u.L[0] == -2065213935 && u.L[1] == 10752)
		return &VAX;
	if (u.L[0] == 1267827943 && u.L[1] == 704643072)
		return &IBM;
	return 0;
	}

 static Akind *
icheck(void)
{
	union {
		double d;
		int L[2];
		} u;
	struct {
		double d;
		int L;
		} x[2];

	if (sizeof(x) > 2*(sizeof(double) + sizeof(int)))
		dalign = 1;
	u.L[0] = u.L[1] = 0;
	u.d = 1e13;
	if (u.L[0] == 1117925532 && u.L[1] == -448790528)
		return &IEEE_MC68k;
	if (u.L[1] == 1117925532 && u.L[0] == -448790528)
		return &IEEE_8087;
	if (u.L[0] == -2065213935 && u.L[1] == 10752)
		return &VAX;
	if (u.L[0] == 1267827943 && u.L[1] == 704643072)
		return &IBM;
	return 0;
	}

char *emptyfmt = "";	/* avoid possible warning message with printf("") */

 static Akind *
ccheck(void)
{
	union {
		double d;
		long L;
		} u;
	long Cray1;

	/* Cray1 = 4617762693716115456 -- without overflow on non-Crays */
	Cray1 = printf(emptyfmt) < 0 ? 0 : 4617762;
	if (printf(emptyfmt, Cray1) >= 0)
		Cray1 = 1000000*Cray1 + 693716;
	if (printf(emptyfmt, Cray1) >= 0)
		Cray1 = 1000000*Cray1 + 115456;
	u.d = 1e13;
	if (u.L == Cray1)
		return &CRAY;
	return 0;
	}

 static int
fzcheck(void)
{
	double a, b;
	int i;

	a = 1.;
	b = .1;
	for(i = 155;; b *= b, i >>= 1) {
		if (i & 1) {
			a *= b;
			if (i == 1)
				break;
			}
		}
	b = a * a;
	return b == 0.;
	}

 static int
need_nancheck(void)
{
	double t;

	errno = 0;
	t = log(t_nan);
	if (errno == 0)
		return 1;
	errno = 0;
	t = sqrt(t_nan);
	return errno == 0;
	}

 void
get_nanbits(unsigned int *b, int k)
{
	union { double d; unsigned int z[2]; } u, u1, u2;

	k = 2 - k;
	u1.z[k] = u2.z[k] = 0x7ff00000;
	u1.z[1-k] = u2.z[1-k] = 0;
	u.d = u1.d - u2.d;	/* Infinity - Infinity */
	b[0] = u.z[0];
	b[1] = u.z[1];
	}

 int
main(void)
{
	FILE *f;
	Akind *a = 0;
	int Ldef = 0;
	unsigned int nanbits[2];

	fpinit_ASL();
#ifdef WRITE_ARITH_H	/* for Symantec's buggy "make" */
	f = fopen("arith.h", "w");
	if (!f) {
		printf("Cannot open arith.h\n");
		return 1;
		}
#else
	f = stdout;
#endif

	if (sizeof(double) == 2*sizeof(long))
		a = Lcheck();
	else if (sizeof(double) == 2*sizeof(int)) {
		Ldef = 1;
		a = icheck();
		}
	else if (sizeof(double) == sizeof(long))
		a = ccheck();
	if (a) {
		fprintf(f, "#define %s\n#define Arith_Kind_ASL %d\n",
			a->name, a->kind);
		if (Ldef)
			fprintf(f, "#define Long int\n#define Intcast (int)(long)\n");
		if (dalign)
			fprintf(f, "#define Double_Align\n");
		if (sizeof(char*) == 8)
			fprintf(f, "#define X64_bit_pointers\n");
#ifndef NO_LONG_LONG
		if (sizeof(long long) < 8)
#endif
			fprintf(f, "#define NO_LONG_LONG\n");
		if (a->kind <= 2) {
			if (fzcheck())
				fprintf(f, "#define Sudden_Underflow\n");
			t_nan = -a->kind;
			if (need_nancheck())
				fprintf(f, "#define NANCHECK\n");
			if (sizeof(double) == 2*sizeof(unsigned int)) {
				get_nanbits(nanbits, a->kind);
				fprintf(f, "#define QNaN0 0x%x\n", nanbits[0]);
				fprintf(f, "#define QNaN1 0x%x\n", nanbits[1]);
				}
			}
		return 0;
		}
	fprintf(f, "/* Unknown arithmetic */\n");
	return 1;
	}

#ifdef __sun
#ifdef __i386
/* kludge for Intel Solaris */
void fpsetprec(int x) { }
#endif
#endif
