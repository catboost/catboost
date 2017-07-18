/*	$NetBSD: strtod.c,v 1.45.2.1 2005/04/19 13:35:54 tron Exp $	*/

/****************************************************************
 *
 * The author of this software is David M. Gay.
 *
 * Copyright (c) 1991 by AT&T.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 *
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHOR NOR AT&T MAKES ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 *
 ***************************************************************/

/* Please send bug reports to
	David M. Gay
	AT&T Bell Laboratories, Room 2C-463
	600 Mountain Avenue
	Murray Hill, NJ 07974-2070
	U.S.A.
	dmg@research.att.com or research!dmg
 */

/* strtod for IEEE-, VAX-, and IBM-arithmetic machines.
 *
 * This strtod returns a nearest machine number to the input decimal
 * string (or sets errno to ERANGE).  With IEEE arithmetic, ties are
 * broken by the IEEE round-even rule.  Otherwise ties are broken by
 * biased rounding (add half and chop).
 *
 * Inspired loosely by William D. Clinger's paper "How to Read Floating
 * Point Numbers Accurately" [Proc. ACM SIGPLAN '90, pp. 92-101].
 *
 * Modifications:
 *
 *	1. We only require IEEE, IBM, or VAX double-precision
 *		arithmetic (not IEEE double-extended).
 *	2. We get by with floating-point arithmetic in a case that
 *		Clinger missed -- when we're computing d * 10^n
 *		for a small integer d and the integer n is not too
 *		much larger than 22 (the maximum integer k for which
 *		we can represent 10^k exactly), we may be able to
 *		compute (d*10^k) * 10^(e-k) with just one roundoff.
 *	3. Rather than a bit-at-a-time adjustment of the binary
 *		result in the hard case, we use floating-point
 *		arithmetic to determine the adjustment to within
 *		one bit; only in really hard cases do we need to
 *		compute a second residual.
 *	4. Because of 3., we don't need a large table of powers of 10
 *		for ten-to-e (just some small tables, e.g. of 10^k
 *		for 0 <= k <= 22).
 */

/*
 * #define IEEE_LITTLE_ENDIAN for IEEE-arithmetic machines where the least
 *	significant byte has the lowest address.
 * #define IEEE_BIG_ENDIAN for IEEE-arithmetic machines where the most
 *	significant byte has the lowest address.
 * #define Long int on machines with 32-bit ints and 64-bit longs.
 * #define Sudden_Underflow for IEEE-format machines without gradual
 *	underflow (i.e., that flush to zero on underflow).
 * #define IBM for IBM mainframe-style floating-point arithmetic.
 * #define VAX for VAX-style floating-point arithmetic.
 * #define Unsigned_Shifts if >> does treats its left operand as unsigned.
 * #define No_leftright to omit left-right logic in fast floating-point
 *	computation of dtoa.
 * #define Check_FLT_ROUNDS if FLT_ROUNDS can assume the values 2 or 3.
 * #define RND_PRODQUOT to use rnd_prod and rnd_quot (assembly routines
 *	that use extended-precision instructions to compute rounded
 *	products and quotients) with IBM.
 * #define ROUND_BIASED for IEEE-format with biased rounding.
 * #define Inaccurate_Divide for IEEE-format with correctly rounded
 *	products but inaccurate quotients, e.g., for Intel i860.
 * #define Just_16 to store 16 bits per 32-bit Long when doing high-precision
 *	integer arithmetic.  Whether this speeds things up or slows things
 *	down depends on the machine and the number being converted.
 * #define KR_headers for old-style C function headers.
 * #define Bad_float_h if your system lacks a float.h or if it does not
 *	define some or all of DBL_DIG, DBL_MAX_10_EXP, DBL_MAX_EXP,
 *	FLT_RADIX, FLT_ROUNDS, and DBL_MAX.
 * #define MALLOC your_malloc, where your_malloc(n) acts like malloc(n)
 *	if memory is available and otherwise does something you deem
 *	appropriate.  If MALLOC is undefined, malloc will be invoked
 *	directly -- and assumed always to succeed.
 */

#define ANDROID_CHANGES

#ifdef ANDROID_CHANGES
/* Needs to be above math.h include below */
#include "fpmath.h"

#include <pthread.h>
#define mutex_lock(x) pthread_mutex_lock(x)
#define mutex_unlock(x) pthread_mutex_unlock(x)
#endif

#include <sys/cdefs.h>
#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: strtod.c,v 1.45.2.1 2005/04/19 13:35:54 tron Exp $");
#endif /* LIBC_SCCS and not lint */

#define Unsigned_Shifts
#if defined(__m68k__) || defined(__sparc__) || defined(__i386__) || \
    defined(__mips__) || defined(__ns32k__) || defined(__alpha__) || \
    defined(__powerpc__) || defined(__sh__) || defined(__x86_64__) || \
    defined(__hppa__) || \
    defined(__arm__) || defined(__aarch64__) || \
    defined(__le32__) || defined(__le64__)
#include <endian.h>
#if BYTE_ORDER == BIG_ENDIAN
#define IEEE_BIG_ENDIAN
#else
#define IEEE_LITTLE_ENDIAN
#endif
#endif

#ifdef __vax__
#define VAX
#endif

#if defined(__hppa__) || defined(__mips__) || defined(__sh__)
#define	NAN_WORD0	0x7ff40000
#else
#define	NAN_WORD0	0x7ff80000
#endif
#define	NAN_WORD1	0

#define Long	int32_t
#define ULong	u_int32_t

#ifdef DEBUG
#include "stdio.h"
#define Bug(x) {fprintf(stderr, "%s\n", x); exit(1);}
#define BugPrintf(x, v) {fprintf(stderr, x, v); exit(1);}
#endif

#ifdef __cplusplus
#include "malloc.h"
#include "memory.h"
#else
#ifndef KR_headers
#include "stdlib.h"
#include "string.h"
#ifndef ANDROID_CHANGES
#include "locale.h"
#endif /* ANDROID_CHANGES */
#else
#include "malloc.h"
#include "memory.h"
#endif
#endif
#ifndef ANDROID_CHANGES
#include "extern.h"
#include "reentrant.h"
#endif /* ANDROID_CHANGES */

#ifdef MALLOC
#ifdef KR_headers
extern char *MALLOC();
#else
extern void *MALLOC(size_t);
#endif
#else
#define MALLOC malloc
#endif

#include "ctype.h"
#include "errno.h"
#include "float.h"

#ifndef __MATH_H__
#include "math.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CONST
#ifdef KR_headers
#define CONST /* blank */
#else
#define CONST const
#endif
#endif

#ifdef Unsigned_Shifts
#define Sign_Extend(a,b) if (b < 0) a |= 0xffff0000;
#else
#define Sign_Extend(a,b) /*no-op*/
#endif

#if defined(IEEE_LITTLE_ENDIAN) + defined(IEEE_BIG_ENDIAN) + defined(VAX) + \
    defined(IBM) != 1
Exactly one of IEEE_LITTLE_ENDIAN IEEE_BIG_ENDIAN, VAX, or
IBM should be defined.
#endif

typedef union {
	double d;
	ULong ul[2];
} _double;
#define value(x) ((x).d)
#ifdef IEEE_LITTLE_ENDIAN
#define word0(x) ((x).ul[1])
#define word1(x) ((x).ul[0])
#else
#define word0(x) ((x).ul[0])
#define word1(x) ((x).ul[1])
#endif

/* The following definition of Storeinc is appropriate for MIPS processors.
 * An alternative that might be better on some machines is
 * #define Storeinc(a,b,c) (*a++ = b << 16 | c & 0xffff)
 */
#if defined(IEEE_LITTLE_ENDIAN) + defined(VAX) + defined(__arm__)
#define Storeinc(a,b,c) \
    (((u_short *)(void *)a)[1] = \
	(u_short)b, ((u_short *)(void *)a)[0] = (u_short)c, a++)
#else
#define Storeinc(a,b,c) \
    (((u_short *)(void *)a)[0] = \
	(u_short)b, ((u_short *)(void *)a)[1] = (u_short)c, a++)
#endif

/* #define P DBL_MANT_DIG */
/* Ten_pmax = floor(P*log(2)/log(5)) */
/* Bletch = (highest power of 2 < DBL_MAX_10_EXP) / 16 */
/* Quick_max = floor((P-1)*log(FLT_RADIX)/log(10) - 1) */
/* Int_max = floor(P*log(FLT_RADIX)/log(10) - 1) */

#if defined(IEEE_LITTLE_ENDIAN) + defined(IEEE_BIG_ENDIAN)
#define Exp_shift  20
#define Exp_shift1 20
#define Exp_msk1    0x100000
#define Exp_msk11   0x100000
#define Exp_mask  0x7ff00000
#define P 53
#define Bias 1023
#define IEEE_Arith
#define Emin (-1022)
#define Exp_1  0x3ff00000
#define Exp_11 0x3ff00000
#define Ebits 11
#define Frac_mask  0xfffff
#define Frac_mask1 0xfffff
#define Ten_pmax 22
#define Bletch 0x10
#define Bndry_mask  0xfffff
#define Bndry_mask1 0xfffff
#define LSB 1
#define Sign_bit 0x80000000
#define Log2P 1
#define Tiny0 0
#define Tiny1 1
#define Quick_max 14
#define Int_max 14
#define Infinite(x) (word0(x) == 0x7ff00000) /* sufficient test for here */
#else
#undef  Sudden_Underflow
#define Sudden_Underflow
#ifdef IBM
#define Exp_shift  24
#define Exp_shift1 24
#define Exp_msk1   0x1000000
#define Exp_msk11  0x1000000
#define Exp_mask  0x7f000000
#define P 14
#define Bias 65
#define Exp_1  0x41000000
#define Exp_11 0x41000000
#define Ebits 8	/* exponent has 7 bits, but 8 is the right value in b2d */
#define Frac_mask  0xffffff
#define Frac_mask1 0xffffff
#define Bletch 4
#define Ten_pmax 22
#define Bndry_mask  0xefffff
#define Bndry_mask1 0xffffff
#define LSB 1
#define Sign_bit 0x80000000
#define Log2P 4
#define Tiny0 0x100000
#define Tiny1 0
#define Quick_max 14
#define Int_max 15
#else /* VAX */
#define Exp_shift  23
#define Exp_shift1 7
#define Exp_msk1    0x80
#define Exp_msk11   0x800000
#define Exp_mask  0x7f80
#define P 56
#define Bias 129
#define Exp_1  0x40800000
#define Exp_11 0x4080
#define Ebits 8
#define Frac_mask  0x7fffff
#define Frac_mask1 0xffff007f
#define Ten_pmax 24
#define Bletch 2
#define Bndry_mask  0xffff007f
#define Bndry_mask1 0xffff007f
#define LSB 0x10000
#define Sign_bit 0x8000
#define Log2P 1
#define Tiny0 0x80
#define Tiny1 0
#define Quick_max 15
#define Int_max 15
#endif
#endif

#ifndef IEEE_Arith
#define ROUND_BIASED
#endif

#ifdef RND_PRODQUOT
#define rounded_product(a,b) a = rnd_prod(a, b)
#define rounded_quotient(a,b) a = rnd_quot(a, b)
#ifdef KR_headers
extern double rnd_prod(), rnd_quot();
#else
extern double rnd_prod(double, double), rnd_quot(double, double);
#endif
#else
#define rounded_product(a,b) a *= b
#define rounded_quotient(a,b) a /= b
#endif

#define Big0 (Frac_mask1 | Exp_msk1*(DBL_MAX_EXP+Bias-1))
#define Big1 0xffffffff

#ifndef Just_16
/* When Pack_32 is not defined, we store 16 bits per 32-bit Long.
 * This makes some inner loops simpler and sometimes saves work
 * during multiplications, but it often seems to make things slightly
 * slower.  Hence the default is now to store 32 bits per Long.
 */
#ifndef Pack_32
#define Pack_32
#endif
#endif

#define Kmax 15

#ifdef Pack_32
#define ULbits 32
#define kshift 5
#define kmask 31
#define ALL_ON 0xffffffff
#else
#define ULbits 16
#define kshift 4
#define kmask 15
#define ALL_ON 0xffff
#endif

#define Kmax 15

 enum {	/* return values from strtodg */
	STRTOG_Zero	= 0,
	STRTOG_Normal	= 1,
	STRTOG_Denormal	= 2,
	STRTOG_Infinite	= 3,
	STRTOG_NaN	= 4,
	STRTOG_NaNbits	= 5,
	STRTOG_NoNumber	= 6,
	STRTOG_Retmask	= 7,

	/* The following may be or-ed into one of the above values. */

	STRTOG_Neg	= 0x08, /* does not affect STRTOG_Inexlo or STRTOG_Inexhi */
	STRTOG_Inexlo	= 0x10,	/* returned result rounded toward zero */
	STRTOG_Inexhi	= 0x20, /* returned result rounded away from zero */
	STRTOG_Inexact	= 0x30,
	STRTOG_Underflow= 0x40,
	STRTOG_Overflow	= 0x80
	};

 typedef struct
FPI {
	int nbits;
	int emin;
	int emax;
	int rounding;
	int sudden_underflow;
	} FPI;

enum {	/* FPI.rounding values: same as FLT_ROUNDS */
	FPI_Round_zero = 0,
	FPI_Round_near = 1,
	FPI_Round_up = 2,
	FPI_Round_down = 3
	};

#undef SI
#ifdef Sudden_Underflow
#define SI 1
#else
#define SI 0
#endif

#ifdef __cplusplus
extern "C" double strtod(const char *s00, char **se);
extern "C" char *__dtoa(double d, int mode, int ndigits,
			int *decpt, int *sign, char **rve);
#endif

 struct
Bigint {
	struct Bigint *next;
	int k, maxwds, sign, wds;
	ULong x[1];
};

 typedef struct Bigint Bigint;

CONST unsigned char hexdig[256] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0, 0, 0, 0, 0, 0,
	0, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static int
gethex(CONST char **, CONST FPI *, Long *, Bigint **, int, locale_t);


 static Bigint *freelist[Kmax+1];

#ifdef ANDROID_CHANGES
 static pthread_mutex_t freelist_mutex = PTHREAD_MUTEX_INITIALIZER;
#else
#ifdef _REENTRANT
 static mutex_t freelist_mutex = MUTEX_INITIALIZER;
#endif
#endif

/* Special value used to indicate an invalid Bigint value,
 * e.g. when a memory allocation fails. The idea is that we
 * want to avoid introducing NULL checks everytime a bigint
 * computation is performed. Also the NULL value can also be
 * already used to indicate "value not initialized yet" and
 * returning NULL might alter the execution code path in
 * case of OOM.
 */
#define  BIGINT_INVALID   ((Bigint *)&bigint_invalid_value)

static const Bigint bigint_invalid_value;


static void
copybits(ULong *c, int n, Bigint *b)
{
	ULong *ce, *x, *xe;
#ifdef Pack_16
	int nw, nw1;
#endif

	ce = c + ((n-1) >> kshift) + 1;
	x = b->x;
#ifdef Pack_32
	xe = x + b->wds;
	while(x < xe)
		*c++ = *x++;
#else
	nw = b->wds;
	nw1 = nw & 1;
	for(xe = x + (nw - nw1); x < xe; x += 2)
		Storeinc(c, x[1], x[0]);
	if (nw1)
		*c++ = *x;
#endif
	while(c < ce)
		*c++ = 0;
	}

 ULong
any_on(Bigint *b, int k)
{
	int n, nwds;
	ULong *x, *x0, x1, x2;

	x = b->x;
	nwds = b->wds;
	n = k >> kshift;
	if (n > nwds)
		n = nwds;
	else if (n < nwds && (k &= kmask)) {
		x1 = x2 = x[n];
		x1 >>= k;
		x1 <<= k;
		if (x1 != x2)
			return 1;
		}
	x0 = x;
	x += n;
	while(x > x0)
		if (*--x)
			return 1;
	return 0;
	}

 void
rshift(Bigint *b, int k)
{
	ULong *x, *x1, *xe, y;
	int n;

	x = x1 = b->x;
	n = k >> kshift;
	if (n < b->wds) {
		xe = x + b->wds;
		x += n;
		if (k &= kmask) {
			n = ULbits - k;
			y = *x++ >> k;
			while(x < xe) {
				*x1++ = (y | (*x << n)) & ALL_ON;
				y = *x++ >> k;
				}
			if ((*x1 = y) !=0)
				x1++;
			}
		else
			while(x < xe)
				*x1++ = *x++;
		}
	if ((b->wds = x1 - b->x) == 0)
		b->x[0] = 0;
	}


typedef union { double d; ULong L[2]; } U;

static void
ULtod(ULong *L, ULong *bits, Long exp, int k)
{
#  define _0 1
#  define _1 0

	switch(k & STRTOG_Retmask) {
	  case STRTOG_NoNumber:
	  case STRTOG_Zero:
		L[0] = L[1] = 0;
		break;

	  case STRTOG_Denormal:
		L[_1] = bits[0];
		L[_0] = bits[1];
		break;

	  case STRTOG_Normal:
	  case STRTOG_NaNbits:
		L[_1] = bits[0];
		L[_0] = (bits[1] & ~0x100000) | ((exp + 0x3ff + 52) << 20);
		break;

	  case STRTOG_Infinite:
		L[_0] = 0x7ff00000;
		L[_1] = 0;
		break;

#define d_QNAN0 0x7ff80000
#define d_QNAN1 0x0
	  case STRTOG_NaN:
		L[0] = d_QNAN0;
		L[1] = d_QNAN1;
	  }
	if (k & STRTOG_Neg)
		L[_0] |= 0x80000000L;
	}



/* Return BIGINT_INVALID on allocation failure.
 *
 * Most of the code here depends on the fact that this function
 * never returns NULL.
 */
 static Bigint *
Balloc
#ifdef KR_headers
	(k) int k;
#else
	(int k)
#endif
{
	int x;
	Bigint *rv;

	mutex_lock(&freelist_mutex);

	if ((rv = freelist[k]) != NULL) {
		freelist[k] = rv->next;
	}
	else {
		x = 1 << k;
		rv = (Bigint *)MALLOC(sizeof(Bigint) + (x-1)*sizeof(Long));
		if (rv == NULL) {
		        rv = BIGINT_INVALID;
			goto EXIT;
		}
		rv->k = k;
		rv->maxwds = x;
	}
	rv->sign = rv->wds = 0;
EXIT:
	mutex_unlock(&freelist_mutex);

	return rv;
}

 static void
Bfree
#ifdef KR_headers
	(v) Bigint *v;
#else
	(Bigint *v)
#endif
{
	if (v && v != BIGINT_INVALID) {
		mutex_lock(&freelist_mutex);

		v->next = freelist[v->k];
		freelist[v->k] = v;

		mutex_unlock(&freelist_mutex);
	}
}

#define Bcopy_valid(x,y) memcpy(&(x)->sign, &(y)->sign, \
    (y)->wds*sizeof(Long) + 2*sizeof(int))

#define Bcopy(x,y)  Bcopy_ptr(&(x),(y))

 static void
Bcopy_ptr(Bigint **px, Bigint *y)
{
	if (*px == BIGINT_INVALID)
		return; /* no space to store copy */
	if (y == BIGINT_INVALID) {
		Bfree(*px); /* invalid input */
		*px = BIGINT_INVALID;
	} else {
		Bcopy_valid(*px,y);
	}
}

 static Bigint *
multadd
#ifdef KR_headers
	(b, m, a) Bigint *b; int m, a;
#else
	(Bigint *b, int m, int a)	/* multiply by m and add a */
#endif
{
	int i, wds;
	ULong *x, y;
#ifdef Pack_32
	ULong xi, z;
#endif
	Bigint *b1;

	if (b == BIGINT_INVALID)
		return b;

	wds = b->wds;
	x = b->x;
	i = 0;
	do {
#ifdef Pack_32
		xi = *x;
		y = (xi & 0xffff) * m + a;
		z = (xi >> 16) * m + (y >> 16);
		a = (int)(z >> 16);
		*x++ = (z << 16) + (y & 0xffff);
#else
		y = *x * m + a;
		a = (int)(y >> 16);
		*x++ = y & 0xffff;
#endif
	}
	while(++i < wds);
	if (a) {
		if (wds >= b->maxwds) {
			b1 = Balloc(b->k+1);
			if (b1 == BIGINT_INVALID) {
				Bfree(b);
				return b1;
			}
			Bcopy_valid(b1, b);
			Bfree(b);
			b = b1;
			}
		b->x[wds++] = a;
		b->wds = wds;
	}
	return b;
}

 Bigint *
increment(Bigint *b)
{
	ULong *x, *xe;
	Bigint *b1;
#ifdef Pack_16
	ULong carry = 1, y;
#endif

	x = b->x;
	xe = x + b->wds;
#ifdef Pack_32
	do {
		if (*x < (ULong)0xffffffffL) {
			++*x;
			return b;
			}
		*x++ = 0;
		} while(x < xe);
#else
	do {
		y = *x + carry;
		carry = y >> 16;
		*x++ = y & 0xffff;
		if (!carry)
			return b;
		} while(x < xe);
	if (carry)
#endif
	{
		if (b->wds >= b->maxwds) {
			b1 = Balloc(b->k+1);
			Bcopy(b1,b);
			Bfree(b);
			b = b1;
			}
		b->x[b->wds++] = 1;
		}
	return b;
	}


 static Bigint *
s2b
#ifdef KR_headers
	(s, nd0, nd, y9) CONST char *s; int nd0, nd; ULong y9;
#else
	(CONST char *s, int nd0, int nd, ULong y9)
#endif
{
	Bigint *b;
	int i, k;
	Long x, y;

	x = (nd + 8) / 9;
	for(k = 0, y = 1; x > y; y <<= 1, k++) ;
#ifdef Pack_32
	b = Balloc(k);
	if (b == BIGINT_INVALID)
		return b;
	b->x[0] = y9;
	b->wds = 1;
#else
	b = Balloc(k+1);
	if (b == BIGINT_INVALID)
		return b;

	b->x[0] = y9 & 0xffff;
	b->wds = (b->x[1] = y9 >> 16) ? 2 : 1;
#endif

	i = 9;
	if (9 < nd0) {
		s += 9;
		do b = multadd(b, 10, *s++ - '0');
			while(++i < nd0);
		s++;
	}
	else
		s += 10;
	for(; i < nd; i++)
		b = multadd(b, 10, *s++ - '0');
	return b;
}

 static int
hi0bits
#ifdef KR_headers
	(x) ULong x;
#else
	(ULong x)
#endif
{
	int k = 0;

	if (!(x & 0xffff0000)) {
		k = 16;
		x <<= 16;
	}
	if (!(x & 0xff000000)) {
		k += 8;
		x <<= 8;
	}
	if (!(x & 0xf0000000)) {
		k += 4;
		x <<= 4;
	}
	if (!(x & 0xc0000000)) {
		k += 2;
		x <<= 2;
	}
	if (!(x & 0x80000000)) {
		k++;
		if (!(x & 0x40000000))
			return 32;
	}
	return k;
}

 static int
lo0bits
#ifdef KR_headers
	(y) ULong *y;
#else
	(ULong *y)
#endif
{
	int k;
	ULong x = *y;

	if (x & 7) {
		if (x & 1)
			return 0;
		if (x & 2) {
			*y = x >> 1;
			return 1;
			}
		*y = x >> 2;
		return 2;
	}
	k = 0;
	if (!(x & 0xffff)) {
		k = 16;
		x >>= 16;
	}
	if (!(x & 0xff)) {
		k += 8;
		x >>= 8;
	}
	if (!(x & 0xf)) {
		k += 4;
		x >>= 4;
	}
	if (!(x & 0x3)) {
		k += 2;
		x >>= 2;
	}
	if (!(x & 1)) {
		k++;
		x >>= 1;
		if (!x & 1)
			return 32;
	}
	*y = x;
	return k;
}

 static Bigint *
i2b
#ifdef KR_headers
	(i) int i;
#else
	(int i)
#endif
{
	Bigint *b;

	b = Balloc(1);
	if (b != BIGINT_INVALID) {
		b->x[0] = i;
		b->wds = 1;
		}
	return b;
}

 static Bigint *
mult
#ifdef KR_headers
	(a, b) Bigint *a, *b;
#else
	(Bigint *a, Bigint *b)
#endif
{
	Bigint *c;
	int k, wa, wb, wc;
	ULong carry, y, z;
	ULong *x, *xa, *xae, *xb, *xbe, *xc, *xc0;
#ifdef Pack_32
	ULong z2;
#endif

	if (a == BIGINT_INVALID || b == BIGINT_INVALID)
		return BIGINT_INVALID;

	if (a->wds < b->wds) {
		c = a;
		a = b;
		b = c;
	}
	k = a->k;
	wa = a->wds;
	wb = b->wds;
	wc = wa + wb;
	if (wc > a->maxwds)
		k++;
	c = Balloc(k);
	if (c == BIGINT_INVALID)
		return c;
	for(x = c->x, xa = x + wc; x < xa; x++)
		*x = 0;
	xa = a->x;
	xae = xa + wa;
	xb = b->x;
	xbe = xb + wb;
	xc0 = c->x;
#ifdef Pack_32
	for(; xb < xbe; xb++, xc0++) {
		if ((y = *xb & 0xffff) != 0) {
			x = xa;
			xc = xc0;
			carry = 0;
			do {
				z = (*x & 0xffff) * y + (*xc & 0xffff) + carry;
				carry = z >> 16;
				z2 = (*x++ >> 16) * y + (*xc >> 16) + carry;
				carry = z2 >> 16;
				Storeinc(xc, z2, z);
			}
			while(x < xae);
			*xc = carry;
		}
		if ((y = *xb >> 16) != 0) {
			x = xa;
			xc = xc0;
			carry = 0;
			z2 = *xc;
			do {
				z = (*x & 0xffff) * y + (*xc >> 16) + carry;
				carry = z >> 16;
				Storeinc(xc, z, z2);
				z2 = (*x++ >> 16) * y + (*xc & 0xffff) + carry;
				carry = z2 >> 16;
			}
			while(x < xae);
			*xc = z2;
		}
	}
#else
	for(; xb < xbe; xc0++) {
		if (y = *xb++) {
			x = xa;
			xc = xc0;
			carry = 0;
			do {
				z = *x++ * y + *xc + carry;
				carry = z >> 16;
				*xc++ = z & 0xffff;
			}
			while(x < xae);
			*xc = carry;
		}
	}
#endif
	for(xc0 = c->x, xc = xc0 + wc; wc > 0 && !*--xc; --wc) ;
	c->wds = wc;
	return c;
}

 static Bigint *p5s;
 static pthread_mutex_t p5s_mutex = PTHREAD_MUTEX_INITIALIZER;

 static Bigint *
pow5mult
#ifdef KR_headers
	(b, k) Bigint *b; int k;
#else
	(Bigint *b, int k)
#endif
{
	Bigint *b1, *p5, *p51;
	int i;
	static const int p05[3] = { 5, 25, 125 };

	if (b == BIGINT_INVALID)
		return b;

	if ((i = k & 3) != 0)
		b = multadd(b, p05[i-1], 0);

	if (!(k = (unsigned int) k >> 2))
		return b;
	mutex_lock(&p5s_mutex);
	if (!(p5 = p5s)) {
		/* first time */
		p5 = i2b(625);
		if (p5 == BIGINT_INVALID) {
			Bfree(b);
			mutex_unlock(&p5s_mutex);
			return p5;
		}
		p5s = p5;
		p5->next = 0;
	}
	for(;;) {
		if (k & 1) {
			b1 = mult(b, p5);
			Bfree(b);
			b = b1;
		}
		if (!(k = (unsigned int) k >> 1))
			break;
		if (!(p51 = p5->next)) {
			p51 = mult(p5,p5);
			if (p51 == BIGINT_INVALID) {
				Bfree(b);
				mutex_unlock(&p5s_mutex);
				return p51;
			}
			p5->next = p51;
			p51->next = 0;
		}
		p5 = p51;
	}
	mutex_unlock(&p5s_mutex);
	return b;
}

 static Bigint *
lshift
#ifdef KR_headers
	(b, k) Bigint *b; int k;
#else
	(Bigint *b, int k)
#endif
{
	int i, k1, n, n1;
	Bigint *b1;
	ULong *x, *x1, *xe, z;

	if (b == BIGINT_INVALID)
		return b;

#ifdef Pack_32
	n = (unsigned int)k >> 5;
#else
	n = (unsigned int)k >> 4;
#endif
	k1 = b->k;
	n1 = n + b->wds + 1;
	for(i = b->maxwds; n1 > i; i <<= 1)
		k1++;
	b1 = Balloc(k1);
	if (b1 == BIGINT_INVALID) {
		Bfree(b);
		return b1;
	}
	x1 = b1->x;
	for(i = 0; i < n; i++)
		*x1++ = 0;
	x = b->x;
	xe = x + b->wds;
#ifdef Pack_32
	if (k &= 0x1f) {
		k1 = 32 - k;
		z = 0;
		do {
			*x1++ = *x << k | z;
			z = *x++ >> k1;
		}
		while(x < xe);
		if ((*x1 = z) != 0)
			++n1;
	}
#else
	if (k &= 0xf) {
		k1 = 16 - k;
		z = 0;
		do {
			*x1++ = *x << k  & 0xffff | z;
			z = *x++ >> k1;
		}
		while(x < xe);
		if (*x1 = z)
			++n1;
	}
#endif
	else do
		*x1++ = *x++;
		while(x < xe);
	b1->wds = n1 - 1;
	Bfree(b);
	return b1;
}

 static int
cmp
#ifdef KR_headers
	(a, b) Bigint *a, *b;
#else
	(Bigint *a, Bigint *b)
#endif
{
	ULong *xa, *xa0, *xb, *xb0;
	int i, j;

	if (a == BIGINT_INVALID || b == BIGINT_INVALID)
#ifdef DEBUG
		Bug("cmp called with a or b invalid");
#else
		return 0; /* equal - the best we can do right now */
#endif

	i = a->wds;
	j = b->wds;
#ifdef DEBUG
	if (i > 1 && !a->x[i-1])
		Bug("cmp called with a->x[a->wds-1] == 0");
	if (j > 1 && !b->x[j-1])
		Bug("cmp called with b->x[b->wds-1] == 0");
#endif
	if (i -= j)
		return i;
	xa0 = a->x;
	xa = xa0 + j;
	xb0 = b->x;
	xb = xb0 + j;
	for(;;) {
		if (*--xa != *--xb)
			return *xa < *xb ? -1 : 1;
		if (xa <= xa0)
			break;
	}
	return 0;
}

 static Bigint *
diff
#ifdef KR_headers
	(a, b) Bigint *a, *b;
#else
	(Bigint *a, Bigint *b)
#endif
{
	Bigint *c;
	int i, wa, wb;
	Long borrow, y;	/* We need signed shifts here. */
	ULong *xa, *xae, *xb, *xbe, *xc;
#ifdef Pack_32
	Long z;
#endif

	if (a == BIGINT_INVALID || b == BIGINT_INVALID)
		return BIGINT_INVALID;

	i = cmp(a,b);
	if (!i) {
		c = Balloc(0);
		if (c != BIGINT_INVALID) {
			c->wds = 1;
			c->x[0] = 0;
			}
		return c;
	}
	if (i < 0) {
		c = a;
		a = b;
		b = c;
		i = 1;
	}
	else
		i = 0;
	c = Balloc(a->k);
	if (c == BIGINT_INVALID)
		return c;
	c->sign = i;
	wa = a->wds;
	xa = a->x;
	xae = xa + wa;
	wb = b->wds;
	xb = b->x;
	xbe = xb + wb;
	xc = c->x;
	borrow = 0;
#ifdef Pack_32
	do {
		y = (*xa & 0xffff) - (*xb & 0xffff) + borrow;
		borrow = (ULong)y >> 16;
		Sign_Extend(borrow, y);
		z = (*xa++ >> 16) - (*xb++ >> 16) + borrow;
		borrow = (ULong)z >> 16;
		Sign_Extend(borrow, z);
		Storeinc(xc, z, y);
	}
	while(xb < xbe);
	while(xa < xae) {
		y = (*xa & 0xffff) + borrow;
		borrow = (ULong)y >> 16;
		Sign_Extend(borrow, y);
		z = (*xa++ >> 16) + borrow;
		borrow = (ULong)z >> 16;
		Sign_Extend(borrow, z);
		Storeinc(xc, z, y);
	}
#else
	do {
		y = *xa++ - *xb++ + borrow;
		borrow = y >> 16;
		Sign_Extend(borrow, y);
		*xc++ = y & 0xffff;
	}
	while(xb < xbe);
	while(xa < xae) {
		y = *xa++ + borrow;
		borrow = y >> 16;
		Sign_Extend(borrow, y);
		*xc++ = y & 0xffff;
	}
#endif
	while(!*--xc)
		wa--;
	c->wds = wa;
	return c;
}

 static double
ulp
#ifdef KR_headers
	(_x) double _x;
#else
	(double _x)
#endif
{
	_double x;
	Long L;
	_double a;

	value(x) = _x;
	L = (word0(x) & Exp_mask) - (P-1)*Exp_msk1;
#ifndef Sudden_Underflow
	if (L > 0) {
#endif
#ifdef IBM
		L |= Exp_msk1 >> 4;
#endif
		word0(a) = L;
		word1(a) = 0;
#ifndef Sudden_Underflow
	}
	else {
		L = (ULong)-L >> Exp_shift;
		if (L < Exp_shift) {
			word0(a) = 0x80000 >> L;
			word1(a) = 0;
		}
		else {
			word0(a) = 0;
			L -= Exp_shift;
			word1(a) = L >= 31 ? 1 : 1 << (31 - L);
		}
	}
#endif
	return value(a);
}

 static double
b2d
#ifdef KR_headers
	(a, e) Bigint *a; int *e;
#else
	(Bigint *a, int *e)
#endif
{
	ULong *xa, *xa0, w, y, z;
	int k;
	_double d;
#ifdef VAX
	ULong d0, d1;
#else
#define d0 word0(d)
#define d1 word1(d)
#endif

	if (a == BIGINT_INVALID)
		return NAN;

	xa0 = a->x;
	xa = xa0 + a->wds;
	y = *--xa;
#ifdef DEBUG
	if (!y) Bug("zero y in b2d");
#endif
	k = hi0bits(y);
	*e = 32 - k;
#ifdef Pack_32
	if (k < Ebits) {
		d0 = Exp_1 | y >> (Ebits - k);
		w = xa > xa0 ? *--xa : 0;
		d1 = y << ((32-Ebits) + k) | w >> (Ebits - k);
		goto ret_d;
	}
	z = xa > xa0 ? *--xa : 0;
	if (k -= Ebits) {
		d0 = Exp_1 | y << k | z >> (32 - k);
		y = xa > xa0 ? *--xa : 0;
		d1 = z << k | y >> (32 - k);
	}
	else {
		d0 = Exp_1 | y;
		d1 = z;
	}
#else
	if (k < Ebits + 16) {
		z = xa > xa0 ? *--xa : 0;
		d0 = Exp_1 | y << k - Ebits | z >> Ebits + 16 - k;
		w = xa > xa0 ? *--xa : 0;
		y = xa > xa0 ? *--xa : 0;
		d1 = z << k + 16 - Ebits | w << k - Ebits | y >> 16 + Ebits - k;
		goto ret_d;
	}
	z = xa > xa0 ? *--xa : 0;
	w = xa > xa0 ? *--xa : 0;
	k -= Ebits + 16;
	d0 = Exp_1 | y << k + 16 | z << k | w >> 16 - k;
	y = xa > xa0 ? *--xa : 0;
	d1 = w << k + 16 | y << k;
#endif
 ret_d:
#ifdef VAX
	word0(d) = d0 >> 16 | d0 << 16;
	word1(d) = d1 >> 16 | d1 << 16;
#else
#undef d0
#undef d1
#endif
	return value(d);
}

 static Bigint *
d2b
#ifdef KR_headers
	(_d, e, bits) double d; int *e, *bits;
#else
	(double _d, int *e, int *bits)
#endif
{
	Bigint *b;
	int de, i, k;
	ULong *x, y, z;
	_double d;
#ifdef VAX
	ULong d0, d1;
#endif

	value(d) = _d;
#ifdef VAX
	d0 = word0(d) >> 16 | word0(d) << 16;
	d1 = word1(d) >> 16 | word1(d) << 16;
#else
#define d0 word0(d)
#define d1 word1(d)
#endif

#ifdef Pack_32
	b = Balloc(1);
#else
	b = Balloc(2);
#endif
	if (b == BIGINT_INVALID)
		return b;
	x = b->x;

	z = d0 & Frac_mask;
	d0 &= 0x7fffffff;	/* clear sign bit, which we ignore */
#ifdef Sudden_Underflow
	de = (int)(d0 >> Exp_shift);
#ifndef IBM
	z |= Exp_msk11;
#endif
#else
	if ((de = (int)(d0 >> Exp_shift)) != 0)
		z |= Exp_msk1;
#endif
#ifdef Pack_32
	if ((y = d1) != 0) {
		if ((k = lo0bits(&y)) != 0) {
			x[0] = y | z << (32 - k);
			z >>= k;
		}
		else
			x[0] = y;
		i = b->wds = (x[1] = z) ? 2 : 1;
	}
	else {
#ifdef DEBUG
		if (!z)
			Bug("Zero passed to d2b");
#endif
		k = lo0bits(&z);
		x[0] = z;
		i = b->wds = 1;
		k += 32;
	}
#else
	if (y = d1) {
		if (k = lo0bits(&y))
			if (k >= 16) {
				x[0] = y | z << 32 - k & 0xffff;
				x[1] = z >> k - 16 & 0xffff;
				x[2] = z >> k;
				i = 2;
			}
			else {
				x[0] = y & 0xffff;
				x[1] = y >> 16 | z << 16 - k & 0xffff;
				x[2] = z >> k & 0xffff;
				x[3] = z >> k+16;
				i = 3;
			}
		else {
			x[0] = y & 0xffff;
			x[1] = y >> 16;
			x[2] = z & 0xffff;
			x[3] = z >> 16;
			i = 3;
		}
	}
	else {
#ifdef DEBUG
		if (!z)
			Bug("Zero passed to d2b");
#endif
		k = lo0bits(&z);
		if (k >= 16) {
			x[0] = z;
			i = 0;
		}
		else {
			x[0] = z & 0xffff;
			x[1] = z >> 16;
			i = 1;
		}
		k += 32;
	}
	while(!x[i])
		--i;
	b->wds = i + 1;
#endif
#ifndef Sudden_Underflow
	if (de) {
#endif
#ifdef IBM
		*e = (de - Bias - (P-1) << 2) + k;
		*bits = 4*P + 8 - k - hi0bits(word0(d) & Frac_mask);
#else
		*e = de - Bias - (P-1) + k;
		*bits = P - k;
#endif
#ifndef Sudden_Underflow
	}
	else {
		*e = de - Bias - (P-1) + 1 + k;
#ifdef Pack_32
		*bits = 32*i - hi0bits(x[i-1]);
#else
		*bits = (i+2)*16 - hi0bits(x[i]);
#endif
		}
#endif
	return b;
}
#undef d0
#undef d1

 static double
ratio
#ifdef KR_headers
	(a, b) Bigint *a, *b;
#else
	(Bigint *a, Bigint *b)
#endif
{
	_double da, db;
	int k, ka, kb;

	if (a == BIGINT_INVALID || b == BIGINT_INVALID)
		return NAN; /* for lack of better value ? */

	value(da) = b2d(a, &ka);
	value(db) = b2d(b, &kb);
#ifdef Pack_32
	k = ka - kb + 32*(a->wds - b->wds);
#else
	k = ka - kb + 16*(a->wds - b->wds);
#endif
#ifdef IBM
	if (k > 0) {
		word0(da) += (k >> 2)*Exp_msk1;
		if (k &= 3)
			da *= 1 << k;
	}
	else {
		k = -k;
		word0(db) += (k >> 2)*Exp_msk1;
		if (k &= 3)
			db *= 1 << k;
	}
#else
	if (k > 0)
		word0(da) += k*Exp_msk1;
	else {
		k = -k;
		word0(db) += k*Exp_msk1;
	}
#endif
	return value(da) / value(db);
}

static CONST double
tens[] = {
		1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
		1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
		1e20, 1e21, 1e22
#ifdef VAX
		, 1e23, 1e24
#endif
};

#ifdef IEEE_Arith
static CONST double bigtens[] = { 1e16, 1e32, 1e64, 1e128, 1e256 };
static CONST double tinytens[] = { 1e-16, 1e-32, 1e-64, 1e-128, 1e-256 };
#define n_bigtens 5
#else
#ifdef IBM
static CONST double bigtens[] = { 1e16, 1e32, 1e64 };
static CONST double tinytens[] = { 1e-16, 1e-32, 1e-64 };
#define n_bigtens 3
#else
static CONST double bigtens[] = { 1e16, 1e32 };
static CONST double tinytens[] = { 1e-16, 1e-32 };
#define n_bigtens 2
#endif
#endif

 double
strtod
#ifdef KR_headers
	(s00, se) CONST char *s00; char **se;
#else
	(CONST char *s00, char **se)
#endif
{
	int bb2, bb5, bbe, bd2, bd5, bbbits, bs2, c, dsign,
		 e, e1, esign, i, j, k, nd, nd0, nf, nz, nz0, sign;
	CONST char *s, *s0, *s1;
	double aadj, aadj1, adj;
	_double rv, rv0;
	Long L;
	ULong y, z;
	Bigint *bb1, *bd0;
	Bigint *bb = NULL, *bd = NULL, *bs = NULL, *delta = NULL;/* pacify gcc */

#ifdef ANDROID_CHANGES
	CONST char decimal_point = '.';
#else /* ANDROID_CHANGES */
#ifndef KR_headers
	CONST char decimal_point = localeconv()->decimal_point[0];
#else
	CONST char decimal_point = '.';
#endif

#endif /* ANDROID_CHANGES */

	sign = nz0 = nz = 0;
	value(rv) = 0.;


	for(s = s00; isspace((unsigned char) *s); s++)
		;

	if (*s == '-') {
		sign = 1;
		s++;
	} else if (*s == '+') {
		s++;
	}

	if (*s == '\0') {
		s = s00;
		goto ret;
	}

	/* "INF" or "INFINITY" */
	if (tolower((unsigned char)*s) == 'i' && strncasecmp(s, "inf", 3) == 0) {
		if (strncasecmp(s + 3, "inity", 5) == 0)
			s += 8;
		else
			s += 3;

		value(rv) = HUGE_VAL;
		goto ret;
	}

#ifdef IEEE_Arith
	/* "NAN" or "NAN(n-char-sequence-opt)" */
	if (tolower((unsigned char)*s) == 'n' && strncasecmp(s, "nan", 3) == 0) {
		/* Build a quiet NaN. */
		word0(rv) = NAN_WORD0;
		word1(rv) = NAN_WORD1;
		s+= 3;

		/* Don't interpret (n-char-sequence-opt), for now. */
		if (*s == '(') {
			s0 = s;
			for (s++; *s != ')' && *s != '\0'; s++)
				;
			if (*s == ')')
				s++;	/* Skip over closing paren ... */
			else
				s = s0;	/* ... otherwise go back. */
		}

		goto ret;
	}
#endif

	if (*s == '0') {
#ifndef NO_HEX_FP /*{*/
		{
		static CONST FPI fpi = { 53, 1-1023-53+1, 2046-1023-53+1, 1, SI };
		Long exp;
		ULong bits[2];
		switch(s[1]) {
		  case 'x':
		  case 'X':
			{
#ifdef Honor_FLT_ROUNDS
			FPI fpi1 = fpi;
			fpi1.rounding = Rounding;
#else
#define fpi1 fpi
#endif
			switch((i = gethex(&s, &fpi1, &exp, &bb, sign, 0)) & STRTOG_Retmask) {
			  case STRTOG_NoNumber:
				s = s00;
				sign = 0;
			  case STRTOG_Zero:
				break;
			  default:
				if (bb) {
					copybits(bits, fpi.nbits, bb);
					Bfree(bb);
					}
				ULtod(((U*)&rv)->L, bits, exp, i);
			  }}
			goto ret;
		  }
		}
#endif /*}*/
		nz0 = 1;
		while(*++s == '0') ;
		if (!*s)
			goto ret;
	}
	s0 = s;
	y = z = 0;
	for(nd = nf = 0; (c = *s) >= '0' && c <= '9'; nd++, s++)
		if (nd < 9)
			y = 10*y + c - '0';
		else if (nd < 16)
			z = 10*z + c - '0';
	nd0 = nd;
	if (c == decimal_point) {
		c = *++s;
		if (!nd) {
			for(; c == '0'; c = *++s)
				nz++;
			if (c > '0' && c <= '9') {
				s0 = s;
				nf += nz;
				nz = 0;
				goto have_dig;
				}
			goto dig_done;
		}
		for(; c >= '0' && c <= '9'; c = *++s) {
 have_dig:
			nz++;
			if (c -= '0') {
				nf += nz;
				for(i = 1; i < nz; i++)
					if (nd++ < 9)
						y *= 10;
					else if (nd <= DBL_DIG + 1)
						z *= 10;
				if (nd++ < 9)
					y = 10*y + c;
				else if (nd <= DBL_DIG + 1)
					z = 10*z + c;
				nz = 0;
			}
		}
	}
 dig_done:
	e = 0;
	if (c == 'e' || c == 'E') {
		if (!nd && !nz && !nz0) {
			s = s00;
			goto ret;
		}
		s00 = s;
		esign = 0;
		switch(c = *++s) {
			case '-':
				esign = 1;
				/* FALLTHROUGH */
			case '+':
				c = *++s;
		}
		if (c >= '0' && c <= '9') {
			while(c == '0')
				c = *++s;
			if (c > '0' && c <= '9') {
				L = c - '0';
				s1 = s;
				while((c = *++s) >= '0' && c <= '9')
					L = 10*L + c - '0';
				if (s - s1 > 8 || L > 19999)
					/* Avoid confusion from exponents
					 * so large that e might overflow.
					 */
					e = 19999; /* safe for 16 bit ints */
				else
					e = (int)L;
				if (esign)
					e = -e;
			}
			else
				e = 0;
		}
		else
			s = s00;
	}
	if (!nd) {
		if (!nz && !nz0)
			s = s00;
		goto ret;
	}
	e1 = e -= nf;

	/* Now we have nd0 digits, starting at s0, followed by a
	 * decimal point, followed by nd-nd0 digits.  The number we're
	 * after is the integer represented by those digits times
	 * 10**e */

	if (!nd0)
		nd0 = nd;
	k = nd < DBL_DIG + 1 ? nd : DBL_DIG + 1;
	value(rv) = y;
	if (k > 9)
		value(rv) = tens[k - 9] * value(rv) + z;
	bd0 = 0;
	if (nd <= DBL_DIG
#ifndef RND_PRODQUOT
		&& FLT_ROUNDS == 1
#endif
		) {
		if (!e)
			goto ret;
		if (e > 0) {
			if (e <= Ten_pmax) {
#ifdef VAX
				goto vax_ovfl_check;
#else
				/* value(rv) = */ rounded_product(value(rv),
				    tens[e]);
				goto ret;
#endif
			}
			i = DBL_DIG - nd;
			if (e <= Ten_pmax + i) {
				/* A fancier test would sometimes let us do
				 * this for larger i values.
				 */
				e -= i;
				value(rv) *= tens[i];
#ifdef VAX
				/* VAX exponent range is so narrow we must
				 * worry about overflow here...
				 */
 vax_ovfl_check:
				word0(rv) -= P*Exp_msk1;
				/* value(rv) = */ rounded_product(value(rv),
				    tens[e]);
				if ((word0(rv) & Exp_mask)
				 > Exp_msk1*(DBL_MAX_EXP+Bias-1-P))
					goto ovfl;
				word0(rv) += P*Exp_msk1;
#else
				/* value(rv) = */ rounded_product(value(rv),
				    tens[e]);
#endif
				goto ret;
			}
		}
#ifndef Inaccurate_Divide
		else if (e >= -Ten_pmax) {
			/* value(rv) = */ rounded_quotient(value(rv),
			    tens[-e]);
			goto ret;
		}
#endif
	}
	e1 += nd - k;

	/* Get starting approximation = rv * 10**e1 */

	if (e1 > 0) {
		if ((i = e1 & 15) != 0)
			value(rv) *= tens[i];
		if (e1 &= ~15) {
			if (e1 > DBL_MAX_10_EXP) {
 ovfl:
				errno = ERANGE;
				value(rv) = HUGE_VAL;
				if (bd0)
					goto retfree;
				goto ret;
			}
			if ((e1 = (unsigned int)e1 >> 4) != 0) {
				for(j = 0; e1 > 1; j++,
				    e1 = (unsigned int)e1 >> 1)
					if (e1 & 1)
						value(rv) *= bigtens[j];
			/* The last multiplication could overflow. */
				word0(rv) -= P*Exp_msk1;
				value(rv) *= bigtens[j];
				if ((z = word0(rv) & Exp_mask)
				 > Exp_msk1*(DBL_MAX_EXP+Bias-P))
					goto ovfl;
				if (z > Exp_msk1*(DBL_MAX_EXP+Bias-1-P)) {
					/* set to largest number */
					/* (Can't trust DBL_MAX) */
					word0(rv) = Big0;
					word1(rv) = Big1;
					}
				else
					word0(rv) += P*Exp_msk1;
			}
		}
	}
	else if (e1 < 0) {
		e1 = -e1;
		if ((i = e1 & 15) != 0)
			value(rv) /= tens[i];
		if (e1 &= ~15) {
			e1 = (unsigned int)e1 >> 4;
			if (e1 >= 1 << n_bigtens)
				goto undfl;
			for(j = 0; e1 > 1; j++,
			    e1 = (unsigned int)e1 >> 1)
				if (e1 & 1)
					value(rv) *= tinytens[j];
			/* The last multiplication could underflow. */
			value(rv0) = value(rv);
			value(rv) *= tinytens[j];
			if (!value(rv)) {
				value(rv) = 2.*value(rv0);
				value(rv) *= tinytens[j];
				if (!value(rv)) {
 undfl:
					value(rv) = 0.;
					errno = ERANGE;
					if (bd0)
						goto retfree;
					goto ret;
				}
				word0(rv) = Tiny0;
				word1(rv) = Tiny1;
				/* The refinement below will clean
				 * this approximation up.
				 */
			}
		}
	}

	/* Now the hard part -- adjusting rv to the correct value.*/

	/* Put digits into bd: true value = bd * 10^e */

	bd0 = s2b(s0, nd0, nd, y);

	for(;;) {
		bd = Balloc(bd0->k);
		Bcopy(bd, bd0);
		bb = d2b(value(rv), &bbe, &bbbits);	/* rv = bb * 2^bbe */
		bs = i2b(1);

		if (e >= 0) {
			bb2 = bb5 = 0;
			bd2 = bd5 = e;
		}
		else {
			bb2 = bb5 = -e;
			bd2 = bd5 = 0;
		}
		if (bbe >= 0)
			bb2 += bbe;
		else
			bd2 -= bbe;
		bs2 = bb2;
#ifdef Sudden_Underflow
#ifdef IBM
		j = 1 + 4*P - 3 - bbbits + ((bbe + bbbits - 1) & 3);
#else
		j = P + 1 - bbbits;
#endif
#else
		i = bbe + bbbits - 1;	/* logb(rv) */
		if (i < Emin)	/* denormal */
			j = bbe + (P-Emin);
		else
			j = P + 1 - bbbits;
#endif
		bb2 += j;
		bd2 += j;
		i = bb2 < bd2 ? bb2 : bd2;
		if (i > bs2)
			i = bs2;
		if (i > 0) {
			bb2 -= i;
			bd2 -= i;
			bs2 -= i;
		}
		if (bb5 > 0) {
			bs = pow5mult(bs, bb5);
			bb1 = mult(bs, bb);
			Bfree(bb);
			bb = bb1;
		}
		if (bb2 > 0)
			bb = lshift(bb, bb2);
		if (bd5 > 0)
			bd = pow5mult(bd, bd5);
		if (bd2 > 0)
			bd = lshift(bd, bd2);
		if (bs2 > 0)
			bs = lshift(bs, bs2);
		delta = diff(bb, bd);
		dsign = delta->sign;
		delta->sign = 0;
		i = cmp(delta, bs);
		if (i < 0) {
			/* Error is less than half an ulp -- check for
			 * special case of mantissa a power of two.
			 */
			if (dsign || word1(rv) || word0(rv) & Bndry_mask)
				break;
			delta = lshift(delta,Log2P);
			if (cmp(delta, bs) > 0)
				goto drop_down;
			break;
		}
		if (i == 0) {
			/* exactly half-way between */
			if (dsign) {
				if ((word0(rv) & Bndry_mask1) == Bndry_mask1
				 &&  word1(rv) == 0xffffffff) {
					/*boundary case -- increment exponent*/
					word0(rv) = (word0(rv) & Exp_mask)
						+ Exp_msk1
#ifdef IBM
						| Exp_msk1 >> 4
#endif
						;
					word1(rv) = 0;
					break;
				}
			}
			else if (!(word0(rv) & Bndry_mask) && !word1(rv)) {
 drop_down:
				/* boundary case -- decrement exponent */
#ifdef Sudden_Underflow
				L = word0(rv) & Exp_mask;
#ifdef IBM
				if (L <  Exp_msk1)
#else
				if (L <= Exp_msk1)
#endif
					goto undfl;
				L -= Exp_msk1;
#else
				L = (word0(rv) & Exp_mask) - Exp_msk1;
#endif
				word0(rv) = L | Bndry_mask1;
				word1(rv) = 0xffffffff;
#ifdef IBM
				goto cont;
#else
				break;
#endif
			}
#ifndef ROUND_BIASED
			if (!(word1(rv) & LSB))
				break;
#endif
			if (dsign)
				value(rv) += ulp(value(rv));
#ifndef ROUND_BIASED
			else {
				value(rv) -= ulp(value(rv));
#ifndef Sudden_Underflow
				if (!value(rv))
					goto undfl;
#endif
			}
#endif
			break;
		}
		if ((aadj = ratio(delta, bs)) <= 2.) {
			if (dsign)
				aadj = aadj1 = 1.;
			else if (word1(rv) || word0(rv) & Bndry_mask) {
#ifndef Sudden_Underflow
				if (word1(rv) == Tiny1 && !word0(rv))
					goto undfl;
#endif
				aadj = 1.;
				aadj1 = -1.;
			}
			else {
				/* special case -- power of FLT_RADIX to be */
				/* rounded down... */

				if (aadj < 2./FLT_RADIX)
					aadj = 1./FLT_RADIX;
				else
					aadj *= 0.5;
				aadj1 = -aadj;
				}
		}
		else {
			aadj *= 0.5;
			aadj1 = dsign ? aadj : -aadj;
#ifdef Check_FLT_ROUNDS
			switch(FLT_ROUNDS) {
				case 2: /* towards +infinity */
					aadj1 -= 0.5;
					break;
				case 0: /* towards 0 */
				case 3: /* towards -infinity */
					aadj1 += 0.5;
			}
#else
			if (FLT_ROUNDS == 0)
				aadj1 += 0.5;
#endif
		}
		y = word0(rv) & Exp_mask;

		/* Check for overflow */

		if (y == Exp_msk1*(DBL_MAX_EXP+Bias-1)) {
			value(rv0) = value(rv);
			word0(rv) -= P*Exp_msk1;
			adj = aadj1 * ulp(value(rv));
			value(rv) += adj;
			if ((word0(rv) & Exp_mask) >=
					Exp_msk1*(DBL_MAX_EXP+Bias-P)) {
				if (word0(rv0) == Big0 && word1(rv0) == Big1)
					goto ovfl;
				word0(rv) = Big0;
				word1(rv) = Big1;
				goto cont;
			}
			else
				word0(rv) += P*Exp_msk1;
		}
		else {
#ifdef Sudden_Underflow
			if ((word0(rv) & Exp_mask) <= P*Exp_msk1) {
				value(rv0) = value(rv);
				word0(rv) += P*Exp_msk1;
				adj = aadj1 * ulp(value(rv));
				value(rv) += adj;
#ifdef IBM
				if ((word0(rv) & Exp_mask) <  P*Exp_msk1)
#else
				if ((word0(rv) & Exp_mask) <= P*Exp_msk1)
#endif
				{
					if (word0(rv0) == Tiny0
					 && word1(rv0) == Tiny1)
						goto undfl;
					word0(rv) = Tiny0;
					word1(rv) = Tiny1;
					goto cont;
				}
				else
					word0(rv) -= P*Exp_msk1;
				}
			else {
				adj = aadj1 * ulp(value(rv));
				value(rv) += adj;
			}
#else
			/* Compute adj so that the IEEE rounding rules will
			 * correctly round rv + adj in some half-way cases.
			 * If rv * ulp(rv) is denormalized (i.e.,
			 * y <= (P-1)*Exp_msk1), we must adjust aadj to avoid
			 * trouble from bits lost to denormalization;
			 * example: 1.2e-307 .
			 */
			if (y <= (P-1)*Exp_msk1 && aadj >= 1.) {
				aadj1 = (double)(int)(aadj + 0.5);
				if (!dsign)
					aadj1 = -aadj1;
			}
			adj = aadj1 * ulp(value(rv));
			value(rv) += adj;
#endif
		}
		z = word0(rv) & Exp_mask;
		if (y == z) {
			/* Can we stop now? */
			L = aadj;
			aadj -= L;
			/* The tolerances below are conservative. */
			if (dsign || word1(rv) || word0(rv) & Bndry_mask) {
				if (aadj < .4999999 || aadj > .5000001)
					break;
			}
			else if (aadj < .4999999/FLT_RADIX)
				break;
		}
 cont:
		Bfree(bb);
		Bfree(bd);
		Bfree(bs);
		Bfree(delta);
	}
 retfree:
	Bfree(bb);
	Bfree(bd);
	Bfree(bs);
	Bfree(bd0);
	Bfree(delta);
 ret:
	if (se)
		/* LINTED interface specification */
		*se = (char *)s;
	return sign ? -value(rv) : value(rv);
}

 static int
quorem
#ifdef KR_headers
	(b, S) Bigint *b, *S;
#else
	(Bigint *b, Bigint *S)
#endif
{
	int n;
	Long borrow, y;
	ULong carry, q, ys;
	ULong *bx, *bxe, *sx, *sxe;
#ifdef Pack_32
	Long z;
	ULong si, zs;
#endif

	if (b == BIGINT_INVALID || S == BIGINT_INVALID)
		return 0;

	n = S->wds;
#ifdef DEBUG
	/*debug*/ if (b->wds > n)
	/*debug*/	Bug("oversize b in quorem");
#endif
	if (b->wds < n)
		return 0;
	sx = S->x;
	sxe = sx + --n;
	bx = b->x;
	bxe = bx + n;
	q = *bxe / (*sxe + 1);	/* ensure q <= true quotient */
#ifdef DEBUG
	/*debug*/ if (q > 9)
	/*debug*/	Bug("oversized quotient in quorem");
#endif
	if (q) {
		borrow = 0;
		carry = 0;
		do {
#ifdef Pack_32
			si = *sx++;
			ys = (si & 0xffff) * q + carry;
			zs = (si >> 16) * q + (ys >> 16);
			carry = zs >> 16;
			y = (*bx & 0xffff) - (ys & 0xffff) + borrow;
			borrow = (ULong)y >> 16;
			Sign_Extend(borrow, y);
			z = (*bx >> 16) - (zs & 0xffff) + borrow;
			borrow = (ULong)z >> 16;
			Sign_Extend(borrow, z);
			Storeinc(bx, z, y);
#else
			ys = *sx++ * q + carry;
			carry = ys >> 16;
			y = *bx - (ys & 0xffff) + borrow;
			borrow = y >> 16;
			Sign_Extend(borrow, y);
			*bx++ = y & 0xffff;
#endif
		}
		while(sx <= sxe);
		if (!*bxe) {
			bx = b->x;
			while(--bxe > bx && !*bxe)
				--n;
			b->wds = n;
		}
	}
	if (cmp(b, S) >= 0) {
		q++;
		borrow = 0;
		carry = 0;
		bx = b->x;
		sx = S->x;
		do {
#ifdef Pack_32
			si = *sx++;
			ys = (si & 0xffff) + carry;
			zs = (si >> 16) + (ys >> 16);
			carry = zs >> 16;
			y = (*bx & 0xffff) - (ys & 0xffff) + borrow;
			borrow = (ULong)y >> 16;
			Sign_Extend(borrow, y);
			z = (*bx >> 16) - (zs & 0xffff) + borrow;
			borrow = (ULong)z >> 16;
			Sign_Extend(borrow, z);
			Storeinc(bx, z, y);
#else
			ys = *sx++ + carry;
			carry = ys >> 16;
			y = *bx - (ys & 0xffff) + borrow;
			borrow = y >> 16;
			Sign_Extend(borrow, y);
			*bx++ = y & 0xffff;
#endif
		}
		while(sx <= sxe);
		bx = b->x;
		bxe = bx + n;
		if (!*bxe) {
			while(--bxe > bx && !*bxe)
				--n;
			b->wds = n;
		}
	}
	return q;
}

/* freedtoa(s) must be used to free values s returned by dtoa
 * when MULTIPLE_THREADS is #defined.  It should be used in all cases,
 * but for consistency with earlier versions of dtoa, it is optional
 * when MULTIPLE_THREADS is not defined.
 */

void
#ifdef KR_headers
freedtoa(s) char *s;
#else
freedtoa(char *s)
#endif
{
	free(s);
}



/* dtoa for IEEE arithmetic (dmg): convert double to ASCII string.
 *
 * Inspired by "How to Print Floating-Point Numbers Accurately" by
 * Guy L. Steele, Jr. and Jon L. White [Proc. ACM SIGPLAN '90, pp. 92-101].
 *
 * Modifications:
 *	1. Rather than iterating, we use a simple numeric overestimate
 *	   to determine k = floor(log10(d)).  We scale relevant
 *	   quantities using O(log2(k)) rather than O(k) multiplications.
 *	2. For some modes > 2 (corresponding to ecvt and fcvt), we don't
 *	   try to generate digits strictly left to right.  Instead, we
 *	   compute with fewer bits and propagate the carry if necessary
 *	   when rounding the final digit up.  This is often faster.
 *	3. Under the assumption that input will be rounded nearest,
 *	   mode 0 renders 1e23 as 1e23 rather than 9.999999999999999e22.
 *	   That is, we allow equality in stopping tests when the
 *	   round-nearest rule will give the same floating-point value
 *	   as would satisfaction of the stopping test with strict
 *	   inequality.
 *	4. We remove common factors of powers of 2 from relevant
 *	   quantities.
 *	5. When converting floating-point integers less than 1e16,
 *	   we use floating-point arithmetic rather than resorting
 *	   to multiple-precision integers.
 *	6. When asked to produce fewer than 15 digits, we first try
 *	   to get by with floating-point arithmetic; we resort to
 *	   multiple-precision integer arithmetic only if we cannot
 *	   guarantee that the floating-point calculation has given
 *	   the correctly rounded result.  For k requested digits and
 *	   "uniformly" distributed input, the probability is
 *	   something like 10^(k-15) that we must resort to the Long
 *	   calculation.
 */

__LIBC_HIDDEN__  char *
__dtoa
#ifdef KR_headers
	(_d, mode, ndigits, decpt, sign, rve)
	double _d; int mode, ndigits, *decpt, *sign; char **rve;
#else
	(double _d, int mode, int ndigits, int *decpt, int *sign, char **rve)
#endif
{
 /*	Arguments ndigits, decpt, sign are similar to those
	of ecvt and fcvt; trailing zeros are suppressed from
	the returned string.  If not null, *rve is set to point
	to the end of the return value.  If d is +-Infinity or NaN,
	then *decpt is set to 9999.

	mode:
		0 ==> shortest string that yields d when read in
			and rounded to nearest.
		1 ==> like 0, but with Steele & White stopping rule;
			e.g. with IEEE P754 arithmetic , mode 0 gives
			1e23 whereas mode 1 gives 9.999999999999999e22.
		2 ==> max(1,ndigits) significant digits.  This gives a
			return value similar to that of ecvt, except
			that trailing zeros are suppressed.
		3 ==> through ndigits past the decimal point.  This
			gives a return value similar to that from fcvt,
			except that trailing zeros are suppressed, and
			ndigits can be negative.
		4-9 should give the same return values as 2-3, i.e.,
			4 <= mode <= 9 ==> same return as mode
			2 + (mode & 1).  These modes are mainly for
			debugging; often they run slower but sometimes
			faster than modes 2-3.
		4,5,8,9 ==> left-to-right digit generation.
		6-9 ==> don't try fast floating-point estimate
			(if applicable).

		Values of mode other than 0-9 are treated as mode 0.

		Sufficient space is allocated to the return value
		to hold the suppressed trailing zeros.
	*/

	int bbits, b2, b5, be, dig, i, ieps, ilim0,
		j, jj1, k, k0, k_check, leftright, m2, m5, s2, s5,
		try_quick;
	int ilim = 0, ilim1 = 0, spec_case = 0;	/* pacify gcc */
	Long L;
#ifndef Sudden_Underflow
	int denorm;
	ULong x;
#endif
	Bigint *b, *b1, *delta, *mhi, *S;
	Bigint *mlo = NULL; /* pacify gcc */
	double ds;
	char *s, *s0;
	Bigint *result = NULL;
	int result_k = 0;
	_double d, d2, eps;

	value(d) = _d;

	if (word0(d) & Sign_bit) {
		/* set sign for everything, including 0's and NaNs */
		*sign = 1;
		word0(d) &= ~Sign_bit;	/* clear sign bit */
	}
	else
		*sign = 0;

#if defined(IEEE_Arith) + defined(VAX)
#ifdef IEEE_Arith
	if ((word0(d) & Exp_mask) == Exp_mask)
#else
	if (word0(d)  == 0x8000)
#endif
	{
		/* Infinity or NaN */
		*decpt = 9999;
		s =
#ifdef IEEE_Arith
			!word1(d) && !(word0(d) & 0xfffff) ? "Infinity" :
#endif
				"NaN";
		result = Balloc(strlen(s)+1);
		if (result == BIGINT_INVALID)
			return NULL;
		s0 = (char *)(void *)result;
		strcpy(s0, s);
		if (rve)
			*rve =
#ifdef IEEE_Arith
				s0[3] ? s0 + 8 :
#endif
				s0 + 3;
		return s0;
	}
#endif
#ifdef IBM
	value(d) += 0; /* normalize */
#endif
	if (!value(d)) {
		*decpt = 1;
		result = Balloc(2);
		if (result == BIGINT_INVALID)
			return NULL;
		s0 = (char *)(void *)result;
		strcpy(s0, "0");
		if (rve)
			*rve = s0 + 1;
		return s0;
	}

	b = d2b(value(d), &be, &bbits);
#ifdef Sudden_Underflow
	i = (int)(word0(d) >> Exp_shift1 & (Exp_mask>>Exp_shift1));
#else
	if ((i = (int)(word0(d) >> Exp_shift1 & (Exp_mask>>Exp_shift1))) != 0) {
#endif
		value(d2) = value(d);
		word0(d2) &= Frac_mask1;
		word0(d2) |= Exp_11;
#ifdef IBM
		if (j = 11 - hi0bits(word0(d2) & Frac_mask))
			value(d2) /= 1 << j;
#endif

		/* log(x)	~=~ log(1.5) + (x-1.5)/1.5
		 * log10(x)	 =  log(x) / log(10)
		 *		~=~ log(1.5)/log(10) + (x-1.5)/(1.5*log(10))
		 * log10(d) = (i-Bias)*log(2)/log(10) + log10(d2)
		 *
		 * This suggests computing an approximation k to log10(d) by
		 *
		 * k = (i - Bias)*0.301029995663981
		 *	+ ( (d2-1.5)*0.289529654602168 + 0.176091259055681 );
		 *
		 * We want k to be too large rather than too small.
		 * The error in the first-order Taylor series approximation
		 * is in our favor, so we just round up the constant enough
		 * to compensate for any error in the multiplication of
		 * (i - Bias) by 0.301029995663981; since |i - Bias| <= 1077,
		 * and 1077 * 0.30103 * 2^-52 ~=~ 7.2e-14,
		 * adding 1e-13 to the constant term more than suffices.
		 * Hence we adjust the constant term to 0.1760912590558.
		 * (We could get a more accurate k by invoking log10,
		 *  but this is probably not worthwhile.)
		 */

		i -= Bias;
#ifdef IBM
		i <<= 2;
		i += j;
#endif
#ifndef Sudden_Underflow
		denorm = 0;
	}
	else {
		/* d is denormalized */

		i = bbits + be + (Bias + (P-1) - 1);
		x = i > 32  ? word0(d) << (64 - i) | word1(d) >> (i - 32)
			    : word1(d) << (32 - i);
		value(d2) = x;
		word0(d2) -= 31*Exp_msk1; /* adjust exponent */
		i -= (Bias + (P-1) - 1) + 1;
		denorm = 1;
	}
#endif
	ds = (value(d2)-1.5)*0.289529654602168 + 0.1760912590558 +
	    i*0.301029995663981;
	k = (int)ds;
	if (ds < 0. && ds != k)
		k--;	/* want k = floor(ds) */
	k_check = 1;
	if (k >= 0 && k <= Ten_pmax) {
		if (value(d) < tens[k])
			k--;
		k_check = 0;
	}
	j = bbits - i - 1;
	if (j >= 0) {
		b2 = 0;
		s2 = j;
	}
	else {
		b2 = -j;
		s2 = 0;
	}
	if (k >= 0) {
		b5 = 0;
		s5 = k;
		s2 += k;
	}
	else {
		b2 -= k;
		b5 = -k;
		s5 = 0;
	}
	if (mode < 0 || mode > 9)
		mode = 0;
	try_quick = 1;
	if (mode > 5) {
		mode -= 4;
		try_quick = 0;
	}
	leftright = 1;
	switch(mode) {
		case 0:
		case 1:
			ilim = ilim1 = -1;
			i = 18;
			ndigits = 0;
			break;
		case 2:
			leftright = 0;
			/* FALLTHROUGH */
		case 4:
			if (ndigits <= 0)
				ndigits = 1;
			ilim = ilim1 = i = ndigits;
			break;
		case 3:
			leftright = 0;
			/* FALLTHROUGH */
		case 5:
			i = ndigits + k + 1;
			ilim = i;
			ilim1 = i - 1;
			if (i <= 0)
				i = 1;
	}
	j = sizeof(ULong);
        for(result_k = 0; (int)(sizeof(Bigint) - sizeof(ULong)) + j <= i;
		j <<= 1) result_k++;
        // this is really a ugly hack, the code uses Balloc
        // instead of malloc, but casts the result into a char*
        // it seems the only reason to do that is due to the
        // complicated way the block size need to be computed
        // buuurk....
	result = Balloc(result_k);
	if (result == BIGINT_INVALID) {
		Bfree(b);
		return NULL;
	}
	s = s0 = (char *)(void *)result;

	if (ilim >= 0 && ilim <= Quick_max && try_quick) {

		/* Try to get by with floating-point arithmetic. */

		i = 0;
		value(d2) = value(d);
		k0 = k;
		ilim0 = ilim;
		ieps = 2; /* conservative */
		if (k > 0) {
			ds = tens[k&0xf];
			j = (unsigned int)k >> 4;
			if (j & Bletch) {
				/* prevent overflows */
				j &= Bletch - 1;
				value(d) /= bigtens[n_bigtens-1];
				ieps++;
				}
			for(; j; j = (unsigned int)j >> 1, i++)
				if (j & 1) {
					ieps++;
					ds *= bigtens[i];
					}
			value(d) /= ds;
		}
		else if ((jj1 = -k) != 0) {
			value(d) *= tens[jj1 & 0xf];
			for(j = (unsigned int)jj1 >> 4; j;
			    j = (unsigned int)j >> 1, i++)
				if (j & 1) {
					ieps++;
					value(d) *= bigtens[i];
				}
		}
		if (k_check && value(d) < 1. && ilim > 0) {
			if (ilim1 <= 0)
				goto fast_failed;
			ilim = ilim1;
			k--;
			value(d) *= 10.;
			ieps++;
		}
		value(eps) = ieps*value(d) + 7.;
		word0(eps) -= (P-1)*Exp_msk1;
		if (ilim == 0) {
			S = mhi = 0;
			value(d) -= 5.;
			if (value(d) > value(eps))
				goto one_digit;
			if (value(d) < -value(eps))
				goto no_digits;
			goto fast_failed;
		}
#ifndef No_leftright
		if (leftright) {
			/* Use Steele & White method of only
			 * generating digits needed.
			 */
			value(eps) = 0.5/tens[ilim-1] - value(eps);
			for(i = 0;;) {
				L = value(d);
				value(d) -= L;
				*s++ = '0' + (int)L;
				if (value(d) < value(eps))
					goto ret1;
				if (1. - value(d) < value(eps))
					goto bump_up;
				if (++i >= ilim)
					break;
				value(eps) *= 10.;
				value(d) *= 10.;
				}
		}
		else {
#endif
			/* Generate ilim digits, then fix them up. */
			value(eps) *= tens[ilim-1];
			for(i = 1;; i++, value(d) *= 10.) {
				L = value(d);
				value(d) -= L;
				*s++ = '0' + (int)L;
				if (i == ilim) {
					if (value(d) > 0.5 + value(eps))
						goto bump_up;
					else if (value(d) < 0.5 - value(eps)) {
						while(*--s == '0');
						s++;
						goto ret1;
						}
					break;
				}
			}
#ifndef No_leftright
		}
#endif
 fast_failed:
		s = s0;
		value(d) = value(d2);
		k = k0;
		ilim = ilim0;
	}

	/* Do we have a "small" integer? */

	if (be >= 0 && k <= Int_max) {
		/* Yes. */
		ds = tens[k];
		if (ndigits < 0 && ilim <= 0) {
			S = mhi = 0;
			if (ilim < 0 || value(d) <= 5*ds)
				goto no_digits;
			goto one_digit;
		}
		for(i = 1;; i++) {
			L = value(d) / ds;
			value(d) -= L*ds;
#ifdef Check_FLT_ROUNDS
			/* If FLT_ROUNDS == 2, L will usually be high by 1 */
			if (value(d) < 0) {
				L--;
				value(d) += ds;
			}
#endif
			*s++ = '0' + (int)L;
			if (i == ilim) {
				value(d) += value(d);
				if (value(d) > ds || (value(d) == ds && L & 1)) {
 bump_up:
					while(*--s == '9')
						if (s == s0) {
							k++;
							*s = '0';
							break;
						}
					++*s++;
				}
				break;
			}
			if (!(value(d) *= 10.))
				break;
			}
		goto ret1;
	}

	m2 = b2;
	m5 = b5;
	mhi = mlo = 0;
	if (leftright) {
		if (mode < 2) {
			i =
#ifndef Sudden_Underflow
				denorm ? be + (Bias + (P-1) - 1 + 1) :
#endif
#ifdef IBM
				1 + 4*P - 3 - bbits + ((bbits + be - 1) & 3);
#else
				1 + P - bbits;
#endif
		}
		else {
			j = ilim - 1;
			if (m5 >= j)
				m5 -= j;
			else {
				s5 += j -= m5;
				b5 += j;
				m5 = 0;
			}
			if ((i = ilim) < 0) {
				m2 -= i;
				i = 0;
			}
		}
		b2 += i;
		s2 += i;
		mhi = i2b(1);
	}
	if (m2 > 0 && s2 > 0) {
		i = m2 < s2 ? m2 : s2;
		b2 -= i;
		m2 -= i;
		s2 -= i;
	}
	if (b5 > 0) {
		if (leftright) {
			if (m5 > 0) {
				mhi = pow5mult(mhi, m5);
				b1 = mult(mhi, b);
				Bfree(b);
				b = b1;
			}
			if ((j = b5 - m5) != 0)
				b = pow5mult(b, j);
			}
		else
			b = pow5mult(b, b5);
	}
	S = i2b(1);
	if (s5 > 0)
		S = pow5mult(S, s5);

	/* Check for special case that d is a normalized power of 2. */

	if (mode < 2) {
		if (!word1(d) && !(word0(d) & Bndry_mask)
#ifndef Sudden_Underflow
		 && word0(d) & Exp_mask
#endif
				) {
			/* The special case */
			b2 += Log2P;
			s2 += Log2P;
			spec_case = 1;
			}
		else
			spec_case = 0;
	}

	/* Arrange for convenient computation of quotients:
	 * shift left if necessary so divisor has 4 leading 0 bits.
	 *
	 * Perhaps we should just compute leading 28 bits of S once
	 * and for all and pass them and a shift to quorem, so it
	 * can do shifts and ors to compute the numerator for q.
	 */
	if (S == BIGINT_INVALID) {
		i = 0;
	} else {
#ifdef Pack_32
		if ((i = ((s5 ? 32 - hi0bits(S->x[S->wds-1]) : 1) + s2) & 0x1f) != 0)
			i = 32 - i;
#else
		if (i = ((s5 ? 32 - hi0bits(S->x[S->wds-1]) : 1) + s2) & 0xf)
			i = 16 - i;
#endif
	}

	if (i > 4) {
		i -= 4;
		b2 += i;
		m2 += i;
		s2 += i;
	}
	else if (i < 4) {
		i += 28;
		b2 += i;
		m2 += i;
		s2 += i;
	}
	if (b2 > 0)
		b = lshift(b, b2);
	if (s2 > 0)
		S = lshift(S, s2);
	if (k_check) {
		if (cmp(b,S) < 0) {
			k--;
			b = multadd(b, 10, 0);	/* we botched the k estimate */
			if (leftright)
				mhi = multadd(mhi, 10, 0);
			ilim = ilim1;
			}
	}
	if (ilim <= 0 && mode > 2) {
		if (ilim < 0 || cmp(b,S = multadd(S,5,0)) <= 0) {
			/* no digits, fcvt style */
 no_digits:
			k = -1 - ndigits;
			goto ret;
		}
 one_digit:
		*s++ = '1';
		k++;
		goto ret;
	}
	if (leftright) {
		if (m2 > 0)
			mhi = lshift(mhi, m2);

		/* Compute mlo -- check for special case
		 * that d is a normalized power of 2.
		 */

		mlo = mhi;
		if (spec_case) {
			mhi = Balloc(mhi->k);
			Bcopy(mhi, mlo);
			mhi = lshift(mhi, Log2P);
		}

		for(i = 1;;i++) {
			dig = quorem(b,S) + '0';
			/* Do we yet have the shortest decimal string
			 * that will round to d?
			 */
			j = cmp(b, mlo);
			delta = diff(S, mhi);
			jj1 = delta->sign ? 1 : cmp(b, delta);
			Bfree(delta);
#ifndef ROUND_BIASED
			if (jj1 == 0 && !mode && !(word1(d) & 1)) {
				if (dig == '9')
					goto round_9_up;
				if (j > 0)
					dig++;
				*s++ = dig;
				goto ret;
			}
#endif
			if (j < 0 || (j == 0 && !mode
#ifndef ROUND_BIASED
							&& !(word1(d) & 1)
#endif
					)) {
				if (jj1 > 0) {
					b = lshift(b, 1);
					jj1 = cmp(b, S);
					if ((jj1 > 0 || (jj1 == 0 && dig & 1))
					&& dig++ == '9')
						goto round_9_up;
					}
				*s++ = dig;
				goto ret;
			}
			if (jj1 > 0) {
				if (dig == '9') { /* possible if i == 1 */
 round_9_up:
					*s++ = '9';
					goto roundoff;
					}
				*s++ = dig + 1;
				goto ret;
			}
			*s++ = dig;
			if (i == ilim)
				break;
			b = multadd(b, 10, 0);
			if (mlo == mhi)
				mlo = mhi = multadd(mhi, 10, 0);
			else {
				mlo = multadd(mlo, 10, 0);
				mhi = multadd(mhi, 10, 0);
			}
		}
	}
	else
		for(i = 1;; i++) {
			*s++ = dig = quorem(b,S) + '0';
			if (i >= ilim)
				break;
			b = multadd(b, 10, 0);
		}

	/* Round off last digit */

	b = lshift(b, 1);
	j = cmp(b, S);
	if (j > 0 || (j == 0 && dig & 1)) {
 roundoff:
		while(*--s == '9')
			if (s == s0) {
				k++;
				*s++ = '1';
				goto ret;
				}
		++*s++;
	}
	else {
		while(*--s == '0');
		s++;
	}
 ret:
	Bfree(S);
	if (mhi) {
		if (mlo && mlo != mhi)
			Bfree(mlo);
		Bfree(mhi);
	}
 ret1:
	Bfree(b);
	if (s == s0) {				/* don't return empty string */
		*s++ = '0';
		k = 0;
	}
	*s = 0;
	*decpt = k + 1;
	if (rve)
		*rve = s;
	return s0;
}

#include <limits.h>

 char *
rv_alloc(int i)
{
	int j, k, *r;

	j = sizeof(ULong);
	for(k = 0;
		sizeof(Bigint) - sizeof(ULong) - sizeof(int) + j <= i;
		j <<= 1)
			k++;
	r = (int*)Balloc(k);
	*r = k;
	return (char *)(r+1);
	}

 char *
nrv_alloc(char *s, char **rve, int n)
{
	char *rv, *t;

	t = rv = rv_alloc(n);
	while((*t = *s++) !=0)
		t++;
	if (rve)
		*rve = t;
	return rv;
	}


/* Strings values used by dtoa() */
#define	INFSTR	"Infinity"
#define	NANSTR	"NaN"

#define	DBL_ADJ		(DBL_MAX_EXP - 2 + ((DBL_MANT_DIG - 1) % 4))
#define	LDBL_ADJ	(LDBL_MAX_EXP - 2 + ((LDBL_MANT_DIG - 1) % 4))

/*
 * Round up the given digit string.  If the digit string is fff...f,
 * this procedure sets it to 100...0 and returns 1 to indicate that
 * the exponent needs to be bumped.  Otherwise, 0 is returned.
 */
static int
roundup(char *s0, int ndigits)
{
	char *s;

	for (s = s0 + ndigits - 1; *s == 0xf; s--) {
		if (s == s0) {
			*s = 1;
			return (1);
		}
		*s = 0;
	}
	++*s;
	return (0);
}

/*
 * Round the given digit string to ndigits digits according to the
 * current rounding mode.  Note that this could produce a string whose
 * value is not representable in the corresponding floating-point
 * type.  The exponent pointed to by decpt is adjusted if necessary.
 */
static void
dorounding(char *s0, int ndigits, int sign, int *decpt)
{
	int adjust = 0;	/* do we need to adjust the exponent? */

	switch (FLT_ROUNDS) {
	case 0:		/* toward zero */
	default:	/* implementation-defined */
		break;
	case 1:		/* to nearest, halfway rounds to even */
		if ((s0[ndigits] > 8) ||
		    (s0[ndigits] == 8 && s0[ndigits + 1] & 1))
			adjust = roundup(s0, ndigits);
		break;
	case 2:		/* toward +inf */
		if (sign == 0)
			adjust = roundup(s0, ndigits);
		break;
	case 3:		/* toward -inf */
		if (sign != 0)
			adjust = roundup(s0, ndigits);
		break;
	}

	if (adjust)
		*decpt += 4;
}

/*
 * This procedure converts a double-precision number in IEEE format
 * into a string of hexadecimal digits and an exponent of 2.  Its
 * behavior is bug-for-bug compatible with dtoa() in mode 2, with the
 * following exceptions:
 *
 * - An ndigits < 0 causes it to use as many digits as necessary to
 *   represent the number exactly.
 * - The additional xdigs argument should point to either the string
 *   "0123456789ABCDEF" or the string "0123456789abcdef", depending on
 *   which case is desired.
 * - This routine does not repeat dtoa's mistake of setting decpt
 *   to 9999 in the case of an infinity or NaN.  INT_MAX is used
 *   for this purpose instead.
 *
 * Note that the C99 standard does not specify what the leading digit
 * should be for non-zero numbers.  For instance, 0x1.3p3 is the same
 * as 0x2.6p2 is the same as 0x4.cp3.  This implementation chooses the
 * first digit so that subsequent digits are aligned on nibble
 * boundaries (before rounding).
 *
 * Inputs:	d, xdigs, ndigits
 * Outputs:	decpt, sign, rve
 */
char *
__hdtoa(double d, const char *xdigs, int ndigits, int *decpt, int *sign,
    char **rve)
{
	static const int sigfigs = (DBL_MANT_DIG + 3) / 4;
	union IEEEd2bits u;
	char *s, *s0;
	int bufsize, f;

	u.d = d;
	*sign = u.bits.sign;

	switch (f = fpclassify(d)) {
	case FP_NORMAL:
		*decpt = u.bits.exp - DBL_ADJ;
		break;
	case FP_ZERO:
return_zero:
		*decpt = 1;
		return (nrv_alloc("0", rve, 1));
	case FP_SUBNORMAL:
		/*
		 * For processors that treat subnormals as zero, comparison
		 * with zero will be equal, so we jump to the FP_ZERO case.
		 */
		if(u.d == 0.0) goto return_zero;
		u.d *= 0x1p514;
		*decpt = u.bits.exp - (514 + DBL_ADJ);
		break;
	case FP_INFINITE:
		*decpt = INT_MAX;
		return (nrv_alloc(INFSTR, rve, sizeof(INFSTR) - 1));
	case FP_NAN:
		*decpt = INT_MAX;
		return (nrv_alloc(NANSTR, rve, sizeof(NANSTR) - 1));
	default:
#ifdef DEBUG
		BugPrintf("fpclassify returned %d\n", f);
#endif
		return 0; // FIXME??
	}

	/* FP_NORMAL or FP_SUBNORMAL */

	if (ndigits == 0)		/* dtoa() compatibility */
		ndigits = 1;

	/*
	 * For simplicity, we generate all the digits even if the
	 * caller has requested fewer.
	 */
	bufsize = (sigfigs > ndigits) ? sigfigs : ndigits;
	s0 = rv_alloc(bufsize);

	/*
	 * We work from right to left, first adding any requested zero
	 * padding, then the least significant portion of the
	 * mantissa, followed by the most significant.  The buffer is
	 * filled with the byte values 0x0 through 0xf, which are
	 * converted to xdigs[0x0] through xdigs[0xf] after the
	 * rounding phase.
	 */
	for (s = s0 + bufsize - 1; s > s0 + sigfigs - 1; s--)
		*s = 0;
	for (; s > s0 + sigfigs - (DBL_MANL_SIZE / 4) - 1 && s > s0; s--) {
		*s = u.bits.manl & 0xf;
		u.bits.manl >>= 4;
	}
	for (; s > s0; s--) {
		*s = u.bits.manh & 0xf;
		u.bits.manh >>= 4;
	}

	/*
	 * At this point, we have snarfed all the bits in the
	 * mantissa, with the possible exception of the highest-order
	 * (partial) nibble, which is dealt with by the next
	 * statement.  We also tack on the implicit normalization bit.
	 */
	*s = u.bits.manh | (1U << ((DBL_MANT_DIG - 1) % 4));

	/* If ndigits < 0, we are expected to auto-size the precision. */
	if (ndigits < 0) {
		for (ndigits = sigfigs; s0[ndigits - 1] == 0; ndigits--)
			;
	}

	if (sigfigs > ndigits && s0[ndigits] != 0)
		dorounding(s0, ndigits, u.bits.sign, decpt);

	s = s0 + ndigits;
	if (rve != NULL)
		*rve = s;
	*s-- = '\0';
	for (; s >= s0; s--)
		*s = xdigs[(unsigned int)*s];

	return (s0);
}

#ifndef NO_HEX_FP /*{*/

static int
gethex( CONST char **sp, CONST FPI *fpi, Long *exp, Bigint **bp, int sign, locale_t loc)
{
	Bigint *b;
	CONST unsigned char *decpt, *s0, *s, *s1;
	unsigned char *strunc;
	int big, esign, havedig, irv, j, k, n, n0, nbits, up, zret;
	ULong L, lostbits, *x;
	Long e, e1;
#ifdef USE_LOCALE
	int i;
	NORMALIZE_LOCALE(loc);
#ifdef NO_LOCALE_CACHE
	const unsigned char *decimalpoint = (unsigned char*)localeconv_l(loc)->decimal_point;
#else
	const unsigned char *decimalpoint;
	static unsigned char *decimalpoint_cache;
	if (!(s0 = decimalpoint_cache)) {
		s0 = (unsigned char*)localeconv_l(loc)->decimal_point;
		if ((decimalpoint_cache = (char*)MALLOC(strlen(s0) + 1))) {
			strcpy(decimalpoint_cache, s0);
			s0 = decimalpoint_cache;
			}
		}
	decimalpoint = s0;
#endif
#endif

#ifndef ANDROID_CHANGES
	if (!hexdig['0'])
		hexdig_init_D2A();
#endif

	*bp = 0;
	havedig = 0;
	s0 = *(CONST unsigned char **)sp + 2;
	while(s0[havedig] == '0')
		havedig++;
	s0 += havedig;
	s = s0;
	decpt = 0;
	zret = 0;
	e = 0;
	if (hexdig[*s])
		havedig++;
	else {
		zret = 1;
#ifdef USE_LOCALE
		for(i = 0; decimalpoint[i]; ++i) {
			if (s[i] != decimalpoint[i])
				goto pcheck;
			}
		decpt = s += i;
#else
		if (*s != '.')
			goto pcheck;
		decpt = ++s;
#endif
		if (!hexdig[*s])
			goto pcheck;
		while(*s == '0')
			s++;
		if (hexdig[*s])
			zret = 0;
		havedig = 1;
		s0 = s;
		}
	while(hexdig[*s])
		s++;
#ifdef USE_LOCALE
	if (*s == *decimalpoint && !decpt) {
		for(i = 1; decimalpoint[i]; ++i) {
			if (s[i] != decimalpoint[i])
				goto pcheck;
			}
		decpt = s += i;
#else
	if (*s == '.' && !decpt) {
		decpt = ++s;
#endif
		while(hexdig[*s])
			s++;
		}/*}*/
	if (decpt)
		e = -(((Long)(s-decpt)) << 2);
 pcheck:
	s1 = s;
	big = esign = 0;
	switch(*s) {
	  case 'p':
	  case 'P':
		switch(*++s) {
		  case '-':
			esign = 1;
			/* no break */
		  case '+':
			s++;
		  }
		if ((n = hexdig[*s]) == 0 || n > 0x19) {
			s = s1;
			break;
			}
		e1 = n - 0x10;
		while((n = hexdig[*++s]) !=0 && n <= 0x19) {
			if (e1 & 0xf8000000)
				big = 1;
			e1 = 10*e1 + n - 0x10;
			}
		if (esign)
			e1 = -e1;
		e += e1;
	  }
	*sp = (char*)s;
	if (!havedig)
		*sp = (char*)s0 - 1;
	if (zret)
		return STRTOG_Zero;
	if (big) {
		if (esign) {
			switch(fpi->rounding) {
			  case FPI_Round_up:
				if (sign)
					break;
				goto ret_tiny;
			  case FPI_Round_down:
				if (!sign)
					break;
				goto ret_tiny;
			  }
			goto retz;
 ret_tiny:
			b = Balloc(0);
			b->wds = 1;
			b->x[0] = 1;
			goto dret;
			}
		switch(fpi->rounding) {
		  case FPI_Round_near:
			goto ovfl1;
		  case FPI_Round_up:
			if (!sign)
				goto ovfl1;
			goto ret_big;
		  case FPI_Round_down:
			if (sign)
				goto ovfl1;
			goto ret_big;
		  }
 ret_big:
		nbits = fpi->nbits;
		n0 = n = nbits >> kshift;
		if (nbits & kmask)
			++n;
		for(j = n, k = 0; j >>= 1; ++k);
		*bp = b = Balloc(k);
		b->wds = n;
		for(j = 0; j < n0; ++j)
			b->x[j] = ALL_ON;
		if (n > n0)
			b->x[j] = ULbits >> (ULbits - (nbits & kmask));
		*exp = fpi->emin;
		return STRTOG_Normal | STRTOG_Inexlo;
		}
	/*
	 * Truncate the hex string if it is longer than the precision needed,
	 * to avoid denial-of-service issues with very large strings.  Use
	 * additional digits to insure precision.  Scan to-be-truncated digits
	 * and replace with either '1' or '0' to ensure proper rounding.
	 */
	{
		int maxdigits = ((fpi->nbits + 3) >> 2) + 2;
		size_t nd = s1 - s0;
#ifdef USE_LOCALE
		int dplen = strlen((const char *)decimalpoint);
#else
		int dplen = 1;
#endif

		if (decpt && s0 < decpt)
			nd -= dplen;
		if (nd > maxdigits && (strunc = alloca(maxdigits + dplen + 2)) != NULL) {
			ssize_t nd0 = decpt ? decpt - s0 - dplen : nd;
			unsigned char *tp = strunc + maxdigits;
			int found = 0;
			if ((nd0 -= maxdigits) >= 0 || s0 >= decpt)
				memcpy(strunc, s0, maxdigits);
			else {
				memcpy(strunc, s0, maxdigits + dplen);
				tp += dplen;
				}
			s0 += maxdigits;
			e += (nd - (maxdigits + 1)) << 2;
			if (nd0 > 0) {
				while(nd0-- > 0)
					if (*s0++ != '0') {
						found++;
						break;
						}
				s0 += dplen;
				}
			if (!found && decpt) {
				while(s0 < s1)
					if(*s0++ != '0') {
						found++;
						break;
						}
				}
			*tp++ = found ? '1' : '0';
			*tp = 0;
			s0 = strunc;
			s1 = tp;
			}
		}

	n = s1 - s0 - 1;
	for(k = 0; n > (1 << (kshift-2)) - 1; n >>= 1)
		k++;
	b = Balloc(k);
	x = b->x;
	n = 0;
	L = 0;
#ifdef USE_LOCALE
	for(i = 0; decimalpoint[i+1]; ++i);
#endif
	while(s1 > s0) {
#ifdef USE_LOCALE
		if (*--s1 == decimalpoint[i]) {
			s1 -= i;
			continue;
			}
#else
		if (*--s1 == '.')
			continue;
#endif
		if (n == ULbits) {
			*x++ = L;
			L = 0;
			n = 0;
			}
		L |= (hexdig[*s1] & 0x0f) << n;
		n += 4;
		}
	*x++ = L;
	b->wds = n = x - b->x;
	n = ULbits*n - hi0bits(L);
	nbits = fpi->nbits;
	lostbits = 0;
	x = b->x;
	if (n > nbits) {
		n -= nbits;
		if (any_on(b,n)) {
			lostbits = 1;
			k = n - 1;
			if (x[k>>kshift] & 1 << (k & kmask)) {
				lostbits = 2;
				if (k > 0 && any_on(b,k))
					lostbits = 3;
				}
			}
		rshift(b, n);
		e += n;
		}
	else if (n < nbits) {
		n = nbits - n;
		b = lshift(b, n);
		e -= n;
		x = b->x;
		}
	if (e > fpi->emax) {
 ovfl:
		Bfree(b);
 ovfl1:
#ifndef NO_ERRNO
		errno = ERANGE;
#endif
		return STRTOG_Infinite | STRTOG_Overflow | STRTOG_Inexhi;
		}
	irv = STRTOG_Normal;
	if (e < fpi->emin) {
		irv = STRTOG_Denormal;
		n = fpi->emin - e;
		if (n >= nbits) {
			switch (fpi->rounding) {
			  case FPI_Round_near:
				if (n == nbits && (n < 2 || any_on(b,n-1)))
					goto one_bit;
				break;
			  case FPI_Round_up:
				if (!sign)
					goto one_bit;
				break;
			  case FPI_Round_down:
				if (sign) {
 one_bit:
					x[0] = b->wds = 1;
 dret:
					*bp = b;
					*exp = fpi->emin;
#ifndef NO_ERRNO
					errno = ERANGE;
#endif
					return STRTOG_Denormal | STRTOG_Inexhi
						| STRTOG_Underflow;
					}
			  }
			Bfree(b);
 retz:
#ifndef NO_ERRNO
			errno = ERANGE;
#endif
			return STRTOG_Zero | STRTOG_Inexlo | STRTOG_Underflow;
			}
		k = n - 1;
		if (lostbits)
			lostbits = 1;
		else if (k > 0)
			lostbits = any_on(b,k);
		if (x[k>>kshift] & 1 << (k & kmask))
			lostbits |= 2;
		nbits -= n;
		rshift(b,n);
		e = fpi->emin;
		}
	if (lostbits) {
		up = 0;
		switch(fpi->rounding) {
		  case FPI_Round_zero:
			break;
		  case FPI_Round_near:
			if (lostbits & 2
			 && (lostbits | x[0]) & 1)
				up = 1;
			break;
		  case FPI_Round_up:
			up = 1 - sign;
			break;
		  case FPI_Round_down:
			up = sign;
		  }
		if (up) {
			k = b->wds;
			b = increment(b);
			x = b->x;
			if (irv == STRTOG_Denormal) {
				if (nbits == fpi->nbits - 1
				 && x[nbits >> kshift] & 1 << (nbits & kmask))
					irv =  STRTOG_Normal;
				}
			else if (b->wds > k
			 || ((n = nbits & kmask) !=0
			      && hi0bits(x[k-1]) < 32-n)) {
				rshift(b,1);
				if (++e > fpi->emax)
					goto ovfl;
				}
			irv |= STRTOG_Inexhi;
			}
		else
			irv |= STRTOG_Inexlo;
		}
	*bp = b;
	*exp = e;
	return irv;
	}

#endif /*}*/

#ifdef __cplusplus
}
#endif
