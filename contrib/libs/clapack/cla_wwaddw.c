/* cla_wwaddw.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"
#include "blaswrap.h"

/* Subroutine */ int cla_wwaddw__(integer *n, complex *x, complex *y, complex 
	*w)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2, q__3;

    /* Local variables */
    integer i__;
    complex s;


/*     -- LAPACK routine (version 3.2)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- November 2008                                                -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     Purpose */
/*     ======= */

/*     CLA_WWADDW adds a vector W into a doubled-single vector (X, Y). */

/*     This works for all extant IBM's hex and binary floating point */
/*     arithmetics, but not for decimal. */

/*     Arguments */
/*     ========= */

/*     N      (input) INTEGER */
/*            The length of vectors X, Y, and W. */

/*     X, Y   (input/output) COMPLEX array, length N */
/*            The doubled-single accumulation vector. */

/*     W      (input) COMPLEX array, length N */
/*            The vector to be added. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --w;
    --y;
    --x;

    /* Function Body */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	i__3 = i__;
	q__1.r = x[i__2].r + w[i__3].r, q__1.i = x[i__2].i + w[i__3].i;
	s.r = q__1.r, s.i = q__1.i;
	q__2.r = s.r + s.r, q__2.i = s.i + s.i;
	q__1.r = q__2.r - s.r, q__1.i = q__2.i - s.i;
	s.r = q__1.r, s.i = q__1.i;
	i__2 = i__;
	i__3 = i__;
	q__3.r = x[i__3].r - s.r, q__3.i = x[i__3].i - s.i;
	i__4 = i__;
	q__2.r = q__3.r + w[i__4].r, q__2.i = q__3.i + w[i__4].i;
	i__5 = i__;
	q__1.r = q__2.r + y[i__5].r, q__1.i = q__2.i + y[i__5].i;
	y[i__2].r = q__1.r, y[i__2].i = q__1.i;
	i__2 = i__;
	x[i__2].r = s.r, x[i__2].i = s.i;
/* L10: */
    }
    return 0;
} /* cla_wwaddw__ */
