/* dla_wwaddw.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dla_wwaddw__(integer *n, doublereal *x, doublereal *y, 
	doublereal *w)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    doublereal s;


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

/*     DLA_WWADDW adds a vector W into a doubled-single vector (X, Y). */

/*     This works for all extant IBM's hex and binary floating point */
/*     arithmetics, but not for decimal. */

/*     Arguments */
/*     ========= */

/*     N      (input) INTEGER */
/*            The length of vectors X, Y, and W. */

/*     X, Y   (input/output) DOUBLE PRECISION array, length N */
/*            The doubled-single accumulation vector. */

/*     W      (input) DOUBLE PRECISION array, length N */
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
	s = x[i__] + w[i__];
	s = s + s - s;
	y[i__] = x[i__] - s + w[i__] + y[i__];
	x[i__] = s;
/* L10: */
    }
    return 0;
} /* dla_wwaddw__ */
