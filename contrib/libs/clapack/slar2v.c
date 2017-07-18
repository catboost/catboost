/* slar2v.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slar2v_(integer *n, real *x, real *y, real *z__, integer 
	*incx, real *c__, real *s, integer *incc)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__;
    real t1, t2, t3, t4, t5, t6;
    integer ic;
    real ci, si;
    integer ix;
    real xi, yi, zi;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAR2V applies a vector of real plane rotations from both sides to */
/*  a sequence of 2-by-2 real symmetric matrices, defined by the elements */
/*  of the vectors x, y and z. For i = 1,2,...,n */

/*     ( x(i)  z(i) ) := (  c(i)  s(i) ) ( x(i)  z(i) ) ( c(i) -s(i) ) */
/*     ( z(i)  y(i) )    ( -s(i)  c(i) ) ( z(i)  y(i) ) ( s(i)  c(i) ) */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of plane rotations to be applied. */

/*  X       (input/output) REAL array, */
/*                         dimension (1+(N-1)*INCX) */
/*          The vector x. */

/*  Y       (input/output) REAL array, */
/*                         dimension (1+(N-1)*INCX) */
/*          The vector y. */

/*  Z       (input/output) REAL array, */
/*                         dimension (1+(N-1)*INCX) */
/*          The vector z. */

/*  INCX    (input) INTEGER */
/*          The increment between elements of X, Y and Z. INCX > 0. */

/*  C       (input) REAL array, dimension (1+(N-1)*INCC) */
/*          The cosines of the plane rotations. */

/*  S       (input) REAL array, dimension (1+(N-1)*INCC) */
/*          The sines of the plane rotations. */

/*  INCC    (input) INTEGER */
/*          The increment between elements of C and S. INCC > 0. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --s;
    --c__;
    --z__;
    --y;
    --x;

    /* Function Body */
    ix = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	xi = x[ix];
	yi = y[ix];
	zi = z__[ix];
	ci = c__[ic];
	si = s[ic];
	t1 = si * zi;
	t2 = ci * zi;
	t3 = t2 - si * xi;
	t4 = t2 + si * yi;
	t5 = ci * xi + t1;
	t6 = ci * yi - t1;
	x[ix] = ci * t5 + si * t4;
	y[ix] = ci * t6 - si * t3;
	z__[ix] = ci * t4 - si * t5;
	ix += *incx;
	ic += *incc;
/* L10: */
    }

/*     End of SLAR2V */

    return 0;
} /* slar2v_ */
