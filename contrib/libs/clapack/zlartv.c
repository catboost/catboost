/* zlartv.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlartv_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublereal *c__, doublecomplex *s, 
	integer *incc)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, ic, ix, iy;
    doublecomplex xi, yi;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLARTV applies a vector of complex plane rotations with real cosines */
/*  to elements of the complex vectors x and y. For i = 1,2,...,n */

/*     ( x(i) ) := (        c(i)   s(i) ) ( x(i) ) */
/*     ( y(i) )    ( -conjg(s(i))  c(i) ) ( y(i) ) */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of plane rotations to be applied. */

/*  X       (input/output) COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/*          The vector x. */

/*  INCX    (input) INTEGER */
/*          The increment between elements of X. INCX > 0. */

/*  Y       (input/output) COMPLEX*16 array, dimension (1+(N-1)*INCY) */
/*          The vector y. */

/*  INCY    (input) INTEGER */
/*          The increment between elements of Y. INCY > 0. */

/*  C       (input) DOUBLE PRECISION array, dimension (1+(N-1)*INCC) */
/*          The cosines of the plane rotations. */

/*  S       (input) COMPLEX*16 array, dimension (1+(N-1)*INCC) */
/*          The sines of the plane rotations. */

/*  INCC    (input) INTEGER */
/*          The increment between elements of C and S. INCC > 0. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --s;
    --c__;
    --y;
    --x;

    /* Function Body */
    ix = 1;
    iy = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	xi.r = x[i__2].r, xi.i = x[i__2].i;
	i__2 = iy;
	yi.r = y[i__2].r, yi.i = y[i__2].i;
	i__2 = ix;
	i__3 = ic;
	z__2.r = c__[i__3] * xi.r, z__2.i = c__[i__3] * xi.i;
	i__4 = ic;
	z__3.r = s[i__4].r * yi.r - s[i__4].i * yi.i, z__3.i = s[i__4].r * 
		yi.i + s[i__4].i * yi.r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	i__2 = iy;
	i__3 = ic;
	z__2.r = c__[i__3] * yi.r, z__2.i = c__[i__3] * yi.i;
	d_cnjg(&z__4, &s[ic]);
	z__3.r = z__4.r * xi.r - z__4.i * xi.i, z__3.i = z__4.r * xi.i + 
		z__4.i * xi.r;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	y[i__2].r = z__1.r, y[i__2].i = z__1.i;
	ix += *incx;
	iy += *incy;
	ic += *incc;
/* L10: */
    }
    return 0;

/*     End of ZLARTV */

} /* zlartv_ */
