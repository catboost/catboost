/* zrot.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zrot_(integer *n, doublecomplex *cx, integer *incx, 
	doublecomplex *cy, integer *incy, doublereal *c__, doublecomplex *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, ix, iy;
    doublecomplex stemp;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZROT   applies a plane rotation, where the cos (C) is real and the */
/*  sin (S) is complex, and the vectors CX and CY are complex. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of elements in the vectors CX and CY. */

/*  CX      (input/output) COMPLEX*16 array, dimension (N) */
/*          On input, the vector X. */
/*          On output, CX is overwritten with C*X + S*Y. */

/*  INCX    (input) INTEGER */
/*          The increment between successive values of CY.  INCX <> 0. */

/*  CY      (input/output) COMPLEX*16 array, dimension (N) */
/*          On input, the vector Y. */
/*          On output, CY is overwritten with -CONJG(S)*X + C*Y. */

/*  INCY    (input) INTEGER */
/*          The increment between successive values of CY.  INCX <> 0. */

/*  C       (input) DOUBLE PRECISION */
/*  S       (input) COMPLEX*16 */
/*          C and S define a rotation */
/*             [  C          S  ] */
/*             [ -conjg(S)   C  ] */
/*          where C*C + S*CONJG(S) = 1.0. */

/* ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --cy;
    --cx;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	goto L20;
    }

/*     Code for unequal increments or equal increments not equal to 1 */

    ix = 1;
    iy = 1;
    if (*incx < 0) {
	ix = (-(*n) + 1) * *incx + 1;
    }
    if (*incy < 0) {
	iy = (-(*n) + 1) * *incy + 1;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
	i__3 = iy;
	z__3.r = s->r * cy[i__3].r - s->i * cy[i__3].i, z__3.i = s->r * cy[
		i__3].i + s->i * cy[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	stemp.r = z__1.r, stemp.i = z__1.i;
	i__2 = iy;
	i__3 = iy;
	z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
	d_cnjg(&z__4, s);
	i__4 = ix;
	z__3.r = z__4.r * cx[i__4].r - z__4.i * cx[i__4].i, z__3.i = z__4.r * 
		cx[i__4].i + z__4.i * cx[i__4].r;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
	i__2 = ix;
	cx[i__2].r = stemp.r, cx[i__2].i = stemp.i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*     Code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	z__2.r = *c__ * cx[i__2].r, z__2.i = *c__ * cx[i__2].i;
	i__3 = i__;
	z__3.r = s->r * cy[i__3].r - s->i * cy[i__3].i, z__3.i = s->r * cy[
		i__3].i + s->i * cy[i__3].r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	stemp.r = z__1.r, stemp.i = z__1.i;
	i__2 = i__;
	i__3 = i__;
	z__2.r = *c__ * cy[i__3].r, z__2.i = *c__ * cy[i__3].i;
	d_cnjg(&z__4, s);
	i__4 = i__;
	z__3.r = z__4.r * cx[i__4].r - z__4.i * cx[i__4].i, z__3.i = z__4.r * 
		cx[i__4].i + z__4.i * cx[i__4].r;
	z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
	cy[i__2].r = z__1.r, cy[i__2].i = z__1.i;
	i__2 = i__;
	cx[i__2].r = stemp.r, cx[i__2].i = stemp.i;
/* L30: */
    }
    return 0;
} /* zrot_ */
