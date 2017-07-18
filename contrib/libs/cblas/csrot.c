/* csrot.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int csrot_(integer *n, complex *cx, integer *incx, complex *
	cy, integer *incy, real *c__, real *s)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    complex q__1, q__2, q__3;

    /* Local variables */
    integer i__, ix, iy;
    complex ctemp;


/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Applies a plane rotation, where the cos and sin (c and s) are real */
/*  and the vectors cx and cy are complex. */
/*  jack dongarra, linpack, 3/11/78. */

/*  Arguments */
/*  ========== */

/*  N        (input) INTEGER */
/*           On entry, N specifies the order of the vectors cx and cy. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  CX       (input) COMPLEX array, dimension at least */
/*           ( 1 + ( N - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array CX must contain the n */
/*           element vector cx. On exit, CX is overwritten by the updated */
/*           vector cx. */

/*  INCX     (input) INTEGER */
/*           On entry, INCX specifies the increment for the elements of */
/*           CX. INCX must not be zero. */
/*           Unchanged on exit. */

/*  CY       (input) COMPLEX array, dimension at least */
/*           ( 1 + ( N - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array CY must contain the n */
/*           element vector cy. On exit, CY is overwritten by the updated */
/*           vector cy. */

/*  INCY     (input) INTEGER */
/*           On entry, INCY specifies the increment for the elements of */
/*           CY. INCY must not be zero. */
/*           Unchanged on exit. */

/*  C        (input) REAL */
/*           On entry, C specifies the cosine, cos. */
/*           Unchanged on exit. */

/*  S        (input) REAL */
/*           On entry, S specifies the sine, sin. */
/*           Unchanged on exit. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
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

/*        code for unequal increments or equal increments not equal */
/*          to 1 */

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
	q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
	i__3 = iy;
	q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
	q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	ctemp.r = q__1.r, ctemp.i = q__1.i;
	i__2 = iy;
	i__3 = iy;
	q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
	i__4 = ix;
	q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
	i__2 = ix;
	cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
	ix += *incx;
	iy += *incy;
/* L10: */
    }
    return 0;

/*        code for both increments equal to 1 */

L20:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	q__2.r = *c__ * cx[i__2].r, q__2.i = *c__ * cx[i__2].i;
	i__3 = i__;
	q__3.r = *s * cy[i__3].r, q__3.i = *s * cy[i__3].i;
	q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	ctemp.r = q__1.r, ctemp.i = q__1.i;
	i__2 = i__;
	i__3 = i__;
	q__2.r = *c__ * cy[i__3].r, q__2.i = *c__ * cy[i__3].i;
	i__4 = i__;
	q__3.r = *s * cx[i__4].r, q__3.i = *s * cx[i__4].i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	cy[i__2].r = q__1.r, cy[i__2].i = q__1.i;
	i__2 = i__;
	cx[i__2].r = ctemp.r, cx[i__2].i = ctemp.i;
/* L30: */
    }
    return 0;
} /* csrot_ */
