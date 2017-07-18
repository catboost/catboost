/* zlassq.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlassq_(integer *n, doublecomplex *x, integer *incx, 
	doublereal *scale, doublereal *sumsq)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    integer ix;
    doublereal temp1;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLASSQ returns the values scl and ssq such that */

/*     ( scl**2 )*ssq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq, */

/*  where x( i ) = abs( X( 1 + ( i - 1 )*INCX ) ). The value of sumsq is */
/*  assumed to be at least unity and the value of ssq will then satisfy */

/*     1.0 .le. ssq .le. ( sumsq + 2*n ). */

/*  scale is assumed to be non-negative and scl returns the value */

/*     scl = max( scale, abs( real( x( i ) ) ), abs( aimag( x( i ) ) ) ), */
/*            i */

/*  scale and sumsq must be supplied in SCALE and SUMSQ respectively. */
/*  SCALE and SUMSQ are overwritten by scl and ssq respectively. */

/*  The routine makes only one pass through the vector X. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of elements to be used from the vector X. */

/*  X       (input) COMPLEX*16 array, dimension (N) */
/*          The vector x as described above. */
/*             x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n. */

/*  INCX    (input) INTEGER */
/*          The increment between successive values of the vector X. */
/*          INCX > 0. */

/*  SCALE   (input/output) DOUBLE PRECISION */
/*          On entry, the value  scale  in the equation above. */
/*          On exit, SCALE is overwritten with the value  scl . */

/*  SUMSQ   (input/output) DOUBLE PRECISION */
/*          On entry, the value  sumsq  in the equation above. */
/*          On exit, SUMSQ is overwritten with the value  ssq . */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n > 0) {
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    i__3 = ix;
	    if (x[i__3].r != 0.) {
		i__3 = ix;
		temp1 = (d__1 = x[i__3].r, abs(d__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    d__1 = *scale / temp1;
		    *sumsq = *sumsq * (d__1 * d__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    d__1 = temp1 / *scale;
		    *sumsq += d__1 * d__1;
		}
	    }
	    if (d_imag(&x[ix]) != 0.) {
		temp1 = (d__1 = d_imag(&x[ix]), abs(d__1));
		if (*scale < temp1) {
/* Computing 2nd power */
		    d__1 = *scale / temp1;
		    *sumsq = *sumsq * (d__1 * d__1) + 1;
		    *scale = temp1;
		} else {
/* Computing 2nd power */
		    d__1 = temp1 / *scale;
		    *sumsq += d__1 * d__1;
		}
	    }
/* L10: */
	}
    }

    return 0;

/*     End of ZLASSQ */

} /* zlassq_ */
