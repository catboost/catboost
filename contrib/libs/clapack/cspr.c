/* cspr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cspr_(char *uplo, integer *n, complex *alpha, complex *x, 
	 integer *incx, complex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2;

    /* Local variables */
    integer i__, j, k, kk, ix, jx, kx, info;
    complex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CSPR    performs the symmetric rank 1 operation */

/*     A := alpha*x*conjg( x' ) + A, */

/*  where alpha is a complex scalar, x is an n element vector and A is an */
/*  n by n symmetric matrix, supplied in packed form. */

/*  Arguments */
/*  ========== */

/*  UPLO     (input) CHARACTER*1 */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N        (input) INTEGER */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA    (input) COMPLEX */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X        (input) COMPLEX array, dimension at least */
/*           ( 1 + ( N - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the N- */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX     (input) INTEGER */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  AP       (input/output) COMPLEX array, dimension at least */
/*           ( ( N*( N + 1 ) )/2 ). */
/*           Before entry, with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the symmetric matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry, with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the symmetric matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    }
    if (info != 0) {
	xerbla_("CSPR  ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
	return 0;
    }

/*     Set the start point in X if the increment is not unity. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (lsame_(uplo, "U")) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = j;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = ap[i__4].r + q__2.r, q__1.i = ap[i__4].i + 
				q__2.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    q__2.r = x[i__4].r * temp.r - x[i__4].i * temp.i, q__2.i =
			     x[i__4].r * temp.i + x[i__4].i * temp.r;
		    q__1.r = ap[i__3].r + q__2.r, q__1.i = ap[i__3].i + 
			    q__2.i;
		    ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    ap[i__2].r = ap[i__3].r, ap[i__2].i = ap[i__3].i;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = jx;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    ix = kx;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = ap[i__4].r + q__2.r, q__1.i = ap[i__4].i + 
				q__2.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			ix += *incx;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    q__2.r = x[i__4].r * temp.r - x[i__4].i * temp.i, q__2.i =
			     x[i__4].r * temp.i + x[i__4].i * temp.r;
		    q__1.r = ap[i__3].r + q__2.r, q__1.i = ap[i__3].i + 
			    q__2.i;
		    ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    ap[i__2].r = ap[i__3].r, ap[i__2].i = ap[i__3].i;
		}
		jx += *incx;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = j;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    q__2.r = temp.r * x[i__4].r - temp.i * x[i__4].i, q__2.i =
			     temp.r * x[i__4].i + temp.i * x[i__4].r;
		    q__1.r = ap[i__3].r + q__2.r, q__1.i = ap[i__3].i + 
			    q__2.i;
		    ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = ap[i__4].r + q__2.r, q__1.i = ap[i__4].i + 
				q__2.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    ap[i__2].r = ap[i__3].r, ap[i__2].i = ap[i__3].i;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    i__2 = jx;
		    q__1.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__1.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    q__2.r = temp.r * x[i__4].r - temp.i * x[i__4].i, q__2.i =
			     temp.r * x[i__4].i + temp.i * x[i__4].r;
		    q__1.r = ap[i__3].r + q__2.r, q__1.i = ap[i__3].i + 
			    q__2.i;
		    ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		    ix = jx;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = ap[i__4].r + q__2.r, q__1.i = ap[i__4].i + 
				q__2.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    ap[i__2].r = ap[i__3].r, ap[i__2].i = ap[i__3].i;
		}
		jx += *incx;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of CSPR */

} /* cspr_ */
