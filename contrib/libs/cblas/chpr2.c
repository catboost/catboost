/* chpr2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int chpr2_(char *uplo, integer *n, complex *alpha, complex *
	x, integer *incx, complex *y, integer *incy, complex *ap)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j, k, kk, ix, iy, jx, jy, kx, ky, info;
    complex temp1, temp2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHPR2  performs the hermitian rank 2 operation */

/*     A := alpha*x*conjg( y' ) + conjg( alpha )*y*conjg( x' ) + A, */

/*  where alpha is a scalar, x and y are n element vectors and A is an */
/*  n by n hermitian matrix, supplied in packed form. */

/*  Arguments */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the matrix A is supplied in the packed */
/*           array AP as follows: */

/*              UPLO = 'U' or 'u'   The upper triangular part of A is */
/*                                  supplied in AP. */

/*              UPLO = 'L' or 'l'   The lower triangular part of A is */
/*                                  supplied in AP. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - COMPLEX         . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  Y      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ). */
/*           Before entry, the incremented array Y must contain the n */
/*           element vector y. */
/*           Unchanged on exit. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX          array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 1, 2 ) */
/*           and a( 2, 2 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the upper triangular part of the */
/*           updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular part of the hermitian matrix */
/*           packed sequentially, column by column, so that AP( 1 ) */
/*           contains a( 1, 1 ), AP( 2 ) and AP( 3 ) contain a( 2, 1 ) */
/*           and a( 3, 1 ) respectively, and so on. On exit, the array */
/*           AP is overwritten by the lower triangular part of the */
/*           updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;
    --y;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("CHPR2 ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || alpha->r == 0.f && alpha->i == 0.f) {
	return 0;
    }

/*     Set up the start points in X and Y if the increments are not both */
/*     unity. */

    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }

/*     Start the operations. In this version the elements of the array AP */
/*     are accessed sequentially with one pass through AP. */

    kk = 1;
    if (lsame_(uplo, "U")) {

/*        Form  A  when upper triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[j]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = j;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    k = kk;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = i__;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L10: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = j;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		kk += j;
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[jy]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = jx;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    ix = kx;
		    iy = ky;
		    i__2 = kk + j - 2;
		    for (k = kk; k <= i__2; ++k) {
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = iy;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			ix += *incx;
			iy += *incy;
/* L30: */
		    }
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    i__4 = jx;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		} else {
		    i__2 = kk + j - 1;
		    i__3 = kk + j - 1;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		jx += *incx;
		jy += *incy;
		kk += j;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when lower triangle is stored in AP. */

	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		i__3 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[j]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = j;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = j;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = j;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		    k = kk + 1;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = k;
			i__4 = k;
			i__5 = i__;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = i__;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
			++k;
/* L50: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		kk = kk + *n - j + 1;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		i__3 = jy;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f || (y[i__3].r != 0.f 
			|| y[i__3].i != 0.f)) {
		    r_cnjg(&q__2, &y[jy]);
		    q__1.r = alpha->r * q__2.r - alpha->i * q__2.i, q__1.i = 
			    alpha->r * q__2.i + alpha->i * q__2.r;
		    temp1.r = q__1.r, temp1.i = q__1.i;
		    i__2 = jx;
		    q__2.r = alpha->r * x[i__2].r - alpha->i * x[i__2].i, 
			    q__2.i = alpha->r * x[i__2].i + alpha->i * x[i__2]
			    .r;
		    r_cnjg(&q__1, &q__2);
		    temp2.r = q__1.r, temp2.i = q__1.i;
		    i__2 = kk;
		    i__3 = kk;
		    i__4 = jx;
		    q__2.r = x[i__4].r * temp1.r - x[i__4].i * temp1.i, 
			    q__2.i = x[i__4].r * temp1.i + x[i__4].i * 
			    temp1.r;
		    i__5 = jy;
		    q__3.r = y[i__5].r * temp2.r - y[i__5].i * temp2.i, 
			    q__3.i = y[i__5].r * temp2.i + y[i__5].i * 
			    temp2.r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    r__1 = ap[i__3].r + q__1.r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		    ix = jx;
		    iy = jy;
		    i__2 = kk + *n - j;
		    for (k = kk + 1; k <= i__2; ++k) {
			ix += *incx;
			iy += *incy;
			i__3 = k;
			i__4 = k;
			i__5 = ix;
			q__3.r = x[i__5].r * temp1.r - x[i__5].i * temp1.i, 
				q__3.i = x[i__5].r * temp1.i + x[i__5].i * 
				temp1.r;
			q__2.r = ap[i__4].r + q__3.r, q__2.i = ap[i__4].i + 
				q__3.i;
			i__6 = iy;
			q__4.r = y[i__6].r * temp2.r - y[i__6].i * temp2.i, 
				q__4.i = y[i__6].r * temp2.i + y[i__6].i * 
				temp2.r;
			q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
			ap[i__3].r = q__1.r, ap[i__3].i = q__1.i;
/* L70: */
		    }
		} else {
		    i__2 = kk;
		    i__3 = kk;
		    r__1 = ap[i__3].r;
		    ap[i__2].r = r__1, ap[i__2].i = 0.f;
		}
		jx += *incx;
		jy += *incy;
		kk = kk + *n - j + 1;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of CHPR2 . */

} /* chpr2_ */
