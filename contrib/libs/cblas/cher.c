/* cher.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cher_(char *uplo, integer *n, real *alpha, complex *x, 
	integer *incx, complex *a, integer *lda)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1;
    complex q__1, q__2;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j, ix, jx, kx, info;
    complex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHER   performs the hermitian rank 1 operation */

/*     A := alpha*x*conjg( x' ) + A, */

/*  where alpha is a real scalar, x is an n element vector and A is an */
/*  n by n hermitian matrix. */

/*  Arguments */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the array A is to be referenced as */
/*           follows: */

/*              UPLO = 'U' or 'u'   Only the upper triangular part of A */
/*                                  is to be referenced. */

/*              UPLO = 'L' or 'l'   Only the lower triangular part of A */
/*                                  is to be referenced. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
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

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry with  UPLO = 'U' or 'u', the leading n by n */
/*           upper triangular part of the array A must contain the upper */
/*           triangular part of the hermitian matrix and the strictly */
/*           lower triangular part of A is not referenced. On exit, the */
/*           upper triangular part of the array A is overwritten by the */
/*           upper triangular part of the updated matrix. */
/*           Before entry with UPLO = 'L' or 'l', the leading n by n */
/*           lower triangular part of the array A must contain the lower */
/*           triangular part of the hermitian matrix and the strictly */
/*           upper triangular part of A is not referenced. On exit, the */
/*           lower triangular part of the array A is overwritten by the */
/*           lower triangular part of the updated matrix. */
/*           Note that the imaginary parts of the diagonal elements need */
/*           not be set, they are assumed to be zero, and on exit they */
/*           are set to zero. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, n ). */
/*           Unchanged on exit. */


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
    --x;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*lda < max(1,*n)) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("CHER  ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.f) {
	return 0;
    }

/*     Set the start point in X if the increment is not unity. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through the triangular part */
/*     of A. */

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in upper triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    r_cnjg(&q__2, &x[j]);
		    q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L10: */
		    }
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    i__4 = j;
		    q__1.r = x[i__4].r * temp.r - x[i__4].i * temp.i, q__1.i =
			     x[i__4].r * temp.i + x[i__4].i * temp.r;
		    r__1 = a[i__3].r + q__1.r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		} else {
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    r__1 = a[i__3].r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		}
/* L20: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    r_cnjg(&q__2, &x[jx]);
		    q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
		    temp.r = q__1.r, temp.i = q__1.i;
		    ix = kx;
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			ix += *incx;
/* L30: */
		    }
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    i__4 = jx;
		    q__1.r = x[i__4].r * temp.r - x[i__4].i * temp.i, q__1.i =
			     x[i__4].r * temp.i + x[i__4].i * temp.r;
		    r__1 = a[i__3].r + q__1.r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		} else {
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    r__1 = a[i__3].r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		}
		jx += *incx;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when A is stored in lower triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    r_cnjg(&q__2, &x[j]);
		    q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    i__4 = j;
		    q__1.r = temp.r * x[i__4].r - temp.i * x[i__4].i, q__1.i =
			     temp.r * x[i__4].i + temp.i * x[i__4].r;
		    r__1 = a[i__3].r + q__1.r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = i__;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L50: */
		    }
		} else {
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    r__1 = a[i__3].r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		}
/* L60: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = jx;
		if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
		    r_cnjg(&q__2, &x[jx]);
		    q__1.r = *alpha * q__2.r, q__1.i = *alpha * q__2.i;
		    temp.r = q__1.r, temp.i = q__1.i;
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    i__4 = jx;
		    q__1.r = temp.r * x[i__4].r - temp.i * x[i__4].i, q__1.i =
			     temp.r * x[i__4].i + temp.i * x[i__4].r;
		    r__1 = a[i__3].r + q__1.r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		    ix = jx;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			ix += *incx;
			i__3 = i__ + j * a_dim1;
			i__4 = i__ + j * a_dim1;
			i__5 = ix;
			q__2.r = x[i__5].r * temp.r - x[i__5].i * temp.i, 
				q__2.i = x[i__5].r * temp.i + x[i__5].i * 
				temp.r;
			q__1.r = a[i__4].r + q__2.r, q__1.i = a[i__4].i + 
				q__2.i;
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
/* L70: */
		    }
		} else {
		    i__2 = j + j * a_dim1;
		    i__3 = j + j * a_dim1;
		    r__1 = a[i__3].r;
		    a[i__2].r = r__1, a[i__2].i = 0.f;
		}
		jx += *incx;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of CHER  . */

} /* cher_ */
