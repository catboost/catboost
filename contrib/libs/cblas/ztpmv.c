/* ztpmv.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ztpmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublecomplex *ap, doublecomplex *x, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j, k, kk, ix, jx, kx, info;
    doublecomplex temp;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTPMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular matrix, supplied in packed form. */

/*  Arguments */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := conjg( A' )*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  AP     - COMPLEX*16       array of DIMENSION at least */
/*           ( ( n*( n + 1 ) )/2 ). */
/*           Before entry with  UPLO = 'U' or 'u', the array AP must */
/*           contain the upper triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 1, 2 ) and a( 2, 2 ) */
/*           respectively, and so on. */
/*           Before entry with UPLO = 'L' or 'l', the array AP must */
/*           contain the lower triangular matrix packed sequentially, */
/*           column by column, so that AP( 1 ) contains a( 1, 1 ), */
/*           AP( 2 ) and AP( 3 ) contain a( 2, 1 ) and a( 3, 1 ) */
/*           respectively, and so on. */
/*           Note that when  DIAG = 'U' or 'u', the diagonal elements of */
/*           A are not referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
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
    --ap;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, 
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, 
	    "N")) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*incx == 0) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("ZTPMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = lsame_(trans, "T");
    nounit = lsame_(diag, "N");

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX  too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of AP are */
/*     accessed sequentially with one pass through AP. */

    if (lsame_(trans, "N")) {

/*        Form  x:= A*x. */

	if (lsame_(uplo, "U")) {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].r != 0. || x[i__2].i != 0.) {
			i__2 = j;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			k = kk;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    i__3 = i__;
			    i__4 = i__;
			    i__5 = k;
			    z__2.r = temp.r * ap[i__5].r - temp.i * ap[i__5]
				    .i, z__2.i = temp.r * ap[i__5].i + temp.i 
				    * ap[i__5].r;
			    z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i + 
				    z__2.i;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
			    ++k;
/* L10: */
			}
			if (nounit) {
			    i__2 = j;
			    i__3 = j;
			    i__4 = kk + j - 1;
			    z__1.r = x[i__3].r * ap[i__4].r - x[i__3].i * ap[
				    i__4].i, z__1.i = x[i__3].r * ap[i__4].i 
				    + x[i__3].i * ap[i__4].r;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			}
		    }
		    kk += j;
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    if (x[i__2].r != 0. || x[i__2].i != 0.) {
			i__2 = jx;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			ix = kx;
			i__2 = kk + j - 2;
			for (k = kk; k <= i__2; ++k) {
			    i__3 = ix;
			    i__4 = ix;
			    i__5 = k;
			    z__2.r = temp.r * ap[i__5].r - temp.i * ap[i__5]
				    .i, z__2.i = temp.r * ap[i__5].i + temp.i 
				    * ap[i__5].r;
			    z__1.r = x[i__4].r + z__2.r, z__1.i = x[i__4].i + 
				    z__2.i;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__2 = jx;
			    i__3 = jx;
			    i__4 = kk + j - 1;
			    z__1.r = x[i__3].r * ap[i__4].r - x[i__3].i * ap[
				    i__4].i, z__1.i = x[i__3].r * ap[i__4].i 
				    + x[i__3].i * ap[i__4].r;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			}
		    }
		    jx += *incx;
		    kk += j;
/* L40: */
		}
	    }
	} else {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].r != 0. || x[i__1].i != 0.) {
			i__1 = j;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			k = kk;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__4 = k;
			    z__2.r = temp.r * ap[i__4].r - temp.i * ap[i__4]
				    .i, z__2.i = temp.r * ap[i__4].i + temp.i 
				    * ap[i__4].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			    --k;
/* L50: */
			}
			if (nounit) {
			    i__1 = j;
			    i__2 = j;
			    i__3 = kk - *n + j;
			    z__1.r = x[i__2].r * ap[i__3].r - x[i__2].i * ap[
				    i__3].i, z__1.i = x[i__2].r * ap[i__3].i 
				    + x[i__2].i * ap[i__3].r;
			    x[i__1].r = z__1.r, x[i__1].i = z__1.i;
			}
		    }
		    kk -= *n - j + 1;
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    if (x[i__1].r != 0. || x[i__1].i != 0.) {
			i__1 = jx;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			ix = kx;
			i__1 = kk - (*n - (j + 1));
			for (k = kk; k >= i__1; --k) {
			    i__2 = ix;
			    i__3 = ix;
			    i__4 = k;
			    z__2.r = temp.r * ap[i__4].r - temp.i * ap[i__4]
				    .i, z__2.i = temp.r * ap[i__4].i + temp.i 
				    * ap[i__4].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__1 = jx;
			    i__2 = jx;
			    i__3 = kk - *n + j;
			    z__1.r = x[i__2].r * ap[i__3].r - x[i__2].i * ap[
				    i__3].i, z__1.i = x[i__2].r * ap[i__3].i 
				    + x[i__2].i * ap[i__3].r;
			    x[i__1].r = z__1.r, x[i__1].i = z__1.i;
			}
		    }
		    jx -= *incx;
		    kk -= *n - j + 1;
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

	if (lsame_(uplo, "U")) {
	    kk = *n * (*n + 1) / 2;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    temp.r = x[i__1].r, temp.i = x[i__1].i;
		    k = kk - 1;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    z__1.r = temp.r * ap[i__1].r - temp.i * ap[i__1]
				    .i, z__1.i = temp.r * ap[i__1].i + temp.i 
				    * ap[i__1].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    i__1 = k;
			    i__2 = i__;
			    z__2.r = ap[i__1].r * x[i__2].r - ap[i__1].i * x[
				    i__2].i, z__2.i = ap[i__1].r * x[i__2].i 
				    + ap[i__1].i * x[i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    --k;
/* L90: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &ap[kk]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			for (i__ = j - 1; i__ >= 1; --i__) {
			    d_cnjg(&z__3, &ap[k]);
			    i__1 = i__;
			    z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i, 
				    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
				    i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    --k;
/* L100: */
			}
		    }
		    i__1 = j;
		    x[i__1].r = temp.r, x[i__1].i = temp.i;
		    kk -= j;
/* L110: */
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    i__1 = jx;
		    temp.r = x[i__1].r, temp.i = x[i__1].i;
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__1 = kk;
			    z__1.r = temp.r * ap[i__1].r - temp.i * ap[i__1]
				    .i, z__1.i = temp.r * ap[i__1].i + temp.i 
				    * ap[i__1].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    i__2 = k;
			    i__3 = ix;
			    z__2.r = ap[i__2].r * x[i__3].r - ap[i__2].i * x[
				    i__3].i, z__2.i = ap[i__2].r * x[i__3].i 
				    + ap[i__2].i * x[i__3].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L120: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &ap[kk]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__1 = kk - j + 1;
			for (k = kk - 1; k >= i__1; --k) {
			    ix -= *incx;
			    d_cnjg(&z__3, &ap[k]);
			    i__2 = ix;
			    z__2.r = z__3.r * x[i__2].r - z__3.i * x[i__2].i, 
				    z__2.i = z__3.r * x[i__2].i + z__3.i * x[
				    i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L130: */
			}
		    }
		    i__1 = jx;
		    x[i__1].r = temp.r, x[i__1].i = temp.i;
		    jx -= *incx;
		    kk -= j;
/* L140: */
		}
	    }
	} else {
	    kk = 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    temp.r = x[i__2].r, temp.i = x[i__2].i;
		    k = kk + 1;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    z__1.r = temp.r * ap[i__2].r - temp.i * ap[i__2]
				    .i, z__1.i = temp.r * ap[i__2].i + temp.i 
				    * ap[i__2].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    i__3 = k;
			    i__4 = i__;
			    z__2.r = ap[i__3].r * x[i__4].r - ap[i__3].i * x[
				    i__4].i, z__2.i = ap[i__3].r * x[i__4].i 
				    + ap[i__3].i * x[i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ++k;
/* L150: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &ap[kk]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = *n;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
			    d_cnjg(&z__3, &ap[k]);
			    i__3 = i__;
			    z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, 
				    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
				    i__3].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ++k;
/* L160: */
			}
		    }
		    i__2 = j;
		    x[i__2].r = temp.r, x[i__2].i = temp.i;
		    kk += *n - j + 1;
/* L170: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = jx;
		    temp.r = x[i__2].r, temp.i = x[i__2].i;
		    ix = jx;
		    if (noconj) {
			if (nounit) {
			    i__2 = kk;
			    z__1.r = temp.r * ap[i__2].r - temp.i * ap[i__2]
				    .i, z__1.i = temp.r * ap[i__2].i + temp.i 
				    * ap[i__2].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    i__3 = k;
			    i__4 = ix;
			    z__2.r = ap[i__3].r * x[i__4].r - ap[i__3].i * x[
				    i__4].i, z__2.i = ap[i__3].r * x[i__4].i 
				    + ap[i__3].i * x[i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L180: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &ap[kk]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
			i__2 = kk + *n - j;
			for (k = kk + 1; k <= i__2; ++k) {
			    ix += *incx;
			    d_cnjg(&z__3, &ap[k]);
			    i__3 = ix;
			    z__2.r = z__3.r * x[i__3].r - z__3.i * x[i__3].i, 
				    z__2.i = z__3.r * x[i__3].i + z__3.i * x[
				    i__3].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L190: */
			}
		    }
		    i__2 = jx;
		    x[i__2].r = temp.r, x[i__2].i = temp.i;
		    jx += *incx;
		    kk += *n - j + 1;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of ZTPMV . */

} /* ztpmv_ */
