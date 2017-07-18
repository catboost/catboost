/* cla_syamv.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cla_syamv__(integer *uplo, integer *n, real *alpha, 
	complex *a, integer *lda, complex *x, integer *incx, real *beta, real 
	*y, integer *incy)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1, r__2;

    /* Builtin functions */
    double r_imag(complex *), r_sign(real *, real *);

    /* Local variables */
    integer i__, j;
    logical symb_zero__;
    integer iy, jx, kx, ky, info;
    real temp, safe1;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilauplo_(char *);


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

/*  Purpose */
/*  ======= */

/*  CLA_SYAMV  performs the matrix-vector operation */

/*          y := alpha*abs(A)*abs(x) + beta*abs(y), */

/*  where alpha and beta are scalars, x and y are vectors and A is an */
/*  n by n symmetric matrix. */

/*  This function is primarily used in calculating error bounds. */
/*  To protect against underflow during evaluation, components in */
/*  the resulting vector are perturbed away from zero by (N+1) */
/*  times the underflow threshold.  To prevent unnecessarily large */
/*  errors for block-structure embedded in general matrices, */
/*  "symbolically" zero components are not perturbed.  A zero */
/*  entry is considered "symbolic" if all multiplications involved */
/*  in computing that entry have at least one zero multiplicand. */

/*  Parameters */
/*  ========== */

/*  UPLO   - INTEGER */
/*           On entry, UPLO specifies whether the upper or lower */
/*           triangular part of the array A is to be referenced as */
/*           follows: */

/*              UPLO = BLAS_UPPER   Only the upper triangular part of A */
/*                                  is to be referenced. */

/*              UPLO = BLAS_LOWER   Only the lower triangular part of A */
/*                                  is to be referenced. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the number of columns of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - COMPLEX             array of DIMENSION ( LDA, n ). */
/*           Before entry, the leading m by n part of the array A must */
/*           contain the matrix of coefficients. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           max( 1, n ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX             array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ) */
/*           Before entry, the incremented array X must contain the */
/*           vector x. */
/*           Unchanged on exit. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */

/*  BETA   - REAL            . */
/*           On entry, BETA specifies the scalar beta. When BETA is */
/*           supplied as zero then Y need not be set on input. */
/*           Unchanged on exit. */

/*  Y      - REAL             array of DIMENSION at least */
/*           ( 1 + ( n - 1 )*abs( INCY ) ) */
/*           Before entry with BETA non-zero, the incremented array Y */
/*           must contain the vector y. On exit, Y is overwritten by the */
/*           updated vector y. */

/*  INCY   - INTEGER. */
/*           On entry, INCY specifies the increment for the elements of */
/*           Y. INCY must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */
/*  -- Modified for the absolute-value product, April 2006 */
/*     Jason Riedy, UC Berkeley */

/*     .. */
/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    /* Function Body */
    info = 0;
    if (*uplo != ilauplo_("U") && *uplo != ilauplo_("L")
	    ) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*lda < max(1,*n)) {
	info = 5;
    } else if (*incx == 0) {
	info = 7;
    } else if (*incy == 0) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("SSYMV ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.f && *beta == 1.f) {
	return 0;
    }

/*     Set up the start points in  X  and  Y. */

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

/*     Set SAFE1 essentially to be the underflow threshold times the */
/*     number of additions in each row. */

    safe1 = slamch_("Safe minimum");
    safe1 = (*n + 1) * safe1;

/*     Form  y := alpha*abs(A)*abs(x) + beta*abs(y). */

/*     The O(N^2) SYMB_ZERO tests could be replaced by O(N) queries to */
/*     the inexact flag.  Still doesn't help change the iteration order */
/*     to per-column. */

    iy = ky;
    if (*incx == 1) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (*beta == 0.f) {
		symb_zero__ = TRUE_;
		y[iy] = 0.f;
	    } else if (y[iy] == 0.f) {
		symb_zero__ = TRUE_;
	    } else {
		symb_zero__ = FALSE_;
		y[iy] = *beta * (r__1 = y[iy], dabs(r__1));
	    }
	    if (*alpha != 0.f) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    if (*uplo == ilauplo_("U")) {
			if (i__ <= j) {
			    i__3 = i__ + j * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[i__ + j * a_dim1]), dabs(r__2));
			} else {
			    i__3 = j + i__ * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[j + i__ * a_dim1]), dabs(r__2));
			}
		    } else {
			if (i__ >= j) {
			    i__3 = i__ + j * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[i__ + j * a_dim1]), dabs(r__2));
			} else {
			    i__3 = j + i__ * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[j + i__ * a_dim1]), dabs(r__2));
			}
		    }
		    i__3 = j;
		    symb_zero__ = symb_zero__ && (x[i__3].r == 0.f && x[i__3]
			    .i == 0.f || temp == 0.f);
		    i__3 = j;
		    y[iy] += *alpha * ((r__1 = x[i__3].r, dabs(r__1)) + (r__2 
			    = r_imag(&x[j]), dabs(r__2))) * temp;
		}
	    }
	    if (! symb_zero__) {
		y[iy] += r_sign(&safe1, &y[iy]);
	    }
	    iy += *incy;
	}
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (*beta == 0.f) {
		symb_zero__ = TRUE_;
		y[iy] = 0.f;
	    } else if (y[iy] == 0.f) {
		symb_zero__ = TRUE_;
	    } else {
		symb_zero__ = FALSE_;
		y[iy] = *beta * (r__1 = y[iy], dabs(r__1));
	    }
	    jx = kx;
	    if (*alpha != 0.f) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    if (*uplo == ilauplo_("U")) {
			if (i__ <= j) {
			    i__3 = i__ + j * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[i__ + j * a_dim1]), dabs(r__2));
			} else {
			    i__3 = j + i__ * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[j + i__ * a_dim1]), dabs(r__2));
			}
		    } else {
			if (i__ >= j) {
			    i__3 = i__ + j * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[i__ + j * a_dim1]), dabs(r__2));
			} else {
			    i__3 = j + i__ * a_dim1;
			    temp = (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&a[j + i__ * a_dim1]), dabs(r__2));
			}
		    }
		    i__3 = j;
		    symb_zero__ = symb_zero__ && (x[i__3].r == 0.f && x[i__3]
			    .i == 0.f || temp == 0.f);
		    i__3 = jx;
		    y[iy] += *alpha * ((r__1 = x[i__3].r, dabs(r__1)) + (r__2 
			    = r_imag(&x[jx]), dabs(r__2))) * temp;
		    jx += *incx;
		}
	    }
	    if (! symb_zero__) {
		y[iy] += r_sign(&safe1, &y[iy]);
	    }
	    iy += *incy;
	}
    }

    return 0;

/*     End of CLA_SYAMV */

} /* cla_syamv__ */
