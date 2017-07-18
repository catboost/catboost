/* cpttrf.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cpttrf_(integer *n, real *d__, complex *e, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    real f, g;
    integer i__, i4;
    real eii, eir;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPTTRF computes the L*D*L' factorization of a complex Hermitian */
/*  positive definite tridiagonal matrix A.  The factorization may also */
/*  be regarded as having the form A = U'*D*U. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  D       (input/output) REAL array, dimension (N) */
/*          On entry, the n diagonal elements of the tridiagonal matrix */
/*          A.  On exit, the n diagonal elements of the diagonal matrix */
/*          D from the L*D*L' factorization of A. */

/*  E       (input/output) COMPLEX array, dimension (N-1) */
/*          On entry, the (n-1) subdiagonal elements of the tridiagonal */
/*          matrix A.  On exit, the (n-1) subdiagonal elements of the */
/*          unit bidiagonal factor L from the L*D*L' factorization of A. */
/*          E can also be regarded as the superdiagonal of the unit */
/*          bidiagonal factor U from the U'*D*U factorization of A. */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -k, the k-th argument had an illegal value */
/*          > 0: if INFO = k, the leading minor of order k is not */
/*               positive definite; if k < N, the factorization could not */
/*               be completed, while if k = N, the factorization was */
/*               completed, but D(N) <= 0. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("CPTTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Compute the L*D*L' (or U'*D*U) factorization of A. */

    i4 = (*n - 1) % 4;
    i__1 = i4;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] <= 0.f) {
	    *info = i__;
	    goto L20;
	}
	i__2 = i__;
	eir = e[i__2].r;
	eii = r_imag(&e[i__]);
	f = eir / d__[i__];
	g = eii / d__[i__];
	i__2 = i__;
	q__1.r = f, q__1.i = g;
	e[i__2].r = q__1.r, e[i__2].i = q__1.i;
	d__[i__ + 1] = d__[i__ + 1] - f * eir - g * eii;
/* L10: */
    }

    i__1 = *n - 4;
    for (i__ = i4 + 1; i__ <= i__1; i__ += 4) {

/*        Drop out of the loop if d(i) <= 0: the matrix is not positive */
/*        definite. */

	if (d__[i__] <= 0.f) {
	    *info = i__;
	    goto L20;
	}

/*        Solve for e(i) and d(i+1). */

	i__2 = i__;
	eir = e[i__2].r;
	eii = r_imag(&e[i__]);
	f = eir / d__[i__];
	g = eii / d__[i__];
	i__2 = i__;
	q__1.r = f, q__1.i = g;
	e[i__2].r = q__1.r, e[i__2].i = q__1.i;
	d__[i__ + 1] = d__[i__ + 1] - f * eir - g * eii;

	if (d__[i__ + 1] <= 0.f) {
	    *info = i__ + 1;
	    goto L20;
	}

/*        Solve for e(i+1) and d(i+2). */

	i__2 = i__ + 1;
	eir = e[i__2].r;
	eii = r_imag(&e[i__ + 1]);
	f = eir / d__[i__ + 1];
	g = eii / d__[i__ + 1];
	i__2 = i__ + 1;
	q__1.r = f, q__1.i = g;
	e[i__2].r = q__1.r, e[i__2].i = q__1.i;
	d__[i__ + 2] = d__[i__ + 2] - f * eir - g * eii;

	if (d__[i__ + 2] <= 0.f) {
	    *info = i__ + 2;
	    goto L20;
	}

/*        Solve for e(i+2) and d(i+3). */

	i__2 = i__ + 2;
	eir = e[i__2].r;
	eii = r_imag(&e[i__ + 2]);
	f = eir / d__[i__ + 2];
	g = eii / d__[i__ + 2];
	i__2 = i__ + 2;
	q__1.r = f, q__1.i = g;
	e[i__2].r = q__1.r, e[i__2].i = q__1.i;
	d__[i__ + 3] = d__[i__ + 3] - f * eir - g * eii;

	if (d__[i__ + 3] <= 0.f) {
	    *info = i__ + 3;
	    goto L20;
	}

/*        Solve for e(i+3) and d(i+4). */

	i__2 = i__ + 3;
	eir = e[i__2].r;
	eii = r_imag(&e[i__ + 3]);
	f = eir / d__[i__ + 3];
	g = eii / d__[i__ + 3];
	i__2 = i__ + 3;
	q__1.r = f, q__1.i = g;
	e[i__2].r = q__1.r, e[i__2].i = q__1.i;
	d__[i__ + 4] = d__[i__ + 4] - f * eir - g * eii;
/* L110: */
    }

/*     Check d(n) for positive definiteness. */

    if (d__[*n] <= 0.f) {
	*info = *n;
    }

L20:
    return 0;

/*     End of CPTTRF */

} /* cpttrf_ */
