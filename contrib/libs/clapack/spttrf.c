/* spttrf.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int spttrf_(integer *n, real *d__, real *e, integer *info)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, i4;
    real ei;
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

/*  SPTTRF computes the L*D*L' factorization of a real symmetric */
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

/*  E       (input/output) REAL array, dimension (N-1) */
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
	xerbla_("SPTTRF", &i__1);
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
	    goto L30;
	}
	ei = e[i__];
	e[i__] = ei / d__[i__];
	d__[i__ + 1] -= e[i__] * ei;
/* L10: */
    }

    i__1 = *n - 4;
    for (i__ = i4 + 1; i__ <= i__1; i__ += 4) {

/*        Drop out of the loop if d(i) <= 0: the matrix is not positive */
/*        definite. */

	if (d__[i__] <= 0.f) {
	    *info = i__;
	    goto L30;
	}

/*        Solve for e(i) and d(i+1). */

	ei = e[i__];
	e[i__] = ei / d__[i__];
	d__[i__ + 1] -= e[i__] * ei;

	if (d__[i__ + 1] <= 0.f) {
	    *info = i__ + 1;
	    goto L30;
	}

/*        Solve for e(i+1) and d(i+2). */

	ei = e[i__ + 1];
	e[i__ + 1] = ei / d__[i__ + 1];
	d__[i__ + 2] -= e[i__ + 1] * ei;

	if (d__[i__ + 2] <= 0.f) {
	    *info = i__ + 2;
	    goto L30;
	}

/*        Solve for e(i+2) and d(i+3). */

	ei = e[i__ + 2];
	e[i__ + 2] = ei / d__[i__ + 2];
	d__[i__ + 3] -= e[i__ + 2] * ei;

	if (d__[i__ + 3] <= 0.f) {
	    *info = i__ + 3;
	    goto L30;
	}

/*        Solve for e(i+3) and d(i+4). */

	ei = e[i__ + 3];
	e[i__ + 3] = ei / d__[i__ + 3];
	d__[i__ + 4] -= e[i__ + 3] * ei;
/* L20: */
    }

/*     Check d(n) for positive definiteness. */

    if (d__[*n] <= 0.f) {
	*info = *n;
    }

L30:
    return 0;

/*     End of SPTTRF */

} /* spttrf_ */
