/* dptts2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dptts2_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    integer i__, j;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DPTTS2 solves a tridiagonal system of the form */
/*     A * X = B */
/*  using the L*D*L' factorization of A computed by DPTTRF.  D is a */
/*  diagonal matrix specified in the vector D, L is a unit bidiagonal */
/*  matrix whose subdiagonal is specified in the vector E, and X and B */
/*  are N by NRHS matrices. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the tridiagonal matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          L*D*L' factorization of A. */

/*  E       (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (n-1) subdiagonal elements of the unit bidiagonal factor */
/*          L from the L*D*L' factorization of A.  E can also be regarded */
/*          as the superdiagonal of the unit bidiagonal factor U from the */
/*          factorization A = U'*D*U. */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*          On entry, the right hand side vectors B for the system of */
/*          linear equations. */
/*          On exit, the solution vectors, X. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    --d__;
    --e;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (*n <= 1) {
	if (*n == 1) {
	    d__1 = 1. / d__[1];
	    dscal_(nrhs, &d__1, &b[b_offset], ldb);
	}
	return 0;
    }

/*     Solve A * X = B using the factorization A = L*D*L', */
/*     overwriting each right hand side vector with its solution. */

    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {

/*           Solve L * x = b. */

	i__2 = *n;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    b[i__ + j * b_dim1] -= b[i__ - 1 + j * b_dim1] * e[i__ - 1];
/* L10: */
	}

/*           Solve D * L' * x = b. */

	b[*n + j * b_dim1] /= d__[*n];
	for (i__ = *n - 1; i__ >= 1; --i__) {
	    b[i__ + j * b_dim1] = b[i__ + j * b_dim1] / d__[i__] - b[i__ + 1 
		    + j * b_dim1] * e[i__];
/* L20: */
	}
/* L30: */
    }

    return 0;

/*     End of DPTTS2 */

} /* dptts2_ */
