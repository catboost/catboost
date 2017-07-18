/* slagtm.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slagtm_(char *trans, integer *n, integer *nrhs, real *
	alpha, real *dl, real *d__, real *du, real *x, integer *ldx, real *
	beta, real *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;
    extern logical lsame_(char *, char *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAGTM performs a matrix-vector product of the form */

/*     B := alpha * A * X + beta * B */

/*  where A is a tridiagonal matrix of order N, B and X are N by NRHS */
/*  matrices, and alpha and beta are real scalars, each of which may be */
/*  0., 1., or -1. */

/*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the operation applied to A. */
/*          = 'N':  No transpose, B := alpha * A * X + beta * B */
/*          = 'T':  Transpose,    B := alpha * A'* X + beta * B */
/*          = 'C':  Conjugate transpose = Transpose */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices X and B. */

/*  ALPHA   (input) REAL */
/*          The scalar alpha.  ALPHA must be 0., 1., or -1.; otherwise, */
/*          it is assumed to be 0. */

/*  DL      (input) REAL array, dimension (N-1) */
/*          The (n-1) sub-diagonal elements of T. */

/*  D       (input) REAL array, dimension (N) */
/*          The diagonal elements of T. */

/*  DU      (input) REAL array, dimension (N-1) */
/*          The (n-1) super-diagonal elements of T. */

/*  X       (input) REAL array, dimension (LDX,NRHS) */
/*          The N by NRHS matrix X. */
/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(N,1). */

/*  BETA    (input) REAL */
/*          The scalar beta.  BETA must be 0., 1., or -1.; otherwise, */
/*          it is assumed to be 1. */

/*  B       (input/output) REAL array, dimension (LDB,NRHS) */
/*          On entry, the N by NRHS matrix B. */
/*          On exit, B is overwritten by the matrix expression */
/*          B := alpha * A * X + beta * B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(N,1). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (*n == 0) {
	return 0;
    }

/*     Multiply B by BETA if BETA.NE.1. */

    if (*beta == 0.f) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = 0.f;
/* L10: */
	    }
/* L20: */
	}
    } else if (*beta == -1.f) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = -b[i__ + j * b_dim1];
/* L30: */
	    }
/* L40: */
	}
    }

    if (*alpha == 1.f) {
	if (lsame_(trans, "N")) {

/*           Compute B := B + A*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    b[j * b_dim1 + 1] += d__[1] * x[j * x_dim1 + 1];
		} else {
		    b[j * b_dim1 + 1] = b[j * b_dim1 + 1] + d__[1] * x[j * 
			    x_dim1 + 1] + du[1] * x[j * x_dim1 + 2];
		    b[*n + j * b_dim1] = b[*n + j * b_dim1] + dl[*n - 1] * x[*
			    n - 1 + j * x_dim1] + d__[*n] * x[*n + j * x_dim1]
			    ;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = b[i__ + j * b_dim1] + dl[i__ - 
				1] * x[i__ - 1 + j * x_dim1] + d__[i__] * x[
				i__ + j * x_dim1] + du[i__] * x[i__ + 1 + j * 
				x_dim1];
/* L50: */
		    }
		}
/* L60: */
	    }
	} else {

/*           Compute B := B + A'*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    b[j * b_dim1 + 1] += d__[1] * x[j * x_dim1 + 1];
		} else {
		    b[j * b_dim1 + 1] = b[j * b_dim1 + 1] + d__[1] * x[j * 
			    x_dim1 + 1] + dl[1] * x[j * x_dim1 + 2];
		    b[*n + j * b_dim1] = b[*n + j * b_dim1] + du[*n - 1] * x[*
			    n - 1 + j * x_dim1] + d__[*n] * x[*n + j * x_dim1]
			    ;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = b[i__ + j * b_dim1] + du[i__ - 
				1] * x[i__ - 1 + j * x_dim1] + d__[i__] * x[
				i__ + j * x_dim1] + dl[i__] * x[i__ + 1 + j * 
				x_dim1];
/* L70: */
		    }
		}
/* L80: */
	    }
	}
    } else if (*alpha == -1.f) {
	if (lsame_(trans, "N")) {

/*           Compute B := B - A*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    b[j * b_dim1 + 1] -= d__[1] * x[j * x_dim1 + 1];
		} else {
		    b[j * b_dim1 + 1] = b[j * b_dim1 + 1] - d__[1] * x[j * 
			    x_dim1 + 1] - du[1] * x[j * x_dim1 + 2];
		    b[*n + j * b_dim1] = b[*n + j * b_dim1] - dl[*n - 1] * x[*
			    n - 1 + j * x_dim1] - d__[*n] * x[*n + j * x_dim1]
			    ;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = b[i__ + j * b_dim1] - dl[i__ - 
				1] * x[i__ - 1 + j * x_dim1] - d__[i__] * x[
				i__ + j * x_dim1] - du[i__] * x[i__ + 1 + j * 
				x_dim1];
/* L90: */
		    }
		}
/* L100: */
	    }
	} else {

/*           Compute B := B - A'*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    b[j * b_dim1 + 1] -= d__[1] * x[j * x_dim1 + 1];
		} else {
		    b[j * b_dim1 + 1] = b[j * b_dim1 + 1] - d__[1] * x[j * 
			    x_dim1 + 1] - dl[1] * x[j * x_dim1 + 2];
		    b[*n + j * b_dim1] = b[*n + j * b_dim1] - du[*n - 1] * x[*
			    n - 1 + j * x_dim1] - d__[*n] * x[*n + j * x_dim1]
			    ;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = b[i__ + j * b_dim1] - du[i__ - 
				1] * x[i__ - 1 + j * x_dim1] - d__[i__] * x[
				i__ + j * x_dim1] - dl[i__] * x[i__ + 1 + j * 
				x_dim1];
/* L110: */
		    }
		}
/* L120: */
	    }
	}
    }
    return 0;

/*     End of SLAGTM */

} /* slagtm_ */
