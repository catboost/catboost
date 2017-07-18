/* cptts2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cptts2_(integer *iuplo, integer *n, integer *nrhs, real *
	d__, complex *e, complex *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j;
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer 
	    *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPTTS2 solves a tridiagonal system of the form */
/*     A * X = B */
/*  using the factorization A = U'*D*U or A = L*D*L' computed by CPTTRF. */
/*  D is a diagonal matrix specified in the vector D, U (or L) is a unit */
/*  bidiagonal matrix whose superdiagonal (subdiagonal) is specified in */
/*  the vector E, and X and B are N by NRHS matrices. */

/*  Arguments */
/*  ========= */

/*  IUPLO   (input) INTEGER */
/*          Specifies the form of the factorization and whether the */
/*          vector E is the superdiagonal of the upper bidiagonal factor */
/*          U or the subdiagonal of the lower bidiagonal factor L. */
/*          = 1:  A = U'*D*U, E is the superdiagonal of U */
/*          = 0:  A = L*D*L', E is the subdiagonal of L */

/*  N       (input) INTEGER */
/*          The order of the tridiagonal matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  D       (input) REAL array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          factorization A = U'*D*U or A = L*D*L'. */

/*  E       (input) COMPLEX array, dimension (N-1) */
/*          If IUPLO = 1, the (n-1) superdiagonal elements of the unit */
/*          bidiagonal factor U from the factorization A = U'*D*U. */
/*          If IUPLO = 0, the (n-1) subdiagonal elements of the unit */
/*          bidiagonal factor L from the factorization A = L*D*L'. */

/*  B       (input/output) REAL array, dimension (LDB,NRHS) */
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
/*     .. Intrinsic Functions .. */
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
	    r__1 = 1.f / d__[1];
	    csscal_(nrhs, &r__1, &b[b_offset], ldb);
	}
	return 0;
    }

    if (*iuplo == 1) {

/*        Solve A * X = B using the factorization A = U'*D*U, */
/*        overwriting each right hand side vector with its solution. */

	if (*nrhs <= 2) {
	    j = 1;
L5:

/*           Solve U' * x = b. */

	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		i__4 = i__ - 1 + j * b_dim1;
		r_cnjg(&q__3, &e[i__ - 1]);
		q__2.r = b[i__4].r * q__3.r - b[i__4].i * q__3.i, q__2.i = b[
			i__4].r * q__3.i + b[i__4].i * q__3.r;
		q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - q__2.i;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L10: */
	    }

/*           Solve D * U * x = b. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		i__4 = i__;
		q__1.r = b[i__3].r / d__[i__4], q__1.i = b[i__3].i / d__[i__4]
			;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L20: */
	    }
	    for (i__ = *n - 1; i__ >= 1; --i__) {
		i__1 = i__ + j * b_dim1;
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + 1 + j * b_dim1;
		i__4 = i__;
		q__2.r = b[i__3].r * e[i__4].r - b[i__3].i * e[i__4].i, 
			q__2.i = b[i__3].r * e[i__4].i + b[i__3].i * e[i__4]
			.r;
		q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
/* L30: */
	    }
	    if (j < *nrhs) {
		++j;
		goto L5;
	    }
	} else {
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {

/*              Solve U' * x = b. */

		i__2 = *n;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ + j * b_dim1;
		    i__5 = i__ - 1 + j * b_dim1;
		    r_cnjg(&q__3, &e[i__ - 1]);
		    q__2.r = b[i__5].r * q__3.r - b[i__5].i * q__3.i, q__2.i =
			     b[i__5].r * q__3.i + b[i__5].i * q__3.r;
		    q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4].i - q__2.i;
		    b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L40: */
		}

/*              Solve D * U * x = b. */

		i__2 = *n + j * b_dim1;
		i__3 = *n + j * b_dim1;
		i__4 = *n;
		q__1.r = b[i__3].r / d__[i__4], q__1.i = b[i__3].i / d__[i__4]
			;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		for (i__ = *n - 1; i__ >= 1; --i__) {
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__;
		    q__2.r = b[i__3].r / d__[i__4], q__2.i = b[i__3].i / d__[
			    i__4];
		    i__5 = i__ + 1 + j * b_dim1;
		    i__6 = i__;
		    q__3.r = b[i__5].r * e[i__6].r - b[i__5].i * e[i__6].i, 
			    q__3.i = b[i__5].r * e[i__6].i + b[i__5].i * e[
			    i__6].r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L50: */
		}
/* L60: */
	    }
	}
    } else {

/*        Solve A * X = B using the factorization A = L*D*L', */
/*        overwriting each right hand side vector with its solution. */

	if (*nrhs <= 2) {
	    j = 1;
L65:

/*           Solve L * x = b. */

	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		i__4 = i__ - 1 + j * b_dim1;
		i__5 = i__ - 1;
		q__2.r = b[i__4].r * e[i__5].r - b[i__4].i * e[i__5].i, 
			q__2.i = b[i__4].r * e[i__5].i + b[i__4].i * e[i__5]
			.r;
		q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - q__2.i;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L70: */
	    }

/*           Solve D * L' * x = b. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		i__4 = i__;
		q__1.r = b[i__3].r / d__[i__4], q__1.i = b[i__3].i / d__[i__4]
			;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L80: */
	    }
	    for (i__ = *n - 1; i__ >= 1; --i__) {
		i__1 = i__ + j * b_dim1;
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + 1 + j * b_dim1;
		r_cnjg(&q__3, &e[i__]);
		q__2.r = b[i__3].r * q__3.r - b[i__3].i * q__3.i, q__2.i = b[
			i__3].r * q__3.i + b[i__3].i * q__3.r;
		q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
/* L90: */
	    }
	    if (j < *nrhs) {
		++j;
		goto L65;
	    }
	} else {
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {

/*              Solve L * x = b. */

		i__2 = *n;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ + j * b_dim1;
		    i__5 = i__ - 1 + j * b_dim1;
		    i__6 = i__ - 1;
		    q__2.r = b[i__5].r * e[i__6].r - b[i__5].i * e[i__6].i, 
			    q__2.i = b[i__5].r * e[i__6].i + b[i__5].i * e[
			    i__6].r;
		    q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4].i - q__2.i;
		    b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L100: */
		}

/*              Solve D * L' * x = b. */

		i__2 = *n + j * b_dim1;
		i__3 = *n + j * b_dim1;
		i__4 = *n;
		q__1.r = b[i__3].r / d__[i__4], q__1.i = b[i__3].i / d__[i__4]
			;
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		for (i__ = *n - 1; i__ >= 1; --i__) {
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__;
		    q__2.r = b[i__3].r / d__[i__4], q__2.i = b[i__3].i / d__[
			    i__4];
		    i__5 = i__ + 1 + j * b_dim1;
		    r_cnjg(&q__4, &e[i__]);
		    q__3.r = b[i__5].r * q__4.r - b[i__5].i * q__4.i, q__3.i =
			     b[i__5].r * q__4.i + b[i__5].i * q__4.r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L110: */
		}
/* L120: */
	    }
	}
    }

    return 0;

/*     End of CPTTS2 */

} /* cptts2_ */
