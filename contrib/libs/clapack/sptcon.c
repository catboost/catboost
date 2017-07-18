/* sptcon.f -- translated by f2c (version 20061008).
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

/* Table of constant values */

static integer c__1 = 1;

/* Subroutine */ int sptcon_(integer *n, real *d__, real *e, real *anorm, 
	real *rcond, real *work, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    integer i__, ix;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    real ainvnm;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SPTCON computes the reciprocal of the condition number (in the */
/*  1-norm) of a real symmetric positive definite tridiagonal matrix */
/*  using the factorization A = L*D*L**T or A = U**T*D*U computed by */
/*  SPTTRF. */

/*  Norm(inv(A)) is computed by a direct method, and the reciprocal of */
/*  the condition number is computed as */
/*               RCOND = 1 / (ANORM * norm(inv(A))). */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  D       (input) REAL array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          factorization of A, as computed by SPTTRF. */

/*  E       (input) REAL array, dimension (N-1) */
/*          The (n-1) off-diagonal elements of the unit bidiagonal factor */
/*          U or L from the factorization of A,  as computed by SPTTRF. */

/*  ANORM   (input) REAL */
/*          The 1-norm of the original matrix A. */

/*  RCOND   (output) REAL */
/*          The reciprocal of the condition number of the matrix A, */
/*          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is the */
/*          1-norm of inv(A) computed in this routine. */

/*  WORK    (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The method used is described in Nicholas J. Higham, "Efficient */
/*  Algorithms for Computing the Condition Number of a Tridiagonal */
/*  Matrix", SIAM J. Sci. Stat. Comput., Vol. 7, No. 1, January 1986. */

/*  ===================================================================== */

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
/*     .. Executable Statements .. */

/*     Test the input arguments. */

    /* Parameter adjustments */
    --work;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*anorm < 0.f) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPTCON", &i__1);
	return 0;
    }

/*     Quick return if possible */

    *rcond = 0.f;
    if (*n == 0) {
	*rcond = 1.f;
	return 0;
    } else if (*anorm == 0.f) {
	return 0;
    }

/*     Check that D(1:N) is positive. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] <= 0.f) {
	    return 0;
	}
/* L10: */
    }

/*     Solve M(A) * x = e, where M(A) = (m(i,j)) is given by */

/*        m(i,j) =  abs(A(i,j)), i = j, */
/*        m(i,j) = -abs(A(i,j)), i .ne. j, */

/*     and e = [ 1, 1, ..., 1 ]'.  Note M(A) = M(L)*D*M(L)'. */

/*     Solve M(L) * x = e. */

    work[1] = 1.f;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	work[i__] = work[i__ - 1] * (r__1 = e[i__ - 1], dabs(r__1)) + 1.f;
/* L20: */
    }

/*     Solve D * M(L)' * x = b. */

    work[*n] /= d__[*n];
    for (i__ = *n - 1; i__ >= 1; --i__) {
	work[i__] = work[i__] / d__[i__] + work[i__ + 1] * (r__1 = e[i__], 
		dabs(r__1));
/* L30: */
    }

/*     Compute AINVNM = max(x(i)), 1<=i<=n. */

    ix = isamax_(n, &work[1], &c__1);
    ainvnm = (r__1 = work[ix], dabs(r__1));

/*     Compute the reciprocal condition number. */

    if (ainvnm != 0.f) {
	*rcond = 1.f / ainvnm / *anorm;
    }

    return 0;

/*     End of SPTCON */

} /* sptcon_ */
