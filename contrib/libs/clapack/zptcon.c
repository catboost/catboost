/* zptcon.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zptcon_(integer *n, doublereal *d__, doublecomplex *e, 
	doublereal *anorm, doublereal *rcond, doublereal *rwork, integer *
	info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;

    /* Builtin functions */
    double z_abs(doublecomplex *);

    /* Local variables */
    integer i__, ix;
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal ainvnm;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZPTCON computes the reciprocal of the condition number (in the */
/*  1-norm) of a complex Hermitian positive definite tridiagonal matrix */
/*  using the factorization A = L*D*L**H or A = U**H*D*U computed by */
/*  ZPTTRF. */

/*  Norm(inv(A)) is computed by a direct method, and the reciprocal of */
/*  the condition number is computed as */
/*                   RCOND = 1 / (ANORM * norm(inv(A))). */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          factorization of A, as computed by ZPTTRF. */

/*  E       (input) COMPLEX*16 array, dimension (N-1) */
/*          The (n-1) off-diagonal elements of the unit bidiagonal factor */
/*          U or L from the factorization of A, as computed by ZPTTRF. */

/*  ANORM   (input) DOUBLE PRECISION */
/*          The 1-norm of the original matrix A. */

/*  RCOND   (output) DOUBLE PRECISION */
/*          The reciprocal of the condition number of the matrix A, */
/*          computed as RCOND = 1/(ANORM * AINVNM), where AINVNM is the */
/*          1-norm of inv(A) computed in this routine. */

/*  RWORK   (workspace) DOUBLE PRECISION array, dimension (N) */

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
    --rwork;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*anorm < 0.) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZPTCON", &i__1);
	return 0;
    }

/*     Quick return if possible */

    *rcond = 0.;
    if (*n == 0) {
	*rcond = 1.;
	return 0;
    } else if (*anorm == 0.) {
	return 0;
    }

/*     Check that D(1:N) is positive. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] <= 0.) {
	    return 0;
	}
/* L10: */
    }

/*     Solve M(A) * x = e, where M(A) = (m(i,j)) is given by */

/*        m(i,j) =  abs(A(i,j)), i = j, */
/*        m(i,j) = -abs(A(i,j)), i .ne. j, */

/*     and e = [ 1, 1, ..., 1 ]'.  Note M(A) = M(L)*D*M(L)'. */

/*     Solve M(L) * x = e. */

    rwork[1] = 1.;
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	rwork[i__] = rwork[i__ - 1] * z_abs(&e[i__ - 1]) + 1.;
/* L20: */
    }

/*     Solve D * M(L)' * x = b. */

    rwork[*n] /= d__[*n];
    for (i__ = *n - 1; i__ >= 1; --i__) {
	rwork[i__] = rwork[i__] / d__[i__] + rwork[i__ + 1] * z_abs(&e[i__]);
/* L30: */
    }

/*     Compute AINVNM = max(x(i)), 1<=i<=n. */

    ix = idamax_(n, &rwork[1], &c__1);
    ainvnm = (d__1 = rwork[ix], abs(d__1));

/*     Compute the reciprocal condition number. */

    if (ainvnm != 0.) {
	*rcond = 1. / ainvnm / *anorm;
    }

    return 0;

/*     End of ZPTCON */

} /* zptcon_ */
