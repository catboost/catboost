/* spteqr.f -- translated by f2c (version 20061008).
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

static real c_b7 = 0.f;
static real c_b8 = 1.f;
static integer c__0 = 0;
static integer c__1 = 1;

/* Subroutine */ int spteqr_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real c__[1]	/* was [1][1] */;
    integer i__;
    real vt[1]	/* was [1][1] */;
    integer nru;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, integer *), slaset_(
	    char *, integer *, integer *, real *, real *, real *, integer *), sbdsqr_(char *, integer *, integer *, integer *, integer 
	    *, real *, real *, real *, integer *, real *, integer *, real *, 
	    integer *, real *, integer *);
    integer icompz;
    extern /* Subroutine */ int spttrf_(integer *, real *, real *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SPTEQR computes all eigenvalues and, optionally, eigenvectors of a */
/*  symmetric positive definite tridiagonal matrix by first factoring the */
/*  matrix using SPTTRF, and then calling SBDSQR to compute the singular */
/*  values of the bidiagonal factor. */

/*  This routine computes the eigenvalues of the positive definite */
/*  tridiagonal matrix to high relative accuracy.  This means that if the */
/*  eigenvalues range over many orders of magnitude in size, then the */
/*  small eigenvalues and corresponding eigenvectors will be computed */
/*  more accurately than, for example, with the standard QR method. */

/*  The eigenvectors of a full or band symmetric positive definite matrix */
/*  can also be found if SSYTRD, SSPTRD, or SSBTRD has been used to */
/*  reduce this matrix to tridiagonal form. (The reduction to tridiagonal */
/*  form, however, may preclude the possibility of obtaining high */
/*  relative accuracy in the small eigenvalues of the original matrix, if */
/*  these eigenvalues range over many orders of magnitude.) */

/*  Arguments */
/*  ========= */

/*  COMPZ   (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only. */
/*          = 'V':  Compute eigenvectors of original symmetric */
/*                  matrix also.  Array Z contains the orthogonal */
/*                  matrix used to reduce the original matrix to */
/*                  tridiagonal form. */
/*          = 'I':  Compute eigenvectors of tridiagonal matrix also. */

/*  N       (input) INTEGER */
/*          The order of the matrix.  N >= 0. */

/*  D       (input/output) REAL array, dimension (N) */
/*          On entry, the n diagonal elements of the tridiagonal */
/*          matrix. */
/*          On normal exit, D contains the eigenvalues, in descending */
/*          order. */

/*  E       (input/output) REAL array, dimension (N-1) */
/*          On entry, the (n-1) subdiagonal elements of the tridiagonal */
/*          matrix. */
/*          On exit, E has been destroyed. */

/*  Z       (input/output) REAL array, dimension (LDZ, N) */
/*          On entry, if COMPZ = 'V', the orthogonal matrix used in the */
/*          reduction to tridiagonal form. */
/*          On exit, if COMPZ = 'V', the orthonormal eigenvectors of the */
/*          original symmetric matrix; */
/*          if COMPZ = 'I', the orthonormal eigenvectors of the */
/*          tridiagonal matrix. */
/*          If INFO > 0 on exit, Z contains the eigenvectors associated */
/*          with only the stored eigenvalues. */
/*          If  COMPZ = 'N', then Z is not referenced. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          COMPZ = 'V' or 'I', LDZ >= max(1,N). */

/*  WORK    (workspace) REAL array, dimension (4*N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  if INFO = i, and i is: */
/*                <= N  the Cholesky factorization of the matrix could */
/*                      not be performed because the i-th principal minor */
/*                      was not positive definite. */
/*                > N   the SVD algorithm failed to converge; */
/*                      if INFO = N+i, i off-diagonal elements of the */
/*                      bidiagonal factor did not converge to zero. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    /* Function Body */
    *info = 0;

    if (lsame_(compz, "N")) {
	icompz = 0;
    } else if (lsame_(compz, "V")) {
	icompz = 1;
    } else if (lsame_(compz, "I")) {
	icompz = 2;
    } else {
	icompz = -1;
    }
    if (icompz < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldz < 1 || icompz > 0 && *ldz < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPTEQR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (icompz > 0) {
	    z__[z_dim1 + 1] = 1.f;
	}
	return 0;
    }
    if (icompz == 2) {
	slaset_("Full", n, n, &c_b7, &c_b8, &z__[z_offset], ldz);
    }

/*     Call SPTTRF to factor the matrix. */

    spttrf_(n, &d__[1], &e[1], info);
    if (*info != 0) {
	return 0;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = sqrt(d__[i__]);
/* L10: */
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	e[i__] *= d__[i__];
/* L20: */
    }

/*     Call SBDSQR to compute the singular values/vectors of the */
/*     bidiagonal factor. */

    if (icompz > 0) {
	nru = *n;
    } else {
	nru = 0;
    }
    sbdsqr_("Lower", n, &c__0, &nru, &c__0, &d__[1], &e[1], vt, &c__1, &z__[
	    z_offset], ldz, c__, &c__1, &work[1], info);

/*     Square the singular values. */

    if (*info == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] *= d__[i__];
/* L30: */
	}
    } else {
	*info = *n + *info;
    }

    return 0;

/*     End of SPTEQR */

} /* spteqr_ */
