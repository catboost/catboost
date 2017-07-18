/* cgtsvx.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, complex *dl, complex *d__, complex *du, complex *dlf, complex *
	df, complex *duf, complex *du2, integer *ipiv, complex *b, integer *
	ldb, complex *x, integer *ldx, real *rcond, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1;

    /* Local variables */
    char norm[1];
    extern logical lsame_(char *, char *);
    real anorm;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *);
    extern doublereal slamch_(char *), clangt_(char *, integer *, 
	    complex *, complex *, complex *);
    logical nofact;
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), cgtcon_(char *, 
	    integer *, complex *, complex *, complex *, complex *, integer *, 
	    real *, real *, complex *, integer *), xerbla_(char *, 
	    integer *), cgtrfs_(char *, integer *, integer *, complex 
	    *, complex *, complex *, complex *, complex *, complex *, complex 
	    *, integer *, complex *, integer *, complex *, integer *, real *, 
	    real *, complex *, real *, integer *), cgttrf_(integer *, 
	    complex *, complex *, complex *, complex *, integer *, integer *);
    logical notran;
    extern /* Subroutine */ int cgttrs_(char *, integer *, integer *, complex 
	    *, complex *, complex *, complex *, integer *, complex *, integer 
	    *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGTSVX uses the LU factorization to compute the solution to a complex */
/*  system of linear equations A * X = B, A**T * X = B, or A**H * X = B, */
/*  where A is a tridiagonal matrix of order N and X and B are N-by-NRHS */
/*  matrices. */

/*  Error bounds on the solution and a condition estimate are also */
/*  provided. */

/*  Description */
/*  =========== */

/*  The following steps are performed: */

/*  1. If FACT = 'N', the LU decomposition is used to factor the matrix A */
/*     as A = L * U, where L is a product of permutation and unit lower */
/*     bidiagonal matrices and U is upper triangular with nonzeros in */
/*     only the main diagonal and first two superdiagonals. */

/*  2. If some U(i,i)=0, so that U is exactly singular, then the routine */
/*     returns with INFO = i. Otherwise, the factored form of A is used */
/*     to estimate the condition number of the matrix A.  If the */
/*     reciprocal of the condition number is less than machine precision, */
/*     INFO = N+1 is returned as a warning, but the routine still goes on */
/*     to solve for X and compute error bounds as described below. */

/*  3. The system of equations is solved for X using the factored form */
/*     of A. */

/*  4. Iterative refinement is applied to improve the computed solution */
/*     matrix and calculate error bounds and backward error estimates */
/*     for it. */

/*  Arguments */
/*  ========= */

/*  FACT    (input) CHARACTER*1 */
/*          Specifies whether or not the factored form of A has been */
/*          supplied on entry. */
/*          = 'F':  DLF, DF, DUF, DU2, and IPIV contain the factored form */
/*                  of A; DL, D, DU, DLF, DF, DUF, DU2 and IPIV will not */
/*                  be modified. */
/*          = 'N':  The matrix will be copied to DLF, DF, and DUF */
/*                  and factored. */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the form of the system of equations: */
/*          = 'N':  A * X = B     (No transpose) */
/*          = 'T':  A**T * X = B  (Transpose) */
/*          = 'C':  A**H * X = B  (Conjugate transpose) */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  DL      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) subdiagonal elements of A. */

/*  D       (input) COMPLEX array, dimension (N) */
/*          The n diagonal elements of A. */

/*  DU      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) superdiagonal elements of A. */

/*  DLF     (input or output) COMPLEX array, dimension (N-1) */
/*          If FACT = 'F', then DLF is an input argument and on entry */
/*          contains the (n-1) multipliers that define the matrix L from */
/*          the LU factorization of A as computed by CGTTRF. */

/*          If FACT = 'N', then DLF is an output argument and on exit */
/*          contains the (n-1) multipliers that define the matrix L from */
/*          the LU factorization of A. */

/*  DF      (input or output) COMPLEX array, dimension (N) */
/*          If FACT = 'F', then DF is an input argument and on entry */
/*          contains the n diagonal elements of the upper triangular */
/*          matrix U from the LU factorization of A. */

/*          If FACT = 'N', then DF is an output argument and on exit */
/*          contains the n diagonal elements of the upper triangular */
/*          matrix U from the LU factorization of A. */

/*  DUF     (input or output) COMPLEX array, dimension (N-1) */
/*          If FACT = 'F', then DUF is an input argument and on entry */
/*          contains the (n-1) elements of the first superdiagonal of U. */

/*          If FACT = 'N', then DUF is an output argument and on exit */
/*          contains the (n-1) elements of the first superdiagonal of U. */

/*  DU2     (input or output) COMPLEX array, dimension (N-2) */
/*          If FACT = 'F', then DU2 is an input argument and on entry */
/*          contains the (n-2) elements of the second superdiagonal of */
/*          U. */

/*          If FACT = 'N', then DU2 is an output argument and on exit */
/*          contains the (n-2) elements of the second superdiagonal of */
/*          U. */

/*  IPIV    (input or output) INTEGER array, dimension (N) */
/*          If FACT = 'F', then IPIV is an input argument and on entry */
/*          contains the pivot indices from the LU factorization of A as */
/*          computed by CGTTRF. */

/*          If FACT = 'N', then IPIV is an output argument and on exit */
/*          contains the pivot indices from the LU factorization of A; */
/*          row i of the matrix was interchanged with row IPIV(i). */
/*          IPIV(i) will always be either i or i+1; IPIV(i) = i indicates */
/*          a row interchange was not required. */

/*  B       (input) COMPLEX array, dimension (LDB,NRHS) */
/*          The N-by-NRHS right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (output) COMPLEX array, dimension (LDX,NRHS) */
/*          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  RCOND   (output) REAL */
/*          The estimate of the reciprocal condition number of the matrix */
/*          A.  If RCOND is less than the machine precision (in */
/*          particular, if RCOND = 0), the matrix is singular to working */
/*          precision.  This condition is indicated by a return code of */
/*          INFO > 0. */

/*  FERR    (output) REAL array, dimension (NRHS) */
/*          The estimated forward error bound for each solution vector */
/*          X(j) (the j-th column of the solution matrix X). */
/*          If XTRUE is the true solution corresponding to X(j), FERR(j) */
/*          is an estimated upper bound for the magnitude of the largest */
/*          element in (X(j) - XTRUE) divided by the magnitude of the */
/*          largest element in X(j).  The estimate is as reliable as */
/*          the estimate for RCOND, and is almost always a slight */
/*          overestimate of the true error. */

/*  BERR    (output) REAL array, dimension (NRHS) */
/*          The componentwise relative backward error of each solution */
/*          vector X(j) (i.e., the smallest relative change in */
/*          any element of A or B that makes X(j) an exact solution). */

/*  WORK    (workspace) COMPLEX array, dimension (2*N) */

/*  RWORK   (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, and i is */
/*                <= N:  U(i,i) is exactly zero.  The factorization */
/*                       has not been completed unless i = N, but the */
/*                       factor U is exactly singular, so the solution */
/*                       and error bounds could not be computed. */
/*                       RCOND = 0 is returned. */
/*                = N+1: U is nonsingular, but RCOND is less than machine */
/*                       precision, meaning that the matrix is singular */
/*                       to working precision.  Nevertheless, the */
/*                       solution and error bounds are computed because */
/*                       there are a number of situations where the */
/*                       computed solution can be more accurate than the */
/*                       value of RCOND would suggest. */

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

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    --dlf;
    --df;
    --duf;
    --du2;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --ferr;
    --berr;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    nofact = lsame_(fact, "N");
    notran = lsame_(trans, "N");
    if (! nofact && ! lsame_(fact, "F")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T") && ! 
	    lsame_(trans, "C")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*ldb < max(1,*n)) {
	*info = -14;
    } else if (*ldx < max(1,*n)) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGTSVX", &i__1);
	return 0;
    }

    if (nofact) {

/*        Compute the LU factorization of A. */

	ccopy_(n, &d__[1], &c__1, &df[1], &c__1);
	if (*n > 1) {
	    i__1 = *n - 1;
	    ccopy_(&i__1, &dl[1], &c__1, &dlf[1], &c__1);
	    i__1 = *n - 1;
	    ccopy_(&i__1, &du[1], &c__1, &duf[1], &c__1);
	}
	cgttrf_(n, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], info);

/*        Return if INFO is non-zero. */

	if (*info > 0) {
	    *rcond = 0.f;
	    return 0;
	}
    }

/*     Compute the norm of the matrix A. */

    if (notran) {
	*(unsigned char *)norm = '1';
    } else {
	*(unsigned char *)norm = 'I';
    }
    anorm = clangt_(norm, n, &dl[1], &d__[1], &du[1]);

/*     Compute the reciprocal of the condition number of A. */

    cgtcon_(norm, n, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], &anorm, 
	    rcond, &work[1], info);

/*     Compute the solution vectors X. */

    clacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    cgttrs_(trans, n, nrhs, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[1], &x[
	    x_offset], ldx, info);

/*     Use iterative refinement to improve the computed solutions and */
/*     compute error bounds and backward error estimates for them. */

    cgtrfs_(trans, n, nrhs, &dl[1], &d__[1], &du[1], &dlf[1], &df[1], &duf[1], 
	     &du2[1], &ipiv[1], &b[b_offset], ldb, &x[x_offset], ldx, &ferr[1]
, &berr[1], &work[1], &rwork[1], info);

/*     Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < slamch_("Epsilon")) {
	*info = *n + 1;
    }

    return 0;

/*     End of CGTSVX */

} /* cgtsvx_ */
