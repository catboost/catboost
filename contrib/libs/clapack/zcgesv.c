/* zcgesv.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {-1.,0.};
static doublecomplex c_b2 = {1.,0.};
static integer c__1 = 1;

/* Subroutine */ int zcgesv_(integer *n, integer *nrhs, doublecomplex *a, 
	integer *lda, integer *ipiv, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublecomplex *work, complex *swork, 
	doublereal *rwork, integer *iter, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, work_dim1, work_offset, 
	    x_dim1, x_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal), d_imag(doublecomplex *);

    /* Local variables */
    integer i__;
    doublereal cte, eps, anrm;
    integer ptsa;
    doublereal rnrm, xnrm;
    integer ptsx, iiter;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *), zaxpy_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), clag2z_(
	    integer *, integer *, complex *, integer *, doublecomplex *, 
	    integer *, integer *), zlag2c_(integer *, integer *, 
	    doublecomplex *, integer *, complex *, integer *, integer *);
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int cgetrf_(integer *, integer *, complex *, 
	    integer *, integer *, integer *), xerbla_(char *, integer *);
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *, 
	    integer *, doublereal *);
    extern /* Subroutine */ int cgetrs_(char *, integer *, integer *, complex 
	    *, integer *, integer *, complex *, integer *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), 
	    zgetrf_(integer *, integer *, doublecomplex *, integer *, integer 
	    *, integer *), zgetrs_(char *, integer *, integer *, 
	    doublecomplex *, integer *, integer *, doublecomplex *, integer *, 
	     integer *);


/*  -- LAPACK PROTOTYPE driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     January 2007 */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZCGESV computes the solution to a complex system of linear equations */
/*     A * X = B, */
/*  where A is an N-by-N matrix and X and B are N-by-NRHS matrices. */

/*  ZCGESV first attempts to factorize the matrix in COMPLEX and use this */
/*  factorization within an iterative refinement procedure to produce a */
/*  solution with COMPLEX*16 normwise backward error quality (see below). */
/*  If the approach fails the method switches to a COMPLEX*16 */
/*  factorization and solve. */

/*  The iterative refinement is not going to be a winning strategy if */
/*  the ratio COMPLEX performance over COMPLEX*16 performance is too */
/*  small. A reasonable strategy should take the number of right-hand */
/*  sides and the size of the matrix into account. This might be done */
/*  with a call to ILAENV in the future. Up to now, we always try */
/*  iterative refinement. */

/*  The iterative refinement process is stopped if */
/*      ITER > ITERMAX */
/*  or for all the RHS we have: */
/*      RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX */
/*  where */
/*      o ITER is the number of the current iteration in the iterative */
/*        refinement process */
/*      o RNRM is the infinity-norm of the residual */
/*      o XNRM is the infinity-norm of the solution */
/*      o ANRM is the infinity-operator-norm of the matrix A */
/*      o EPS is the machine epsilon returned by DLAMCH('Epsilon') */
/*  The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 */
/*  respectively. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of linear equations, i.e., the order of the */
/*          matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  A       (input or input/ouptut) COMPLEX*16 array, */
/*          dimension (LDA,N) */
/*          On entry, the N-by-N coefficient matrix A. */
/*          On exit, if iterative refinement has been successfully used */
/*          (INFO.EQ.0 and ITER.GE.0, see description below), then A is */
/*          unchanged, if double precision factorization has been used */
/*          (INFO.EQ.0 and ITER.LT.0, see description below), then the */
/*          array A contains the factors L and U from the factorization */
/*          A = P*L*U; the unit diagonal elements of L are not stored. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  IPIV    (output) INTEGER array, dimension (N) */
/*          The pivot indices that define the permutation matrix P; */
/*          row i of the matrix was interchanged with row IPIV(i). */
/*          Corresponds either to the single precision factorization */
/*          (if INFO.EQ.0 and ITER.GE.0) or the double precision */
/*          factorization (if INFO.EQ.0 and ITER.LT.0). */

/*  B       (input) COMPLEX*16 array, dimension (LDB,NRHS) */
/*          The N-by-NRHS right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (output) COMPLEX*16 array, dimension (LDX,NRHS) */
/*          If INFO = 0, the N-by-NRHS solution matrix X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  WORK    (workspace) COMPLEX*16 array, dimension (N*NRHS) */
/*          This array is used to hold the residual vectors. */

/*  SWORK   (workspace) COMPLEX array, dimension (N*(N+NRHS)) */
/*          This array is used to use the single precision matrix and the */
/*          right-hand sides or solutions in single precision. */

/*  RWORK   (workspace) DOUBLE PRECISION array, dimension (N) */

/*  ITER    (output) INTEGER */
/*          < 0: iterative refinement has failed, COMPLEX*16 */
/*               factorization has been performed */
/*               -1 : the routine fell back to full precision for */
/*                    implementation- or machine-specific reasons */
/*               -2 : narrowing the precision induced an overflow, */
/*                    the routine fell back to full precision */
/*               -3 : failure of CGETRF */
/*               -31: stop the iterative refinement after the 30th */
/*                    iterations */
/*          > 0: iterative refinement has been sucessfully used. */
/*               Returns the number of iterations */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, U(i,i) computed in COMPLEX*16 is exactly */
/*                zero.  The factorization has been completed, but the */
/*                factor U is exactly singular, so the solution */
/*                could not be computed. */

/*  ========= */

/*     .. Parameters .. */




/*     .. Local Scalars .. */

/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    work_dim1 = *n;
    work_offset = 1 + work_dim1;
    work -= work_offset;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --swork;
    --rwork;

    /* Function Body */
    *info = 0;
    *iter = 0;

/*     Test the input parameters. */

    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    } else if (*ldx < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZCGESV", &i__1);
	return 0;
    }

/*     Quick return if (N.EQ.0). */

    if (*n == 0) {
	return 0;
    }

/*     Skip single precision iterative refinement if a priori slower */
/*     than double precision factorization. */

    if (FALSE_) {
	*iter = -1;
	goto L40;
    }

/*     Compute some constants. */

    anrm = zlange_("I", n, n, &a[a_offset], lda, &rwork[1]);
    eps = dlamch_("Epsilon");
    cte = anrm * eps * sqrt((doublereal) (*n)) * 1.;

/*     Set the indices PTSA, PTSX for referencing SA and SX in SWORK. */

    ptsa = 1;
    ptsx = ptsa + *n * *n;

/*     Convert B from double precision to single precision and store the */
/*     result in SX. */

    zlag2c_(n, nrhs, &b[b_offset], ldb, &swork[ptsx], n, info);

    if (*info != 0) {
	*iter = -2;
	goto L40;
    }

/*     Convert A from double precision to single precision and store the */
/*     result in SA. */

    zlag2c_(n, n, &a[a_offset], lda, &swork[ptsa], n, info);

    if (*info != 0) {
	*iter = -2;
	goto L40;
    }

/*     Compute the LU factorization of SA. */

    cgetrf_(n, n, &swork[ptsa], n, &ipiv[1], info);

    if (*info != 0) {
	*iter = -3;
	goto L40;
    }

/*     Solve the system SA*SX = SB. */

    cgetrs_("No transpose", n, nrhs, &swork[ptsa], n, &ipiv[1], &swork[ptsx], 
	    n, info);

/*     Convert SX back to double precision */

    clag2z_(n, nrhs, &swork[ptsx], n, &x[x_offset], ldx, info);

/*     Compute R = B - AX (R is WORK). */

    zlacpy_("All", n, nrhs, &b[b_offset], ldb, &work[work_offset], n);

    zgemm_("No Transpose", "No Transpose", n, nrhs, n, &c_b1, &a[a_offset], 
	    lda, &x[x_offset], ldx, &c_b2, &work[work_offset], n);

/*     Check whether the NRHS normwise backward errors satisfy the */
/*     stopping criterion. If yes, set ITER=0 and return. */

    i__1 = *nrhs;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = izamax_(n, &x[i__ * x_dim1 + 1], &c__1) + i__ * x_dim1;
	xnrm = (d__1 = x[i__2].r, abs(d__1)) + (d__2 = d_imag(&x[izamax_(n, &
		x[i__ * x_dim1 + 1], &c__1) + i__ * x_dim1]), abs(d__2));
	i__2 = izamax_(n, &work[i__ * work_dim1 + 1], &c__1) + i__ * 
		work_dim1;
	rnrm = (d__1 = work[i__2].r, abs(d__1)) + (d__2 = d_imag(&work[
		izamax_(n, &work[i__ * work_dim1 + 1], &c__1) + i__ * 
		work_dim1]), abs(d__2));
	if (rnrm > xnrm * cte) {
	    goto L10;
	}
    }

/*     If we are here, the NRHS normwise backward errors satisfy the */
/*     stopping criterion. We are good to exit. */

    *iter = 0;
    return 0;

L10:

    for (iiter = 1; iiter <= 30; ++iiter) {

/*        Convert R (in WORK) from double precision to single precision */
/*        and store the result in SX. */

	zlag2c_(n, nrhs, &work[work_offset], n, &swork[ptsx], n, info);

	if (*info != 0) {
	    *iter = -2;
	    goto L40;
	}

/*        Solve the system SA*SX = SR. */

	cgetrs_("No transpose", n, nrhs, &swork[ptsa], n, &ipiv[1], &swork[
		ptsx], n, info);

/*        Convert SX back to double precision and update the current */
/*        iterate. */

	clag2z_(n, nrhs, &swork[ptsx], n, &work[work_offset], n, info);

	i__1 = *nrhs;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    zaxpy_(n, &c_b2, &work[i__ * work_dim1 + 1], &c__1, &x[i__ * 
		    x_dim1 + 1], &c__1);
	}

/*        Compute R = B - AX (R is WORK). */

	zlacpy_("All", n, nrhs, &b[b_offset], ldb, &work[work_offset], n);

	zgemm_("No Transpose", "No Transpose", n, nrhs, n, &c_b1, &a[a_offset]
, lda, &x[x_offset], ldx, &c_b2, &work[work_offset], n);

/*        Check whether the NRHS normwise backward errors satisfy the */
/*        stopping criterion. If yes, set ITER=IITER>0 and return. */

	i__1 = *nrhs;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = izamax_(n, &x[i__ * x_dim1 + 1], &c__1) + i__ * x_dim1;
	    xnrm = (d__1 = x[i__2].r, abs(d__1)) + (d__2 = d_imag(&x[izamax_(
		    n, &x[i__ * x_dim1 + 1], &c__1) + i__ * x_dim1]), abs(
		    d__2));
	    i__2 = izamax_(n, &work[i__ * work_dim1 + 1], &c__1) + i__ * 
		    work_dim1;
	    rnrm = (d__1 = work[i__2].r, abs(d__1)) + (d__2 = d_imag(&work[
		    izamax_(n, &work[i__ * work_dim1 + 1], &c__1) + i__ * 
		    work_dim1]), abs(d__2));
	    if (rnrm > xnrm * cte) {
		goto L20;
	    }
	}

/*        If we are here, the NRHS normwise backward errors satisfy the */
/*        stopping criterion, we are good to exit. */

	*iter = iiter;

	return 0;

L20:

/* L30: */
	;
    }

/*     If we are at this place of the code, this is because we have */
/*     performed ITER=ITERMAX iterations and never satisified the stopping */
/*     criterion, set up the ITER flag accordingly and follow up on double */
/*     precision routine. */

    *iter = -31;

L40:

/*     Single-precision iterative refinement failed to converge to a */
/*     satisfactory solution, so we resort to double precision. */

    zgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);

    if (*info != 0) {
	return 0;
    }

    zlacpy_("All", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    zgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &x[x_offset]
, ldx, info);

    return 0;

/*     End of ZCGESV. */

} /* zcgesv_ */
