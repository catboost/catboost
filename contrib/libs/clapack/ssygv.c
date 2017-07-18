/* ssygv.f -- translated by f2c (version 20061008).
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
static integer c_n1 = -1;
static real c_b16 = 1.f;

/* Subroutine */ int ssygv_(integer *itype, char *jobz, char *uplo, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *w, real *work, 
	integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    integer nb, neig;
    extern logical lsame_(char *, char *);
    char trans[1];
    logical upper;
    extern /* Subroutine */ int strmm_(char *, char *, char *, char *, 
	    integer *, integer *, real *, real *, integer *, real *, integer *
);
    logical wantz;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, 
	    integer *, integer *, real *, real *, integer *, real *, integer *
), ssyev_(char *, char *, integer 
	    *, real *, integer *, real *, real *, integer *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    integer lwkmin;
    extern /* Subroutine */ int spotrf_(char *, integer *, real *, integer *, 
	    integer *);
    integer lwkopt;
    logical lquery;
    extern /* Subroutine */ int ssygst_(integer *, char *, integer *, real *, 
	    integer *, real *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SSYGV computes all the eigenvalues, and optionally, the eigenvectors */
/*  of a real generalized symmetric-definite eigenproblem, of the form */
/*  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x. */
/*  Here A and B are assumed to be symmetric and B is also */
/*  positive definite. */

/*  Arguments */
/*  ========= */

/*  ITYPE   (input) INTEGER */
/*          Specifies the problem type to be solved: */
/*          = 1:  A*x = (lambda)*B*x */
/*          = 2:  A*B*x = (lambda)*x */
/*          = 3:  B*A*x = (lambda)*x */

/*  JOBZ    (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only; */
/*          = 'V':  Compute eigenvalues and eigenvectors. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangles of A and B are stored; */
/*          = 'L':  Lower triangles of A and B are stored. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  A       (input/output) REAL array, dimension (LDA, N) */
/*          On entry, the symmetric matrix A.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of A contains the */
/*          upper triangular part of the matrix A.  If UPLO = 'L', */
/*          the leading N-by-N lower triangular part of A contains */
/*          the lower triangular part of the matrix A. */

/*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
/*          matrix Z of eigenvectors.  The eigenvectors are normalized */
/*          as follows: */
/*          if ITYPE = 1 or 2, Z**T*B*Z = I; */
/*          if ITYPE = 3, Z**T*inv(B)*Z = I. */
/*          If JOBZ = 'N', then on exit the upper triangle (if UPLO='U') */
/*          or the lower triangle (if UPLO='L') of A, including the */
/*          diagonal, is destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  B       (input/output) REAL array, dimension (LDB, N) */
/*          On entry, the symmetric positive definite matrix B. */
/*          If UPLO = 'U', the leading N-by-N upper triangular part of B */
/*          contains the upper triangular part of the matrix B. */
/*          If UPLO = 'L', the leading N-by-N lower triangular part of B */
/*          contains the lower triangular part of the matrix B. */

/*          On exit, if INFO <= N, the part of B containing the matrix is */
/*          overwritten by the triangular factor U or L from the Cholesky */
/*          factorization B = U**T*U or B = L*L**T. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  W       (output) REAL array, dimension (N) */
/*          If INFO = 0, the eigenvalues in ascending order. */

/*  WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The length of the array WORK.  LWORK >= max(1,3*N-1). */
/*          For optimal efficiency, LWORK >= (NB+2)*N, */
/*          where NB is the blocksize for SSYTRD returned by ILAENV. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  SPOTRF or SSYEV returned an error code: */
/*             <= N:  if INFO = i, SSYEV failed to converge; */
/*                    i off-diagonal elements of an intermediate */
/*                    tridiagonal form did not converge to zero; */
/*             > N:   if INFO = N + i, for 1 <= i <= N, then the leading */
/*                    minor of order i of B is not positive definite. */
/*                    The factorization of B could not be completed and */
/*                    no eigenvalues or eigenvectors were computed. */

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

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --w;
    --work;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;

    *info = 0;
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! (wantz || lsame_(jobz, "N"))) {
	*info = -2;
    } else if (! (upper || lsame_(uplo, "L"))) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lda < max(1,*n)) {
	*info = -6;
    } else if (*ldb < max(1,*n)) {
	*info = -8;
    }

    if (*info == 0) {
/* Computing MAX */
	i__1 = 1, i__2 = *n * 3 - 1;
	lwkmin = max(i__1,i__2);
	nb = ilaenv_(&c__1, "SSYTRD", uplo, n, &c_n1, &c_n1, &c_n1);
/* Computing MAX */
	i__1 = lwkmin, i__2 = (nb + 2) * *n;
	lwkopt = max(i__1,i__2);
	work[1] = (real) lwkopt;

	if (*lwork < lwkmin && ! lquery) {
	    *info = -11;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYGV ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Form a Cholesky factorization of B. */

    spotrf_(uplo, n, &b[b_offset], ldb, info);
    if (*info != 0) {
	*info = *n + *info;
	return 0;
    }

/*     Transform problem to standard eigenvalue problem and solve. */

    ssygst_(itype, uplo, n, &a[a_offset], lda, &b[b_offset], ldb, info);
    ssyev_(jobz, uplo, n, &a[a_offset], lda, &w[1], &work[1], lwork, info);

    if (wantz) {

/*        Backtransform eigenvectors to the original problem. */

	neig = *n;
	if (*info > 0) {
	    neig = *info - 1;
	}
	if (*itype == 1 || *itype == 2) {

/*           For A*x=(lambda)*B*x and A*B*x=(lambda)*x; */
/*           backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */

	    if (upper) {
		*(unsigned char *)trans = 'N';
	    } else {
		*(unsigned char *)trans = 'T';
	    }

	    strsm_("Left", uplo, trans, "Non-unit", n, &neig, &c_b16, &b[
		    b_offset], ldb, &a[a_offset], lda);

	} else if (*itype == 3) {

/*           For B*A*x=(lambda)*x; */
/*           backtransform eigenvectors: x = L*y or U'*y */

	    if (upper) {
		*(unsigned char *)trans = 'T';
	    } else {
		*(unsigned char *)trans = 'N';
	    }

	    strmm_("Left", uplo, trans, "Non-unit", n, &neig, &c_b16, &b[
		    b_offset], ldb, &a[a_offset], lda);
	}
    }

    work[1] = (real) lwkopt;
    return 0;

/*     End of SSYGV */

} /* ssygv_ */
