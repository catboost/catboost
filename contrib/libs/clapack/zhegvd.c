/* zhegvd.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {1.,0.};

/* Subroutine */ int zhegvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublereal *w, doublecomplex *work, integer *lwork, doublereal *rwork, 
	 integer *lrwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;
    doublereal d__1, d__2;

    /* Local variables */
    integer lopt;
    extern logical lsame_(char *, char *);
    integer lwmin;
    char trans[1];
    integer liopt;
    logical upper;
    integer lropt;
    logical wantz;
    extern /* Subroutine */ int ztrmm_(char *, char *, char *, char *, 
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *, 
	     doublecomplex *, integer *), 
	    ztrsm_(char *, char *, char *, char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *), xerbla_(char *, 
	    integer *), zheevd_(char *, char *, integer *, 
	    doublecomplex *, integer *, doublereal *, doublecomplex *, 
	    integer *, doublereal *, integer *, integer *, integer *, integer 
	    *);
    integer liwmin;
    extern /* Subroutine */ int zhegst_(integer *, char *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *);
    integer lrwmin;
    logical lquery;
    extern /* Subroutine */ int zpotrf_(char *, integer *, doublecomplex *, 
	    integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHEGVD computes all the eigenvalues, and optionally, the eigenvectors */
/*  of a complex generalized Hermitian-definite eigenproblem, of the form */
/*  A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and */
/*  B are assumed to be Hermitian and B is also positive definite. */
/*  If eigenvectors are desired, it uses a divide and conquer algorithm. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

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

/*  A       (input/output) COMPLEX*16 array, dimension (LDA, N) */
/*          On entry, the Hermitian matrix A.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of A contains the */
/*          upper triangular part of the matrix A.  If UPLO = 'L', */
/*          the leading N-by-N lower triangular part of A contains */
/*          the lower triangular part of the matrix A. */

/*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
/*          matrix Z of eigenvectors.  The eigenvectors are normalized */
/*          as follows: */
/*          if ITYPE = 1 or 2, Z**H*B*Z = I; */
/*          if ITYPE = 3, Z**H*inv(B)*Z = I. */
/*          If JOBZ = 'N', then on exit the upper triangle (if UPLO='U') */
/*          or the lower triangle (if UPLO='L') of A, including the */
/*          diagonal, is destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  B       (input/output) COMPLEX*16 array, dimension (LDB, N) */
/*          On entry, the Hermitian matrix B.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of B contains the */
/*          upper triangular part of the matrix B.  If UPLO = 'L', */
/*          the leading N-by-N lower triangular part of B contains */
/*          the lower triangular part of the matrix B. */

/*          On exit, if INFO <= N, the part of B containing the matrix is */
/*          overwritten by the triangular factor U or L from the Cholesky */
/*          factorization B = U**H*U or B = L*L**H. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          If INFO = 0, the eigenvalues in ascending order. */

/*  WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The length of the array WORK. */
/*          If N <= 1,                LWORK >= 1. */
/*          If JOBZ  = 'N' and N > 1, LWORK >= N + 1. */
/*          If JOBZ  = 'V' and N > 1, LWORK >= 2*N + N**2. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal sizes of the WORK, RWORK and */
/*          IWORK arrays, returns these values as the first entries of */
/*          the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  RWORK   (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LRWORK)) */
/*          On exit, if INFO = 0, RWORK(1) returns the optimal LRWORK. */

/*  LRWORK  (input) INTEGER */
/*          The dimension of the array RWORK. */
/*          If N <= 1,                LRWORK >= 1. */
/*          If JOBZ  = 'N' and N > 1, LRWORK >= N. */
/*          If JOBZ  = 'V' and N > 1, LRWORK >= 1 + 5*N + 2*N**2. */

/*          If LRWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK, RWORK */
/*          and IWORK arrays, returns these values as the first entries */
/*          of the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK. */
/*          If N <= 1,                LIWORK >= 1. */
/*          If JOBZ  = 'N' and N > 1, LIWORK >= 1. */
/*          If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N. */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK, RWORK */
/*          and IWORK arrays, returns these values as the first entries */
/*          of the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  ZPOTRF or ZHEEVD returned an error code: */
/*             <= N:  if INFO = i and JOBZ = 'N', then the algorithm */
/*                    failed to converge; i off-diagonal elements of an */
/*                    intermediate tridiagonal form did not converge to */
/*                    zero; */
/*                    if INFO = i and JOBZ = 'V', then the algorithm */
/*                    failed to compute an eigenvalue while working on */
/*                    the submatrix lying in rows and columns INFO/(N+1) */
/*                    through mod(INFO,N+1); */
/*             > N:   if INFO = N + i, for 1 <= i <= N, then the leading */
/*                    minor of order i of B is not positive definite. */
/*                    The factorization of B could not be completed and */
/*                    no eigenvalues or eigenvectors were computed. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA */

/*  Modified so that no backsubstitution is performed if ZHEEVD fails to */
/*  converge (NEIG in old code could be greater than N causing out of */
/*  bounds reference to A - reported by Ralf Meyer).  Also corrected the */
/*  description of INFO and the test on ITYPE. Sven, 16 Feb 05. */
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
    --rwork;
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1 || *lrwork == -1 || *liwork == -1;

    *info = 0;
    if (*n <= 1) {
	lwmin = 1;
	lrwmin = 1;
	liwmin = 1;
    } else if (wantz) {
	lwmin = (*n << 1) + *n * *n;
	lrwmin = *n * 5 + 1 + (*n << 1) * *n;
	liwmin = *n * 5 + 3;
    } else {
	lwmin = *n + 1;
	lrwmin = *n;
	liwmin = 1;
    }
    lopt = lwmin;
    lropt = lrwmin;
    liopt = liwmin;
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
	work[1].r = (doublereal) lopt, work[1].i = 0.;
	rwork[1] = (doublereal) lropt;
	iwork[1] = liopt;

	if (*lwork < lwmin && ! lquery) {
	    *info = -11;
	} else if (*lrwork < lrwmin && ! lquery) {
	    *info = -13;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -15;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHEGVD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Form a Cholesky factorization of B. */

    zpotrf_(uplo, n, &b[b_offset], ldb, info);
    if (*info != 0) {
	*info = *n + *info;
	return 0;
    }

/*     Transform problem to standard eigenvalue problem and solve. */

    zhegst_(itype, uplo, n, &a[a_offset], lda, &b[b_offset], ldb, info);
    zheevd_(jobz, uplo, n, &a[a_offset], lda, &w[1], &work[1], lwork, &rwork[
	    1], lrwork, &iwork[1], liwork, info);
/* Computing MAX */
    d__1 = (doublereal) lopt, d__2 = work[1].r;
    lopt = (integer) max(d__1,d__2);
/* Computing MAX */
    d__1 = (doublereal) lropt;
    lropt = (integer) max(d__1,rwork[1]);
/* Computing MAX */
    d__1 = (doublereal) liopt, d__2 = (doublereal) iwork[1];
    liopt = (integer) max(d__1,d__2);

    if (wantz && *info == 0) {

/*        Backtransform eigenvectors to the original problem. */

	if (*itype == 1 || *itype == 2) {

/*           For A*x=(lambda)*B*x and A*B*x=(lambda)*x; */
/*           backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */

	    if (upper) {
		*(unsigned char *)trans = 'N';
	    } else {
		*(unsigned char *)trans = 'C';
	    }

	    ztrsm_("Left", uplo, trans, "Non-unit", n, n, &c_b1, &b[b_offset], 
		     ldb, &a[a_offset], lda);

	} else if (*itype == 3) {

/*           For B*A*x=(lambda)*x; */
/*           backtransform eigenvectors: x = L*y or U'*y */

	    if (upper) {
		*(unsigned char *)trans = 'C';
	    } else {
		*(unsigned char *)trans = 'N';
	    }

	    ztrmm_("Left", uplo, trans, "Non-unit", n, n, &c_b1, &b[b_offset], 
		     ldb, &a[a_offset], lda);
	}
    }

    work[1].r = (doublereal) lopt, work[1].i = 0.;
    rwork[1] = (doublereal) lropt;
    iwork[1] = liopt;

    return 0;

/*     End of ZHEGVD */

} /* zhegvd_ */
