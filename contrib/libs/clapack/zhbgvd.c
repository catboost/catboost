/* zhbgvd.f -- translated by f2c (version 20061008).
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
static doublecomplex c_b2 = {0.,0.};

/* Subroutine */ int zhbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublecomplex *ab, integer *ldab, doublecomplex *bb, 
	integer *ldbb, doublereal *w, doublecomplex *z__, integer *ldz, 
	doublecomplex *work, integer *lwork, doublereal *rwork, integer *
	lrwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, bb_dim1, bb_offset, z_dim1, z_offset, i__1;

    /* Local variables */
    integer inde;
    char vect[1];
    integer llwk2;
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *);
    integer lwmin;
    logical upper;
    integer llrwk;
    logical wantz;
    integer indwk2;
    extern /* Subroutine */ int xerbla_(char *, integer *), dsterf_(
	    integer *, doublereal *, doublereal *, integer *), zstedc_(char *, 
	     integer *, doublereal *, doublereal *, doublecomplex *, integer *
, doublecomplex *, integer *, doublereal *, integer *, integer *, 
	    integer *, integer *), zhbtrd_(char *, char *, integer *, 
	    integer *, doublecomplex *, integer *, doublereal *, doublereal *, 
	     doublecomplex *, integer *, doublecomplex *, integer *);
    integer indwrk, liwmin;
    extern /* Subroutine */ int zhbgst_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *, 
	     doublecomplex *, integer *, doublecomplex *, doublereal *, 
	    integer *), zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    integer lrwmin;
    extern /* Subroutine */ int zpbstf_(char *, integer *, integer *, 
	    doublecomplex *, integer *, integer *);
    logical lquery;


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHBGVD computes all the eigenvalues, and optionally, the eigenvectors */
/*  of a complex generalized Hermitian-definite banded eigenproblem, of */
/*  the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian */
/*  and banded, and B is also positive definite.  If eigenvectors are */
/*  desired, it uses a divide and conquer algorithm. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only; */
/*          = 'V':  Compute eigenvalues and eigenvectors. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangles of A and B are stored; */
/*          = 'L':  Lower triangles of A and B are stored. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  KA      (input) INTEGER */
/*          The number of superdiagonals of the matrix A if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'. KA >= 0. */

/*  KB      (input) INTEGER */
/*          The number of superdiagonals of the matrix B if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'. KB >= 0. */

/*  AB      (input/output) COMPLEX*16 array, dimension (LDAB, N) */
/*          On entry, the upper or lower triangle of the Hermitian band */
/*          matrix A, stored in the first ka+1 rows of the array.  The */
/*          j-th column of A is stored in the j-th column of the array AB */
/*          as follows: */
/*          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka). */

/*          On exit, the contents of AB are destroyed. */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KA+1. */

/*  BB      (input/output) COMPLEX*16 array, dimension (LDBB, N) */
/*          On entry, the upper or lower triangle of the Hermitian band */
/*          matrix B, stored in the first kb+1 rows of the array.  The */
/*          j-th column of B is stored in the j-th column of the array BB */
/*          as follows: */
/*          if UPLO = 'U', BB(kb+1+i-j,j) = B(i,j) for max(1,j-kb)<=i<=j; */
/*          if UPLO = 'L', BB(1+i-j,j)    = B(i,j) for j<=i<=min(n,j+kb). */

/*          On exit, the factor S from the split Cholesky factorization */
/*          B = S**H*S, as returned by ZPBSTF. */

/*  LDBB    (input) INTEGER */
/*          The leading dimension of the array BB.  LDBB >= KB+1. */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          If INFO = 0, the eigenvalues in ascending order. */

/*  Z       (output) COMPLEX*16 array, dimension (LDZ, N) */
/*          If JOBZ = 'V', then if INFO = 0, Z contains the matrix Z of */
/*          eigenvectors, with the i-th column of Z holding the */
/*          eigenvector associated with W(i). The eigenvectors are */
/*          normalized so that Z**H*B*Z = I. */
/*          If JOBZ = 'N', then Z is not referenced. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          JOBZ = 'V', LDZ >= N. */

/*  WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO=0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          If N <= 1,               LWORK >= 1. */
/*          If JOBZ = 'N' and N > 1, LWORK >= N. */
/*          If JOBZ = 'V' and N > 1, LWORK >= 2*N**2. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal sizes of the WORK, RWORK and */
/*          IWORK arrays, returns these values as the first entries of */
/*          the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  RWORK   (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LRWORK)) */
/*          On exit, if INFO=0, RWORK(1) returns the optimal LRWORK. */

/*  LRWORK  (input) INTEGER */
/*          The dimension of array RWORK. */
/*          If N <= 1,               LRWORK >= 1. */
/*          If JOBZ = 'N' and N > 1, LRWORK >= N. */
/*          If JOBZ = 'V' and N > 1, LRWORK >= 1 + 5*N + 2*N**2. */

/*          If LRWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK, RWORK */
/*          and IWORK arrays, returns these values as the first entries */
/*          of the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          On exit, if INFO=0, IWORK(1) returns the optimal LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of array IWORK. */
/*          If JOBZ = 'N' or N <= 1, LIWORK >= 1. */
/*          If JOBZ = 'V' and N > 1, LIWORK >= 3 + 5*N. */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK, RWORK */
/*          and IWORK arrays, returns these values as the first entries */
/*          of the WORK, RWORK and IWORK arrays, and no error message */
/*          related to LWORK or LRWORK or LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, and i is: */
/*             <= N:  the algorithm failed to converge: */
/*                    i off-diagonal elements of an intermediate */
/*                    tridiagonal form did not converge to zero; */
/*             > N:   if INFO = N + i, for 1 <= i <= N, then ZPBSTF */
/*                    returned INFO = i: B is not positive definite. */
/*                    The factorization of B could not be completed and */
/*                    no eigenvalues or eigenvectors were computed. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    bb_dim1 = *ldbb;
    bb_offset = 1 + bb_dim1;
    bb -= bb_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
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
/* Computing 2nd power */
	i__1 = *n;
	lwmin = i__1 * i__1 << 1;
/* Computing 2nd power */
	i__1 = *n;
	lrwmin = *n * 5 + 1 + (i__1 * i__1 << 1);
	liwmin = *n * 5 + 3;
    } else {
	lwmin = *n;
	lrwmin = *n;
	liwmin = 1;
    }
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (upper || lsame_(uplo, "L"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ka < 0) {
	*info = -4;
    } else if (*kb < 0 || *kb > *ka) {
	*info = -5;
    } else if (*ldab < *ka + 1) {
	*info = -7;
    } else if (*ldbb < *kb + 1) {
	*info = -9;
    } else if (*ldz < 1 || wantz && *ldz < *n) {
	*info = -12;
    }

    if (*info == 0) {
	work[1].r = (doublereal) lwmin, work[1].i = 0.;
	rwork[1] = (doublereal) lrwmin;
	iwork[1] = liwmin;

	if (*lwork < lwmin && ! lquery) {
	    *info = -14;
	} else if (*lrwork < lrwmin && ! lquery) {
	    *info = -16;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -18;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHBGVD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Form a split Cholesky factorization of B. */

    zpbstf_(uplo, n, kb, &bb[bb_offset], ldbb, info);
    if (*info != 0) {
	*info = *n + *info;
	return 0;
    }

/*     Transform problem to standard eigenvalue problem. */

    inde = 1;
    indwrk = inde + *n;
    indwk2 = *n * *n + 1;
    llwk2 = *lwork - indwk2 + 2;
    llrwk = *lrwork - indwrk + 2;
    zhbgst_(jobz, uplo, n, ka, kb, &ab[ab_offset], ldab, &bb[bb_offset], ldbb, 
	     &z__[z_offset], ldz, &work[1], &rwork[indwrk], &iinfo);

/*     Reduce Hermitian band matrix to tridiagonal form. */

    if (wantz) {
	*(unsigned char *)vect = 'U';
    } else {
	*(unsigned char *)vect = 'N';
    }
    zhbtrd_(vect, uplo, n, ka, &ab[ab_offset], ldab, &w[1], &rwork[inde], &
	    z__[z_offset], ldz, &work[1], &iinfo);

/*     For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC. */

    if (! wantz) {
	dsterf_(n, &w[1], &rwork[inde], info);
    } else {
	zstedc_("I", n, &w[1], &rwork[inde], &work[1], n, &work[indwk2], &
		llwk2, &rwork[indwrk], &llrwk, &iwork[1], liwork, info);
	zgemm_("N", "N", n, n, n, &c_b1, &z__[z_offset], ldz, &work[1], n, &
		c_b2, &work[indwk2], n);
	zlacpy_("A", n, n, &work[indwk2], n, &z__[z_offset], ldz);
    }

    work[1].r = (doublereal) lwmin, work[1].i = 0.;
    rwork[1] = (doublereal) lrwmin;
    iwork[1] = liwmin;
    return 0;

/*     End of ZHBGVD */

} /* zhbgvd_ */
