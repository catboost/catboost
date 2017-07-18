/* zgesdd.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {0.,0.};
static doublecomplex c_b2 = {1.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__0 = 0;

/* Subroutine */ int zgesdd_(char *jobz, integer *m, integer *n, 
	doublecomplex *a, integer *lda, doublereal *s, doublecomplex *u, 
	integer *ldu, doublecomplex *vt, integer *ldvt, doublecomplex *work, 
	integer *lwork, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1, 
	    i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, ie, il, ir, iu, blk;
    doublereal dum[1], eps;
    integer iru, ivt, iscl;
    doublereal anrm;
    integer idum[1], ierr, itau, irvt;
    extern logical lsame_(char *, char *);
    integer chunk, minmn;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *);
    integer wrkbl, itaup, itauq;
    logical wntqa;
    integer nwork;
    logical wntqn, wntqo, wntqs;
    extern /* Subroutine */ int zlacp2_(char *, integer *, integer *, 
	    doublereal *, integer *, doublecomplex *, integer *);
    integer mnthr1, mnthr2;
    extern /* Subroutine */ int dbdsdc_(char *, char *, integer *, doublereal 
	    *, doublereal *, doublereal *, integer *, doublereal *, integer *, 
	     doublereal *, integer *, doublereal *, integer *, integer *);
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlascl_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, integer *), xerbla_(char *, integer *),
	     zgebrd_(integer *, integer *, doublecomplex *, integer *, 
	    doublereal *, doublereal *, doublecomplex *, doublecomplex *, 
	    doublecomplex *, integer *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    doublereal bignum;
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *, 
	    integer *, doublereal *);
    extern /* Subroutine */ int zgelqf_(integer *, integer *, doublecomplex *, 
	     integer *, doublecomplex *, doublecomplex *, integer *, integer *
), zlacrm_(integer *, integer *, doublecomplex *, integer *, 
	    doublereal *, integer *, doublecomplex *, integer *, doublereal *)
	    , zlarcm_(integer *, integer *, doublereal *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublereal *), zlascl_(char *, integer *, integer *, doublereal *, 
	     doublereal *, integer *, integer *, doublecomplex *, integer *, 
	    integer *), zgeqrf_(integer *, integer *, doublecomplex *, 
	     integer *, doublecomplex *, doublecomplex *, integer *, integer *
);
    integer ldwrkl;
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), 
	    zlaset_(char *, integer *, integer *, doublecomplex *, 
	    doublecomplex *, doublecomplex *, integer *);
    integer ldwrkr, minwrk, ldwrku, maxwrk;
    extern /* Subroutine */ int zungbr_(char *, integer *, integer *, integer 
	    *, doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, integer *);
    integer ldwkvt;
    doublereal smlnum;
    logical wntqas;
    extern /* Subroutine */ int zunmbr_(char *, char *, char *, integer *, 
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *, 
	     doublecomplex *, integer *, doublecomplex *, integer *, integer *
), zunglq_(integer *, integer *, integer *
, doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, integer *);
    integer nrwork;
    extern /* Subroutine */ int zungqr_(integer *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */
/*     8-15-00:  Improve consistency of WS calculations (eca) */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZGESDD computes the singular value decomposition (SVD) of a complex */
/*  M-by-N matrix A, optionally computing the left and/or right singular */
/*  vectors, by using divide-and-conquer method. The SVD is written */

/*       A = U * SIGMA * conjugate-transpose(V) */

/*  where SIGMA is an M-by-N matrix which is zero except for its */
/*  min(m,n) diagonal elements, U is an M-by-M unitary matrix, and */
/*  V is an N-by-N unitary matrix.  The diagonal elements of SIGMA */
/*  are the singular values of A; they are real and non-negative, and */
/*  are returned in descending order.  The first min(m,n) columns of */
/*  U and V are the left and right singular vectors of A. */

/*  Note that the routine returns VT = V**H, not V. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          Specifies options for computing all or part of the matrix U: */
/*          = 'A':  all M columns of U and all N rows of V**H are */
/*                  returned in the arrays U and VT; */
/*          = 'S':  the first min(M,N) columns of U and the first */
/*                  min(M,N) rows of V**H are returned in the arrays U */
/*                  and VT; */
/*          = 'O':  If M >= N, the first N columns of U are overwritten */
/*                  in the array A and all rows of V**H are returned in */
/*                  the array VT; */
/*                  otherwise, all columns of U are returned in the */
/*                  array U and the first M rows of V**H are overwritten */
/*                  in the array A; */
/*          = 'N':  no columns of U or rows of V**H are computed. */

/*  M       (input) INTEGER */
/*          The number of rows of the input matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the input matrix A.  N >= 0. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, */
/*          if JOBZ = 'O',  A is overwritten with the first N columns */
/*                          of U (the left singular vectors, stored */
/*                          columnwise) if M >= N; */
/*                          A is overwritten with the first M rows */
/*                          of V**H (the right singular vectors, stored */
/*                          rowwise) otherwise. */
/*          if JOBZ .ne. 'O', the contents of A are destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  S       (output) DOUBLE PRECISION array, dimension (min(M,N)) */
/*          The singular values of A, sorted so that S(i) >= S(i+1). */

/*  U       (output) COMPLEX*16 array, dimension (LDU,UCOL) */
/*          UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N; */
/*          UCOL = min(M,N) if JOBZ = 'S'. */
/*          If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M */
/*          unitary matrix U; */
/*          if JOBZ = 'S', U contains the first min(M,N) columns of U */
/*          (the left singular vectors, stored columnwise); */
/*          if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced. */

/*  LDU     (input) INTEGER */
/*          The leading dimension of the array U.  LDU >= 1; if */
/*          JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M. */

/*  VT      (output) COMPLEX*16 array, dimension (LDVT,N) */
/*          If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the */
/*          N-by-N unitary matrix V**H; */
/*          if JOBZ = 'S', VT contains the first min(M,N) rows of */
/*          V**H (the right singular vectors, stored rowwise); */
/*          if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced. */

/*  LDVT    (input) INTEGER */
/*          The leading dimension of the array VT.  LDVT >= 1; if */
/*          JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N; */
/*          if JOBZ = 'S', LDVT >= min(M,N). */

/*  WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= 1. */
/*          if JOBZ = 'N', LWORK >= 2*min(M,N)+max(M,N). */
/*          if JOBZ = 'O', */
/*                LWORK >= 2*min(M,N)*min(M,N)+2*min(M,N)+max(M,N). */
/*          if JOBZ = 'S' or 'A', */
/*                LWORK >= min(M,N)*min(M,N)+2*min(M,N)+max(M,N). */
/*          For good performance, LWORK should generally be larger. */

/*          If LWORK = -1, a workspace query is assumed.  The optimal */
/*          size for the WORK array is calculated and stored in WORK(1), */
/*          and no other work except argument checking is performed. */

/*  RWORK   (workspace) DOUBLE PRECISION array, dimension (MAX(1,LRWORK)) */
/*          If JOBZ = 'N', LRWORK >= 5*min(M,N). */
/*          Otherwise, LRWORK >= 5*min(M,N)*min(M,N) + 7*min(M,N) */

/*  IWORK   (workspace) INTEGER array, dimension (8*min(M,N)) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  The updating process of DBDSDC did not converge. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Huan Ren, Computer Science Division, University of */
/*     California at Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    mnthr1 = (integer) (minmn * 17. / 9.);
    mnthr2 = (integer) (minmn * 5. / 3.);
    wntqa = lsame_(jobz, "A");
    wntqs = lsame_(jobz, "S");
    wntqas = wntqa || wntqs;
    wntqo = lsame_(jobz, "O");
    wntqn = lsame_(jobz, "N");
    minwrk = 1;
    maxwrk = 1;

    if (! (wntqa || wntqs || wntqo || wntqn)) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldu < 1 || wntqas && *ldu < *m || wntqo && *m < *n && *ldu < *
	    m) {
	*info = -8;
    } else if (*ldvt < 1 || wntqa && *ldvt < *n || wntqs && *ldvt < minmn || 
	    wntqo && *m >= *n && *ldvt < *n) {
	*info = -10;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       CWorkspace refers to complex workspace, and RWorkspace to */
/*       real workspace. NB refers to the optimal block size for the */
/*       immediately following subroutine, as returned by ILAENV.) */

    if (*info == 0 && *m > 0 && *n > 0) {
	if (*m >= *n) {

/*           There is no complex work space needed for bidiagonal SVD */
/*           The real work space needed for bidiagonal SVD is BDSPAC */
/*           for computing singular values and singular vectors; BDSPAN */
/*           for computing singular values only. */
/*           BDSPAC = 5*N*N + 7*N */
/*           BDSPAN = MAX(7*N+4, 3*N+2+SMLSIZ*(SMLSIZ+8)) */

	    if (*m >= mnthr1) {
		if (wntqn) {

/*                 Path 1 (M much larger than N, JOBZ='N') */

		    maxwrk = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1);
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3;
		} else if (wntqo) {

/*                 Path 2 (M much larger than N, JOBZ='O') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "ZUNGQR", 
			    " ", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *n * *n + wrkbl;
		    minwrk = (*n << 1) * *n + *n * 3;
		} else if (wntqs) {

/*                 Path 3 (M much larger than N, JOBZ='S') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "ZUNGQR", 
			    " ", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + *n * 3;
		} else if (wntqa) {

/*                 Path 4 (M much larger than N, JOBZ='A') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "ZGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *m * ilaenv_(&c__1, "ZUNGQR", 
			    " ", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + (*n << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *n * *n + wrkbl;
		    minwrk = *n * *n + (*n << 1) + *m;
		}
	    } else if (*m >= mnthr2) {

/*              Path 5 (M much larger than N, but not as much as MNTHR1) */

		maxwrk = (*n << 1) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD", 
			" ", m, n, &c_n1, &c_n1);
		minwrk = (*n << 1) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "P", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "P", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "P", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6 (M at least N, but not much larger) */

		maxwrk = (*n << 1) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD", 
			" ", m, n, &c_n1, &c_n1);
		minwrk = (*n << 1) + *m;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *n * *n;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "PRC", n, n, n, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*n << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "QLN", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		}
	    }
	} else {

/*           There is no complex work space needed for bidiagonal SVD */
/*           The real work space needed for bidiagonal SVD is BDSPAC */
/*           for computing singular values and singular vectors; BDSPAN */
/*           for computing singular values only. */
/*           BDSPAC = 5*M*M + 7*M */
/*           BDSPAN = MAX(7*M+4, 3*M+2+SMLSIZ*(SMLSIZ+8)) */

	    if (*n >= mnthr1) {
		if (wntqn) {

/*                 Path 1t (N much larger than M, JOBZ='N') */

		    maxwrk = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1);
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3;
		} else if (wntqo) {

/*                 Path 2t (N much larger than M, JOBZ='O') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "ZUNGLQ", 
			    " ", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *n + *m * *m + wrkbl;
		    minwrk = (*m << 1) * *m + *m * 3;
		} else if (wntqs) {

/*                 Path 3t (N much larger than M, JOBZ='S') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "ZUNGLQ", 
			    " ", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + *m * 3;
		} else if (wntqa) {

/*                 Path 4t (N much larger than M, JOBZ='A') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "ZGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *n * ilaenv_(&c__1, "ZUNGLQ", 
			    " ", n, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + (*m << 1) * ilaenv_(&
			    c__1, "ZGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
		    maxwrk = *m * *m + wrkbl;
		    minwrk = *m * *m + (*m << 1) + *n;
		}
	    } else if (*n >= mnthr2) {

/*              Path 5t (N much larger than M, but not as much as MNTHR1) */

		maxwrk = (*m << 1) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD", 
			" ", m, n, &c_n1, &c_n1);
		minwrk = (*m << 1) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "P", m, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "P", m, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "P", n, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "Q", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		}
	    } else {

/*              Path 6t (N greater than M, but not much larger) */

		maxwrk = (*m << 1) + (*m + *n) * ilaenv_(&c__1, "ZGEBRD", 
			" ", m, n, &c_n1, &c_n1);
		minwrk = (*m << 1) + *n;
		if (wntqo) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "PRC", m, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNMBR", "QLN", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		    maxwrk += *m * *n;
		    minwrk += *m * *m;
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "PRC", m, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "QLN", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *n * ilaenv_(&c__1, 
			    "ZUNGBR", "PRC", n, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "ZUNGBR", "QLN", m, m, n, &c_n1);
		    maxwrk = max(i__1,i__2);
		}
	    }
	}
	maxwrk = max(maxwrk,minwrk);
    }
    if (*info == 0) {
	work[1].r = (doublereal) maxwrk, work[1].i = 0.;
	if (*lwork < minwrk && *lwork != -1) {
	    *info = -13;
	}
    }

/*     Quick returns */

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGESDD", &i__1);
	return 0;
    }
    if (*lwork == -1) {
	return 0;
    }
    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Get machine constants */

    eps = dlamch_("P");
    smlnum = sqrt(dlamch_("S")) / eps;
    bignum = 1. / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = zlange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
	iscl = 1;
	zlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	zlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
		ierr);
    }

    if (*m >= *n) {

/*        A has at least as many rows as columns. If A has sufficiently */
/*        more rows than columns, first reduce using the QR */
/*        decomposition (if sufficient workspace available) */

	if (*m >= mnthr1) {

	    if (wntqn) {

/*              Path 1 (M much larger than N, JOBZ='N') */
/*              No singular vectors to be computed */

		itau = 1;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (CWorkspace: need 2*N, prefer N+N*NB) */
/*              (RWorkspace: need 0) */

		i__1 = *lwork - nwork + 1;
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out below R */

		i__1 = *n - 1;
		i__2 = *n - 1;
		zlaset_("L", &i__1, &i__2, &c_b1, &c_b1, &a[a_dim1 + 2], lda);
		ie = 1;
		itauq = 1;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in A */
/*              (CWorkspace: need 3*N, prefer 2*N+2*N*NB) */
/*              (RWorkspace: need N) */

		i__1 = *lwork - nwork + 1;
		zgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nrwork = ie + *n;

/*              Perform bidiagonal SVD, compute singular values only */
/*              (CWorkspace: 0) */
/*              (RWorkspace: need BDSPAN) */

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);

	    } else if (wntqo) {

/*              Path 2 (M much larger than N, JOBZ='O') */
/*              N left singular vectors to be overwritten on A and */
/*              N right singular vectors to be computed in VT */

		iu = 1;

/*              WORK(IU) is N by N */

		ldwrku = *n;
		ir = iu + ldwrku * *n;
		if (*lwork >= *m * *n + *n * *n + *n * 3) {

/*                 WORK(IR) is M by N */

		    ldwrkr = *m;
		} else {
		    ldwrkr = (*lwork - *n * *n - *n * 3) / *n;
		}
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (CWorkspace: need N*N+2*N, prefer M*N+N+N*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy R to WORK( IR ), zeroing out below it */

		zlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		zlaset_("L", &i__1, &i__2, &c_b1, &c_b1, &work[ir + 1], &
			ldwrkr);

/*              Generate Q in A */
/*              (CWorkspace: need 2*N, prefer N+N*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__1, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in WORK(IR) */
/*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+2*N*NB) */
/*              (RWorkspace: need N) */

		i__1 = *lwork - nwork + 1;
		zgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of R in WORK(IRU) and computing right singular vectors */
/*              of R in WORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU) */
/*              Overwrite WORK(IU) by the left singular vectors of R */
/*              (CWorkspace: need 2*N*N+3*N, prefer M*N+N*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by the right singular vectors of R */
/*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*              Multiply Q in A by left singular vectors of R in */
/*              WORK(IU), storing result in WORK(IR) and copying to A */
/*              (CWorkspace: need 2*N*N, prefer N*N+M*N) */
/*              (RWorkspace: 0) */

		i__1 = *m;
		i__2 = ldwrkr;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrkr);
		    zgemm_("N", "N", &chunk, n, n, &c_b2, &a[i__ + a_dim1], 
			    lda, &work[iu], &ldwrku, &c_b1, &work[ir], &
			    ldwrkr);
		    zlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ + 
			    a_dim1], lda);
/* L10: */
		}

	    } else if (wntqs) {

/*              Path 3 (M much larger than N, JOBZ='S') */
/*              N left singular vectors to be computed in U and */
/*              N right singular vectors to be computed in VT */

		ir = 1;

/*              WORK(IR) is N by N */

		ldwrkr = *n;
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		zlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		zlaset_("L", &i__2, &i__1, &c_b1, &c_b1, &work[ir + 1], &
			ldwrkr);

/*              Generate Q in A */
/*              (CWorkspace: need 2*N, prefer N+N*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zungqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__2, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in WORK(IR) */
/*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB) */
/*              (RWorkspace: need N) */

		i__2 = *lwork - nwork + 1;
		zgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of R */
/*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of R */
/*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply Q in A by left singular vectors of R in */
/*              WORK(IR), storing result in U */
/*              (CWorkspace: need N*N) */
/*              (RWorkspace: 0) */

		zlacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		zgemm_("N", "N", m, n, n, &c_b2, &a[a_offset], lda, &work[ir], 
			 &ldwrkr, &c_b1, &u[u_offset], ldu);

	    } else if (wntqa) {

/*              Path 4 (M much larger than N, JOBZ='A') */
/*              M left singular vectors to be computed in U and */
/*              N right singular vectors to be computed in VT */

		iu = 1;

/*              WORK(IU) is N by N */

		ldwrku = *n;
		itau = iu + ldwrku * *n;
		nwork = itau + *n;

/*              Compute A=Q*R, copying result to U */
/*              (CWorkspace: need 2*N, prefer N+N*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);

/*              Generate Q in U */
/*              (CWorkspace: need N+M, prefer N+M*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zungqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork], 
			 &i__2, &ierr);

/*              Produce R in A, zeroing out below it */

		i__2 = *n - 1;
		i__1 = *n - 1;
		zlaset_("L", &i__2, &i__1, &c_b1, &c_b1, &a[a_dim1 + 2], lda);
		ie = 1;
		itauq = itau;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in A */
/*              (CWorkspace: need 3*N, prefer 2*N+2*N*NB) */
/*              (RWorkspace: need N) */

		i__2 = *lwork - nwork + 1;
		zgebrd_(n, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		iru = ie + *n;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU) */
/*              Overwrite WORK(IU) by left singular vectors of R */
/*              (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of R */
/*              (CWorkspace: need 3*N, prefer 2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply Q in U by left singular vectors of R in */
/*              WORK(IU), storing result in A */
/*              (CWorkspace: need N*N) */
/*              (RWorkspace: 0) */

		zgemm_("N", "N", m, n, n, &c_b2, &u[u_offset], ldu, &work[iu], 
			 &ldwrku, &c_b1, &a[a_offset], lda);

/*              Copy left singular vectors of A from A to U */

		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);

	    }

	} else if (*m >= mnthr2) {

/*           MNTHR2 <= M < MNTHR1 */

/*           Path 5 (M much larger than N, but not as much as MNTHR1) */
/*           Reduce to bidiagonal form without QR decomposition, use */
/*           ZUNGBR and matrix multiplication to compute singular vectors */

	    ie = 1;
	    nrwork = ie + *n;
	    itauq = 1;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*           Bidiagonalize A */
/*           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB) */
/*           (RWorkspace: need N) */

	    i__2 = *lwork - nwork + 1;
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], 
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*              Compute singular values only */
/*              (Cworkspace: 0) */
/*              (Rworkspace: need BDSPAN) */

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;

/*              Copy A to VT, generate P**H */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*              Generate Q in A */
/*              (CWorkspace: need 2*N, prefer N+N*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &work[
			nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *n * 3) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		} else {

/*                 WORK(IU) is LDWRKU by N */

		    ldwrku = (*lwork - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply real matrix RWORK(IRVT) by P**H in VT, */
/*              storing the result in WORK(IU), copying to VT */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need 3*N*N) */

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &work[iu]
, &ldwrku, &rwork[nrwork]);
		zlacpy_("F", n, n, &work[iu], &ldwrku, &vt[vt_offset], ldvt);

/*              Multiply Q in A by real matrix RWORK(IRU), storing the */
/*              result in WORK(IU), copying to A */
/*              (CWorkspace: need N*N, prefer M*N) */
/*              (Rworkspace: need 3*N*N, prefer N*N+2*M*N) */

		nrwork = irvt;
		i__2 = *m;
		i__1 = ldwrku;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += 
			i__1) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrku);
		    zlacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru], n, 
			    &work[iu], &ldwrku, &rwork[nrwork]);
		    zlacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ + 
			    a_dim1], lda);
/* L20: */
		}

	    } else if (wntqs) {

/*              Copy A to VT, generate P**H */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*              Copy A to U, generate Q */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, n, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply real matrix RWORK(IRVT) by P**H in VT, */
/*              storing the result in A, copying to VT */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need 3*N*N) */

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*              Multiply Q in U by real matrix RWORK(IRU), storing the */
/*              result in A, copying to U */
/*              (CWorkspace: need 0) */
/*              (Rworkspace: need N*N+2*M*N) */

		nrwork = irvt;
		zlacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset], 
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    } else {

/*              Copy A to VT, generate P**H */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("U", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zungbr_("P", n, n, n, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__1, &ierr);

/*              Copy A to U, generate Q */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply real matrix RWORK(IRVT) by P**H in VT, */
/*              storing the result in A, copying to VT */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need 3*N*N) */

		zlarcm_(n, n, &rwork[irvt], n, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", n, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*              Multiply Q in U by real matrix RWORK(IRU), storing the */
/*              result in A, copying to U */
/*              (CWorkspace: 0) */
/*              (Rworkspace: need 3*N*N) */

		nrwork = irvt;
		zlacrm_(m, n, &u[u_offset], ldu, &rwork[iru], n, &a[a_offset], 
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    }

	} else {

/*           M .LT. MNTHR2 */

/*           Path 6 (M at least N, but not much larger) */
/*           Reduce to bidiagonal form without QR decomposition */
/*           Use ZUNMBR to compute singular vectors */

	    ie = 1;
	    nrwork = ie + *n;
	    itauq = 1;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*           Bidiagonalize A */
/*           (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB) */
/*           (RWorkspace: need N) */

	    i__1 = *lwork - nwork + 1;
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], 
		    &work[itaup], &work[nwork], &i__1, &ierr);
	    if (wntqn) {

/*              Compute singular values only */
/*              (Cworkspace: 0) */
/*              (Rworkspace: need BDSPAN) */

		dbdsdc_("U", "N", n, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		if (*lwork >= *m * *n + *n * 3) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		} else {

/*                 WORK( IU ) is LDWRKU by N */

		    ldwrku = (*lwork - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of A */
/*              (Cworkspace: need 2*N, prefer N+N*NB) */
/*              (Rworkspace: need 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

		if (*lwork >= *m * *n + *n * 3) {

/*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU) */
/*              Overwrite WORK(IU) by left singular vectors of A, copying */
/*              to A */
/*              (Cworkspace: need M*N+2*N, prefer M*N+N+N*NB) */
/*              (Rworkspace: need 0) */

		    zlaset_("F", m, n, &c_b1, &c_b1, &work[iu], &ldwrku);
		    zlacp2_("F", n, n, &rwork[iru], n, &work[iu], &ldwrku);
		    i__1 = *lwork - nwork + 1;
		    zunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &
			    ierr);
		    zlacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {

/*                 Generate Q in A */
/*                 (Cworkspace: need 2*N, prefer N+N*NB) */
/*                 (Rworkspace: need 0) */

		    i__1 = *lwork - nwork + 1;
		    zungbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
			    work[nwork], &i__1, &ierr);

/*                 Multiply Q in A by real matrix RWORK(IRU), storing the */
/*                 result in WORK(IU), copying to A */
/*                 (CWorkspace: need N*N, prefer M*N) */
/*                 (Rworkspace: need 3*N*N, prefer N*N+2*M*N) */

		    nrwork = irvt;
		    i__1 = *m;
		    i__2 = ldwrku;
		    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			     i__2) {
/* Computing MIN */
			i__3 = *m - i__ + 1;
			chunk = min(i__3,ldwrku);
			zlacrm_(&chunk, n, &a[i__ + a_dim1], lda, &rwork[iru], 
				 n, &work[iu], &ldwrku, &rwork[nrwork]);
			zlacpy_("F", &chunk, n, &work[iu], &ldwrku, &a[i__ + 
				a_dim1], lda);
/* L30: */
		    }
		}

	    } else if (wntqs) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of A */
/*              (CWorkspace: need 3*N, prefer 2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlaset_("F", m, n, &c_b1, &c_b1, &u[u_offset], ldu)
			;
		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of A */
/*              (CWorkspace: need 3*N, prefer 2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
	    } else {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = nrwork;
		irvt = iru + *n * *n;
		nrwork = irvt + *n * *n;
		dbdsdc_("U", "I", n, &s[1], &rwork[ie], &rwork[iru], n, &
			rwork[irvt], n, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Set the right corner of U to identity matrix */

		zlaset_("F", m, m, &c_b1, &c_b1, &u[u_offset], ldu)
			;
		if (*m > *n) {
		    i__2 = *m - *n;
		    i__1 = *m - *n;
		    zlaset_("F", &i__2, &i__1, &c_b1, &c_b2, &u[*n + 1 + (*n 
			    + 1) * u_dim1], ldu);
		}

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of A */
/*              (CWorkspace: need 2*N+M, prefer 2*N+M*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[iru], n, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of A */
/*              (CWorkspace: need 3*N, prefer 2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", n, n, &rwork[irvt], n, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
	    }

	}

    } else {

/*        A has more columns than rows. If A has sufficiently more */
/*        columns than rows, first reduce using the LQ decomposition (if */
/*        sufficient workspace available) */

	if (*n >= mnthr1) {

	    if (wntqn) {

/*              Path 1t (N much larger than M, JOBZ='N') */
/*              No singular vectors to be computed */

		itau = 1;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (CWorkspace: need 2*M, prefer M+M*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Zero out above L */

		i__2 = *m - 1;
		i__1 = *m - 1;
		zlaset_("U", &i__2, &i__1, &c_b1, &c_b1, &a[(a_dim1 << 1) + 1]
, lda);
		ie = 1;
		itauq = 1;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in A */
/*              (CWorkspace: need 3*M, prefer 2*M+2*M*NB) */
/*              (RWorkspace: need M) */

		i__2 = *lwork - nwork + 1;
		zgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		nrwork = ie + *m;

/*              Perform bidiagonal SVD, compute singular values only */
/*              (CWorkspace: 0) */
/*              (RWorkspace: need BDSPAN) */

		dbdsdc_("U", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);

	    } else if (wntqo) {

/*              Path 2t (N much larger than M, JOBZ='O') */
/*              M right singular vectors to be overwritten on A and */
/*              M left singular vectors to be computed in U */

		ivt = 1;
		ldwkvt = *m;

/*              WORK(IVT) is M by M */

		il = ivt + ldwkvt * *m;
		if (*lwork >= *m * *n + *m * *m + *m * 3) {

/*                 WORK(IL) M by N */

		    ldwrkl = *m;
		    chunk = *n;
		} else {

/*                 WORK(IL) is M by CHUNK */

		    ldwrkl = *m;
		    chunk = (*lwork - *m * *m - *m * 3) / *m;
		}
		itau = il + ldwrkl * chunk;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (CWorkspace: need 2*M, prefer M+M*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy L to WORK(IL), zeroing about above it */

		zlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		zlaset_("U", &i__2, &i__1, &c_b1, &c_b1, &work[il + ldwrkl], &
			ldwrkl);

/*              Generate Q in A */
/*              (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB) */
/*              (RWorkspace: 0) */

		i__2 = *lwork - nwork + 1;
		zunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__2, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in WORK(IL) */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
/*              (RWorkspace: need M) */

		i__2 = *lwork - nwork + 1;
		zgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix WORK(IU) */
/*              Overwrite WORK(IU) by the left singular vectors of L */
/*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT) */
/*              Overwrite WORK(IVT) by the right singular vectors of L */
/*              (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__2 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply right singular vectors of L in WORK(IL) by Q */
/*              in A, storing result in WORK(IL) and copying to A */
/*              (CWorkspace: need 2*M*M, prefer M*M+M*N)) */
/*              (RWorkspace: 0) */

		i__2 = *n;
		i__1 = chunk;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += 
			i__1) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    zgemm_("N", "N", m, &blk, m, &c_b2, &work[ivt], m, &a[i__ 
			    * a_dim1 + 1], lda, &c_b1, &work[il], &ldwrkl);
		    zlacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1 
			    + 1], lda);
/* L40: */
		}

	    } else if (wntqs) {

/*             Path 3t (N much larger than M, JOBZ='S') */
/*             M right singular vectors to be computed in VT and */
/*             M left singular vectors to be computed in U */

		il = 1;

/*              WORK(IL) is M by M */

		ldwrkl = *m;
		itau = il + ldwrkl * *m;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (CWorkspace: need 2*M, prefer M+M*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy L to WORK(IL), zeroing out above it */

		zlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		zlaset_("U", &i__1, &i__2, &c_b1, &c_b1, &work[il + ldwrkl], &
			ldwrkl);

/*              Generate Q in A */
/*              (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zunglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__1, &ierr);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in WORK(IL) */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
/*              (RWorkspace: need M) */

		i__1 = *lwork - nwork + 1;
		zgebrd_(m, m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of L */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by left singular vectors of L */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*              Copy VT to WORK(IL), multiply right singular vectors of L */
/*              in WORK(IL) by Q in A, storing result in VT */
/*              (CWorkspace: need M*M) */
/*              (RWorkspace: 0) */

		zlacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		zgemm_("N", "N", m, n, m, &c_b2, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b1, &vt[vt_offset], ldvt);

	    } else if (wntqa) {

/*              Path 9t (N much larger than M, JOBZ='A') */
/*              N right singular vectors to be computed in VT and */
/*              M left singular vectors to be computed in U */

		ivt = 1;

/*              WORK(IVT) is M by M */

		ldwkvt = *m;
		itau = ivt + ldwkvt * *m;
		nwork = itau + *m;

/*              Compute A=L*Q, copying result to VT */
/*              (CWorkspace: need 2*M, prefer M+M*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*              Generate Q in VT */
/*              (CWorkspace: need M+N, prefer M+N*NB) */
/*              (RWorkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zunglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__1, &ierr);

/*              Produce L in A, zeroing out above it */

		i__1 = *m - 1;
		i__2 = *m - 1;
		zlaset_("U", &i__1, &i__2, &c_b1, &c_b1, &a[(a_dim1 << 1) + 1]
, lda);
		ie = 1;
		itauq = itau;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in A */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
/*              (RWorkspace: need M) */

		i__1 = *lwork - nwork + 1;
		zgebrd_(m, m, &a[a_offset], lda, &s[1], &rwork[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		iru = ie + *m;
		irvt = iru + *m * *m;
		nrwork = irvt + *m * *m;
		dbdsdc_("U", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of L */
/*              (CWorkspace: need 3*M, prefer 2*M+M*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT) */
/*              Overwrite WORK(IVT) by right singular vectors of L */
/*              (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
/*              (RWorkspace: 0) */

		zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__1, &
			ierr);

/*              Multiply right singular vectors of L in WORK(IVT) by */
/*              Q in VT, storing result in A */
/*              (CWorkspace: need M*M) */
/*              (RWorkspace: 0) */

		zgemm_("N", "N", m, n, m, &c_b2, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b1, &a[a_offset], lda);

/*              Copy right singular vectors of A from A to VT */

		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

	    }

	} else if (*n >= mnthr2) {

/*           MNTHR2 <= N < MNTHR1 */

/*           Path 5t (N much larger than M, but not as much as MNTHR1) */
/*           Reduce to bidiagonal form without QR decomposition, use */
/*           ZUNGBR and matrix multiplication to compute singular vectors */


	    ie = 1;
	    nrwork = ie + *m;
	    itauq = 1;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*           Bidiagonalize A */
/*           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB) */
/*           (RWorkspace: M) */

	    i__1 = *lwork - nwork + 1;
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], 
		    &work[itaup], &work[nwork], &i__1, &ierr);

	    if (wntqn) {

/*              Compute singular values only */
/*              (Cworkspace: 0) */
/*              (Rworkspace: need BDSPAN) */

		dbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		ivt = nwork;

/*              Copy A to U, generate Q */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__1, &ierr);

/*              Generate P**H in A */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		i__1 = *lwork - nwork + 1;
		zungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &work[
			nwork], &i__1, &ierr);

		ldwkvt = *m;
		if (*lwork >= *m * *n + *m * 3) {

/*                 WORK( IVT ) is M by N */

		    nwork = ivt + ldwkvt * *n;
		    chunk = *n;
		} else {

/*                 WORK( IVT ) is M by CHUNK */

		    chunk = (*lwork - *m * 3) / *m;
		    nwork = ivt + ldwkvt * chunk;
		}

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply Q in U by real matrix RWORK(IRVT) */
/*              storing the result in WORK(IVT), copying to U */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need 2*M*M) */

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &work[ivt], &
			ldwkvt, &rwork[nrwork]);
		zlacpy_("F", m, m, &work[ivt], &ldwkvt, &u[u_offset], ldu);

/*              Multiply RWORK(IRVT) by P**H in A, storing the */
/*              result in WORK(IVT), copying to A */
/*              (CWorkspace: need M*M, prefer M*N) */
/*              (Rworkspace: need 2*M*M, prefer 2*M*N) */

		nrwork = iru;
		i__1 = *n;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    zlarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1], 
			    lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
		    zlacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ * 
			    a_dim1 + 1], lda);
/* L50: */
		}
	    } else if (wntqs) {

/*              Copy A to U, generate Q */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*              Copy A to VT, generate P**H */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", m, n, m, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply Q in U by real matrix RWORK(IRU), storing the */
/*              result in A, copying to U */
/*              (CWorkspace: need 0) */
/*              (Rworkspace: need 3*M*M) */

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset], 
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*              Multiply real matrix RWORK(IRVT) by P**H in VT, */
/*              storing the result in A, copying to VT */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need M*M+2*M*N) */

		nrwork = iru;
		zlarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    } else {

/*              Copy A to U, generate Q */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("L", m, m, &a[a_offset], lda, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zungbr_("Q", m, m, n, &u[u_offset], ldu, &work[itauq], &work[
			nwork], &i__2, &ierr);

/*              Copy A to VT, generate P**H */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: 0) */

		zlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		i__2 = *lwork - nwork + 1;
		zungbr_("P", n, n, m, &vt[vt_offset], ldvt, &work[itaup], &
			work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Multiply Q in U by real matrix RWORK(IRU), storing the */
/*              result in A, copying to U */
/*              (CWorkspace: need 0) */
/*              (Rworkspace: need 3*M*M) */

		zlacrm_(m, m, &u[u_offset], ldu, &rwork[iru], m, &a[a_offset], 
			 lda, &rwork[nrwork]);
		zlacpy_("F", m, m, &a[a_offset], lda, &u[u_offset], ldu);

/*              Multiply real matrix RWORK(IRVT) by P**H in VT, */
/*              storing the result in A, copying to VT */
/*              (Cworkspace: need 0) */
/*              (Rworkspace: need M*M+2*M*N) */

		zlarcm_(m, n, &rwork[irvt], m, &vt[vt_offset], ldvt, &a[
			a_offset], lda, &rwork[nrwork]);
		zlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    }

	} else {

/*           N .LT. MNTHR2 */

/*           Path 6t (N greater than M, but not much larger) */
/*           Reduce to bidiagonal form without LQ decomposition */
/*           Use ZUNMBR to compute singular vectors */

	    ie = 1;
	    nrwork = ie + *m;
	    itauq = 1;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*           Bidiagonalize A */
/*           (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB) */
/*           (RWorkspace: M) */

	    i__2 = *lwork - nwork + 1;
	    zgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], 
		    &work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*              Compute singular values only */
/*              (Cworkspace: 0) */
/*              (Rworkspace: need BDSPAN) */

		dbdsdc_("L", "N", m, &s[1], &rwork[ie], dum, &c__1, dum, &
			c__1, dum, idum, &rwork[nrwork], &iwork[1], info);
	    } else if (wntqo) {
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3) {

/*                 WORK( IVT ) is M by N */

		    zlaset_("F", m, n, &c_b1, &c_b1, &work[ivt], &ldwkvt);
		    nwork = ivt + ldwkvt * *n;
		} else {

/*                 WORK( IVT ) is M by CHUNK */

		    chunk = (*lwork - *m * 3) / *m;
		    nwork = ivt + ldwkvt * chunk;
		}

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of A */
/*              (Cworkspace: need 2*M, prefer M+M*NB) */
/*              (Rworkspace: need 0) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__2 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *m * 3) {

/*              Copy real matrix RWORK(IRVT) to complex matrix WORK(IVT) */
/*              Overwrite WORK(IVT) by right singular vectors of A, */
/*              copying to A */
/*              (Cworkspace: need M*N+2*M, prefer M*N+M+M*NB) */
/*              (Rworkspace: need 0) */

		    zlacp2_("F", m, m, &rwork[irvt], m, &work[ivt], &ldwkvt);
		    i__2 = *lwork - nwork + 1;
		    zunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, 
			    &ierr);
		    zlacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda);
		} else {

/*                 Generate P**H in A */
/*                 (Cworkspace: need 2*M, prefer M+M*NB) */
/*                 (Rworkspace: need 0) */

		    i__2 = *lwork - nwork + 1;
		    zungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
			    work[nwork], &i__2, &ierr);

/*                 Multiply Q in A by real matrix RWORK(IRU), storing the */
/*                 result in WORK(IU), copying to A */
/*                 (CWorkspace: need M*M, prefer M*N) */
/*                 (Rworkspace: need 3*M*M, prefer M*M+2*M*N) */

		    nrwork = iru;
		    i__2 = *n;
		    i__1 = chunk;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *n - i__ + 1;
			blk = min(i__3,chunk);
			zlarcm_(m, &blk, &rwork[irvt], m, &a[i__ * a_dim1 + 1]
, lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
			zlacpy_("F", m, &blk, &work[ivt], &ldwkvt, &a[i__ * 
				a_dim1 + 1], lda);
/* L60: */
		    }
		}
	    } else if (wntqs) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;
		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of A */
/*              (CWorkspace: need 3*M, prefer 2*M+M*NB) */
/*              (RWorkspace: M*M) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of A */
/*              (CWorkspace: need 3*M, prefer 2*M+M*NB) */
/*              (RWorkspace: M*M) */

		zlaset_("F", m, n, &c_b1, &c_b1, &vt[vt_offset], ldvt);
		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", m, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in RWORK(IRU) and computing right */
/*              singular vectors of bidiagonal matrix in RWORK(IRVT) */
/*              (CWorkspace: need 0) */
/*              (RWorkspace: need BDSPAC) */

		irvt = nrwork;
		iru = irvt + *m * *m;
		nrwork = iru + *m * *m;

		dbdsdc_("L", "I", m, &s[1], &rwork[ie], &rwork[iru], m, &
			rwork[irvt], m, dum, idum, &rwork[nrwork], &iwork[1], 
			info);

/*              Copy real matrix RWORK(IRU) to complex matrix U */
/*              Overwrite U by left singular vectors of A */
/*              (CWorkspace: need 3*M, prefer 2*M+M*NB) */
/*              (RWorkspace: M*M) */

		zlacp2_("F", m, m, &rwork[iru], m, &u[u_offset], ldu);
		i__1 = *lwork - nwork + 1;
		zunmbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);

/*              Set all of VT to identity matrix */

		zlaset_("F", n, n, &c_b1, &c_b2, &vt[vt_offset], ldvt);

/*              Copy real matrix RWORK(IRVT) to complex matrix VT */
/*              Overwrite VT by right singular vectors of A */
/*              (CWorkspace: need 2*M+N, prefer 2*M+N*NB) */
/*              (RWorkspace: M*M) */

		zlacp2_("F", m, m, &rwork[irvt], m, &vt[vt_offset], ldvt);
		i__1 = *lwork - nwork + 1;
		zunmbr_("P", "R", "C", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    }

/*     Undo scaling if necessary */

    if (iscl == 1) {
	if (anrm > bignum) {
	    dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (*info != 0 && anrm > bignum) {
	    i__1 = minmn - 1;
	    dlascl_("G", &c__0, &c__0, &bignum, &anrm, &i__1, &c__1, &rwork[
		    ie], &minmn, &ierr);
	}
	if (anrm < smlnum) {
	    dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (*info != 0 && anrm < smlnum) {
	    i__1 = minmn - 1;
	    dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &i__1, &c__1, &rwork[
		    ie], &minmn, &ierr);
	}
    }

/*     Return optimal workspace in WORK(1) */

    work[1].r = (doublereal) maxwrk, work[1].i = 0.;

    return 0;

/*     End of ZGESDD */

} /* zgesdd_ */
