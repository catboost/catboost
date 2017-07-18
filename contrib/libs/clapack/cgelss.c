/* cgelss.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {0.f,0.f};
static complex c_b2 = {1.f,0.f};
static integer c__6 = 6;
static integer c_n1 = -1;
static integer c__1 = 1;
static integer c__0 = 0;
static real c_b78 = 0.f;

/* Subroutine */ int cgelss_(integer *m, integer *n, integer *nrhs, complex *
	a, integer *lda, complex *b, integer *ldb, real *s, real *rcond, 
	integer *rank, complex *work, integer *lwork, real *rwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    real r__1;

    /* Local variables */
    integer i__, bl, ie, il, mm;
    real eps, thr, anrm, bnrm;
    integer itau;
    complex vdum[1];
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *);
    integer iascl, ibscl;
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
, complex *, integer *, complex *, integer *, complex *, complex *
, integer *);
    integer chunk;
    real sfmin;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *);
    integer minmn, maxmn, itaup, itauq, mnthr, iwork;
    extern /* Subroutine */ int cgebrd_(integer *, integer *, complex *, 
	    integer *, real *, real *, complex *, complex *, complex *, 
	    integer *, integer *), slabad_(real *, real *);
    extern doublereal clange_(char *, integer *, integer *, complex *, 
	    integer *, real *);
    extern /* Subroutine */ int cgelqf_(integer *, integer *, complex *, 
	    integer *, complex *, complex *, integer *, integer *), clascl_(
	    char *, integer *, integer *, real *, real *, integer *, integer *
, complex *, integer *, integer *), cgeqrf_(integer *, 
	    integer *, complex *, integer *, complex *, complex *, integer *, 
	    integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), claset_(char *, 
	    integer *, integer *, complex *, complex *, complex *, integer *), xerbla_(char *, integer *), cbdsqr_(char *, 
	    integer *, integer *, integer *, integer *, real *, real *, 
	    complex *, integer *, complex *, integer *, complex *, integer *, 
	    real *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    real bignum;
    extern /* Subroutine */ int cungbr_(char *, integer *, integer *, integer 
	    *, complex *, integer *, complex *, complex *, integer *, integer 
	    *), slascl_(char *, integer *, integer *, real *, real *, 
	    integer *, integer *, real *, integer *, integer *), 
	    cunmbr_(char *, char *, char *, integer *, integer *, integer *, 
	    complex *, integer *, complex *, complex *, integer *, complex *, 
	    integer *, integer *), csrscl_(integer *, 
	    real *, complex *, integer *), slaset_(char *, integer *, integer 
	    *, real *, real *, real *, integer *), cunmlq_(char *, 
	    char *, integer *, integer *, integer *, complex *, integer *, 
	    complex *, complex *, integer *, complex *, integer *, integer *);
    integer ldwork;
    extern /* Subroutine */ int cunmqr_(char *, char *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, complex *, integer *, 
	    complex *, integer *, integer *);
    integer minwrk, maxwrk;
    real smlnum;
    integer irwork;
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

/*  CGELSS computes the minimum norm solution to a complex linear */
/*  least squares problem: */

/*  Minimize 2-norm(| b - A*x |). */

/*  using the singular value decomposition (SVD) of A. A is an M-by-N */
/*  matrix which may be rank-deficient. */

/*  Several right hand side vectors b and solution vectors x can be */
/*  handled in a single call; they are stored as the columns of the */
/*  M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix */
/*  X. */

/*  The effective rank of A is determined by treating as zero those */
/*  singular values which are less than RCOND times the largest singular */
/*  value. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A. M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A. N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices B and X. NRHS >= 0. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, the first min(m,n) rows of A are overwritten with */
/*          its right singular vectors, stored rowwise. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  B       (input/output) COMPLEX array, dimension (LDB,NRHS) */
/*          On entry, the M-by-NRHS right hand side matrix B. */
/*          On exit, B is overwritten by the N-by-NRHS solution matrix X. */
/*          If m >= n and RANK = n, the residual sum-of-squares for */
/*          the solution in the i-th column is given by the sum of */
/*          squares of the modulus of elements n+1:m in that column. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,M,N). */

/*  S       (output) REAL array, dimension (min(M,N)) */
/*          The singular values of A in decreasing order. */
/*          The condition number of A in the 2-norm = S(1)/S(min(m,n)). */

/*  RCOND   (input) REAL */
/*          RCOND is used to determine the effective rank of A. */
/*          Singular values S(i) <= RCOND*S(1) are treated as zero. */
/*          If RCOND < 0, machine precision is used instead. */

/*  RANK    (output) INTEGER */
/*          The effective rank of A, i.e., the number of singular values */
/*          which are greater than RCOND*S(1). */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= 1, and also: */
/*          LWORK >=  2*min(M,N) + max(M,N,NRHS) */
/*          For good performance, LWORK should generally be larger. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  RWORK   (workspace) REAL array, dimension (5*min(M,N)) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  the algorithm for computing the SVD failed to converge; */
/*                if INFO = i, i off-diagonal elements of an intermediate */
/*                bidiagonal form did not converge to zero. */

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
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --s;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    maxmn = max(*m,*n);
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldb < max(1,maxmn)) {
	*info = -7;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       CWorkspace refers to complex workspace, and RWorkspace refers */
/*       to real workspace. NB refers to the optimal block size for the */
/*       immediately following subroutine, as returned by ILAENV.) */

    if (*info == 0) {
	minwrk = 1;
	maxwrk = 1;
	if (minmn > 0) {
	    mm = *m;
	    mnthr = ilaenv_(&c__6, "CGELSS", " ", m, n, nrhs, &c_n1);
	    if (*m >= *n && *m >= mnthr) {

/*              Path 1a - overdetermined, with many more rows than */
/*                        columns */

		mm = *n;
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + *n * ilaenv_(&c__1, "CGEQRF", 
			" ", m, n, &c_n1, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n + *nrhs * ilaenv_(&c__1, "CUNMQR", 
			"LC", m, nrhs, n, &c_n1);
		maxwrk = max(i__1,i__2);
	    }
	    if (*m >= *n) {

/*              Path 1 - overdetermined or exactly determined */

/* Computing MAX */
		i__1 = maxwrk, i__2 = (*n << 1) + (mm + *n) * ilaenv_(&c__1, 
			"CGEBRD", " ", &mm, n, &c_n1, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = (*n << 1) + *nrhs * ilaenv_(&c__1, 
			"CUNMBR", "QLC", &mm, nrhs, n, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = (*n << 1) + (*n - 1) * ilaenv_(&c__1, 
			"CUNGBR", "P", n, n, n, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *n * *nrhs;
		maxwrk = max(i__1,i__2);
		minwrk = (*n << 1) + max(*nrhs,*m);
	    }
	    if (*n > *m) {
		minwrk = (*m << 1) + max(*nrhs,*n);
		if (*n >= mnthr) {

/*                 Path 2a - underdetermined, with many more columns */
/*                 than rows */

		    maxwrk = *m + *m * ilaenv_(&c__1, "CGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * 3 + *m * *m + (*m << 1) * 
			    ilaenv_(&c__1, "CGEBRD", " ", m, m, &c_n1, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * 3 + *m * *m + *nrhs * ilaenv_(&
			    c__1, "CUNMBR", "QLC", m, nrhs, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * 3 + *m * *m + (*m - 1) * 
			    ilaenv_(&c__1, "CUNGBR", "P", m, m, m, &c_n1);
		    maxwrk = max(i__1,i__2);
		    if (*nrhs > 1) {
/* Computing MAX */
			i__1 = maxwrk, i__2 = *m * *m + *m + *m * *nrhs;
			maxwrk = max(i__1,i__2);
		    } else {
/* Computing MAX */
			i__1 = maxwrk, i__2 = *m * *m + (*m << 1);
			maxwrk = max(i__1,i__2);
		    }
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m + *nrhs * ilaenv_(&c__1, "CUNMLQ"
, "LC", n, nrhs, m, &c_n1);
		    maxwrk = max(i__1,i__2);
		} else {

/*                 Path 2 - underdetermined */

		    maxwrk = (*m << 1) + (*n + *m) * ilaenv_(&c__1, "CGEBRD", 
			    " ", m, n, &c_n1, &c_n1);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *nrhs * ilaenv_(&c__1, 
			    "CUNMBR", "QLC", m, nrhs, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = (*m << 1) + *m * ilaenv_(&c__1, 
			    "CUNGBR", "P", m, n, m, &c_n1);
		    maxwrk = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *n * *nrhs;
		    maxwrk = max(i__1,i__2);
		}
	    }
	    maxwrk = max(minwrk,maxwrk);
	}
	work[1].r = (real) maxwrk, work[1].i = 0.f;

	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGELSS", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	*rank = 0;
	return 0;
    }

/*     Get machine parameters */

    eps = slamch_("P");
    sfmin = slamch_("S");
    smlnum = sfmin / eps;
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = clange_("M", m, n, &a[a_offset], lda, &rwork[1]);
    iascl = 0;
    if (anrm > 0.f && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	clascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, 
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	clascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, 
		info);
	iascl = 2;
    } else if (anrm == 0.f) {

/*        Matrix all zero. Return zero solution. */

	i__1 = max(*m,*n);
	claset_("F", &i__1, nrhs, &c_b1, &c_b1, &b[b_offset], ldb);
	slaset_("F", &minmn, &c__1, &c_b78, &c_b78, &s[1], &minmn);
	*rank = 0;
	goto L70;
    }

/*     Scale B if max element outside range [SMLNUM,BIGNUM] */

    bnrm = clange_("M", m, nrhs, &b[b_offset], ldb, &rwork[1]);
    ibscl = 0;
    if (bnrm > 0.f && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM */

	clascl_("G", &c__0, &c__0, &bnrm, &smlnum, m, nrhs, &b[b_offset], ldb, 
		 info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM */

	clascl_("G", &c__0, &c__0, &bnrm, &bignum, m, nrhs, &b[b_offset], ldb, 
		 info);
	ibscl = 2;
    }

/*     Overdetermined case */

    if (*m >= *n) {

/*        Path 1 - overdetermined or exactly determined */

	mm = *m;
	if (*m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns */

	    mm = *n;
	    itau = 1;
	    iwork = itau + *n;

/*           Compute A=Q*R */
/*           (CWorkspace: need 2*N, prefer N+N*NB) */
/*           (RWorkspace: none) */

	    i__1 = *lwork - iwork + 1;
	    cgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[iwork], &i__1, 
		     info);

/*           Multiply B by transpose(Q) */
/*           (CWorkspace: need N+NRHS, prefer N+NRHS*NB) */
/*           (RWorkspace: none) */

	    i__1 = *lwork - iwork + 1;
	    cunmqr_("L", "C", m, nrhs, n, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[iwork], &i__1, info);

/*           Zero out below R */

	    if (*n > 1) {
		i__1 = *n - 1;
		i__2 = *n - 1;
		claset_("L", &i__1, &i__2, &c_b1, &c_b1, &a[a_dim1 + 2], lda);
	    }
	}

	ie = 1;
	itauq = 1;
	itaup = itauq + *n;
	iwork = itaup + *n;

/*        Bidiagonalize R in A */
/*        (CWorkspace: need 2*N+MM, prefer 2*N+(MM+N)*NB) */
/*        (RWorkspace: need N) */

	i__1 = *lwork - iwork + 1;
	cgebrd_(&mm, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], &
		work[itaup], &work[iwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of R */
/*        (CWorkspace: need 2*N+NRHS, prefer 2*N+NRHS*NB) */
/*        (RWorkspace: none) */

	i__1 = *lwork - iwork + 1;
	cunmbr_("Q", "L", "C", &mm, nrhs, n, &a[a_offset], lda, &work[itauq], 
		&b[b_offset], ldb, &work[iwork], &i__1, info);

/*        Generate right bidiagonalizing vectors of R in A */
/*        (CWorkspace: need 3*N-1, prefer 2*N+(N-1)*NB) */
/*        (RWorkspace: none) */

	i__1 = *lwork - iwork + 1;
	cungbr_("P", n, n, n, &a[a_offset], lda, &work[itaup], &work[iwork], &
		i__1, info);
	irwork = ie + *n;

/*        Perform bidiagonal QR iteration */
/*          multiply B by transpose of left singular vectors */
/*          compute right singular vectors in A */
/*        (CWorkspace: none) */
/*        (RWorkspace: need BDSPAC) */

	cbdsqr_("U", n, n, &c__0, nrhs, &s[1], &rwork[ie], &a[a_offset], lda, 
		vdum, &c__1, &b[b_offset], ldb, &rwork[irwork], info);
	if (*info != 0) {
	    goto L70;
	}

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	r__1 = *rcond * s[1];
	thr = dmax(r__1,sfmin);
	if (*rcond < 0.f) {
/* Computing MAX */
	    r__1 = eps * s[1];
	    thr = dmax(r__1,sfmin);
	}
	*rank = 0;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (s[i__] > thr) {
		csrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		++(*rank);
	    } else {
		claset_("F", &c__1, nrhs, &c_b1, &c_b1, &b[i__ + b_dim1], ldb);
	    }
/* L10: */
	}

/*        Multiply B by right singular vectors */
/*        (CWorkspace: need N, prefer N*NRHS) */
/*        (RWorkspace: none) */

	if (*lwork >= *ldb * *nrhs && *nrhs > 1) {
	    cgemm_("C", "N", n, nrhs, n, &c_b2, &a[a_offset], lda, &b[
		    b_offset], ldb, &c_b1, &work[1], ldb);
	    clacpy_("G", n, nrhs, &work[1], ldb, &b[b_offset], ldb)
		    ;
	} else if (*nrhs > 1) {
	    chunk = *lwork / *n;
	    i__1 = *nrhs;
	    i__2 = chunk;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = *nrhs - i__ + 1;
		bl = min(i__3,chunk);
		cgemm_("C", "N", n, &bl, n, &c_b2, &a[a_offset], lda, &b[i__ *
			 b_dim1 + 1], ldb, &c_b1, &work[1], n);
		clacpy_("G", n, &bl, &work[1], n, &b[i__ * b_dim1 + 1], ldb);
/* L20: */
	    }
	} else {
	    cgemv_("C", n, n, &c_b2, &a[a_offset], lda, &b[b_offset], &c__1, &
		    c_b1, &work[1], &c__1);
	    ccopy_(n, &work[1], &c__1, &b[b_offset], &c__1);
	}

    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__2 = max(*m,*nrhs), i__1 = *n - (*m << 1);
	if (*n >= mnthr && *lwork >= *m * 3 + *m * *m + max(i__2,i__1)) {

/*        Underdetermined case, M much less than N */

/*        Path 2a - underdetermined, with many more columns than rows */
/*        and sufficient workspace for an efficient algorithm */

	    ldwork = *m;
/* Computing MAX */
	    i__2 = max(*m,*nrhs), i__1 = *n - (*m << 1);
	    if (*lwork >= *m * 3 + *m * *lda + max(i__2,i__1)) {
		ldwork = *lda;
	    }
	    itau = 1;
	    iwork = *m + 1;

/*        Compute A=L*Q */
/*        (CWorkspace: need 2*M, prefer M+M*NB) */
/*        (RWorkspace: none) */

	    i__2 = *lwork - iwork + 1;
	    cgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[iwork], &i__2, 
		     info);
	    il = iwork;

/*        Copy L to WORK(IL), zeroing out above it */

	    clacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwork);
	    i__2 = *m - 1;
	    i__1 = *m - 1;
	    claset_("U", &i__2, &i__1, &c_b1, &c_b1, &work[il + ldwork], &
		    ldwork);
	    ie = 1;
	    itauq = il + ldwork * *m;
	    itaup = itauq + *m;
	    iwork = itaup + *m;

/*        Bidiagonalize L in WORK(IL) */
/*        (CWorkspace: need M*M+4*M, prefer M*M+3*M+2*M*NB) */
/*        (RWorkspace: need M) */

	    i__2 = *lwork - iwork + 1;
	    cgebrd_(m, m, &work[il], &ldwork, &s[1], &rwork[ie], &work[itauq], 
		     &work[itaup], &work[iwork], &i__2, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of L */
/*        (CWorkspace: need M*M+3*M+NRHS, prefer M*M+3*M+NRHS*NB) */
/*        (RWorkspace: none) */

	    i__2 = *lwork - iwork + 1;
	    cunmbr_("Q", "L", "C", m, nrhs, m, &work[il], &ldwork, &work[
		    itauq], &b[b_offset], ldb, &work[iwork], &i__2, info);

/*        Generate right bidiagonalizing vectors of R in WORK(IL) */
/*        (CWorkspace: need M*M+4*M-1, prefer M*M+3*M+(M-1)*NB) */
/*        (RWorkspace: none) */

	    i__2 = *lwork - iwork + 1;
	    cungbr_("P", m, m, m, &work[il], &ldwork, &work[itaup], &work[
		    iwork], &i__2, info);
	    irwork = ie + *m;

/*        Perform bidiagonal QR iteration, computing right singular */
/*        vectors of L in WORK(IL) and multiplying B by transpose of */
/*        left singular vectors */
/*        (CWorkspace: need M*M) */
/*        (RWorkspace: need BDSPAC) */

	    cbdsqr_("U", m, m, &c__0, nrhs, &s[1], &rwork[ie], &work[il], &
		    ldwork, &a[a_offset], lda, &b[b_offset], ldb, &rwork[
		    irwork], info);
	    if (*info != 0) {
		goto L70;
	    }

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	    r__1 = *rcond * s[1];
	    thr = dmax(r__1,sfmin);
	    if (*rcond < 0.f) {
/* Computing MAX */
		r__1 = eps * s[1];
		thr = dmax(r__1,sfmin);
	    }
	    *rank = 0;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		if (s[i__] > thr) {
		    csrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		    ++(*rank);
		} else {
		    claset_("F", &c__1, nrhs, &c_b1, &c_b1, &b[i__ + b_dim1], 
			    ldb);
		}
/* L30: */
	    }
	    iwork = il + *m * ldwork;

/*        Multiply B by right singular vectors of L in WORK(IL) */
/*        (CWorkspace: need M*M+2*M, prefer M*M+M+M*NRHS) */
/*        (RWorkspace: none) */

	    if (*lwork >= *ldb * *nrhs + iwork - 1 && *nrhs > 1) {
		cgemm_("C", "N", m, nrhs, m, &c_b2, &work[il], &ldwork, &b[
			b_offset], ldb, &c_b1, &work[iwork], ldb);
		clacpy_("G", m, nrhs, &work[iwork], ldb, &b[b_offset], ldb);
	    } else if (*nrhs > 1) {
		chunk = (*lwork - iwork + 1) / *m;
		i__2 = *nrhs;
		i__1 = chunk;
		for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += 
			i__1) {
/* Computing MIN */
		    i__3 = *nrhs - i__ + 1;
		    bl = min(i__3,chunk);
		    cgemm_("C", "N", m, &bl, m, &c_b2, &work[il], &ldwork, &b[
			    i__ * b_dim1 + 1], ldb, &c_b1, &work[iwork], m);
		    clacpy_("G", m, &bl, &work[iwork], m, &b[i__ * b_dim1 + 1]
, ldb);
/* L40: */
		}
	    } else {
		cgemv_("C", m, m, &c_b2, &work[il], &ldwork, &b[b_dim1 + 1], &
			c__1, &c_b1, &work[iwork], &c__1);
		ccopy_(m, &work[iwork], &c__1, &b[b_dim1 + 1], &c__1);
	    }

/*        Zero out below first M rows of B */

	    i__1 = *n - *m;
	    claset_("F", &i__1, nrhs, &c_b1, &c_b1, &b[*m + 1 + b_dim1], ldb);
	    iwork = itau + *m;

/*        Multiply transpose(Q) by B */
/*        (CWorkspace: need M+NRHS, prefer M+NHRS*NB) */
/*        (RWorkspace: none) */

	    i__1 = *lwork - iwork + 1;
	    cunmlq_("L", "C", n, nrhs, m, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[iwork], &i__1, info);

	} else {

/*        Path 2 - remaining underdetermined cases */

	    ie = 1;
	    itauq = 1;
	    itaup = itauq + *m;
	    iwork = itaup + *m;

/*        Bidiagonalize A */
/*        (CWorkspace: need 3*M, prefer 2*M+(M+N)*NB) */
/*        (RWorkspace: need N) */

	    i__1 = *lwork - iwork + 1;
	    cgebrd_(m, n, &a[a_offset], lda, &s[1], &rwork[ie], &work[itauq], 
		    &work[itaup], &work[iwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors */
/*        (CWorkspace: need 2*M+NRHS, prefer 2*M+NRHS*NB) */
/*        (RWorkspace: none) */

	    i__1 = *lwork - iwork + 1;
	    cunmbr_("Q", "L", "C", m, nrhs, n, &a[a_offset], lda, &work[itauq]
, &b[b_offset], ldb, &work[iwork], &i__1, info);

/*        Generate right bidiagonalizing vectors in A */
/*        (CWorkspace: need 3*M, prefer 2*M+M*NB) */
/*        (RWorkspace: none) */

	    i__1 = *lwork - iwork + 1;
	    cungbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &work[
		    iwork], &i__1, info);
	    irwork = ie + *m;

/*        Perform bidiagonal QR iteration, */
/*           computing right singular vectors of A in A and */
/*           multiplying B by transpose of left singular vectors */
/*        (CWorkspace: none) */
/*        (RWorkspace: need BDSPAC) */

	    cbdsqr_("L", m, n, &c__0, nrhs, &s[1], &rwork[ie], &a[a_offset], 
		    lda, vdum, &c__1, &b[b_offset], ldb, &rwork[irwork], info);
	    if (*info != 0) {
		goto L70;
	    }

/*        Multiply B by reciprocals of singular values */

/* Computing MAX */
	    r__1 = *rcond * s[1];
	    thr = dmax(r__1,sfmin);
	    if (*rcond < 0.f) {
/* Computing MAX */
		r__1 = eps * s[1];
		thr = dmax(r__1,sfmin);
	    }
	    *rank = 0;
	    i__1 = *m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		if (s[i__] > thr) {
		    csrscl_(nrhs, &s[i__], &b[i__ + b_dim1], ldb);
		    ++(*rank);
		} else {
		    claset_("F", &c__1, nrhs, &c_b1, &c_b1, &b[i__ + b_dim1], 
			    ldb);
		}
/* L50: */
	    }

/*        Multiply B by right singular vectors of A */
/*        (CWorkspace: need N, prefer N*NRHS) */
/*        (RWorkspace: none) */

	    if (*lwork >= *ldb * *nrhs && *nrhs > 1) {
		cgemm_("C", "N", n, nrhs, m, &c_b2, &a[a_offset], lda, &b[
			b_offset], ldb, &c_b1, &work[1], ldb);
		clacpy_("G", n, nrhs, &work[1], ldb, &b[b_offset], ldb);
	    } else if (*nrhs > 1) {
		chunk = *lwork / *n;
		i__1 = *nrhs;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *nrhs - i__ + 1;
		    bl = min(i__3,chunk);
		    cgemm_("C", "N", n, &bl, m, &c_b2, &a[a_offset], lda, &b[
			    i__ * b_dim1 + 1], ldb, &c_b1, &work[1], n);
		    clacpy_("F", n, &bl, &work[1], n, &b[i__ * b_dim1 + 1], 
			    ldb);
/* L60: */
		}
	    } else {
		cgemv_("C", m, n, &c_b2, &a[a_offset], lda, &b[b_offset], &
			c__1, &c_b1, &work[1], &c__1);
		ccopy_(n, &work[1], &c__1, &b[b_offset], &c__1);
	    }
	}
    }

/*     Undo scaling */

    if (iascl == 1) {
	clascl_("G", &c__0, &c__0, &anrm, &smlnum, n, nrhs, &b[b_offset], ldb, 
		 info);
	slascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    } else if (iascl == 2) {
	clascl_("G", &c__0, &c__0, &anrm, &bignum, n, nrhs, &b[b_offset], ldb, 
		 info);
	slascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    }
    if (ibscl == 1) {
	clascl_("G", &c__0, &c__0, &smlnum, &bnrm, n, nrhs, &b[b_offset], ldb, 
		 info);
    } else if (ibscl == 2) {
	clascl_("G", &c__0, &c__0, &bignum, &bnrm, n, nrhs, &b[b_offset], ldb, 
		 info);
    }
L70:
    work[1].r = (real) maxwrk, work[1].i = 0.f;
    return 0;

/*     End of CGELSS */

} /* cgelss_ */
