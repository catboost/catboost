/* ctrsen.f -- translated by f2c (version 20061008).
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

static integer c_n1 = -1;

/* Subroutine */ int ctrsen_(char *job, char *compq, logical *select, integer 
	*n, complex *t, integer *ldt, complex *q, integer *ldq, complex *w, 
	integer *m, real *s, real *sep, complex *work, integer *lwork, 
	integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, t_dim1, t_offset, i__1, i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer k, n1, n2, nn, ks;
    real est;
    integer kase, ierr;
    real scale;
    extern logical lsame_(char *, char *);
    integer isave[3], lwmin;
    logical wantq, wants;
    real rnorm;
    extern /* Subroutine */ int clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *);
    real rwork[1];
    extern doublereal clange_(char *, integer *, integer *, complex *, 
	    integer *, real *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), xerbla_(char *, 
	    integer *);
    logical wantbh;
    extern /* Subroutine */ int ctrexc_(char *, integer *, complex *, integer 
	    *, complex *, integer *, integer *, integer *, integer *);
    logical wantsp;
    extern /* Subroutine */ int ctrsyl_(char *, char *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, integer *, complex *, 
	    integer *, real *, integer *);
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     Modified to call CLACN2 in place of CLACON, 10 Feb 03, SJH. */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTRSEN reorders the Schur factorization of a complex matrix */
/*  A = Q*T*Q**H, so that a selected cluster of eigenvalues appears in */
/*  the leading positions on the diagonal of the upper triangular matrix */
/*  T, and the leading columns of Q form an orthonormal basis of the */
/*  corresponding right invariant subspace. */

/*  Optionally the routine computes the reciprocal condition numbers of */
/*  the cluster of eigenvalues and/or the invariant subspace. */

/*  Arguments */
/*  ========= */

/*  JOB     (input) CHARACTER*1 */
/*          Specifies whether condition numbers are required for the */
/*          cluster of eigenvalues (S) or the invariant subspace (SEP): */
/*          = 'N': none; */
/*          = 'E': for eigenvalues only (S); */
/*          = 'V': for invariant subspace only (SEP); */
/*          = 'B': for both eigenvalues and invariant subspace (S and */
/*                 SEP). */

/*  COMPQ   (input) CHARACTER*1 */
/*          = 'V': update the matrix Q of Schur vectors; */
/*          = 'N': do not update Q. */

/*  SELECT  (input) LOGICAL array, dimension (N) */
/*          SELECT specifies the eigenvalues in the selected cluster. To */
/*          select the j-th eigenvalue, SELECT(j) must be set to .TRUE.. */

/*  N       (input) INTEGER */
/*          The order of the matrix T. N >= 0. */

/*  T       (input/output) COMPLEX array, dimension (LDT,N) */
/*          On entry, the upper triangular matrix T. */
/*          On exit, T is overwritten by the reordered matrix T, with the */
/*          selected eigenvalues as the leading diagonal elements. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= max(1,N). */

/*  Q       (input/output) COMPLEX array, dimension (LDQ,N) */
/*          On entry, if COMPQ = 'V', the matrix Q of Schur vectors. */
/*          On exit, if COMPQ = 'V', Q has been postmultiplied by the */
/*          unitary transformation matrix which reorders T; the leading M */
/*          columns of Q form an orthonormal basis for the specified */
/*          invariant subspace. */
/*          If COMPQ = 'N', Q is not referenced. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. */
/*          LDQ >= 1; and if COMPQ = 'V', LDQ >= N. */

/*  W       (output) COMPLEX array, dimension (N) */
/*          The reordered eigenvalues of T, in the same order as they */
/*          appear on the diagonal of T. */

/*  M       (output) INTEGER */
/*          The dimension of the specified invariant subspace. */
/*          0 <= M <= N. */

/*  S       (output) REAL */
/*          If JOB = 'E' or 'B', S is a lower bound on the reciprocal */
/*          condition number for the selected cluster of eigenvalues. */
/*          S cannot underestimate the true reciprocal condition number */
/*          by more than a factor of sqrt(N). If M = 0 or N, S = 1. */
/*          If JOB = 'N' or 'V', S is not referenced. */

/*  SEP     (output) REAL */
/*          If JOB = 'V' or 'B', SEP is the estimated reciprocal */
/*          condition number of the specified invariant subspace. If */
/*          M = 0 or N, SEP = norm(T). */
/*          If JOB = 'N' or 'E', SEP is not referenced. */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          If JOB = 'N', LWORK >= 1; */
/*          if JOB = 'E', LWORK = max(1,M*(N-M)); */
/*          if JOB = 'V' or 'B', LWORK >= max(1,2*M*(N-M)). */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  CTRSEN first collects the selected eigenvalues by computing a unitary */
/*  transformation Z to move them to the top left corner of T. In other */
/*  words, the selected eigenvalues are the eigenvalues of T11 in: */

/*                Z'*T*Z = ( T11 T12 ) n1 */
/*                         (  0  T22 ) n2 */
/*                            n1  n2 */

/*  where N = n1+n2 and Z' means the conjugate transpose of Z. The first */
/*  n1 columns of Z span the specified invariant subspace of T. */

/*  If T has been obtained from the Schur factorization of a matrix */
/*  A = Q*T*Q', then the reordered Schur factorization of A is given by */
/*  A = (Q*Z)*(Z'*T*Z)*(Q*Z)', and the first n1 columns of Q*Z span the */
/*  corresponding invariant subspace of A. */

/*  The reciprocal condition number of the average of the eigenvalues of */
/*  T11 may be returned in S. S lies between 0 (very badly conditioned) */
/*  and 1 (very well conditioned). It is computed as follows. First we */
/*  compute R so that */

/*                         P = ( I  R ) n1 */
/*                             ( 0  0 ) n2 */
/*                               n1 n2 */

/*  is the projector on the invariant subspace associated with T11. */
/*  R is the solution of the Sylvester equation: */

/*                        T11*R - R*T22 = T12. */

/*  Let F-norm(M) denote the Frobenius-norm of M and 2-norm(M) denote */
/*  the two-norm of M. Then S is computed as the lower bound */

/*                      (1 + F-norm(R)**2)**(-1/2) */

/*  on the reciprocal of 2-norm(P), the true reciprocal condition number. */
/*  S cannot underestimate 1 / 2-norm(P) by more than a factor of */
/*  sqrt(N). */

/*  An approximate error bound for the computed average of the */
/*  eigenvalues of T11 is */

/*                         EPS * norm(T) / S */

/*  where EPS is the machine precision. */

/*  The reciprocal condition number of the right invariant subspace */
/*  spanned by the first n1 columns of Z (or of Q*Z) is returned in SEP. */
/*  SEP is defined as the separation of T11 and T22: */

/*                     sep( T11, T22 ) = sigma-min( C ) */

/*  where sigma-min(C) is the smallest singular value of the */
/*  n1*n2-by-n1*n2 matrix */

/*     C  = kprod( I(n2), T11 ) - kprod( transpose(T22), I(n1) ) */

/*  I(m) is an m by m identity matrix, and kprod denotes the Kronecker */
/*  product. We estimate sigma-min(C) by the reciprocal of an estimate of */
/*  the 1-norm of inverse(C). The true reciprocal 1-norm of inverse(C) */
/*  cannot differ from sigma-min(C) by more than a factor of sqrt(n1*n2). */

/*  When SEP is small, small changes in T can cause large changes in */
/*  the invariant subspace. An approximate bound on the maximum angular */
/*  error in the computed right invariant subspace is */

/*                      EPS * norm(T) / SEP */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Decode and test the input parameters. */

    /* Parameter adjustments */
    --select;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --w;
    --work;

    /* Function Body */
    wantbh = lsame_(job, "B");
    wants = lsame_(job, "E") || wantbh;
    wantsp = lsame_(job, "V") || wantbh;
    wantq = lsame_(compq, "V");

/*     Set M to the number of selected eigenvalues. */

    *m = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (select[k]) {
	    ++(*m);
	}
/* L10: */
    }

    n1 = *m;
    n2 = *n - *m;
    nn = n1 * n2;

    *info = 0;
    lquery = *lwork == -1;

    if (wantsp) {
/* Computing MAX */
	i__1 = 1, i__2 = nn << 1;
	lwmin = max(i__1,i__2);
    } else if (lsame_(job, "N")) {
	lwmin = 1;
    } else if (lsame_(job, "E")) {
	lwmin = max(1,nn);
    }

    if (! lsame_(job, "N") && ! wants && ! wantsp) {
	*info = -1;
    } else if (! lsame_(compq, "N") && ! wantq) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ldt < max(1,*n)) {
	*info = -6;
    } else if (*ldq < 1 || wantq && *ldq < *n) {
	*info = -8;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -14;
    }

    if (*info == 0) {
	work[1].r = (real) lwmin, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTRSEN", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == *n || *m == 0) {
	if (wants) {
	    *s = 1.f;
	}
	if (wantsp) {
	    *sep = clange_("1", n, n, &t[t_offset], ldt, rwork);
	}
	goto L40;
    }

/*     Collect the selected eigenvalues at the top left corner of T. */

    ks = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (select[k]) {
	    ++ks;

/*           Swap the K-th eigenvalue to position KS. */

	    if (k != ks) {
		ctrexc_(compq, n, &t[t_offset], ldt, &q[q_offset], ldq, &k, &
			ks, &ierr);
	    }
	}
/* L20: */
    }

    if (wants) {

/*        Solve the Sylvester equation for R: */

/*           T11*R - R*T22 = scale*T12 */

	clacpy_("F", &n1, &n2, &t[(n1 + 1) * t_dim1 + 1], ldt, &work[1], &n1);
	ctrsyl_("N", "N", &c_n1, &n1, &n2, &t[t_offset], ldt, &t[n1 + 1 + (n1 
		+ 1) * t_dim1], ldt, &work[1], &n1, &scale, &ierr);

/*        Estimate the reciprocal of the condition number of the cluster */
/*        of eigenvalues. */

	rnorm = clange_("F", &n1, &n2, &work[1], &n1, rwork);
	if (rnorm == 0.f) {
	    *s = 1.f;
	} else {
	    *s = scale / (sqrt(scale * scale / rnorm + rnorm) * sqrt(rnorm));
	}
    }

    if (wantsp) {

/*        Estimate sep(T11,T22). */

	est = 0.f;
	kase = 0;
L30:
	clacn2_(&nn, &work[nn + 1], &work[1], &est, &kase, isave);
	if (kase != 0) {
	    if (kase == 1) {

/*              Solve T11*R - R*T22 = scale*X. */

		ctrsyl_("N", "N", &c_n1, &n1, &n2, &t[t_offset], ldt, &t[n1 + 
			1 + (n1 + 1) * t_dim1], ldt, &work[1], &n1, &scale, &
			ierr);
	    } else {

/*              Solve T11'*R - R*T22' = scale*X. */

		ctrsyl_("C", "C", &c_n1, &n1, &n2, &t[t_offset], ldt, &t[n1 + 
			1 + (n1 + 1) * t_dim1], ldt, &work[1], &n1, &scale, &
			ierr);
	    }
	    goto L30;
	}

	*sep = scale / est;
    }

L40:

/*     Copy reordered eigenvalues to W. */

    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	i__2 = k;
	i__3 = k + k * t_dim1;
	w[i__2].r = t[i__3].r, w[i__2].i = t[i__3].i;
/* L50: */
    }

    work[1].r = (real) lwmin, work[1].i = 0.f;

    return 0;

/*     End of CTRSEN */

} /* ctrsen_ */
