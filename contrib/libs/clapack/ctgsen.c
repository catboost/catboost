/* ctgsen.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ctgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, complex *a, integer *lda, complex *b, 
	integer *ldb, complex *alpha, complex *beta, complex *q, integer *ldq, 
	 complex *z__, integer *ldz, integer *m, real *pl, real *pr, real *
	dif, complex *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, z_dim1, 
	    z_offset, i__1, i__2, i__3;
    complex q__1, q__2;

    /* Builtin functions */
    double sqrt(doublereal), c_abs(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, k, n1, n2, ks, mn2, ijb, kase, ierr;
    real dsum;
    logical swap;
    complex temp1, temp2;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *, 
	    integer *);
    integer isave[3];
    logical wantd;
    integer lwmin;
    logical wantp;
    extern /* Subroutine */ int clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *);
    logical wantd1, wantd2;
    real dscale;
    extern doublereal slamch_(char *);
    real rdscal;
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *);
    real safmin;
    extern /* Subroutine */ int ctgexc_(logical *, logical *, integer *, 
	    complex *, integer *, complex *, integer *, complex *, integer *, 
	    complex *, integer *, integer *, integer *, integer *), xerbla_(
	    char *, integer *), classq_(integer *, complex *, integer 
	    *, real *, real *);
    integer liwmin;
    extern /* Subroutine */ int ctgsyl_(char *, integer *, integer *, integer 
	    *, complex *, integer *, complex *, integer *, complex *, integer 
	    *, complex *, integer *, complex *, integer *, complex *, integer 
	    *, real *, real *, complex *, integer *, integer *, integer *);
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     January 2007 */

/*     Modified to call CLACN2 in place of CLACON, 10 Feb 03, SJH. */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTGSEN reorders the generalized Schur decomposition of a complex */
/*  matrix pair (A, B) (in terms of an unitary equivalence trans- */
/*  formation Q' * (A, B) * Z), so that a selected cluster of eigenvalues */
/*  appears in the leading diagonal blocks of the pair (A,B). The leading */
/*  columns of Q and Z form unitary bases of the corresponding left and */
/*  right eigenspaces (deflating subspaces). (A, B) must be in */
/*  generalized Schur canonical form, that is, A and B are both upper */
/*  triangular. */

/*  CTGSEN also computes the generalized eigenvalues */

/*           w(j)= ALPHA(j) / BETA(j) */

/*  of the reordered matrix pair (A, B). */

/*  Optionally, the routine computes estimates of reciprocal condition */
/*  numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11), */
/*  (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s) */
/*  between the matrix pairs (A11, B11) and (A22,B22) that correspond to */
/*  the selected cluster and the eigenvalues outside the cluster, resp., */
/*  and norms of "projections" onto left and right eigenspaces w.r.t. */
/*  the selected cluster in the (1,1)-block. */


/*  Arguments */
/*  ========= */

/*  IJOB    (input) integer */
/*          Specifies whether condition numbers are required for the */
/*          cluster of eigenvalues (PL and PR) or the deflating subspaces */
/*          (Difu and Difl): */
/*           =0: Only reorder w.r.t. SELECT. No extras. */
/*           =1: Reciprocal of norms of "projections" onto left and right */
/*               eigenspaces w.r.t. the selected cluster (PL and PR). */
/*           =2: Upper bounds on Difu and Difl. F-norm-based estimate */
/*               (DIF(1:2)). */
/*           =3: Estimate of Difu and Difl. 1-norm-based estimate */
/*               (DIF(1:2)). */
/*               About 5 times as expensive as IJOB = 2. */
/*           =4: Compute PL, PR and DIF (i.e. 0, 1 and 2 above): Economic */
/*               version to get it all. */
/*           =5: Compute PL, PR and DIF (i.e. 0, 1 and 3 above) */

/*  WANTQ   (input) LOGICAL */
/*          .TRUE. : update the left transformation matrix Q; */
/*          .FALSE.: do not update Q. */

/*  WANTZ   (input) LOGICAL */
/*          .TRUE. : update the right transformation matrix Z; */
/*          .FALSE.: do not update Z. */

/*  SELECT  (input) LOGICAL array, dimension (N) */
/*          SELECT specifies the eigenvalues in the selected cluster. To */
/*          select an eigenvalue w(j), SELECT(j) must be set to */
/*          .TRUE.. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B. N >= 0. */

/*  A       (input/output) COMPLEX array, dimension(LDA,N) */
/*          On entry, the upper triangular matrix A, in generalized */
/*          Schur canonical form. */
/*          On exit, A is overwritten by the reordered matrix A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,N). */

/*  B       (input/output) COMPLEX array, dimension(LDB,N) */
/*          On entry, the upper triangular matrix B, in generalized */
/*          Schur canonical form. */
/*          On exit, B is overwritten by the reordered matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  ALPHA   (output) COMPLEX array, dimension (N) */
/*  BETA    (output) COMPLEX array, dimension (N) */
/*          The diagonal elements of A and B, respectively, */
/*          when the pair (A,B) has been reduced to generalized Schur */
/*          form.  ALPHA(i)/BETA(i) i=1,...,N are the generalized */
/*          eigenvalues. */

/*  Q       (input/output) COMPLEX array, dimension (LDQ,N) */
/*          On entry, if WANTQ = .TRUE., Q is an N-by-N matrix. */
/*          On exit, Q has been postmultiplied by the left unitary */
/*          transformation matrix which reorder (A, B); The leading M */
/*          columns of Q form orthonormal bases for the specified pair of */
/*          left eigenspaces (deflating subspaces). */
/*          If WANTQ = .FALSE., Q is not referenced. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. LDQ >= 1. */
/*          If WANTQ = .TRUE., LDQ >= N. */

/*  Z       (input/output) COMPLEX array, dimension (LDZ,N) */
/*          On entry, if WANTZ = .TRUE., Z is an N-by-N matrix. */
/*          On exit, Z has been postmultiplied by the left unitary */
/*          transformation matrix which reorder (A, B); The leading M */
/*          columns of Z form orthonormal bases for the specified pair of */
/*          left eigenspaces (deflating subspaces). */
/*          If WANTZ = .FALSE., Z is not referenced. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z. LDZ >= 1. */
/*          If WANTZ = .TRUE., LDZ >= N. */

/*  M       (output) INTEGER */
/*          The dimension of the specified pair of left and right */
/*          eigenspaces, (deflating subspaces) 0 <= M <= N. */

/*  PL	   (output) REAL */
/*  PR	   (output) REAL */
/*          If IJOB = 1, 4 or 5, PL, PR are lower bounds on the */
/*          reciprocal  of the norm of "projections" onto left and right */
/*          eigenspace with respect to the selected cluster. */
/*          0 < PL, PR <= 1. */
/*          If M = 0 or M = N, PL = PR  = 1. */
/*          If IJOB = 0, 2 or 3 PL, PR are not referenced. */

/*  DIF     (output) REAL array, dimension (2). */
/*          If IJOB >= 2, DIF(1:2) store the estimates of Difu and Difl. */
/*          If IJOB = 2 or 4, DIF(1:2) are F-norm-based upper bounds on */
/*          Difu and Difl. If IJOB = 3 or 5, DIF(1:2) are 1-norm-based */
/*          estimates of Difu and Difl, computed using reversed */
/*          communication with CLACN2. */
/*          If M = 0 or N, DIF(1:2) = F-norm([A, B]). */
/*          If IJOB = 0 or 1, DIF is not referenced. */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          IF IJOB = 0, WORK is not referenced.  Otherwise, */
/*          on exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >=  1 */
/*          If IJOB = 1, 2 or 4, LWORK >=  2*M*(N-M) */
/*          If IJOB = 3 or 5, LWORK >=  4*M*(N-M) */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          IF IJOB = 0, IWORK is not referenced.  Otherwise, */
/*          on exit, if INFO = 0, IWORK(1) returns the optimal LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK. LIWORK >= 1. */
/*          If IJOB = 1, 2 or 4, LIWORK >=  N+2; */
/*          If IJOB = 3 or 5, LIWORK >= MAX(N+2, 2*M*(N-M)); */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal size of the IWORK array, */
/*          returns this value as the first entry of the IWORK array, and */
/*          no error message related to LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*            =0: Successful exit. */
/*            <0: If INFO = -i, the i-th argument had an illegal value. */
/*            =1: Reordering of (A, B) failed because the transformed */
/*                matrix pair (A, B) would be too far from generalized */
/*                Schur form; the problem is very ill-conditioned. */
/*                (A, B) may have been partially reordered. */
/*                If requested, 0 is returned in DIF(*), PL and PR. */


/*  Further Details */
/*  =============== */

/*  CTGSEN first collects the selected eigenvalues by computing unitary */
/*  U and W that move them to the top left corner of (A, B). In other */
/*  words, the selected eigenvalues are the eigenvalues of (A11, B11) in */

/*                U'*(A, B)*W = (A11 A12) (B11 B12) n1 */
/*                              ( 0  A22),( 0  B22) n2 */
/*                                n1  n2    n1  n2 */

/*  where N = n1+n2 and U' means the conjugate transpose of U. The first */
/*  n1 columns of U and W span the specified pair of left and right */
/*  eigenspaces (deflating subspaces) of (A, B). */

/*  If (A, B) has been obtained from the generalized real Schur */
/*  decomposition of a matrix pair (C, D) = Q*(A, B)*Z', then the */
/*  reordered generalized Schur form of (C, D) is given by */

/*           (C, D) = (Q*U)*(U'*(A, B)*W)*(Z*W)', */

/*  and the first n1 columns of Q*U and Z*W span the corresponding */
/*  deflating subspaces of (C, D) (Q and Z store Q*U and Z*W, resp.). */

/*  Note that if the selected eigenvalue is sufficiently ill-conditioned, */
/*  then its value may differ significantly from its value before */
/*  reordering. */

/*  The reciprocal condition numbers of the left and right eigenspaces */
/*  spanned by the first n1 columns of U and W (or Q*U and Z*W) may */
/*  be returned in DIF(1:2), corresponding to Difu and Difl, resp. */

/*  The Difu and Difl are defined as: */

/*       Difu[(A11, B11), (A22, B22)] = sigma-min( Zu ) */
/*  and */
/*       Difl[(A11, B11), (A22, B22)] = Difu[(A22, B22), (A11, B11)], */

/*  where sigma-min(Zu) is the smallest singular value of the */
/*  (2*n1*n2)-by-(2*n1*n2) matrix */

/*       Zu = [ kron(In2, A11)  -kron(A22', In1) ] */
/*            [ kron(In2, B11)  -kron(B22', In1) ]. */

/*  Here, Inx is the identity matrix of size nx and A22' is the */
/*  transpose of A22. kron(X, Y) is the Kronecker product between */
/*  the matrices X and Y. */

/*  When DIF(2) is small, small changes in (A, B) can cause large changes */
/*  in the deflating subspace. An approximate (asymptotic) bound on the */
/*  maximum angular error in the computed deflating subspaces is */

/*       EPS * norm((A, B)) / DIF(2), */

/*  where EPS is the machine precision. */

/*  The reciprocal norm of the projectors on the left and right */
/*  eigenspaces associated with (A11, B11) may be returned in PL and PR. */
/*  They are computed as follows. First we compute L and R so that */
/*  P*(A, B)*Q is block diagonal, where */

/*       P = ( I -L ) n1           Q = ( I R ) n1 */
/*           ( 0  I ) n2    and        ( 0 I ) n2 */
/*             n1 n2                    n1 n2 */

/*  and (L, R) is the solution to the generalized Sylvester equation */

/*       A11*R - L*A22 = -A12 */
/*       B11*R - L*B22 = -B12 */

/*  Then PL = (F-norm(L)**2+1)**(-1/2) and PR = (F-norm(R)**2+1)**(-1/2). */
/*  An approximate (asymptotic) bound on the average absolute error of */
/*  the selected eigenvalues is */

/*       EPS * norm((A, B)) / PL. */

/*  There are also global error bounds which valid for perturbations up */
/*  to a certain restriction:  A lower bound (x) on the smallest */
/*  F-norm(E,F) for which an eigenvalue of (A11, B11) may move and */
/*  coalesce with an eigenvalue of (A22, B22) under perturbation (E,F), */
/*  (i.e. (A + E, B + F), is */

/*   x = min(Difu,Difl)/((1/(PL*PL)+1/(PR*PR))**(1/2)+2*max(1/PL,1/PR)). */

/*  An approximate bound on x can be computed from DIF(1:2), PL and PR. */

/*  If y = ( F-norm(E,F) / x) <= 1, the angles between the perturbed */
/*  (L', R') and unperturbed (L, R) left and right deflating subspaces */
/*  associated with the selected cluster in the (1,1)-blocks can be */
/*  bounded as */

/*   max-angle(L, L') <= arctan( y * PL / (1 - y * (1 - PL * PL)**(1/2)) */
/*   max-angle(R, R') <= arctan( y * PR / (1 - y * (1 - PR * PR)**(1/2)) */

/*  See LAPACK User's Guide section 4.11 or the following references */
/*  for more information. */

/*  Note that if the default method for computing the Frobenius-norm- */
/*  based estimate DIF is not wanted (see CLATDF), then the parameter */
/*  IDIFJB (see below) should be changed from 3 to 4 (routine CLATDF */
/*  (IJOB = 2 will be used)). See CTGSYL for more details. */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  References */
/*  ========== */

/*  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the */
/*      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in */
/*      M.S. Moonen et al (eds), Linear Algebra for Large Scale and */
/*      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218. */

/*  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified */
/*      Eigenvalues of a Regular Matrix Pair (A, B) and Condition */
/*      Estimation: Theory, Algorithms and Software, Report */
/*      UMINF - 94.04, Department of Computing Science, Umea University, */
/*      S-901 87 Umea, Sweden, 1994. Also as LAPACK Working Note 87. */
/*      To appear in Numerical Algorithms, 1996. */

/*  [3] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software */
/*      for Solving the Generalized Sylvester Equation and Estimating the */
/*      Separation between Regular Matrix Pairs, Report UMINF - 93.23, */
/*      Department of Computing Science, Umea University, S-901 87 Umea, */
/*      Sweden, December 1993, Revised April 1994, Also as LAPACK working */
/*      Note 75. To appear in ACM Trans. on Math. Software, Vol 22, No 1, */
/*      1996. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Decode and test the input parameters */

    /* Parameter adjustments */
    --select;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --alpha;
    --beta;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --dif;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1 || *liwork == -1;

    if (*ijob < 0 || *ijob > 5) {
	*info = -1;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,*n)) {
	*info = -7;
    } else if (*ldb < max(1,*n)) {
	*info = -9;
    } else if (*ldq < 1 || *wantq && *ldq < *n) {
	*info = -13;
    } else if (*ldz < 1 || *wantz && *ldz < *n) {
	*info = -15;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTGSEN", &i__1);
	return 0;
    }

    ierr = 0;

    wantp = *ijob == 1 || *ijob >= 4;
    wantd1 = *ijob == 2 || *ijob == 4;
    wantd2 = *ijob == 3 || *ijob == 5;
    wantd = wantd1 || wantd2;

/*     Set M to the dimension of the specified pair of deflating */
/*     subspaces. */

    *m = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	i__2 = k;
	i__3 = k + k * a_dim1;
	alpha[i__2].r = a[i__3].r, alpha[i__2].i = a[i__3].i;
	i__2 = k;
	i__3 = k + k * b_dim1;
	beta[i__2].r = b[i__3].r, beta[i__2].i = b[i__3].i;
	if (k < *n) {
	    if (select[k]) {
		++(*m);
	    }
	} else {
	    if (select[*n]) {
		++(*m);
	    }
	}
/* L10: */
    }

    if (*ijob == 1 || *ijob == 2 || *ijob == 4) {
/* Computing MAX */
	i__1 = 1, i__2 = (*m << 1) * (*n - *m);
	lwmin = max(i__1,i__2);
/* Computing MAX */
	i__1 = 1, i__2 = *n + 2;
	liwmin = max(i__1,i__2);
    } else if (*ijob == 3 || *ijob == 5) {
/* Computing MAX */
	i__1 = 1, i__2 = (*m << 2) * (*n - *m);
	lwmin = max(i__1,i__2);
/* Computing MAX */
	i__1 = 1, i__2 = (*m << 1) * (*n - *m), i__1 = max(i__1,i__2), i__2 = 
		*n + 2;
	liwmin = max(i__1,i__2);
    } else {
	lwmin = 1;
	liwmin = 1;
    }

    work[1].r = (real) lwmin, work[1].i = 0.f;
    iwork[1] = liwmin;

    if (*lwork < lwmin && ! lquery) {
	*info = -21;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -23;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTGSEN", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible. */

    if (*m == *n || *m == 0) {
	if (wantp) {
	    *pl = 1.f;
	    *pr = 1.f;
	}
	if (wantd) {
	    dscale = 0.f;
	    dsum = 1.f;
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		classq_(n, &a[i__ * a_dim1 + 1], &c__1, &dscale, &dsum);
		classq_(n, &b[i__ * b_dim1 + 1], &c__1, &dscale, &dsum);
/* L20: */
	    }
	    dif[1] = dscale * sqrt(dsum);
	    dif[2] = dif[1];
	}
	goto L70;
    }

/*     Get machine constant */

    safmin = slamch_("S");

/*     Collect the selected blocks at the top-left corner of (A, B). */

    ks = 0;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	swap = select[k];
	if (swap) {
	    ++ks;

/*           Swap the K-th block to position KS. Compute unitary Q */
/*           and Z that will swap adjacent diagonal blocks in (A, B). */

	    if (k != ks) {
		ctgexc_(wantq, wantz, n, &a[a_offset], lda, &b[b_offset], ldb, 
			 &q[q_offset], ldq, &z__[z_offset], ldz, &k, &ks, &
			ierr);
	    }

	    if (ierr > 0) {

/*              Swap is rejected: exit. */

		*info = 1;
		if (wantp) {
		    *pl = 0.f;
		    *pr = 0.f;
		}
		if (wantd) {
		    dif[1] = 0.f;
		    dif[2] = 0.f;
		}
		goto L70;
	    }
	}
/* L30: */
    }
    if (wantp) {

/*        Solve generalized Sylvester equation for R and L: */
/*                   A11 * R - L * A22 = A12 */
/*                   B11 * R - L * B22 = B12 */

	n1 = *m;
	n2 = *n - *m;
	i__ = n1 + 1;
	clacpy_("Full", &n1, &n2, &a[i__ * a_dim1 + 1], lda, &work[1], &n1);
	clacpy_("Full", &n1, &n2, &b[i__ * b_dim1 + 1], ldb, &work[n1 * n2 + 
		1], &n1);
	ijb = 0;
	i__1 = *lwork - (n1 << 1) * n2;
	ctgsyl_("N", &ijb, &n1, &n2, &a[a_offset], lda, &a[i__ + i__ * a_dim1]
, lda, &work[1], &n1, &b[b_offset], ldb, &b[i__ + i__ * 
		b_dim1], ldb, &work[n1 * n2 + 1], &n1, &dscale, &dif[1], &
		work[(n1 * n2 << 1) + 1], &i__1, &iwork[1], &ierr);

/*        Estimate the reciprocal of norms of "projections" onto */
/*        left and right eigenspaces */

	rdscal = 0.f;
	dsum = 1.f;
	i__1 = n1 * n2;
	classq_(&i__1, &work[1], &c__1, &rdscal, &dsum);
	*pl = rdscal * sqrt(dsum);
	if (*pl == 0.f) {
	    *pl = 1.f;
	} else {
	    *pl = dscale / (sqrt(dscale * dscale / *pl + *pl) * sqrt(*pl));
	}
	rdscal = 0.f;
	dsum = 1.f;
	i__1 = n1 * n2;
	classq_(&i__1, &work[n1 * n2 + 1], &c__1, &rdscal, &dsum);
	*pr = rdscal * sqrt(dsum);
	if (*pr == 0.f) {
	    *pr = 1.f;
	} else {
	    *pr = dscale / (sqrt(dscale * dscale / *pr + *pr) * sqrt(*pr));
	}
    }
    if (wantd) {

/*        Compute estimates Difu and Difl. */

	if (wantd1) {
	    n1 = *m;
	    n2 = *n - *m;
	    i__ = n1 + 1;
	    ijb = 3;

/*           Frobenius norm-based Difu estimate. */

	    i__1 = *lwork - (n1 << 1) * n2;
	    ctgsyl_("N", &ijb, &n1, &n2, &a[a_offset], lda, &a[i__ + i__ * 
		    a_dim1], lda, &work[1], &n1, &b[b_offset], ldb, &b[i__ + 
		    i__ * b_dim1], ldb, &work[n1 * n2 + 1], &n1, &dscale, &
		    dif[1], &work[(n1 * n2 << 1) + 1], &i__1, &iwork[1], &
		    ierr);

/*           Frobenius norm-based Difl estimate. */

	    i__1 = *lwork - (n1 << 1) * n2;
	    ctgsyl_("N", &ijb, &n2, &n1, &a[i__ + i__ * a_dim1], lda, &a[
		    a_offset], lda, &work[1], &n2, &b[i__ + i__ * b_dim1], 
		    ldb, &b[b_offset], ldb, &work[n1 * n2 + 1], &n2, &dscale, 
		    &dif[2], &work[(n1 * n2 << 1) + 1], &i__1, &iwork[1], &
		    ierr);
	} else {

/*           Compute 1-norm-based estimates of Difu and Difl using */
/*           reversed communication with CLACN2. In each step a */
/*           generalized Sylvester equation or a transposed variant */
/*           is solved. */

	    kase = 0;
	    n1 = *m;
	    n2 = *n - *m;
	    i__ = n1 + 1;
	    ijb = 0;
	    mn2 = (n1 << 1) * n2;

/*           1-norm-based estimate of Difu. */

L40:
	    clacn2_(&mn2, &work[mn2 + 1], &work[1], &dif[1], &kase, isave);
	    if (kase != 0) {
		if (kase == 1) {

/*                 Solve generalized Sylvester equation */

		    i__1 = *lwork - (n1 << 1) * n2;
		    ctgsyl_("N", &ijb, &n1, &n2, &a[a_offset], lda, &a[i__ + 
			    i__ * a_dim1], lda, &work[1], &n1, &b[b_offset], 
			    ldb, &b[i__ + i__ * b_dim1], ldb, &work[n1 * n2 + 
			    1], &n1, &dscale, &dif[1], &work[(n1 * n2 << 1) + 
			    1], &i__1, &iwork[1], &ierr);
		} else {

/*                 Solve the transposed variant. */

		    i__1 = *lwork - (n1 << 1) * n2;
		    ctgsyl_("C", &ijb, &n1, &n2, &a[a_offset], lda, &a[i__ + 
			    i__ * a_dim1], lda, &work[1], &n1, &b[b_offset], 
			    ldb, &b[i__ + i__ * b_dim1], ldb, &work[n1 * n2 + 
			    1], &n1, &dscale, &dif[1], &work[(n1 * n2 << 1) + 
			    1], &i__1, &iwork[1], &ierr);
		}
		goto L40;
	    }
	    dif[1] = dscale / dif[1];

/*           1-norm-based estimate of Difl. */

L50:
	    clacn2_(&mn2, &work[mn2 + 1], &work[1], &dif[2], &kase, isave);
	    if (kase != 0) {
		if (kase == 1) {

/*                 Solve generalized Sylvester equation */

		    i__1 = *lwork - (n1 << 1) * n2;
		    ctgsyl_("N", &ijb, &n2, &n1, &a[i__ + i__ * a_dim1], lda, 
			    &a[a_offset], lda, &work[1], &n2, &b[i__ + i__ * 
			    b_dim1], ldb, &b[b_offset], ldb, &work[n1 * n2 + 
			    1], &n2, &dscale, &dif[2], &work[(n1 * n2 << 1) + 
			    1], &i__1, &iwork[1], &ierr);
		} else {

/*                 Solve the transposed variant. */

		    i__1 = *lwork - (n1 << 1) * n2;
		    ctgsyl_("C", &ijb, &n2, &n1, &a[i__ + i__ * a_dim1], lda, 
			    &a[a_offset], lda, &work[1], &n2, &b[b_offset], 
			    ldb, &b[i__ + i__ * b_dim1], ldb, &work[n1 * n2 + 
			    1], &n2, &dscale, &dif[2], &work[(n1 * n2 << 1) + 
			    1], &i__1, &iwork[1], &ierr);
		}
		goto L50;
	    }
	    dif[2] = dscale / dif[2];
	}
    }

/*     If B(K,K) is complex, make it real and positive (normalization */
/*     of the generalized Schur form) and Store the generalized */
/*     eigenvalues of reordered pair (A, B) */

    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	dscale = c_abs(&b[k + k * b_dim1]);
	if (dscale > safmin) {
	    i__2 = k + k * b_dim1;
	    q__2.r = b[i__2].r / dscale, q__2.i = b[i__2].i / dscale;
	    r_cnjg(&q__1, &q__2);
	    temp1.r = q__1.r, temp1.i = q__1.i;
	    i__2 = k + k * b_dim1;
	    q__1.r = b[i__2].r / dscale, q__1.i = b[i__2].i / dscale;
	    temp2.r = q__1.r, temp2.i = q__1.i;
	    i__2 = k + k * b_dim1;
	    b[i__2].r = dscale, b[i__2].i = 0.f;
	    i__2 = *n - k;
	    cscal_(&i__2, &temp1, &b[k + (k + 1) * b_dim1], ldb);
	    i__2 = *n - k + 1;
	    cscal_(&i__2, &temp1, &a[k + k * a_dim1], lda);
	    if (*wantq) {
		cscal_(n, &temp2, &q[k * q_dim1 + 1], &c__1);
	    }
	} else {
	    i__2 = k + k * b_dim1;
	    b[i__2].r = 0.f, b[i__2].i = 0.f;
	}

	i__2 = k;
	i__3 = k + k * a_dim1;
	alpha[i__2].r = a[i__3].r, alpha[i__2].i = a[i__3].i;
	i__2 = k;
	i__3 = k + k * b_dim1;
	beta[i__2].r = b[i__3].r, beta[i__2].i = b[i__3].i;

/* L60: */
    }

L70:

    work[1].r = (real) lwmin, work[1].i = 0.f;
    iwork[1] = liwmin;

    return 0;

/*     End of CTGSEN */

} /* ctgsen_ */
