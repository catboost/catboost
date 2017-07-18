/* dtgex2.f -- translated by f2c (version 20061008).
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

static integer c__4 = 4;
static doublereal c_b5 = 0.;
static integer c__1 = 1;
static integer c__2 = 2;
static doublereal c_b42 = 1.;
static doublereal c_b48 = -1.;
static integer c__0 = 0;

/* Subroutine */ int dtgex2_(logical *wantq, logical *wantz, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	q, integer *ldq, doublereal *z__, integer *ldz, integer *j1, integer *
	n1, integer *n2, doublereal *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, z_dim1, 
	    z_offset, i__1, i__2;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    doublereal f, g;
    integer i__, m;
    doublereal s[16]	/* was [4][4] */, t[16]	/* was [4][4] */, be[2], ai[2]
	    , ar[2], sa, sb, li[16]	/* was [4][4] */, ir[16]	/* 
	    was [4][4] */, ss, ws, eps;
    logical weak;
    doublereal ddum;
    integer idum;
    doublereal taul[4], dsum;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    doublereal taur[4], scpy[16]	/* was [4][4] */, tcpy[16]	/* 
	    was [4][4] */;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal scale, bqra21, brqa21;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    doublereal licop[16]	/* was [4][4] */;
    integer linfo;
    doublereal ircop[16]	/* was [4][4] */, dnorm;
    integer iwork[4];
    extern /* Subroutine */ int dlagv2_(doublereal *, integer *, doublereal *, 
	     integer *, doublereal *, doublereal *, doublereal *, doublereal *
, doublereal *, doublereal *, doublereal *), dgeqr2_(integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *), dgerq2_(integer *, integer *, doublereal *, integer *, 
	     doublereal *, doublereal *, integer *), dorg2r_(integer *, 
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *), dorgr2_(integer *, integer *, integer *, 
	     doublereal *, integer *, doublereal *, doublereal *, integer *), 
	    dorm2r_(char *, char *, integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *), dormr2_(char *, char *, 
	    integer *, integer *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *), dtgsy2_(char *, integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *, 
	     integer *, integer *, integer *);
    extern doublereal dlamch_(char *);
    doublereal dscale;
    extern /* Subroutine */ int dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *), 
	    dlartg_(doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), dlaset_(char *, integer *, integer *, doublereal *, 
	     doublereal *, doublereal *, integer *), dlassq_(integer *
, doublereal *, integer *, doublereal *, doublereal *);
    logical dtrong;
    doublereal thresh, smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTGEX2 swaps adjacent diagonal blocks (A11, B11) and (A22, B22) */
/*  of size 1-by-1 or 2-by-2 in an upper (quasi) triangular matrix pair */
/*  (A, B) by an orthogonal equivalence transformation. */

/*  (A, B) must be in generalized real Schur canonical form (as returned */
/*  by DGGES), i.e. A is block upper triangular with 1-by-1 and 2-by-2 */
/*  diagonal blocks. B is upper triangular. */

/*  Optionally, the matrices Q and Z of generalized Schur vectors are */
/*  updated. */

/*         Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)' */
/*         Q(in) * B(in) * Z(in)' = Q(out) * B(out) * Z(out)' */


/*  Arguments */
/*  ========= */

/*  WANTQ   (input) LOGICAL */
/*          .TRUE. : update the left transformation matrix Q; */
/*          .FALSE.: do not update Q. */

/*  WANTZ   (input) LOGICAL */
/*          .TRUE. : update the right transformation matrix Z; */
/*          .FALSE.: do not update Z. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B. N >= 0. */

/*  A      (input/output) DOUBLE PRECISION arrays, dimensions (LDA,N) */
/*          On entry, the matrix A in the pair (A, B). */
/*          On exit, the updated matrix A. */

/*  LDA     (input)  INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,N). */

/*  B      (input/output) DOUBLE PRECISION arrays, dimensions (LDB,N) */
/*          On entry, the matrix B in the pair (A, B). */
/*          On exit, the updated matrix B. */

/*  LDB     (input)  INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  Q       (input/output) DOUBLE PRECISION array, dimension (LDZ,N) */
/*          On entry, if WANTQ = .TRUE., the orthogonal matrix Q. */
/*          On exit, the updated matrix Q. */
/*          Not referenced if WANTQ = .FALSE.. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. LDQ >= 1. */
/*          If WANTQ = .TRUE., LDQ >= N. */

/*  Z       (input/output) DOUBLE PRECISION array, dimension (LDZ,N) */
/*          On entry, if WANTZ =.TRUE., the orthogonal matrix Z. */
/*          On exit, the updated matrix Z. */
/*          Not referenced if WANTZ = .FALSE.. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z. LDZ >= 1. */
/*          If WANTZ = .TRUE., LDZ >= N. */

/*  J1      (input) INTEGER */
/*          The index to the first block (A11, B11). 1 <= J1 <= N. */

/*  N1      (input) INTEGER */
/*          The order of the first block (A11, B11). N1 = 0, 1 or 2. */

/*  N2      (input) INTEGER */
/*          The order of the second block (A22, B22). N2 = 0, 1 or 2. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)). */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          LWORK >=  MAX( 1, N*(N2+N1), (N2+N1)*(N2+N1)*2 ) */

/*  INFO    (output) INTEGER */
/*            =0: Successful exit */
/*            >0: If INFO = 1, the transformed matrix (A, B) would be */
/*                too far from generalized Schur form; the blocks are */
/*                not swapped and (A, B) and (Q, Z) are unchanged. */
/*                The problem of swapping is too ill-conditioned. */
/*            <0: If INFO = -16: LWORK is too small. Appropriate value */
/*                for LWORK is returned in WORK(1). */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  In the current code both weak and strong stability tests are */
/*  performed. The user can omit the strong stability test by changing */
/*  the internal logical parameter WANDS to .FALSE.. See ref. [2] for */
/*  details. */

/*  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the */
/*      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in */
/*      M.S. Moonen et al (eds), Linear Algebra for Large Scale and */
/*      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218. */

/*  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified */
/*      Eigenvalues of a Regular Matrix Pair (A, B) and Condition */
/*      Estimation: Theory, Algorithms and Software, */
/*      Report UMINF - 94.04, Department of Computing Science, Umea */
/*      University, S-901 87 Umea, Sweden, 1994. Also as LAPACK Working */
/*      Note 87. To appear in Numerical Algorithms, 1996. */

/*  ===================================================================== */
/*  Replaced various illegal calls to DCOPY by calls to DLASET, or by DO */
/*  loops. Sven Hammarling, 1/5/02. */

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

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n <= 1 || *n1 <= 0 || *n2 <= 0) {
	return 0;
    }
    if (*n1 > *n || *j1 + *n1 > *n) {
	return 0;
    }
    m = *n1 + *n2;
/* Computing MAX */
    i__1 = 1, i__2 = *n * m, i__1 = max(i__1,i__2), i__2 = m * m << 1;
    if (*lwork < max(i__1,i__2)) {
	*info = -16;
/* Computing MAX */
	i__1 = 1, i__2 = *n * m, i__1 = max(i__1,i__2), i__2 = m * m << 1;
	work[1] = (doublereal) max(i__1,i__2);
	return 0;
    }

    weak = FALSE_;
    dtrong = FALSE_;

/*     Make a local copy of selected block */

    dlaset_("Full", &c__4, &c__4, &c_b5, &c_b5, li, &c__4);
    dlaset_("Full", &c__4, &c__4, &c_b5, &c_b5, ir, &c__4);
    dlacpy_("Full", &m, &m, &a[*j1 + *j1 * a_dim1], lda, s, &c__4);
    dlacpy_("Full", &m, &m, &b[*j1 + *j1 * b_dim1], ldb, t, &c__4);

/*     Compute threshold for testing acceptance of swapping. */

    eps = dlamch_("P");
    smlnum = dlamch_("S") / eps;
    dscale = 0.;
    dsum = 1.;
    dlacpy_("Full", &m, &m, s, &c__4, &work[1], &m);
    i__1 = m * m;
    dlassq_(&i__1, &work[1], &c__1, &dscale, &dsum);
    dlacpy_("Full", &m, &m, t, &c__4, &work[1], &m);
    i__1 = m * m;
    dlassq_(&i__1, &work[1], &c__1, &dscale, &dsum);
    dnorm = dscale * sqrt(dsum);
/* Computing MAX */
    d__1 = eps * 10. * dnorm;
    thresh = max(d__1,smlnum);

    if (m == 2) {

/*        CASE 1: Swap 1-by-1 and 1-by-1 blocks. */

/*        Compute orthogonal QL and RQ that swap 1-by-1 and 1-by-1 blocks */
/*        using Givens rotations and perform the swap tentatively. */

	f = s[5] * t[0] - t[5] * s[0];
	g = s[5] * t[4] - t[5] * s[4];
	sb = abs(t[5]);
	sa = abs(s[5]);
	dlartg_(&f, &g, &ir[4], ir, &ddum);
	ir[1] = -ir[4];
	ir[5] = ir[0];
	drot_(&c__2, s, &c__1, &s[4], &c__1, ir, &ir[1]);
	drot_(&c__2, t, &c__1, &t[4], &c__1, ir, &ir[1]);
	if (sa >= sb) {
	    dlartg_(s, &s[1], li, &li[1], &ddum);
	} else {
	    dlartg_(t, &t[1], li, &li[1], &ddum);
	}
	drot_(&c__2, s, &c__4, &s[1], &c__4, li, &li[1]);
	drot_(&c__2, t, &c__4, &t[1], &c__4, li, &li[1]);
	li[5] = li[0];
	li[4] = -li[1];

/*        Weak stability test: */
/*           |S21| + |T21| <= O(EPS * F-norm((S, T))) */

	ws = abs(s[1]) + abs(t[1]);
	weak = ws <= thresh;
	if (! weak) {
	    goto L70;
	}

	if (TRUE_) {

/*           Strong stability test: */
/*             F-norm((A-QL'*S*QR, B-QL'*T*QR)) <= O(EPS*F-norm((A,B))) */

	    dlacpy_("Full", &m, &m, &a[*j1 + *j1 * a_dim1], lda, &work[m * m 
		    + 1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b42, li, &c__4, s, &c__4, &c_b5, &
		    work[1], &m);
	    dgemm_("N", "T", &m, &m, &m, &c_b48, &work[1], &m, ir, &c__4, &
		    c_b42, &work[m * m + 1], &m);
	    dscale = 0.;
	    dsum = 1.;
	    i__1 = m * m;
	    dlassq_(&i__1, &work[m * m + 1], &c__1, &dscale, &dsum);

	    dlacpy_("Full", &m, &m, &b[*j1 + *j1 * b_dim1], ldb, &work[m * m 
		    + 1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b42, li, &c__4, t, &c__4, &c_b5, &
		    work[1], &m);
	    dgemm_("N", "T", &m, &m, &m, &c_b48, &work[1], &m, ir, &c__4, &
		    c_b42, &work[m * m + 1], &m);
	    i__1 = m * m;
	    dlassq_(&i__1, &work[m * m + 1], &c__1, &dscale, &dsum);
	    ss = dscale * sqrt(dsum);
	    dtrong = ss <= thresh;
	    if (! dtrong) {
		goto L70;
	    }
	}

/*        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and */
/*               (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)). */

	i__1 = *j1 + 1;
	drot_(&i__1, &a[*j1 * a_dim1 + 1], &c__1, &a[(*j1 + 1) * a_dim1 + 1], 
		&c__1, ir, &ir[1]);
	i__1 = *j1 + 1;
	drot_(&i__1, &b[*j1 * b_dim1 + 1], &c__1, &b[(*j1 + 1) * b_dim1 + 1], 
		&c__1, ir, &ir[1]);
	i__1 = *n - *j1 + 1;
	drot_(&i__1, &a[*j1 + *j1 * a_dim1], lda, &a[*j1 + 1 + *j1 * a_dim1], 
		lda, li, &li[1]);
	i__1 = *n - *j1 + 1;
	drot_(&i__1, &b[*j1 + *j1 * b_dim1], ldb, &b[*j1 + 1 + *j1 * b_dim1], 
		ldb, li, &li[1]);

/*        Set  N1-by-N2 (2,1) - blocks to ZERO. */

	a[*j1 + 1 + *j1 * a_dim1] = 0.;
	b[*j1 + 1 + *j1 * b_dim1] = 0.;

/*        Accumulate transformations into Q and Z if requested. */

	if (*wantz) {
	    drot_(n, &z__[*j1 * z_dim1 + 1], &c__1, &z__[(*j1 + 1) * z_dim1 + 
		    1], &c__1, ir, &ir[1]);
	}
	if (*wantq) {
	    drot_(n, &q[*j1 * q_dim1 + 1], &c__1, &q[(*j1 + 1) * q_dim1 + 1], 
		    &c__1, li, &li[1]);
	}

/*        Exit with INFO = 0 if swap was successfully performed. */

	return 0;

    } else {

/*        CASE 2: Swap 1-by-1 and 2-by-2 blocks, or 2-by-2 */
/*                and 2-by-2 blocks. */

/*        Solve the generalized Sylvester equation */
/*                 S11 * R - L * S22 = SCALE * S12 */
/*                 T11 * R - L * T22 = SCALE * T12 */
/*        for R and L. Solutions in LI and IR. */

	dlacpy_("Full", n1, n2, &t[(*n1 + 1 << 2) - 4], &c__4, li, &c__4);
	dlacpy_("Full", n1, n2, &s[(*n1 + 1 << 2) - 4], &c__4, &ir[*n2 + 1 + (
		*n1 + 1 << 2) - 5], &c__4);
	dtgsy2_("N", &c__0, n1, n2, s, &c__4, &s[*n1 + 1 + (*n1 + 1 << 2) - 5]
, &c__4, &ir[*n2 + 1 + (*n1 + 1 << 2) - 5], &c__4, t, &c__4, &
		t[*n1 + 1 + (*n1 + 1 << 2) - 5], &c__4, li, &c__4, &scale, &
		dsum, &dscale, iwork, &idum, &linfo);

/*        Compute orthogonal matrix QL: */

/*                    QL' * LI = [ TL ] */
/*                               [ 0  ] */
/*        where */
/*                    LI =  [      -L              ] */
/*                          [ SCALE * identity(N2) ] */

	i__1 = *n2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dscal_(n1, &c_b48, &li[(i__ << 2) - 4], &c__1);
	    li[*n1 + i__ + (i__ << 2) - 5] = scale;
/* L10: */
	}
	dgeqr2_(&m, n2, li, &c__4, taul, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}
	dorg2r_(&m, &m, n2, li, &c__4, taul, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}

/*        Compute orthogonal matrix RQ: */

/*                    IR * RQ' =   [ 0  TR], */

/*         where IR = [ SCALE * identity(N1), R ] */

	i__1 = *n1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    ir[*n2 + i__ + (i__ << 2) - 5] = scale;
/* L20: */
	}
	dgerq2_(n1, &m, &ir[*n2], &c__4, taur, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}
	dorgr2_(&m, &m, n1, ir, &c__4, taur, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}

/*        Perform the swapping tentatively: */

	dgemm_("T", "N", &m, &m, &m, &c_b42, li, &c__4, s, &c__4, &c_b5, &
		work[1], &m);
	dgemm_("N", "T", &m, &m, &m, &c_b42, &work[1], &m, ir, &c__4, &c_b5, 
		s, &c__4);
	dgemm_("T", "N", &m, &m, &m, &c_b42, li, &c__4, t, &c__4, &c_b5, &
		work[1], &m);
	dgemm_("N", "T", &m, &m, &m, &c_b42, &work[1], &m, ir, &c__4, &c_b5, 
		t, &c__4);
	dlacpy_("F", &m, &m, s, &c__4, scpy, &c__4);
	dlacpy_("F", &m, &m, t, &c__4, tcpy, &c__4);
	dlacpy_("F", &m, &m, ir, &c__4, ircop, &c__4);
	dlacpy_("F", &m, &m, li, &c__4, licop, &c__4);

/*        Triangularize the B-part by an RQ factorization. */
/*        Apply transformation (from left) to A-part, giving S. */

	dgerq2_(&m, &m, t, &c__4, taur, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}
	dormr2_("R", "T", &m, &m, &m, t, &c__4, taur, s, &c__4, &work[1], &
		linfo);
	if (linfo != 0) {
	    goto L70;
	}
	dormr2_("L", "N", &m, &m, &m, t, &c__4, taur, ir, &c__4, &work[1], &
		linfo);
	if (linfo != 0) {
	    goto L70;
	}

/*        Compute F-norm(S21) in BRQA21. (T21 is 0.) */

	dscale = 0.;
	dsum = 1.;
	i__1 = *n2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlassq_(n1, &s[*n2 + 1 + (i__ << 2) - 5], &c__1, &dscale, &dsum);
/* L30: */
	}
	brqa21 = dscale * sqrt(dsum);

/*        Triangularize the B-part by a QR factorization. */
/*        Apply transformation (from right) to A-part, giving S. */

	dgeqr2_(&m, &m, tcpy, &c__4, taul, &work[1], &linfo);
	if (linfo != 0) {
	    goto L70;
	}
	dorm2r_("L", "T", &m, &m, &m, tcpy, &c__4, taul, scpy, &c__4, &work[1]
, info);
	dorm2r_("R", "N", &m, &m, &m, tcpy, &c__4, taul, licop, &c__4, &work[
		1], info);
	if (linfo != 0) {
	    goto L70;
	}

/*        Compute F-norm(S21) in BQRA21. (T21 is 0.) */

	dscale = 0.;
	dsum = 1.;
	i__1 = *n2;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlassq_(n1, &scpy[*n2 + 1 + (i__ << 2) - 5], &c__1, &dscale, &
		    dsum);
/* L40: */
	}
	bqra21 = dscale * sqrt(dsum);

/*        Decide which method to use. */
/*          Weak stability test: */
/*             F-norm(S21) <= O(EPS * F-norm((S, T))) */

	if (bqra21 <= brqa21 && bqra21 <= thresh) {
	    dlacpy_("F", &m, &m, scpy, &c__4, s, &c__4);
	    dlacpy_("F", &m, &m, tcpy, &c__4, t, &c__4);
	    dlacpy_("F", &m, &m, ircop, &c__4, ir, &c__4);
	    dlacpy_("F", &m, &m, licop, &c__4, li, &c__4);
	} else if (brqa21 >= thresh) {
	    goto L70;
	}

/*        Set lower triangle of B-part to zero */

	i__1 = m - 1;
	i__2 = m - 1;
	dlaset_("Lower", &i__1, &i__2, &c_b5, &c_b5, &t[1], &c__4);

	if (TRUE_) {

/*           Strong stability test: */
/*              F-norm((A-QL*S*QR', B-QL*T*QR')) <= O(EPS*F-norm((A,B))) */

	    dlacpy_("Full", &m, &m, &a[*j1 + *j1 * a_dim1], lda, &work[m * m 
		    + 1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b42, li, &c__4, s, &c__4, &c_b5, &
		    work[1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b48, &work[1], &m, ir, &c__4, &
		    c_b42, &work[m * m + 1], &m);
	    dscale = 0.;
	    dsum = 1.;
	    i__1 = m * m;
	    dlassq_(&i__1, &work[m * m + 1], &c__1, &dscale, &dsum);

	    dlacpy_("Full", &m, &m, &b[*j1 + *j1 * b_dim1], ldb, &work[m * m 
		    + 1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b42, li, &c__4, t, &c__4, &c_b5, &
		    work[1], &m);
	    dgemm_("N", "N", &m, &m, &m, &c_b48, &work[1], &m, ir, &c__4, &
		    c_b42, &work[m * m + 1], &m);
	    i__1 = m * m;
	    dlassq_(&i__1, &work[m * m + 1], &c__1, &dscale, &dsum);
	    ss = dscale * sqrt(dsum);
	    dtrong = ss <= thresh;
	    if (! dtrong) {
		goto L70;
	    }

	}

/*        If the swap is accepted ("weakly" and "strongly"), apply the */
/*        transformations and set N1-by-N2 (2,1)-block to zero. */

	dlaset_("Full", n1, n2, &c_b5, &c_b5, &s[*n2], &c__4);

/*        copy back M-by-M diagonal block starting at index J1 of (A, B) */

	dlacpy_("F", &m, &m, s, &c__4, &a[*j1 + *j1 * a_dim1], lda)
		;
	dlacpy_("F", &m, &m, t, &c__4, &b[*j1 + *j1 * b_dim1], ldb)
		;
	dlaset_("Full", &c__4, &c__4, &c_b5, &c_b5, t, &c__4);

/*        Standardize existing 2-by-2 blocks. */

	i__1 = m * m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.;
/* L50: */
	}
	work[1] = 1.;
	t[0] = 1.;
	idum = *lwork - m * m - 2;
	if (*n2 > 1) {
	    dlagv2_(&a[*j1 + *j1 * a_dim1], lda, &b[*j1 + *j1 * b_dim1], ldb, 
		    ar, ai, be, &work[1], &work[2], t, &t[1]);
	    work[m + 1] = -work[2];
	    work[m + 2] = work[1];
	    t[*n2 + (*n2 << 2) - 5] = t[0];
	    t[4] = -t[1];
	}
	work[m * m] = 1.;
	t[m + (m << 2) - 5] = 1.;

	if (*n1 > 1) {
	    dlagv2_(&a[*j1 + *n2 + (*j1 + *n2) * a_dim1], lda, &b[*j1 + *n2 + 
		    (*j1 + *n2) * b_dim1], ldb, taur, taul, &work[m * m + 1], 
		    &work[*n2 * m + *n2 + 1], &work[*n2 * m + *n2 + 2], &t[*
		    n2 + 1 + (*n2 + 1 << 2) - 5], &t[m + (m - 1 << 2) - 5]);
	    work[m * m] = work[*n2 * m + *n2 + 1];
	    work[m * m - 1] = -work[*n2 * m + *n2 + 2];
	    t[m + (m << 2) - 5] = t[*n2 + 1 + (*n2 + 1 << 2) - 5];
	    t[m - 1 + (m << 2) - 5] = -t[m + (m - 1 << 2) - 5];
	}
	dgemm_("T", "N", n2, n1, n2, &c_b42, &work[1], &m, &a[*j1 + (*j1 + *
		n2) * a_dim1], lda, &c_b5, &work[m * m + 1], n2);
	dlacpy_("Full", n2, n1, &work[m * m + 1], n2, &a[*j1 + (*j1 + *n2) * 
		a_dim1], lda);
	dgemm_("T", "N", n2, n1, n2, &c_b42, &work[1], &m, &b[*j1 + (*j1 + *
		n2) * b_dim1], ldb, &c_b5, &work[m * m + 1], n2);
	dlacpy_("Full", n2, n1, &work[m * m + 1], n2, &b[*j1 + (*j1 + *n2) * 
		b_dim1], ldb);
	dgemm_("N", "N", &m, &m, &m, &c_b42, li, &c__4, &work[1], &m, &c_b5, &
		work[m * m + 1], &m);
	dlacpy_("Full", &m, &m, &work[m * m + 1], &m, li, &c__4);
	dgemm_("N", "N", n2, n1, n1, &c_b42, &a[*j1 + (*j1 + *n2) * a_dim1], 
		lda, &t[*n2 + 1 + (*n2 + 1 << 2) - 5], &c__4, &c_b5, &work[1], 
		 n2);
	dlacpy_("Full", n2, n1, &work[1], n2, &a[*j1 + (*j1 + *n2) * a_dim1], 
		lda);
	dgemm_("N", "N", n2, n1, n1, &c_b42, &b[*j1 + (*j1 + *n2) * b_dim1], 
		ldb, &t[*n2 + 1 + (*n2 + 1 << 2) - 5], &c__4, &c_b5, &work[1], 
		 n2);
	dlacpy_("Full", n2, n1, &work[1], n2, &b[*j1 + (*j1 + *n2) * b_dim1], 
		ldb);
	dgemm_("T", "N", &m, &m, &m, &c_b42, ir, &c__4, t, &c__4, &c_b5, &
		work[1], &m);
	dlacpy_("Full", &m, &m, &work[1], &m, ir, &c__4);

/*        Accumulate transformations into Q and Z if requested. */

	if (*wantq) {
	    dgemm_("N", "N", n, &m, &m, &c_b42, &q[*j1 * q_dim1 + 1], ldq, li, 
		     &c__4, &c_b5, &work[1], n);
	    dlacpy_("Full", n, &m, &work[1], n, &q[*j1 * q_dim1 + 1], ldq);

	}

	if (*wantz) {
	    dgemm_("N", "N", n, &m, &m, &c_b42, &z__[*j1 * z_dim1 + 1], ldz, 
		    ir, &c__4, &c_b5, &work[1], n);
	    dlacpy_("Full", n, &m, &work[1], n, &z__[*j1 * z_dim1 + 1], ldz);

	}

/*        Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and */
/*                (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)). */

	i__ = *j1 + m;
	if (i__ <= *n) {
	    i__1 = *n - i__ + 1;
	    dgemm_("T", "N", &m, &i__1, &m, &c_b42, li, &c__4, &a[*j1 + i__ * 
		    a_dim1], lda, &c_b5, &work[1], &m);
	    i__1 = *n - i__ + 1;
	    dlacpy_("Full", &m, &i__1, &work[1], &m, &a[*j1 + i__ * a_dim1], 
		    lda);
	    i__1 = *n - i__ + 1;
	    dgemm_("T", "N", &m, &i__1, &m, &c_b42, li, &c__4, &b[*j1 + i__ * 
		    b_dim1], lda, &c_b5, &work[1], &m);
	    i__1 = *n - i__ + 1;
	    dlacpy_("Full", &m, &i__1, &work[1], &m, &b[*j1 + i__ * b_dim1], 
		    ldb);
	}
	i__ = *j1 - 1;
	if (i__ > 0) {
	    dgemm_("N", "N", &i__, &m, &m, &c_b42, &a[*j1 * a_dim1 + 1], lda, 
		    ir, &c__4, &c_b5, &work[1], &i__);
	    dlacpy_("Full", &i__, &m, &work[1], &i__, &a[*j1 * a_dim1 + 1], 
		    lda);
	    dgemm_("N", "N", &i__, &m, &m, &c_b42, &b[*j1 * b_dim1 + 1], ldb, 
		    ir, &c__4, &c_b5, &work[1], &i__);
	    dlacpy_("Full", &i__, &m, &work[1], &i__, &b[*j1 * b_dim1 + 1], 
		    ldb);
	}

/*        Exit with INFO = 0 if swap was successfully performed. */

	return 0;

    }

/*     Exit with INFO = 1 if swap was rejected. */

L70:

    *info = 1;
    return 0;

/*     End of DTGEX2 */

} /* dtgex2_ */
