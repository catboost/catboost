/* dtgsy2.f -- translated by f2c (version 20061008).
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

static integer c__8 = 8;
static integer c__1 = 1;
static doublereal c_b27 = -1.;
static doublereal c_b42 = 1.;
static doublereal c_b56 = 0.;

/* Subroutine */ int dtgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *c__, integer *ldc, doublereal *d__, integer *ldd, 
	doublereal *e, integer *lde, doublereal *f, integer *ldf, doublereal *
	scale, doublereal *rdsum, doublereal *rdscal, integer *iwork, integer 
	*pq, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, d_dim1, 
	    d_offset, e_dim1, e_offset, f_dim1, f_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, k, p, q;
    doublereal z__[64]	/* was [8][8] */;
    integer ie, je, mb, nb, ii, jj, is, js;
    doublereal rhs[8];
    integer isp1, jsp1;
    extern /* Subroutine */ int dger_(integer *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    integer ierr, zdim, ipiv[8], jpiv[8];
    doublereal alpha;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *), dgemm_(char *, char *, integer *, integer *, integer *
, doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *), dcopy_(integer *, 
	    doublereal *, integer *, doublereal *, integer *), daxpy_(integer 
	    *, doublereal *, doublereal *, integer *, doublereal *, integer *)
	    , dgesc2_(integer *, doublereal *, integer *, doublereal *, 
	    integer *, integer *, doublereal *), dgetc2_(integer *, 
	    doublereal *, integer *, integer *, integer *, integer *), 
	    dlatdf_(integer *, integer *, doublereal *, integer *, doublereal 
	    *, doublereal *, doublereal *, integer *, integer *);
    doublereal scaloc;
    extern /* Subroutine */ int dlaset_(char *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *), 
	    xerbla_(char *, integer *);
    logical notran;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     January 2007 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTGSY2 solves the generalized Sylvester equation: */

/*              A * R - L * B = scale * C                (1) */
/*              D * R - L * E = scale * F, */

/*  using Level 1 and 2 BLAS. where R and L are unknown M-by-N matrices, */
/*  (A, D), (B, E) and (C, F) are given matrix pairs of size M-by-M, */
/*  N-by-N and M-by-N, respectively, with real entries. (A, D) and (B, E) */
/*  must be in generalized Schur canonical form, i.e. A, B are upper */
/*  quasi triangular and D, E are upper triangular. The solution (R, L) */
/*  overwrites (C, F). 0 <= SCALE <= 1 is an output scaling factor */
/*  chosen to avoid overflow. */

/*  In matrix notation solving equation (1) corresponds to solve */
/*  Z*x = scale*b, where Z is defined as */

/*         Z = [ kron(In, A)  -kron(B', Im) ]             (2) */
/*             [ kron(In, D)  -kron(E', Im) ], */

/*  Ik is the identity matrix of size k and X' is the transpose of X. */
/*  kron(X, Y) is the Kronecker product between the matrices X and Y. */
/*  In the process of solving (1), we solve a number of such systems */
/*  where Dim(In), Dim(In) = 1 or 2. */

/*  If TRANS = 'T', solve the transposed system Z'*y = scale*b for y, */
/*  which is equivalent to solve for R and L in */

/*              A' * R  + D' * L   = scale *  C           (3) */
/*              R  * B' + L  * E'  = scale * -F */

/*  This case is used to compute an estimate of Dif[(A, D), (B, E)] = */
/*  sigma_min(Z) using reverse communicaton with DLACON. */

/*  DTGSY2 also (IJOB >= 1) contributes to the computation in DTGSYL */
/*  of an upper bound on the separation between to matrix pairs. Then */
/*  the input (A, D), (B, E) are sub-pencils of the matrix pair in */
/*  DTGSYL. See DTGSYL for details. */

/*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER*1 */
/*          = 'N', solve the generalized Sylvester equation (1). */
/*          = 'T': solve the 'transposed' system (3). */

/*  IJOB    (input) INTEGER */
/*          Specifies what kind of functionality to be performed. */
/*          = 0: solve (1) only. */
/*          = 1: A contribution from this subsystem to a Frobenius */
/*               norm-based estimate of the separation between two matrix */
/*               pairs is computed. (look ahead strategy is used). */
/*          = 2: A contribution from this subsystem to a Frobenius */
/*               norm-based estimate of the separation between two matrix */
/*               pairs is computed. (DGECON on sub-systems is used.) */
/*          Not referenced if TRANS = 'T'. */

/*  M       (input) INTEGER */
/*          On entry, M specifies the order of A and D, and the row */
/*          dimension of C, F, R and L. */

/*  N       (input) INTEGER */
/*          On entry, N specifies the order of B and E, and the column */
/*          dimension of C, F, R and L. */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA, M) */
/*          On entry, A contains an upper quasi triangular matrix. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the matrix A. LDA >= max(1, M). */

/*  B       (input) DOUBLE PRECISION array, dimension (LDB, N) */
/*          On entry, B contains an upper quasi triangular matrix. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the matrix B. LDB >= max(1, N). */

/*  C       (input/output) DOUBLE PRECISION array, dimension (LDC, N) */
/*          On entry, C contains the right-hand-side of the first matrix */
/*          equation in (1). */
/*          On exit, if IJOB = 0, C has been overwritten by the */
/*          solution R. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the matrix C. LDC >= max(1, M). */

/*  D       (input) DOUBLE PRECISION array, dimension (LDD, M) */
/*          On entry, D contains an upper triangular matrix. */

/*  LDD     (input) INTEGER */
/*          The leading dimension of the matrix D. LDD >= max(1, M). */

/*  E       (input) DOUBLE PRECISION array, dimension (LDE, N) */
/*          On entry, E contains an upper triangular matrix. */

/*  LDE     (input) INTEGER */
/*          The leading dimension of the matrix E. LDE >= max(1, N). */

/*  F       (input/output) DOUBLE PRECISION array, dimension (LDF, N) */
/*          On entry, F contains the right-hand-side of the second matrix */
/*          equation in (1). */
/*          On exit, if IJOB = 0, F has been overwritten by the */
/*          solution L. */

/*  LDF     (input) INTEGER */
/*          The leading dimension of the matrix F. LDF >= max(1, M). */

/*  SCALE   (output) DOUBLE PRECISION */
/*          On exit, 0 <= SCALE <= 1. If 0 < SCALE < 1, the solutions */
/*          R and L (C and F on entry) will hold the solutions to a */
/*          slightly perturbed system but the input matrices A, B, D and */
/*          E have not been changed. If SCALE = 0, R and L will hold the */
/*          solutions to the homogeneous system with C = F = 0. Normally, */
/*          SCALE = 1. */

/*  RDSUM   (input/output) DOUBLE PRECISION */
/*          On entry, the sum of squares of computed contributions to */
/*          the Dif-estimate under computation by DTGSYL, where the */
/*          scaling factor RDSCAL (see below) has been factored out. */
/*          On exit, the corresponding sum of squares updated with the */
/*          contributions from the current sub-system. */
/*          If TRANS = 'T' RDSUM is not touched. */
/*          NOTE: RDSUM only makes sense when DTGSY2 is called by DTGSYL. */

/*  RDSCAL  (input/output) DOUBLE PRECISION */
/*          On entry, scaling factor used to prevent overflow in RDSUM. */
/*          On exit, RDSCAL is updated w.r.t. the current contributions */
/*          in RDSUM. */
/*          If TRANS = 'T', RDSCAL is not touched. */
/*          NOTE: RDSCAL only makes sense when DTGSY2 is called by */
/*                DTGSYL. */

/*  IWORK   (workspace) INTEGER array, dimension (M+N+2) */

/*  PQ      (output) INTEGER */
/*          On exit, the number of subsystems (of size 2-by-2, 4-by-4 and */
/*          8-by-8) solved by this routine. */

/*  INFO    (output) INTEGER */
/*          On exit, if INFO is set to */
/*            =0: Successful exit */
/*            <0: If INFO = -i, the i-th argument had an illegal value. */
/*            >0: The matrix pairs (A, D) and (B, E) have common or very */
/*                close eigenvalues. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  ===================================================================== */
/*  Replaced various illegal calls to DCOPY by calls to DLASET. */
/*  Sven Hammarling, 27/5/02. */

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

/*     Decode and test input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    d_dim1 = *ldd;
    d_offset = 1 + d_dim1;
    d__ -= d_offset;
    e_dim1 = *lde;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    f_dim1 = *ldf;
    f_offset = 1 + f_dim1;
    f -= f_offset;
    --iwork;

    /* Function Body */
    *info = 0;
    ierr = 0;
    notran = lsame_(trans, "N");
    if (! notran && ! lsame_(trans, "T")) {
	*info = -1;
    } else if (notran) {
	if (*ijob < 0 || *ijob > 2) {
	    *info = -2;
	}
    }
    if (*info == 0) {
	if (*m <= 0) {
	    *info = -3;
	} else if (*n <= 0) {
	    *info = -4;
	} else if (*lda < max(1,*m)) {
	    *info = -5;
	} else if (*ldb < max(1,*n)) {
	    *info = -8;
	} else if (*ldc < max(1,*m)) {
	    *info = -10;
	} else if (*ldd < max(1,*m)) {
	    *info = -12;
	} else if (*lde < max(1,*n)) {
	    *info = -14;
	} else if (*ldf < max(1,*m)) {
	    *info = -16;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTGSY2", &i__1);
	return 0;
    }

/*     Determine block structure of A */

    *pq = 0;
    p = 0;
    i__ = 1;
L10:
    if (i__ > *m) {
	goto L20;
    }
    ++p;
    iwork[p] = i__;
    if (i__ == *m) {
	goto L20;
    }
    if (a[i__ + 1 + i__ * a_dim1] != 0.) {
	i__ += 2;
    } else {
	++i__;
    }
    goto L10;
L20:
    iwork[p + 1] = *m + 1;

/*     Determine block structure of B */

    q = p + 1;
    j = 1;
L30:
    if (j > *n) {
	goto L40;
    }
    ++q;
    iwork[q] = j;
    if (j == *n) {
	goto L40;
    }
    if (b[j + 1 + j * b_dim1] != 0.) {
	j += 2;
    } else {
	++j;
    }
    goto L30;
L40:
    iwork[q + 1] = *n + 1;
    *pq = p * (q - p - 1);

    if (notran) {

/*        Solve (I, J) - subsystem */
/*           A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J) */
/*           D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J) */
/*        for I = P, P - 1, ..., 1; J = 1, 2, ..., Q */

	*scale = 1.;
	scaloc = 1.;
	i__1 = q;
	for (j = p + 2; j <= i__1; ++j) {
	    js = iwork[j];
	    jsp1 = js + 1;
	    je = iwork[j + 1] - 1;
	    nb = je - js + 1;
	    for (i__ = p; i__ >= 1; --i__) {

		is = iwork[i__];
		isp1 = is + 1;
		ie = iwork[i__ + 1] - 1;
		mb = ie - is + 1;
		zdim = mb * nb << 1;

		if (mb == 1 && nb == 1) {

/*                 Build a 2-by-2 system Z * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = d__[is + is * d_dim1];
		    z__[8] = -b[js + js * b_dim1];
		    z__[9] = -e[js + js * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = f[is + js * f_dim1];

/*                 Solve Z * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }

		    if (*ijob == 0) {
			dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
			if (scaloc != 1.) {
			    i__2 = *n;
			    for (k = 1; k <= i__2; ++k) {
				dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &
					c__1);
				dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L50: */
			    }
			    *scale *= scaloc;
			}
		    } else {
			dlatdf_(ijob, &zdim, z__, &c__8, rhs, rdsum, rdscal, 
				ipiv, jpiv);
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    f[is + js * f_dim1] = rhs[1];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (i__ > 1) {
			alpha = -rhs[0];
			i__2 = is - 1;
			daxpy_(&i__2, &alpha, &a[is * a_dim1 + 1], &c__1, &
				c__[js * c_dim1 + 1], &c__1);
			i__2 = is - 1;
			daxpy_(&i__2, &alpha, &d__[is * d_dim1 + 1], &c__1, &
				f[js * f_dim1 + 1], &c__1);
		    }
		    if (j < q) {
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[1], &b[js + (je + 1) * b_dim1], 
				ldb, &c__[is + (je + 1) * c_dim1], ldc);
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[1], &e[js + (je + 1) * e_dim1], 
				lde, &f[is + (je + 1) * f_dim1], ldf);
		    }

		} else if (mb == 1 && nb == 2) {

/*                 Build a 4-by-4 system Z * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = 0.;
		    z__[2] = d__[is + is * d_dim1];
		    z__[3] = 0.;

		    z__[8] = 0.;
		    z__[9] = a[is + is * a_dim1];
		    z__[10] = 0.;
		    z__[11] = d__[is + is * d_dim1];

		    z__[16] = -b[js + js * b_dim1];
		    z__[17] = -b[js + jsp1 * b_dim1];
		    z__[18] = -e[js + js * e_dim1];
		    z__[19] = -e[js + jsp1 * e_dim1];

		    z__[24] = -b[jsp1 + js * b_dim1];
		    z__[25] = -b[jsp1 + jsp1 * b_dim1];
		    z__[26] = 0.;
		    z__[27] = -e[jsp1 + jsp1 * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = c__[is + jsp1 * c_dim1];
		    rhs[2] = f[is + js * f_dim1];
		    rhs[3] = f[is + jsp1 * f_dim1];

/*                 Solve Z * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }

		    if (*ijob == 0) {
			dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
			if (scaloc != 1.) {
			    i__2 = *n;
			    for (k = 1; k <= i__2; ++k) {
				dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &
					c__1);
				dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L60: */
			    }
			    *scale *= scaloc;
			}
		    } else {
			dlatdf_(ijob, &zdim, z__, &c__8, rhs, rdsum, rdscal, 
				ipiv, jpiv);
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    c__[is + jsp1 * c_dim1] = rhs[1];
		    f[is + js * f_dim1] = rhs[2];
		    f[is + jsp1 * f_dim1] = rhs[3];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (i__ > 1) {
			i__2 = is - 1;
			dger_(&i__2, &nb, &c_b27, &a[is * a_dim1 + 1], &c__1, 
				rhs, &c__1, &c__[js * c_dim1 + 1], ldc);
			i__2 = is - 1;
			dger_(&i__2, &nb, &c_b27, &d__[is * d_dim1 + 1], &
				c__1, rhs, &c__1, &f[js * f_dim1 + 1], ldf);
		    }
		    if (j < q) {
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[2], &b[js + (je + 1) * b_dim1], 
				ldb, &c__[is + (je + 1) * c_dim1], ldc);
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[2], &e[js + (je + 1) * e_dim1], 
				lde, &f[is + (je + 1) * f_dim1], ldf);
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[3], &b[jsp1 + (je + 1) * b_dim1], 
				ldb, &c__[is + (je + 1) * c_dim1], ldc);
			i__2 = *n - je;
			daxpy_(&i__2, &rhs[3], &e[jsp1 + (je + 1) * e_dim1], 
				lde, &f[is + (je + 1) * f_dim1], ldf);
		    }

		} else if (mb == 2 && nb == 1) {

/*                 Build a 4-by-4 system Z * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = a[isp1 + is * a_dim1];
		    z__[2] = d__[is + is * d_dim1];
		    z__[3] = 0.;

		    z__[8] = a[is + isp1 * a_dim1];
		    z__[9] = a[isp1 + isp1 * a_dim1];
		    z__[10] = d__[is + isp1 * d_dim1];
		    z__[11] = d__[isp1 + isp1 * d_dim1];

		    z__[16] = -b[js + js * b_dim1];
		    z__[17] = 0.;
		    z__[18] = -e[js + js * e_dim1];
		    z__[19] = 0.;

		    z__[24] = 0.;
		    z__[25] = -b[js + js * b_dim1];
		    z__[26] = 0.;
		    z__[27] = -e[js + js * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = c__[isp1 + js * c_dim1];
		    rhs[2] = f[is + js * f_dim1];
		    rhs[3] = f[isp1 + js * f_dim1];

/*                 Solve Z * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }
		    if (*ijob == 0) {
			dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
			if (scaloc != 1.) {
			    i__2 = *n;
			    for (k = 1; k <= i__2; ++k) {
				dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &
					c__1);
				dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L70: */
			    }
			    *scale *= scaloc;
			}
		    } else {
			dlatdf_(ijob, &zdim, z__, &c__8, rhs, rdsum, rdscal, 
				ipiv, jpiv);
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    c__[isp1 + js * c_dim1] = rhs[1];
		    f[is + js * f_dim1] = rhs[2];
		    f[isp1 + js * f_dim1] = rhs[3];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (i__ > 1) {
			i__2 = is - 1;
			dgemv_("N", &i__2, &mb, &c_b27, &a[is * a_dim1 + 1], 
				lda, rhs, &c__1, &c_b42, &c__[js * c_dim1 + 1]
, &c__1);
			i__2 = is - 1;
			dgemv_("N", &i__2, &mb, &c_b27, &d__[is * d_dim1 + 1], 
				 ldd, rhs, &c__1, &c_b42, &f[js * f_dim1 + 1], 
				 &c__1);
		    }
		    if (j < q) {
			i__2 = *n - je;
			dger_(&mb, &i__2, &c_b42, &rhs[2], &c__1, &b[js + (je 
				+ 1) * b_dim1], ldb, &c__[is + (je + 1) * 
				c_dim1], ldc);
			i__2 = *n - je;
			dger_(&mb, &i__2, &c_b42, &rhs[2], &c__1, &e[js + (je 
				+ 1) * e_dim1], lde, &f[is + (je + 1) * 
				f_dim1], ldf);
		    }

		} else if (mb == 2 && nb == 2) {

/*                 Build an 8-by-8 system Z * x = RHS */

		    dlaset_("F", &c__8, &c__8, &c_b56, &c_b56, z__, &c__8);

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = a[isp1 + is * a_dim1];
		    z__[4] = d__[is + is * d_dim1];

		    z__[8] = a[is + isp1 * a_dim1];
		    z__[9] = a[isp1 + isp1 * a_dim1];
		    z__[12] = d__[is + isp1 * d_dim1];
		    z__[13] = d__[isp1 + isp1 * d_dim1];

		    z__[18] = a[is + is * a_dim1];
		    z__[19] = a[isp1 + is * a_dim1];
		    z__[22] = d__[is + is * d_dim1];

		    z__[26] = a[is + isp1 * a_dim1];
		    z__[27] = a[isp1 + isp1 * a_dim1];
		    z__[30] = d__[is + isp1 * d_dim1];
		    z__[31] = d__[isp1 + isp1 * d_dim1];

		    z__[32] = -b[js + js * b_dim1];
		    z__[34] = -b[js + jsp1 * b_dim1];
		    z__[36] = -e[js + js * e_dim1];
		    z__[38] = -e[js + jsp1 * e_dim1];

		    z__[41] = -b[js + js * b_dim1];
		    z__[43] = -b[js + jsp1 * b_dim1];
		    z__[45] = -e[js + js * e_dim1];
		    z__[47] = -e[js + jsp1 * e_dim1];

		    z__[48] = -b[jsp1 + js * b_dim1];
		    z__[50] = -b[jsp1 + jsp1 * b_dim1];
		    z__[54] = -e[jsp1 + jsp1 * e_dim1];

		    z__[57] = -b[jsp1 + js * b_dim1];
		    z__[59] = -b[jsp1 + jsp1 * b_dim1];
		    z__[63] = -e[jsp1 + jsp1 * e_dim1];

/*                 Set up right hand side(s) */

		    k = 1;
		    ii = mb * nb + 1;
		    i__2 = nb - 1;
		    for (jj = 0; jj <= i__2; ++jj) {
			dcopy_(&mb, &c__[is + (js + jj) * c_dim1], &c__1, &
				rhs[k - 1], &c__1);
			dcopy_(&mb, &f[is + (js + jj) * f_dim1], &c__1, &rhs[
				ii - 1], &c__1);
			k += mb;
			ii += mb;
/* L80: */
		    }

/*                 Solve Z * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }
		    if (*ijob == 0) {
			dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
			if (scaloc != 1.) {
			    i__2 = *n;
			    for (k = 1; k <= i__2; ++k) {
				dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &
					c__1);
				dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L90: */
			    }
			    *scale *= scaloc;
			}
		    } else {
			dlatdf_(ijob, &zdim, z__, &c__8, rhs, rdsum, rdscal, 
				ipiv, jpiv);
		    }

/*                 Unpack solution vector(s) */

		    k = 1;
		    ii = mb * nb + 1;
		    i__2 = nb - 1;
		    for (jj = 0; jj <= i__2; ++jj) {
			dcopy_(&mb, &rhs[k - 1], &c__1, &c__[is + (js + jj) * 
				c_dim1], &c__1);
			dcopy_(&mb, &rhs[ii - 1], &c__1, &f[is + (js + jj) * 
				f_dim1], &c__1);
			k += mb;
			ii += mb;
/* L100: */
		    }

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (i__ > 1) {
			i__2 = is - 1;
			dgemm_("N", "N", &i__2, &nb, &mb, &c_b27, &a[is * 
				a_dim1 + 1], lda, rhs, &mb, &c_b42, &c__[js * 
				c_dim1 + 1], ldc);
			i__2 = is - 1;
			dgemm_("N", "N", &i__2, &nb, &mb, &c_b27, &d__[is * 
				d_dim1 + 1], ldd, rhs, &mb, &c_b42, &f[js * 
				f_dim1 + 1], ldf);
		    }
		    if (j < q) {
			k = mb * nb + 1;
			i__2 = *n - je;
			dgemm_("N", "N", &mb, &i__2, &nb, &c_b42, &rhs[k - 1], 
				 &mb, &b[js + (je + 1) * b_dim1], ldb, &c_b42, 
				 &c__[is + (je + 1) * c_dim1], ldc);
			i__2 = *n - je;
			dgemm_("N", "N", &mb, &i__2, &nb, &c_b42, &rhs[k - 1], 
				 &mb, &e[js + (je + 1) * e_dim1], lde, &c_b42, 
				 &f[is + (je + 1) * f_dim1], ldf);
		    }

		}

/* L110: */
	    }
/* L120: */
	}
    } else {

/*        Solve (I, J) - subsystem */
/*             A(I, I)' * R(I, J) + D(I, I)' * L(J, J)  =  C(I, J) */
/*             R(I, I)  * B(J, J) + L(I, J)  * E(J, J)  = -F(I, J) */
/*        for I = 1, 2, ..., P, J = Q, Q - 1, ..., 1 */

	*scale = 1.;
	scaloc = 1.;
	i__1 = p;
	for (i__ = 1; i__ <= i__1; ++i__) {

	    is = iwork[i__];
	    isp1 = is + 1;
	    ie = i__;
	    mb = ie - is + 1;
	    i__2 = p + 2;
	    for (j = q; j >= i__2; --j) {

		js = iwork[j];
		jsp1 = js + 1;
		je = iwork[j + 1] - 1;
		nb = je - js + 1;
		zdim = mb * nb << 1;
		if (mb == 1 && nb == 1) {

/*                 Build a 2-by-2 system Z' * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = -b[js + js * b_dim1];
		    z__[8] = d__[is + is * d_dim1];
		    z__[9] = -e[js + js * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = f[is + js * f_dim1];

/*                 Solve Z' * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }

		    dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
		    if (scaloc != 1.) {
			i__3 = *n;
			for (k = 1; k <= i__3; ++k) {
			    dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &c__1);
			    dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L130: */
			}
			*scale *= scaloc;
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    f[is + js * f_dim1] = rhs[1];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (j > p + 2) {
			alpha = rhs[0];
			i__3 = js - 1;
			daxpy_(&i__3, &alpha, &b[js * b_dim1 + 1], &c__1, &f[
				is + f_dim1], ldf);
			alpha = rhs[1];
			i__3 = js - 1;
			daxpy_(&i__3, &alpha, &e[js * e_dim1 + 1], &c__1, &f[
				is + f_dim1], ldf);
		    }
		    if (i__ < p) {
			alpha = -rhs[0];
			i__3 = *m - ie;
			daxpy_(&i__3, &alpha, &a[is + (ie + 1) * a_dim1], lda, 
				 &c__[ie + 1 + js * c_dim1], &c__1);
			alpha = -rhs[1];
			i__3 = *m - ie;
			daxpy_(&i__3, &alpha, &d__[is + (ie + 1) * d_dim1], 
				ldd, &c__[ie + 1 + js * c_dim1], &c__1);
		    }

		} else if (mb == 1 && nb == 2) {

/*                 Build a 4-by-4 system Z' * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = 0.;
		    z__[2] = -b[js + js * b_dim1];
		    z__[3] = -b[jsp1 + js * b_dim1];

		    z__[8] = 0.;
		    z__[9] = a[is + is * a_dim1];
		    z__[10] = -b[js + jsp1 * b_dim1];
		    z__[11] = -b[jsp1 + jsp1 * b_dim1];

		    z__[16] = d__[is + is * d_dim1];
		    z__[17] = 0.;
		    z__[18] = -e[js + js * e_dim1];
		    z__[19] = 0.;

		    z__[24] = 0.;
		    z__[25] = d__[is + is * d_dim1];
		    z__[26] = -e[js + jsp1 * e_dim1];
		    z__[27] = -e[jsp1 + jsp1 * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = c__[is + jsp1 * c_dim1];
		    rhs[2] = f[is + js * f_dim1];
		    rhs[3] = f[is + jsp1 * f_dim1];

/*                 Solve Z' * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }
		    dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
		    if (scaloc != 1.) {
			i__3 = *n;
			for (k = 1; k <= i__3; ++k) {
			    dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &c__1);
			    dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L140: */
			}
			*scale *= scaloc;
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    c__[is + jsp1 * c_dim1] = rhs[1];
		    f[is + js * f_dim1] = rhs[2];
		    f[is + jsp1 * f_dim1] = rhs[3];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (j > p + 2) {
			i__3 = js - 1;
			daxpy_(&i__3, rhs, &b[js * b_dim1 + 1], &c__1, &f[is 
				+ f_dim1], ldf);
			i__3 = js - 1;
			daxpy_(&i__3, &rhs[1], &b[jsp1 * b_dim1 + 1], &c__1, &
				f[is + f_dim1], ldf);
			i__3 = js - 1;
			daxpy_(&i__3, &rhs[2], &e[js * e_dim1 + 1], &c__1, &f[
				is + f_dim1], ldf);
			i__3 = js - 1;
			daxpy_(&i__3, &rhs[3], &e[jsp1 * e_dim1 + 1], &c__1, &
				f[is + f_dim1], ldf);
		    }
		    if (i__ < p) {
			i__3 = *m - ie;
			dger_(&i__3, &nb, &c_b27, &a[is + (ie + 1) * a_dim1], 
				lda, rhs, &c__1, &c__[ie + 1 + js * c_dim1], 
				ldc);
			i__3 = *m - ie;
			dger_(&i__3, &nb, &c_b27, &d__[is + (ie + 1) * d_dim1]
, ldd, &rhs[2], &c__1, &c__[ie + 1 + js * 
				c_dim1], ldc);
		    }

		} else if (mb == 2 && nb == 1) {

/*                 Build a 4-by-4 system Z' * x = RHS */

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = a[is + isp1 * a_dim1];
		    z__[2] = -b[js + js * b_dim1];
		    z__[3] = 0.;

		    z__[8] = a[isp1 + is * a_dim1];
		    z__[9] = a[isp1 + isp1 * a_dim1];
		    z__[10] = 0.;
		    z__[11] = -b[js + js * b_dim1];

		    z__[16] = d__[is + is * d_dim1];
		    z__[17] = d__[is + isp1 * d_dim1];
		    z__[18] = -e[js + js * e_dim1];
		    z__[19] = 0.;

		    z__[24] = 0.;
		    z__[25] = d__[isp1 + isp1 * d_dim1];
		    z__[26] = 0.;
		    z__[27] = -e[js + js * e_dim1];

/*                 Set up right hand side(s) */

		    rhs[0] = c__[is + js * c_dim1];
		    rhs[1] = c__[isp1 + js * c_dim1];
		    rhs[2] = f[is + js * f_dim1];
		    rhs[3] = f[isp1 + js * f_dim1];

/*                 Solve Z' * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }

		    dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
		    if (scaloc != 1.) {
			i__3 = *n;
			for (k = 1; k <= i__3; ++k) {
			    dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &c__1);
			    dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L150: */
			}
			*scale *= scaloc;
		    }

/*                 Unpack solution vector(s) */

		    c__[is + js * c_dim1] = rhs[0];
		    c__[isp1 + js * c_dim1] = rhs[1];
		    f[is + js * f_dim1] = rhs[2];
		    f[isp1 + js * f_dim1] = rhs[3];

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (j > p + 2) {
			i__3 = js - 1;
			dger_(&mb, &i__3, &c_b42, rhs, &c__1, &b[js * b_dim1 
				+ 1], &c__1, &f[is + f_dim1], ldf);
			i__3 = js - 1;
			dger_(&mb, &i__3, &c_b42, &rhs[2], &c__1, &e[js * 
				e_dim1 + 1], &c__1, &f[is + f_dim1], ldf);
		    }
		    if (i__ < p) {
			i__3 = *m - ie;
			dgemv_("T", &mb, &i__3, &c_b27, &a[is + (ie + 1) * 
				a_dim1], lda, rhs, &c__1, &c_b42, &c__[ie + 1 
				+ js * c_dim1], &c__1);
			i__3 = *m - ie;
			dgemv_("T", &mb, &i__3, &c_b27, &d__[is + (ie + 1) * 
				d_dim1], ldd, &rhs[2], &c__1, &c_b42, &c__[ie 
				+ 1 + js * c_dim1], &c__1);
		    }

		} else if (mb == 2 && nb == 2) {

/*                 Build an 8-by-8 system Z' * x = RHS */

		    dlaset_("F", &c__8, &c__8, &c_b56, &c_b56, z__, &c__8);

		    z__[0] = a[is + is * a_dim1];
		    z__[1] = a[is + isp1 * a_dim1];
		    z__[4] = -b[js + js * b_dim1];
		    z__[6] = -b[jsp1 + js * b_dim1];

		    z__[8] = a[isp1 + is * a_dim1];
		    z__[9] = a[isp1 + isp1 * a_dim1];
		    z__[13] = -b[js + js * b_dim1];
		    z__[15] = -b[jsp1 + js * b_dim1];

		    z__[18] = a[is + is * a_dim1];
		    z__[19] = a[is + isp1 * a_dim1];
		    z__[20] = -b[js + jsp1 * b_dim1];
		    z__[22] = -b[jsp1 + jsp1 * b_dim1];

		    z__[26] = a[isp1 + is * a_dim1];
		    z__[27] = a[isp1 + isp1 * a_dim1];
		    z__[29] = -b[js + jsp1 * b_dim1];
		    z__[31] = -b[jsp1 + jsp1 * b_dim1];

		    z__[32] = d__[is + is * d_dim1];
		    z__[33] = d__[is + isp1 * d_dim1];
		    z__[36] = -e[js + js * e_dim1];

		    z__[41] = d__[isp1 + isp1 * d_dim1];
		    z__[45] = -e[js + js * e_dim1];

		    z__[50] = d__[is + is * d_dim1];
		    z__[51] = d__[is + isp1 * d_dim1];
		    z__[52] = -e[js + jsp1 * e_dim1];
		    z__[54] = -e[jsp1 + jsp1 * e_dim1];

		    z__[59] = d__[isp1 + isp1 * d_dim1];
		    z__[61] = -e[js + jsp1 * e_dim1];
		    z__[63] = -e[jsp1 + jsp1 * e_dim1];

/*                 Set up right hand side(s) */

		    k = 1;
		    ii = mb * nb + 1;
		    i__3 = nb - 1;
		    for (jj = 0; jj <= i__3; ++jj) {
			dcopy_(&mb, &c__[is + (js + jj) * c_dim1], &c__1, &
				rhs[k - 1], &c__1);
			dcopy_(&mb, &f[is + (js + jj) * f_dim1], &c__1, &rhs[
				ii - 1], &c__1);
			k += mb;
			ii += mb;
/* L160: */
		    }


/*                 Solve Z' * x = RHS */

		    dgetc2_(&zdim, z__, &c__8, ipiv, jpiv, &ierr);
		    if (ierr > 0) {
			*info = ierr;
		    }

		    dgesc2_(&zdim, z__, &c__8, rhs, ipiv, jpiv, &scaloc);
		    if (scaloc != 1.) {
			i__3 = *n;
			for (k = 1; k <= i__3; ++k) {
			    dscal_(m, &scaloc, &c__[k * c_dim1 + 1], &c__1);
			    dscal_(m, &scaloc, &f[k * f_dim1 + 1], &c__1);
/* L170: */
			}
			*scale *= scaloc;
		    }

/*                 Unpack solution vector(s) */

		    k = 1;
		    ii = mb * nb + 1;
		    i__3 = nb - 1;
		    for (jj = 0; jj <= i__3; ++jj) {
			dcopy_(&mb, &rhs[k - 1], &c__1, &c__[is + (js + jj) * 
				c_dim1], &c__1);
			dcopy_(&mb, &rhs[ii - 1], &c__1, &f[is + (js + jj) * 
				f_dim1], &c__1);
			k += mb;
			ii += mb;
/* L180: */
		    }

/*                 Substitute R(I, J) and L(I, J) into remaining */
/*                 equation. */

		    if (j > p + 2) {
			i__3 = js - 1;
			dgemm_("N", "T", &mb, &i__3, &nb, &c_b42, &c__[is + 
				js * c_dim1], ldc, &b[js * b_dim1 + 1], ldb, &
				c_b42, &f[is + f_dim1], ldf);
			i__3 = js - 1;
			dgemm_("N", "T", &mb, &i__3, &nb, &c_b42, &f[is + js *
				 f_dim1], ldf, &e[js * e_dim1 + 1], lde, &
				c_b42, &f[is + f_dim1], ldf);
		    }
		    if (i__ < p) {
			i__3 = *m - ie;
			dgemm_("T", "N", &i__3, &nb, &mb, &c_b27, &a[is + (ie 
				+ 1) * a_dim1], lda, &c__[is + js * c_dim1], 
				ldc, &c_b42, &c__[ie + 1 + js * c_dim1], ldc);
			i__3 = *m - ie;
			dgemm_("T", "N", &i__3, &nb, &mb, &c_b27, &d__[is + (
				ie + 1) * d_dim1], ldd, &f[is + js * f_dim1], 
				ldf, &c_b42, &c__[ie + 1 + js * c_dim1], ldc);
		    }

		}

/* L190: */
	    }
/* L200: */
	}

    }
    return 0;

/*     End of DTGSY2 */

} /* dtgsy2_ */
