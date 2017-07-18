/* zlatdf.f -- translated by f2c (version 20061008).
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
static integer c__1 = 1;
static integer c_n1 = -1;
static doublereal c_b24 = 1.;

/* Subroutine */ int zlatdf_(integer *ijob, integer *n, doublecomplex *z__, 
	integer *ldz, doublecomplex *rhs, doublereal *rdsum, doublereal *
	rdscal, integer *ipiv, integer *jpiv)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void z_div(doublecomplex *, doublecomplex *, doublecomplex *);
    double z_abs(doublecomplex *);
    void z_sqrt(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j, k;
    doublecomplex bm, bp, xm[2], xp[2];
    integer info;
    doublecomplex temp, work[8];
    doublereal scale;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *);
    doublecomplex pmone;
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    doublereal rtemp, sminu, rwork[2];
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *);
    doublereal splus;
    extern /* Subroutine */ int zaxpy_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), zgesc2_(
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *, 
	     integer *, doublereal *), zgecon_(char *, integer *, 
	    doublecomplex *, integer *, doublereal *, doublereal *, 
	    doublecomplex *, doublereal *, integer *);
    extern doublereal dzasum_(integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zlassq_(integer *, doublecomplex *, integer *, 
	     doublereal *, doublereal *), zlaswp_(integer *, doublecomplex *, 
	    integer *, integer *, integer *, integer *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLATDF computes the contribution to the reciprocal Dif-estimate */
/*  by solving for x in Z * x = b, where b is chosen such that the norm */
/*  of x is as large as possible. It is assumed that LU decomposition */
/*  of Z has been computed by ZGETC2. On entry RHS = f holds the */
/*  contribution from earlier solved sub-systems, and on return RHS = x. */

/*  The factorization of Z returned by ZGETC2 has the form */
/*  Z = P * L * U * Q, where P and Q are permutation matrices. L is lower */
/*  triangular with unit diagonal elements and U is upper triangular. */

/*  Arguments */
/*  ========= */

/*  IJOB    (input) INTEGER */
/*          IJOB = 2: First compute an approximative null-vector e */
/*              of Z using ZGECON, e is normalized and solve for */
/*              Zx = +-e - f with the sign giving the greater value of */
/*              2-norm(x).  About 5 times as expensive as Default. */
/*          IJOB .ne. 2: Local look ahead strategy where */
/*              all entries of the r.h.s. b is choosen as either +1 or */
/*              -1.  Default. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix Z. */

/*  Z       (input) DOUBLE PRECISION array, dimension (LDZ, N) */
/*          On entry, the LU part of the factorization of the n-by-n */
/*          matrix Z computed by ZGETC2:  Z = P * L * U * Q */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDA >= max(1, N). */

/*  RHS     (input/output) DOUBLE PRECISION array, dimension (N). */
/*          On entry, RHS contains contributions from other subsystems. */
/*          On exit, RHS contains the solution of the subsystem with */
/*          entries according to the value of IJOB (see above). */

/*  RDSUM   (input/output) DOUBLE PRECISION */
/*          On entry, the sum of squares of computed contributions to */
/*          the Dif-estimate under computation by ZTGSYL, where the */
/*          scaling factor RDSCAL (see below) has been factored out. */
/*          On exit, the corresponding sum of squares updated with the */
/*          contributions from the current sub-system. */
/*          If TRANS = 'T' RDSUM is not touched. */
/*          NOTE: RDSUM only makes sense when ZTGSY2 is called by CTGSYL. */

/*  RDSCAL  (input/output) DOUBLE PRECISION */
/*          On entry, scaling factor used to prevent overflow in RDSUM. */
/*          On exit, RDSCAL is updated w.r.t. the current contributions */
/*          in RDSUM. */
/*          If TRANS = 'T', RDSCAL is not touched. */
/*          NOTE: RDSCAL only makes sense when ZTGSY2 is called by */
/*          ZTGSYL. */

/*  IPIV    (input) INTEGER array, dimension (N). */
/*          The pivot indices; for 1 <= i <= N, row i of the */
/*          matrix has been interchanged with row IPIV(i). */

/*  JPIV    (input) INTEGER array, dimension (N). */
/*          The pivot indices; for 1 <= j <= N, column j of the */
/*          matrix has been interchanged with column JPIV(j). */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  This routine is a further developed implementation of algorithm */
/*  BSOLVE in [1] using complete pivoting in the LU factorization. */

/*   [1]   Bo Kagstrom and Lars Westin, */
/*         Generalized Schur Methods with Condition Estimators for */
/*         Solving the Generalized Sylvester Equation, IEEE Transactions */
/*         on Automatic Control, Vol. 34, No. 7, July 1989, pp 745-751. */

/*   [2]   Peter Poromaa, */
/*         On Efficient and Robust Estimators for the Separation */
/*         between two Regular Matrix Pairs with Applications in */
/*         Condition Estimation. Report UMINF-95.05, Department of */
/*         Computing Science, Umea University, S-901 87 Umea, Sweden, */
/*         1995. */

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

    /* Parameter adjustments */
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --rhs;
    --ipiv;
    --jpiv;

    /* Function Body */
    if (*ijob != 2) {

/*        Apply permutations IPIV to RHS */

	i__1 = *n - 1;
	zlaswp_(&c__1, &rhs[1], ldz, &c__1, &i__1, &ipiv[1], &c__1);

/*        Solve for L-part choosing RHS either to +1 or -1. */

	z__1.r = -1., z__1.i = -0.;
	pmone.r = z__1.r, pmone.i = z__1.i;
	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    z__1.r = rhs[i__2].r + 1., z__1.i = rhs[i__2].i + 0.;
	    bp.r = z__1.r, bp.i = z__1.i;
	    i__2 = j;
	    z__1.r = rhs[i__2].r - 1., z__1.i = rhs[i__2].i - 0.;
	    bm.r = z__1.r, bm.i = z__1.i;
	    splus = 1.;

/*           Lockahead for L- part RHS(1:N-1) = +-1 */
/*           SPLUS and SMIN computed more efficiently than in BSOLVE[1]. */

	    i__2 = *n - j;
	    zdotc_(&z__1, &i__2, &z__[j + 1 + j * z_dim1], &c__1, &z__[j + 1 
		    + j * z_dim1], &c__1);
	    splus += z__1.r;
	    i__2 = *n - j;
	    zdotc_(&z__1, &i__2, &z__[j + 1 + j * z_dim1], &c__1, &rhs[j + 1], 
		     &c__1);
	    sminu = z__1.r;
	    i__2 = j;
	    splus *= rhs[i__2].r;
	    if (splus > sminu) {
		i__2 = j;
		rhs[i__2].r = bp.r, rhs[i__2].i = bp.i;
	    } else if (sminu > splus) {
		i__2 = j;
		rhs[i__2].r = bm.r, rhs[i__2].i = bm.i;
	    } else {

/*              In this case the updating sums are equal and we can */
/*              choose RHS(J) +1 or -1. The first time this happens we */
/*              choose -1, thereafter +1. This is a simple way to get */
/*              good estimates of matrices like Byers well-known example */
/*              (see [1]). (Not done in BSOLVE.) */

		i__2 = j;
		i__3 = j;
		z__1.r = rhs[i__3].r + pmone.r, z__1.i = rhs[i__3].i + 
			pmone.i;
		rhs[i__2].r = z__1.r, rhs[i__2].i = z__1.i;
		pmone.r = 1., pmone.i = 0.;
	    }

/*           Compute the remaining r.h.s. */

	    i__2 = j;
	    z__1.r = -rhs[i__2].r, z__1.i = -rhs[i__2].i;
	    temp.r = z__1.r, temp.i = z__1.i;
	    i__2 = *n - j;
	    zaxpy_(&i__2, &temp, &z__[j + 1 + j * z_dim1], &c__1, &rhs[j + 1], 
		     &c__1);
/* L10: */
	}

/*        Solve for U- part, lockahead for RHS(N) = +-1. This is not done */
/*        In BSOLVE and will hopefully give us a better estimate because */
/*        any ill-conditioning of the original matrix is transfered to U */
/*        and not to L. U(N, N) is an approximation to sigma_min(LU). */

	i__1 = *n - 1;
	zcopy_(&i__1, &rhs[1], &c__1, work, &c__1);
	i__1 = *n - 1;
	i__2 = *n;
	z__1.r = rhs[i__2].r + 1., z__1.i = rhs[i__2].i + 0.;
	work[i__1].r = z__1.r, work[i__1].i = z__1.i;
	i__1 = *n;
	i__2 = *n;
	z__1.r = rhs[i__2].r - 1., z__1.i = rhs[i__2].i - 0.;
	rhs[i__1].r = z__1.r, rhs[i__1].i = z__1.i;
	splus = 0.;
	sminu = 0.;
	for (i__ = *n; i__ >= 1; --i__) {
	    z_div(&z__1, &c_b1, &z__[i__ + i__ * z_dim1]);
	    temp.r = z__1.r, temp.i = z__1.i;
	    i__1 = i__ - 1;
	    i__2 = i__ - 1;
	    z__1.r = work[i__2].r * temp.r - work[i__2].i * temp.i, z__1.i = 
		    work[i__2].r * temp.i + work[i__2].i * temp.r;
	    work[i__1].r = z__1.r, work[i__1].i = z__1.i;
	    i__1 = i__;
	    i__2 = i__;
	    z__1.r = rhs[i__2].r * temp.r - rhs[i__2].i * temp.i, z__1.i = 
		    rhs[i__2].r * temp.i + rhs[i__2].i * temp.r;
	    rhs[i__1].r = z__1.r, rhs[i__1].i = z__1.i;
	    i__1 = *n;
	    for (k = i__ + 1; k <= i__1; ++k) {
		i__2 = i__ - 1;
		i__3 = i__ - 1;
		i__4 = k - 1;
		i__5 = i__ + k * z_dim1;
		z__3.r = z__[i__5].r * temp.r - z__[i__5].i * temp.i, z__3.i =
			 z__[i__5].r * temp.i + z__[i__5].i * temp.r;
		z__2.r = work[i__4].r * z__3.r - work[i__4].i * z__3.i, 
			z__2.i = work[i__4].r * z__3.i + work[i__4].i * 
			z__3.r;
		z__1.r = work[i__3].r - z__2.r, z__1.i = work[i__3].i - 
			z__2.i;
		work[i__2].r = z__1.r, work[i__2].i = z__1.i;
		i__2 = i__;
		i__3 = i__;
		i__4 = k;
		i__5 = i__ + k * z_dim1;
		z__3.r = z__[i__5].r * temp.r - z__[i__5].i * temp.i, z__3.i =
			 z__[i__5].r * temp.i + z__[i__5].i * temp.r;
		z__2.r = rhs[i__4].r * z__3.r - rhs[i__4].i * z__3.i, z__2.i =
			 rhs[i__4].r * z__3.i + rhs[i__4].i * z__3.r;
		z__1.r = rhs[i__3].r - z__2.r, z__1.i = rhs[i__3].i - z__2.i;
		rhs[i__2].r = z__1.r, rhs[i__2].i = z__1.i;
/* L20: */
	    }
	    splus += z_abs(&work[i__ - 1]);
	    sminu += z_abs(&rhs[i__]);
/* L30: */
	}
	if (splus > sminu) {
	    zcopy_(n, work, &c__1, &rhs[1], &c__1);
	}

/*        Apply the permutations JPIV to the computed solution (RHS) */

	i__1 = *n - 1;
	zlaswp_(&c__1, &rhs[1], ldz, &c__1, &i__1, &jpiv[1], &c_n1);

/*        Compute the sum of squares */

	zlassq_(n, &rhs[1], &c__1, rdscal, rdsum);
	return 0;
    }

/*     ENTRY IJOB = 2 */

/*     Compute approximate nullvector XM of Z */

    zgecon_("I", n, &z__[z_offset], ldz, &c_b24, &rtemp, work, rwork, &info);
    zcopy_(n, &work[*n], &c__1, xm, &c__1);

/*     Compute RHS */

    i__1 = *n - 1;
    zlaswp_(&c__1, xm, ldz, &c__1, &i__1, &ipiv[1], &c_n1);
    zdotc_(&z__3, n, xm, &c__1, xm, &c__1);
    z_sqrt(&z__2, &z__3);
    z_div(&z__1, &c_b1, &z__2);
    temp.r = z__1.r, temp.i = z__1.i;
    zscal_(n, &temp, xm, &c__1);
    zcopy_(n, xm, &c__1, xp, &c__1);
    zaxpy_(n, &c_b1, &rhs[1], &c__1, xp, &c__1);
    z__1.r = -1., z__1.i = -0.;
    zaxpy_(n, &z__1, xm, &c__1, &rhs[1], &c__1);
    zgesc2_(n, &z__[z_offset], ldz, &rhs[1], &ipiv[1], &jpiv[1], &scale);
    zgesc2_(n, &z__[z_offset], ldz, xp, &ipiv[1], &jpiv[1], &scale);
    if (dzasum_(n, xp, &c__1) > dzasum_(n, &rhs[1], &c__1)) {
	zcopy_(n, xp, &c__1, &rhs[1], &c__1);
    }

/*     Compute the sum of squares */

    zlassq_(n, &rhs[1], &c__1, rdscal, rdsum);
    return 0;

/*     End of ZLATDF */

} /* zlatdf_ */
