/* dlagv2.f -- translated by f2c (version 20061008).
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

static integer c__2 = 2;
static integer c__1 = 1;

/* Subroutine */ int dlagv2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *csl, doublereal *snl, doublereal *csr, doublereal *
	snr)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Local variables */
    doublereal r__, t, h1, h2, h3, wi, qq, rr, wr1, wr2, ulp;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *), dlag2_(
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *);
    doublereal anorm, bnorm, scale1, scale2;
    extern /* Subroutine */ int dlasv2_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *);
    extern doublereal dlapy2_(doublereal *, doublereal *);
    doublereal ascale, bscale;
    extern doublereal dlamch_(char *);
    doublereal safmin;
    extern /* Subroutine */ int dlartg_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAGV2 computes the Generalized Schur factorization of a real 2-by-2 */
/*  matrix pencil (A,B) where B is upper triangular. This routine */
/*  computes orthogonal (rotation) matrices given by CSL, SNL and CSR, */
/*  SNR such that */

/*  1) if the pencil (A,B) has two real eigenvalues (include 0/0 or 1/0 */
/*     types), then */

/*     [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ] */
/*     [  0  a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ] */

/*     [ b11 b12 ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ] */
/*     [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ], */

/*  2) if the pencil (A,B) has a pair of complex conjugate eigenvalues, */
/*     then */

/*     [ a11 a12 ] := [  CSL  SNL ] [ a11 a12 ] [  CSR -SNR ] */
/*     [ a21 a22 ]    [ -SNL  CSL ] [ a21 a22 ] [  SNR  CSR ] */

/*     [ b11  0  ] := [  CSL  SNL ] [ b11 b12 ] [  CSR -SNR ] */
/*     [  0  b22 ]    [ -SNL  CSL ] [  0  b22 ] [  SNR  CSR ] */

/*     where b11 >= b22 > 0. */


/*  Arguments */
/*  ========= */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, 2) */
/*          On entry, the 2 x 2 matrix A. */
/*          On exit, A is overwritten by the ``A-part'' of the */
/*          generalized Schur form. */

/*  LDA     (input) INTEGER */
/*          THe leading dimension of the array A.  LDA >= 2. */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB, 2) */
/*          On entry, the upper triangular 2 x 2 matrix B. */
/*          On exit, B is overwritten by the ``B-part'' of the */
/*          generalized Schur form. */

/*  LDB     (input) INTEGER */
/*          THe leading dimension of the array B.  LDB >= 2. */

/*  ALPHAR  (output) DOUBLE PRECISION array, dimension (2) */
/*  ALPHAI  (output) DOUBLE PRECISION array, dimension (2) */
/*  BETA    (output) DOUBLE PRECISION array, dimension (2) */
/*          (ALPHAR(k)+i*ALPHAI(k))/BETA(k) are the eigenvalues of the */
/*          pencil (A,B), k=1,2, i = sqrt(-1).  Note that BETA(k) may */
/*          be zero. */

/*  CSL     (output) DOUBLE PRECISION */
/*          The cosine of the left rotation matrix. */

/*  SNL     (output) DOUBLE PRECISION */
/*          The sine of the left rotation matrix. */

/*  CSR     (output) DOUBLE PRECISION */
/*          The cosine of the right rotation matrix. */

/*  SNR     (output) DOUBLE PRECISION */
/*          The sine of the right rotation matrix. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
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
    --alphar;
    --alphai;
    --beta;

    /* Function Body */
    safmin = dlamch_("S");
    ulp = dlamch_("P");

/*     Scale A */

/* Computing MAX */
    d__5 = (d__1 = a[a_dim1 + 1], abs(d__1)) + (d__2 = a[a_dim1 + 2], abs(
	    d__2)), d__6 = (d__3 = a[(a_dim1 << 1) + 1], abs(d__3)) + (d__4 = 
	    a[(a_dim1 << 1) + 2], abs(d__4)), d__5 = max(d__5,d__6);
    anorm = max(d__5,safmin);
    ascale = 1. / anorm;
    a[a_dim1 + 1] = ascale * a[a_dim1 + 1];
    a[(a_dim1 << 1) + 1] = ascale * a[(a_dim1 << 1) + 1];
    a[a_dim1 + 2] = ascale * a[a_dim1 + 2];
    a[(a_dim1 << 1) + 2] = ascale * a[(a_dim1 << 1) + 2];

/*     Scale B */

/* Computing MAX */
    d__4 = (d__3 = b[b_dim1 + 1], abs(d__3)), d__5 = (d__1 = b[(b_dim1 << 1) 
	    + 1], abs(d__1)) + (d__2 = b[(b_dim1 << 1) + 2], abs(d__2)), d__4 
	    = max(d__4,d__5);
    bnorm = max(d__4,safmin);
    bscale = 1. / bnorm;
    b[b_dim1 + 1] = bscale * b[b_dim1 + 1];
    b[(b_dim1 << 1) + 1] = bscale * b[(b_dim1 << 1) + 1];
    b[(b_dim1 << 1) + 2] = bscale * b[(b_dim1 << 1) + 2];

/*     Check if A can be deflated */

    if ((d__1 = a[a_dim1 + 2], abs(d__1)) <= ulp) {
	*csl = 1.;
	*snl = 0.;
	*csr = 1.;
	*snr = 0.;
	a[a_dim1 + 2] = 0.;
	b[b_dim1 + 2] = 0.;

/*     Check if B is singular */

    } else if ((d__1 = b[b_dim1 + 1], abs(d__1)) <= ulp) {
	dlartg_(&a[a_dim1 + 1], &a[a_dim1 + 2], csl, snl, &r__);
	*csr = 1.;
	*snr = 0.;
	drot_(&c__2, &a[a_dim1 + 1], lda, &a[a_dim1 + 2], lda, csl, snl);
	drot_(&c__2, &b[b_dim1 + 1], ldb, &b[b_dim1 + 2], ldb, csl, snl);
	a[a_dim1 + 2] = 0.;
	b[b_dim1 + 1] = 0.;
	b[b_dim1 + 2] = 0.;

    } else if ((d__1 = b[(b_dim1 << 1) + 2], abs(d__1)) <= ulp) {
	dlartg_(&a[(a_dim1 << 1) + 2], &a[a_dim1 + 2], csr, snr, &t);
	*snr = -(*snr);
	drot_(&c__2, &a[a_dim1 + 1], &c__1, &a[(a_dim1 << 1) + 1], &c__1, csr, 
		 snr);
	drot_(&c__2, &b[b_dim1 + 1], &c__1, &b[(b_dim1 << 1) + 1], &c__1, csr, 
		 snr);
	*csl = 1.;
	*snl = 0.;
	a[a_dim1 + 2] = 0.;
	b[b_dim1 + 2] = 0.;
	b[(b_dim1 << 1) + 2] = 0.;

    } else {

/*        B is nonsingular, first compute the eigenvalues of (A,B) */

	dlag2_(&a[a_offset], lda, &b[b_offset], ldb, &safmin, &scale1, &
		scale2, &wr1, &wr2, &wi);

	if (wi == 0.) {

/*           two real eigenvalues, compute s*A-w*B */

	    h1 = scale1 * a[a_dim1 + 1] - wr1 * b[b_dim1 + 1];
	    h2 = scale1 * a[(a_dim1 << 1) + 1] - wr1 * b[(b_dim1 << 1) + 1];
	    h3 = scale1 * a[(a_dim1 << 1) + 2] - wr1 * b[(b_dim1 << 1) + 2];

	    rr = dlapy2_(&h1, &h2);
	    d__1 = scale1 * a[a_dim1 + 2];
	    qq = dlapy2_(&d__1, &h3);

	    if (rr > qq) {

/*              find right rotation matrix to zero 1,1 element of */
/*              (sA - wB) */

		dlartg_(&h2, &h1, csr, snr, &t);

	    } else {

/*              find right rotation matrix to zero 2,1 element of */
/*              (sA - wB) */

		d__1 = scale1 * a[a_dim1 + 2];
		dlartg_(&h3, &d__1, csr, snr, &t);

	    }

	    *snr = -(*snr);
	    drot_(&c__2, &a[a_dim1 + 1], &c__1, &a[(a_dim1 << 1) + 1], &c__1, 
		    csr, snr);
	    drot_(&c__2, &b[b_dim1 + 1], &c__1, &b[(b_dim1 << 1) + 1], &c__1, 
		    csr, snr);

/*           compute inf norms of A and B */

/* Computing MAX */
	    d__5 = (d__1 = a[a_dim1 + 1], abs(d__1)) + (d__2 = a[(a_dim1 << 1)
		     + 1], abs(d__2)), d__6 = (d__3 = a[a_dim1 + 2], abs(d__3)
		    ) + (d__4 = a[(a_dim1 << 1) + 2], abs(d__4));
	    h1 = max(d__5,d__6);
/* Computing MAX */
	    d__5 = (d__1 = b[b_dim1 + 1], abs(d__1)) + (d__2 = b[(b_dim1 << 1)
		     + 1], abs(d__2)), d__6 = (d__3 = b[b_dim1 + 2], abs(d__3)
		    ) + (d__4 = b[(b_dim1 << 1) + 2], abs(d__4));
	    h2 = max(d__5,d__6);

	    if (scale1 * h1 >= abs(wr1) * h2) {

/*              find left rotation matrix Q to zero out B(2,1) */

		dlartg_(&b[b_dim1 + 1], &b[b_dim1 + 2], csl, snl, &r__);

	    } else {

/*              find left rotation matrix Q to zero out A(2,1) */

		dlartg_(&a[a_dim1 + 1], &a[a_dim1 + 2], csl, snl, &r__);

	    }

	    drot_(&c__2, &a[a_dim1 + 1], lda, &a[a_dim1 + 2], lda, csl, snl);
	    drot_(&c__2, &b[b_dim1 + 1], ldb, &b[b_dim1 + 2], ldb, csl, snl);

	    a[a_dim1 + 2] = 0.;
	    b[b_dim1 + 2] = 0.;

	} else {

/*           a pair of complex conjugate eigenvalues */
/*           first compute the SVD of the matrix B */

	    dlasv2_(&b[b_dim1 + 1], &b[(b_dim1 << 1) + 1], &b[(b_dim1 << 1) + 
		    2], &r__, &t, snr, csr, snl, csl);

/*           Form (A,B) := Q(A,B)Z' where Q is left rotation matrix and */
/*           Z is right rotation matrix computed from DLASV2 */

	    drot_(&c__2, &a[a_dim1 + 1], lda, &a[a_dim1 + 2], lda, csl, snl);
	    drot_(&c__2, &b[b_dim1 + 1], ldb, &b[b_dim1 + 2], ldb, csl, snl);
	    drot_(&c__2, &a[a_dim1 + 1], &c__1, &a[(a_dim1 << 1) + 1], &c__1, 
		    csr, snr);
	    drot_(&c__2, &b[b_dim1 + 1], &c__1, &b[(b_dim1 << 1) + 1], &c__1, 
		    csr, snr);

	    b[b_dim1 + 2] = 0.;
	    b[(b_dim1 << 1) + 1] = 0.;

	}

    }

/*     Unscaling */

    a[a_dim1 + 1] = anorm * a[a_dim1 + 1];
    a[a_dim1 + 2] = anorm * a[a_dim1 + 2];
    a[(a_dim1 << 1) + 1] = anorm * a[(a_dim1 << 1) + 1];
    a[(a_dim1 << 1) + 2] = anorm * a[(a_dim1 << 1) + 2];
    b[b_dim1 + 1] = bnorm * b[b_dim1 + 1];
    b[b_dim1 + 2] = bnorm * b[b_dim1 + 2];
    b[(b_dim1 << 1) + 1] = bnorm * b[(b_dim1 << 1) + 1];
    b[(b_dim1 << 1) + 2] = bnorm * b[(b_dim1 << 1) + 2];

    if (wi == 0.) {
	alphar[1] = a[a_dim1 + 1];
	alphar[2] = a[(a_dim1 << 1) + 2];
	alphai[1] = 0.;
	alphai[2] = 0.;
	beta[1] = b[b_dim1 + 1];
	beta[2] = b[(b_dim1 << 1) + 2];
    } else {
	alphar[1] = anorm * wr1 / scale1 / bnorm;
	alphai[1] = anorm * wi / scale1 / bnorm;
	alphar[2] = alphar[1];
	alphai[2] = -alphai[1];
	beta[1] = 1.;
	beta[2] = 1.;
    }

    return 0;

/*     End of DLAGV2 */

} /* dlagv2_ */
