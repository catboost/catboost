/* zlabrd.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlabrd_(integer *m, integer *n, integer *nb, 
	doublecomplex *a, integer *lda, doublereal *d__, doublereal *e, 
	doublecomplex *tauq, doublecomplex *taup, doublecomplex *x, integer *
	ldx, doublecomplex *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2, 
	    i__3;
    doublecomplex z__1;

    /* Local variables */
    integer i__;
    doublecomplex alpha;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, doublecomplex *, integer *), 
	    zlarfg_(integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *), zlacgv_(integer *, doublecomplex *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLABRD reduces the first NB rows and columns of a complex general */
/*  m by n matrix A to upper or lower real bidiagonal form by a unitary */
/*  transformation Q' * A * P, and returns the matrices X and Y which */
/*  are needed to apply the transformation to the unreduced part of A. */

/*  If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower */
/*  bidiagonal form. */

/*  This is an auxiliary routine called by ZGEBRD */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows in the matrix A. */

/*  N       (input) INTEGER */
/*          The number of columns in the matrix A. */

/*  NB      (input) INTEGER */
/*          The number of leading rows and columns of A to be reduced. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the m by n general matrix to be reduced. */
/*          On exit, the first NB rows and columns of the matrix are */
/*          overwritten; the rest of the array is unchanged. */
/*          If m >= n, elements on and below the diagonal in the first NB */
/*            columns, with the array TAUQ, represent the unitary */
/*            matrix Q as a product of elementary reflectors; and */
/*            elements above the diagonal in the first NB rows, with the */
/*            array TAUP, represent the unitary matrix P as a product */
/*            of elementary reflectors. */
/*          If m < n, elements below the diagonal in the first NB */
/*            columns, with the array TAUQ, represent the unitary */
/*            matrix Q as a product of elementary reflectors, and */
/*            elements on and above the diagonal in the first NB rows, */
/*            with the array TAUP, represent the unitary matrix P as */
/*            a product of elementary reflectors. */
/*          See Further Details. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  D       (output) DOUBLE PRECISION array, dimension (NB) */
/*          The diagonal elements of the first NB rows and columns of */
/*          the reduced matrix.  D(i) = A(i,i). */

/*  E       (output) DOUBLE PRECISION array, dimension (NB) */
/*          The off-diagonal elements of the first NB rows and columns of */
/*          the reduced matrix. */

/*  TAUQ    (output) COMPLEX*16 array dimension (NB) */
/*          The scalar factors of the elementary reflectors which */
/*          represent the unitary matrix Q. See Further Details. */

/*  TAUP    (output) COMPLEX*16 array, dimension (NB) */
/*          The scalar factors of the elementary reflectors which */
/*          represent the unitary matrix P. See Further Details. */

/*  X       (output) COMPLEX*16 array, dimension (LDX,NB) */
/*          The m-by-nb matrix X required to update the unreduced part */
/*          of A. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X. LDX >= max(1,M). */

/*  Y       (output) COMPLEX*16 array, dimension (LDY,NB) */
/*          The n-by-nb matrix Y required to update the unreduced part */
/*          of A. */

/*  LDY     (input) INTEGER */
/*          The leading dimension of the array Y. LDY >= max(1,N). */

/*  Further Details */
/*  =============== */

/*  The matrices Q and P are represented as products of elementary */
/*  reflectors: */

/*     Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb) */

/*  Each H(i) and G(i) has the form: */

/*     H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u' */

/*  where tauq and taup are complex scalars, and v and u are complex */
/*  vectors. */

/*  If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in */
/*  A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in */
/*  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i). */

/*  If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in */
/*  A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in */
/*  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i). */

/*  The elements of the vectors v and u together form the m-by-nb matrix */
/*  V and the nb-by-n matrix U' which are needed, with X and Y, to apply */
/*  the transformation to the unreduced part of the matrix, using a block */
/*  update of the form:  A := A - V*Y' - X*U'. */

/*  The contents of A on exit are illustrated by the following examples */
/*  with nb = 2: */

/*  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n): */

/*    (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 ) */
/*    (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 ) */
/*    (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  ) */
/*    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  ) */
/*    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  ) */
/*    (  v1  v2  a   a   a  ) */

/*  where a denotes an element of the original matrix which is unchanged, */
/*  vi denotes an element of the vector defining H(i), and ui an element */
/*  of the vector defining G(i). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:m,i) */

	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + a_dim1], lda, 
		     &y[i__ + y_dim1], ldy, &c_b2, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + x_dim1], ldx, 
		     &a[i__ * a_dim1 + 1], &c__1, &c_b2, &a[i__ + i__ * 
		    a_dim1], &c__1);

/*           Generate reflection Q(i) to annihilate A(i+1:m,i) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    zlarfg_(&i__2, &alpha, &a[min(i__3, *m)+ i__ * a_dim1], &c__1, &
		    tauq[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *n) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + (
			i__ + 1) * a_dim1], lda, &a[i__ + i__ * a_dim1], &
			c__1, &c_b1, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + 
			a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b1, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + 1 + 
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b2, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &x[i__ + 
			x_dim1], ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b1, &
			y[i__ * y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[(i__ + 
			1) * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b2, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);

/*              Update A(i,i+1:n) */

		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		zlacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &y[i__ + 1 + 
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b2, &a[i__ + (
			i__ + 1) * a_dim1], lda);
		zlacgv_(&i__, &a[i__ + a_dim1], lda);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &x[i__ + x_dim1], ldx);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[(i__ + 
			1) * a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b2, &
			a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*              Generate reflection P(i) to annihilate A(i,i+2:n) */

		i__2 = i__ + (i__ + 1) * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[i__ + min(i__3, *n)* a_dim1], lda, &
			taup[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + (i__ + 1) * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		zgemv_("No transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 + (i__ 
			+ 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1], 
			lda, &c_b1, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__, &c_b2, &y[i__ + 1 
			+ y_dim1], ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b1, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &a[i__ + 1 + 
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b2, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		zgemv_("No transpose", &i__2, &i__3, &c_b2, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b1, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + 1 + 
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b2, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		zscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i,i:n) */

	    i__2 = *n - i__ + 1;
	    zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + y_dim1], ldy, 
		     &a[i__ + a_dim1], lda, &c_b2, &a[i__ + i__ * a_dim1], 
		    lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &x[i__ + x_dim1], ldx);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("Conjugate transpose", &i__2, &i__3, &z__1, &a[i__ * 
		    a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b2, &a[i__ + 
		    i__ * a_dim1], lda);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &x[i__ + x_dim1], ldx);

/*           Generate reflection P(i) to annihilate A(i,i+1:n) */

	    i__2 = i__ + i__ * a_dim1;
	    alpha.r = a[i__2].r, alpha.i = a[i__2].i;
	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    zlarfg_(&i__2, &alpha, &a[i__ + min(i__3, *n)* a_dim1], lda, &
		    taup[i__]);
	    i__2 = i__;
	    d__[i__2] = alpha.r;
	    if (i__ < *m) {
		i__2 = i__ + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		zgemv_("No transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 + i__ *
			 a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b1, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &y[i__ + 
			y_dim1], ldy, &a[i__ + i__ * a_dim1], lda, &c_b1, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 + 
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b2, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		zgemv_("No transpose", &i__2, &i__3, &c_b2, &a[i__ * a_dim1 + 
			1], lda, &a[i__ + i__ * a_dim1], lda, &c_b1, &x[i__ * 
			x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &x[i__ + 1 + 
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b2, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		zscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);

/*              Update A(i+1:m,i) */

		i__2 = i__ - 1;
		zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 + 
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b2, &a[i__ + 
			1 + i__ * a_dim1], &c__1);
		i__2 = i__ - 1;
		zlacgv_(&i__2, &y[i__ + y_dim1], ldy);
		i__2 = *m - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__, &z__1, &x[i__ + 1 + 
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b2, &a[
			i__ + 1 + i__ * a_dim1], &c__1);

/*              Generate reflection Q(i) to annihilate A(i+2:m,i) */

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[min(i__3, *m)+ i__ * a_dim1], &c__1, 
			 &tauq[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 
			+ (i__ + 1) * a_dim1], lda, &a[i__ + 1 + i__ * a_dim1]
, &c__1, &c_b1, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 
			+ a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b1, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &y[i__ + 1 + 
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b2, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		zgemv_("Conjugate transpose", &i__2, &i__, &c_b2, &x[i__ + 1 
			+ x_dim1], ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b1, &y[i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Conjugate transpose", &i__, &i__2, &z__1, &a[(i__ + 1)
			 * a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &
			c_b2, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    } else {
		i__2 = *n - i__ + 1;
		zlacgv_(&i__2, &a[i__ + i__ * a_dim1], lda);
	    }
/* L20: */
	}
    }
    return 0;

/*     End of ZLABRD */

} /* zlabrd_ */
