/* slahr2.f -- translated by f2c (version 20061008).
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

static real c_b4 = -1.f;
static real c_b5 = 1.f;
static integer c__1 = 1;
static real c_b38 = 0.f;

/* Subroutine */ int slahr2_(integer *n, integer *k, integer *nb, real *a, 
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__1, i__2, 
	    i__3;
    real r__1;

    /* Local variables */
    integer i__;
    real ei;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *), 
	    sgemm_(char *, char *, integer *, integer *, integer *, real *, 
	    real *, integer *, real *, integer *, real *, real *, integer *), sgemv_(char *, integer *, integer *, real *, 
	    real *, integer *, real *, integer *, real *, real *, integer *), scopy_(integer *, real *, integer *, real *, integer *), 
	    strmm_(char *, char *, char *, char *, integer *, integer *, real 
	    *, real *, integer *, real *, integer *), saxpy_(integer *, real *, real *, integer *, real *, 
	    integer *), strmv_(char *, char *, char *, integer *, real *, 
	    integer *, real *, integer *), slarfg_(
	    integer *, real *, real *, integer *, real *), slacpy_(char *, 
	    integer *, integer *, real *, integer *, real *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAHR2 reduces the first NB columns of A real general n-BY-(n-k+1) */
/*  matrix A so that elements below the k-th subdiagonal are zero. The */
/*  reduction is performed by an orthogonal similarity transformation */
/*  Q' * A * Q. The routine returns the matrices V and T which determine */
/*  Q as a block reflector I - V*T*V', and also the matrix Y = A * V * T. */

/*  This is an auxiliary routine called by SGEHRD. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A. */

/*  K       (input) INTEGER */
/*          The offset for the reduction. Elements below the k-th */
/*          subdiagonal in the first NB columns are reduced to zero. */
/*          K < N. */

/*  NB      (input) INTEGER */
/*          The number of columns to be reduced. */

/*  A       (input/output) REAL array, dimension (LDA,N-K+1) */
/*          On entry, the n-by-(n-k+1) general matrix A. */
/*          On exit, the elements on and above the k-th subdiagonal in */
/*          the first NB columns are overwritten with the corresponding */
/*          elements of the reduced matrix; the elements below the k-th */
/*          subdiagonal, with the array TAU, represent the matrix Q as a */
/*          product of elementary reflectors. The other columns of A are */
/*          unchanged. See Further Details. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  TAU     (output) REAL array, dimension (NB) */
/*          The scalar factors of the elementary reflectors. See Further */
/*          Details. */

/*  T       (output) REAL array, dimension (LDT,NB) */
/*          The upper triangular matrix T. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T.  LDT >= NB. */

/*  Y       (output) REAL array, dimension (LDY,NB) */
/*          The n-by-nb matrix Y. */

/*  LDY     (input) INTEGER */
/*          The leading dimension of the array Y. LDY >= N. */

/*  Further Details */
/*  =============== */

/*  The matrix Q is represented as a product of nb elementary reflectors */

/*     Q = H(1) H(2) . . . H(nb). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a real scalar, and v is a real vector with */
/*  v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in */
/*  A(i+k+1:n,i), and tau in TAU(i). */

/*  The elements of the vectors v together form the (n-k+1)-by-nb matrix */
/*  V which is needed, with T and Y, to apply the transformation to the */
/*  unreduced part of the matrix, using an update of the form: */
/*  A := (I - V*T*V') * (A - Y*V'). */

/*  The contents of A on exit are illustrated by the following example */
/*  with n = 7, k = 3 and nb = 2: */

/*     ( a   a   a   a   a ) */
/*     ( a   a   a   a   a ) */
/*     ( a   a   a   a   a ) */
/*     ( h   h   a   a   a ) */
/*     ( v1  h   a   a   a ) */
/*     ( v1  v2  a   a   a ) */
/*     ( v1  v2  a   a   a ) */

/*  where a denotes an element of the original matrix A, h denotes a */
/*  modified element of the upper Hessenberg matrix H, and vi denotes an */
/*  element of the vector defining H(i). */

/*  This file is a slight modification of LAPACK-3.0's SLAHRD */
/*  incorporating improvements proposed by Quintana-Orti and Van de */
/*  Gejin. Note that the entries of A(1:K,2:NB) differ from those */
/*  returned by the original LAPACK routine. This function is */
/*  not backward compatible with LAPACK3.0. */

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
    --tau;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*n <= 1) {
	return 0;
    }

    i__1 = *nb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ > 1) {

/*           Update A(K+1:N,I) */

/*           Update I-th column of A - Y * V' */

	    i__2 = *n - *k;
	    i__3 = i__ - 1;
	    sgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &y[*k + 1 + y_dim1], 
		    ldy, &a[*k + i__ - 1 + a_dim1], lda, &c_b5, &a[*k + 1 + 
		    i__ * a_dim1], &c__1);

/*           Apply I - V * T' * V' to this column (call it b) from the */
/*           left, using the last column of T as workspace */

/*           Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows) */
/*                    ( V2 )             ( b2 ) */

/*           where V1 is unit lower triangular */

/*           w := V1' * b1 */

	    i__2 = i__ - 1;
	    scopy_(&i__2, &a[*k + 1 + i__ * a_dim1], &c__1, &t[*nb * t_dim1 + 
		    1], &c__1);
	    i__2 = i__ - 1;
	    strmv_("Lower", "Transpose", "UNIT", &i__2, &a[*k + 1 + a_dim1], 
		    lda, &t[*nb * t_dim1 + 1], &c__1);

/*           w := w + V2'*b2 */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("Transpose", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1], 
		    lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b5, &t[*nb * 
		    t_dim1 + 1], &c__1);

/*           w := T'*w */

	    i__2 = i__ - 1;
	    strmv_("Upper", "Transpose", "NON-UNIT", &i__2, &t[t_offset], ldt, 
		     &t[*nb * t_dim1 + 1], &c__1);

/*           b2 := b2 - V2*w */

	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    sgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &a[*k + i__ + a_dim1], 
		     lda, &t[*nb * t_dim1 + 1], &c__1, &c_b5, &a[*k + i__ + 
		    i__ * a_dim1], &c__1);

/*           b1 := b1 - V1*w */

	    i__2 = i__ - 1;
	    strmv_("Lower", "NO TRANSPOSE", "UNIT", &i__2, &a[*k + 1 + a_dim1]
, lda, &t[*nb * t_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    saxpy_(&i__2, &c_b4, &t[*nb * t_dim1 + 1], &c__1, &a[*k + 1 + i__ 
		    * a_dim1], &c__1);

	    a[*k + i__ - 1 + (i__ - 1) * a_dim1] = ei;
	}

/*        Generate the elementary reflector H(I) to annihilate */
/*        A(K+I+1:N,I) */

	i__2 = *n - *k - i__ + 1;
/* Computing MIN */
	i__3 = *k + i__ + 1;
	slarfg_(&i__2, &a[*k + i__ + i__ * a_dim1], &a[min(i__3, *n)+ i__ * 
		a_dim1], &c__1, &tau[i__]);
	ei = a[*k + i__ + i__ * a_dim1];
	a[*k + i__ + i__ * a_dim1] = 1.f;

/*        Compute  Y(K+1:N,I) */

	i__2 = *n - *k;
	i__3 = *n - *k - i__ + 1;
	sgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b5, &a[*k + 1 + (i__ + 1) * 
		a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b38, &y[*
		k + 1 + i__ * y_dim1], &c__1);
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	sgemv_("Transpose", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1], lda, &
		a[*k + i__ + i__ * a_dim1], &c__1, &c_b38, &t[i__ * t_dim1 + 
		1], &c__1);
	i__2 = *n - *k;
	i__3 = i__ - 1;
	sgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &y[*k + 1 + y_dim1], ldy, 
		&t[i__ * t_dim1 + 1], &c__1, &c_b5, &y[*k + 1 + i__ * y_dim1], 
		 &c__1);
	i__2 = *n - *k;
	sscal_(&i__2, &tau[i__], &y[*k + 1 + i__ * y_dim1], &c__1);

/*        Compute T(1:I,I) */

	i__2 = i__ - 1;
	r__1 = -tau[i__];
	sscal_(&i__2, &r__1, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	strmv_("Upper", "No Transpose", "NON-UNIT", &i__2, &t[t_offset], ldt, 
		&t[i__ * t_dim1 + 1], &c__1)
		;
	t[i__ + i__ * t_dim1] = tau[i__];

/* L10: */
    }
    a[*k + *nb + *nb * a_dim1] = ei;

/*     Compute Y(1:K,1:NB) */

    slacpy_("ALL", k, nb, &a[(a_dim1 << 1) + 1], lda, &y[y_offset], ldy);
    strmm_("RIGHT", "Lower", "NO TRANSPOSE", "UNIT", k, nb, &c_b5, &a[*k + 1 
	    + a_dim1], lda, &y[y_offset], ldy);
    if (*n > *k + *nb) {
	i__1 = *n - *k - *nb;
	sgemm_("NO TRANSPOSE", "NO TRANSPOSE", k, nb, &i__1, &c_b5, &a[(*nb + 
		2) * a_dim1 + 1], lda, &a[*k + 1 + *nb + a_dim1], lda, &c_b5, 
		&y[y_offset], ldy);
    }
    strmm_("RIGHT", "Upper", "NO TRANSPOSE", "NON-UNIT", k, nb, &c_b5, &t[
	    t_offset], ldt, &y[y_offset], ldy);

    return 0;

/*     End of SLAHR2 */

} /* slahr2_ */
