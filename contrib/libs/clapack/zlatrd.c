/* zlatrd.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlatrd_(char *uplo, integer *n, integer *nb, 
	doublecomplex *a, integer *lda, doublereal *e, doublecomplex *tau, 
	doublecomplex *w, integer *ldw)
{
    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    integer i__, iw;
    doublecomplex alpha;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, doublecomplex *, integer *), 
	    zhemv_(char *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, doublecomplex *, integer *, doublecomplex *, 
	    doublecomplex *, integer *), zaxpy_(integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *), zlarfg_(integer *, doublecomplex *, doublecomplex *, 
	    integer *, doublecomplex *), zlacgv_(integer *, doublecomplex *, 
	    integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLATRD reduces NB rows and columns of a complex Hermitian matrix A to */
/*  Hermitian tridiagonal form by a unitary similarity */
/*  transformation Q' * A * Q, and returns the matrices V and W which are */
/*  needed to apply the transformation to the unreduced part of A. */

/*  If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a */
/*  matrix, of which the upper triangle is supplied; */
/*  if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a */
/*  matrix, of which the lower triangle is supplied. */

/*  This is an auxiliary routine called by ZHETRD. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          Hermitian matrix A is stored: */
/*          = 'U': Upper triangular */
/*          = 'L': Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A. */

/*  NB      (input) INTEGER */
/*          The number of rows and columns to be reduced. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/*          n-by-n upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading n-by-n lower triangular part of A contains the lower */
/*          triangular part of the matrix A, and the strictly upper */
/*          triangular part of A is not referenced. */
/*          On exit: */
/*          if UPLO = 'U', the last NB columns have been reduced to */
/*            tridiagonal form, with the diagonal elements overwriting */
/*            the diagonal elements of A; the elements above the diagonal */
/*            with the array TAU, represent the unitary matrix Q as a */
/*            product of elementary reflectors; */
/*          if UPLO = 'L', the first NB columns have been reduced to */
/*            tridiagonal form, with the diagonal elements overwriting */
/*            the diagonal elements of A; the elements below the diagonal */
/*            with the array TAU, represent the  unitary matrix Q as a */
/*            product of elementary reflectors. */
/*          See Further Details. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  E       (output) DOUBLE PRECISION array, dimension (N-1) */
/*          If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal */
/*          elements of the last NB columns of the reduced matrix; */
/*          if UPLO = 'L', E(1:nb) contains the subdiagonal elements of */
/*          the first NB columns of the reduced matrix. */

/*  TAU     (output) COMPLEX*16 array, dimension (N-1) */
/*          The scalar factors of the elementary reflectors, stored in */
/*          TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'. */
/*          See Further Details. */

/*  W       (output) COMPLEX*16 array, dimension (LDW,NB) */
/*          The n-by-nb matrix W required to update the unreduced part */
/*          of A. */

/*  LDW     (input) INTEGER */
/*          The leading dimension of the array W. LDW >= max(1,N). */

/*  Further Details */
/*  =============== */

/*  If UPLO = 'U', the matrix Q is represented as a product of elementary */
/*  reflectors */

/*     Q = H(n) H(n-1) . . . H(n-nb+1). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a complex scalar, and v is a complex vector with */
/*  v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i), */
/*  and tau in TAU(i-1). */

/*  If UPLO = 'L', the matrix Q is represented as a product of elementary */
/*  reflectors */

/*     Q = H(1) H(2) . . . H(nb). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a complex scalar, and v is a complex vector with */
/*  v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i), */
/*  and tau in TAU(i). */

/*  The elements of the vectors v together form the n-by-nb matrix V */
/*  which is needed, with W, to apply the transformation to the unreduced */
/*  part of the matrix, using a Hermitian rank-2k update of the form: */
/*  A := A - V*W' - W*V'. */

/*  The contents of A on exit are illustrated by the following examples */
/*  with n = 5 and nb = 2: */

/*  if UPLO = 'U':                       if UPLO = 'L': */

/*    (  a   a   a   v4  v5 )              (  d                  ) */
/*    (      a   a   v4  v5 )              (  1   d              ) */
/*    (          a   1   v5 )              (  v1  1   a          ) */
/*    (              d   1  )              (  v1  v2  a   a      ) */
/*    (                  d  )              (  v1  v2  a   a   a  ) */

/*  where d denotes a diagonal element of the reduced matrix, a denotes */
/*  an element of the original matrix that is unchanged, and vi denotes */
/*  an element of the vector defining H(i). */

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

/*     Quick return if possible */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --e;
    --tau;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;

    /* Function Body */
    if (*n <= 0) {
	return 0;
    }

    if (lsame_(uplo, "U")) {

/*        Reduce last NB columns of upper triangle */

	i__1 = *n - *nb + 1;
	for (i__ = *n; i__ >= i__1; --i__) {
	    iw = i__ - *n + *nb;
	    if (i__ < *n) {

/*              Update A(1:i,i) */

		i__2 = i__ + i__ * a_dim1;
		i__3 = i__ + i__ * a_dim1;
		d__1 = a[i__3].r;
		a[i__2].r = d__1, a[i__2].i = 0.;
		i__2 = *n - i__;
		zlacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__, &i__2, &z__1, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, &
			c_b2, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &w[i__ + (iw + 1) * w_dim1], ldw);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__, &i__2, &z__1, &w[(iw + 1) * 
			w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b2, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		zlacgv_(&i__2, &a[i__ + (i__ + 1) * a_dim1], lda);
		i__2 = i__ + i__ * a_dim1;
		i__3 = i__ + i__ * a_dim1;
		d__1 = a[i__3].r;
		a[i__2].r = d__1, a[i__2].i = 0.;
	    }
	    if (i__ > 1) {

/*              Generate elementary reflector H(i) to annihilate */
/*              A(1:i-2,i) */

		i__2 = i__ - 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = i__ - 1;
		zlarfg_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &tau[i__ 
			- 1]);
		i__2 = i__ - 1;
		e[i__2] = alpha.r;
		i__2 = i__ - 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute W(1:i-1,i) */

		i__2 = i__ - 1;
		zhemv_("Upper", &i__2, &c_b2, &a[a_offset], lda, &a[i__ * 
			a_dim1 + 1], &c__1, &c_b1, &w[iw * w_dim1 + 1], &c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &w[(iw 
			    + 1) * w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1], &
			    c__1, &c_b1, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    z__1.r = -1., z__1.i = -0.;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[(i__ + 1) *
			     a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b2, &w[iw * w_dim1 + 1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[(
			    i__ + 1) * a_dim1 + 1], lda, &a[i__ * a_dim1 + 1], 
			     &c__1, &c_b1, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    z__1.r = -1., z__1.i = -0.;
		    zgemv_("No transpose", &i__2, &i__3, &z__1, &w[(iw + 1) * 
			    w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b2, &w[iw * w_dim1 + 1], &c__1);
		}
		i__2 = i__ - 1;
		zscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
		z__3.r = -.5, z__3.i = -0.;
		i__2 = i__ - 1;
		z__2.r = z__3.r * tau[i__2].r - z__3.i * tau[i__2].i, z__2.i =
			 z__3.r * tau[i__2].i + z__3.i * tau[i__2].r;
		i__3 = i__ - 1;
		zdotc_(&z__4, &i__3, &w[iw * w_dim1 + 1], &c__1, &a[i__ * 
			a_dim1 + 1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * 
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = i__ - 1;
		zaxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw * 
			w_dim1 + 1], &c__1);
	    }

/* L10: */
	}
    } else {

/*        Reduce first NB columns of lower triangle */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:n,i) */

	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__ + i__ * a_dim1;
	    d__1 = a[i__3].r;
	    a[i__2].r = d__1, a[i__2].i = 0.;
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + a_dim1], lda, 
		     &w[i__ + w_dim1], ldw, &c_b2, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &w[i__ + w_dim1], ldw);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("No transpose", &i__2, &i__3, &z__1, &w[i__ + w_dim1], ldw, 
		     &a[i__ + a_dim1], lda, &c_b2, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = i__ - 1;
	    zlacgv_(&i__2, &a[i__ + a_dim1], lda);
	    i__2 = i__ + i__ * a_dim1;
	    i__3 = i__ + i__ * a_dim1;
	    d__1 = a[i__3].r;
	    a[i__2].r = d__1, a[i__2].i = 0.;
	    if (i__ < *n) {

/*              Generate elementary reflector H(i) to annihilate */
/*              A(i+2:n,i) */

		i__2 = i__ + 1 + i__ * a_dim1;
		alpha.r = a[i__2].r, alpha.i = a[i__2].i;
		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		zlarfg_(&i__2, &alpha, &a[min(i__3, *n)+ i__ * a_dim1], &c__1, 
			 &tau[i__]);
		i__2 = i__;
		e[i__2] = alpha.r;
		i__2 = i__ + 1 + i__ * a_dim1;
		a[i__2].r = 1., a[i__2].i = 0.;

/*              Compute W(i+1:n,i) */

		i__2 = *n - i__;
		zhemv_("Lower", &i__2, &c_b2, &a[i__ + 1 + (i__ + 1) * a_dim1]
, lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &w[i__ + 1 
			+ w_dim1], ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b1, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &a[i__ + 1 + 
			a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b2, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		zgemv_("Conjugate transpose", &i__2, &i__3, &c_b2, &a[i__ + 1 
			+ a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &
			c_b1, &w[i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("No transpose", &i__2, &i__3, &z__1, &w[i__ + 1 + 
			w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b2, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		zscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
		z__3.r = -.5, z__3.i = -0.;
		i__2 = i__;
		z__2.r = z__3.r * tau[i__2].r - z__3.i * tau[i__2].i, z__2.i =
			 z__3.r * tau[i__2].i + z__3.i * tau[i__2].r;
		i__3 = *n - i__;
		zdotc_(&z__4, &i__3, &w[i__ + 1 + i__ * w_dim1], &c__1, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * 
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = *n - i__;
		zaxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
	    }

/* L20: */
	}
    }

    return 0;

/*     End of ZLATRD */

} /* zlatrd_ */
