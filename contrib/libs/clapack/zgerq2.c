/* zgerq2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zgerq2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublecomplex *tau, doublecomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__, k;
    doublecomplex alpha;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, doublecomplex *), xerbla_(char *, integer *), zlacgv_(integer *, doublecomplex *, integer *), zlarfp_(
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZGERQ2 computes an RQ factorization of a complex m by n matrix A: */
/*  A = R * Q. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the m by n matrix A. */
/*          On exit, if m <= n, the upper triangle of the subarray */
/*          A(1:m,n-m+1:n) contains the m by m upper triangular matrix R; */
/*          if m >= n, the elements on and above the (m-n)-th subdiagonal */
/*          contain the m by n upper trapezoidal matrix R; the remaining */
/*          elements, with the array TAU, represent the unitary matrix */
/*          Q as a product of elementary reflectors (see Further */
/*          Details). */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) COMPLEX*16 array, dimension (min(M,N)) */
/*          The scalar factors of the elementary reflectors (see Further */
/*          Details). */

/*  WORK    (workspace) COMPLEX*16 array, dimension (M) */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The matrix Q is represented as a product of elementary reflectors */

/*     Q = H(1)' H(2)' . . . H(k)', where k = min(m,n). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a complex scalar, and v is a complex vector with */
/*  v(n-k+i+1:n) = 0 and v(n-k+i) = 1; conjg(v(1:n-k+i-1)) is stored on */
/*  exit in A(m-k+i,1:n-k+i-1), and tau in TAU(i). */

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

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGERQ2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    for (i__ = k; i__ >= 1; --i__) {

/*        Generate elementary reflector H(i) to annihilate */
/*        A(m-k+i,1:n-k+i-1) */

	i__1 = *n - k + i__;
	zlacgv_(&i__1, &a[*m - k + i__ + a_dim1], lda);
	i__1 = *m - k + i__ + (*n - k + i__) * a_dim1;
	alpha.r = a[i__1].r, alpha.i = a[i__1].i;
	i__1 = *n - k + i__;
	zlarfp_(&i__1, &alpha, &a[*m - k + i__ + a_dim1], lda, &tau[i__]);

/*        Apply H(i) to A(1:m-k+i-1,1:n-k+i) from the right */

	i__1 = *m - k + i__ + (*n - k + i__) * a_dim1;
	a[i__1].r = 1., a[i__1].i = 0.;
	i__1 = *m - k + i__ - 1;
	i__2 = *n - k + i__;
	zlarf_("Right", &i__1, &i__2, &a[*m - k + i__ + a_dim1], lda, &tau[
		i__], &a[a_offset], lda, &work[1]);
	i__1 = *m - k + i__ + (*n - k + i__) * a_dim1;
	a[i__1].r = alpha.r, a[i__1].i = alpha.i;
	i__1 = *n - k + i__ - 1;
	zlacgv_(&i__1, &a[*m - k + i__ + a_dim1], lda);
/* L10: */
    }
    return 0;

/*     End of ZGERQ2 */

} /* zgerq2_ */
