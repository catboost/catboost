/* chetri.f -- translated by f2c (version 20061008).
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

static complex c_b2 = {0.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int chetri_(char *uplo, integer *n, complex *a, integer *lda, 
	 integer *ipiv, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2;

    /* Builtin functions */
    double c_abs(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    real d__;
    integer j, k;
    real t, ak;
    integer kp;
    real akp1;
    complex temp, akkp1;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int chemv_(char *, integer *, complex *, complex *
, integer *, complex *, integer *, complex *, complex *, integer *
), ccopy_(integer *, complex *, integer *, complex *, 
	    integer *), cswap_(integer *, complex *, integer *, complex *, 
	    integer *);
    integer kstep;
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHETRI computes the inverse of a complex Hermitian indefinite matrix */
/*  A using the factorization A = U*D*U**H or A = L*D*L**H computed by */
/*  CHETRF. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the details of the factorization are stored */
/*          as an upper or lower triangular matrix. */
/*          = 'U':  Upper triangular, form is A = U*D*U**H; */
/*          = 'L':  Lower triangular, form is A = L*D*L**H. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the block diagonal matrix D and the multipliers */
/*          used to obtain the factor U or L as computed by CHETRF. */

/*          On exit, if INFO = 0, the (Hermitian) inverse of the original */
/*          matrix.  If UPLO = 'U', the upper triangular part of the */
/*          inverse is formed and the part of A below the diagonal is not */
/*          referenced; if UPLO = 'L' the lower triangular part of the */
/*          inverse is formed and the part of A above the diagonal is */
/*          not referenced. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          Details of the interchanges and the block structure of D */
/*          as determined by CHETRF. */

/*  WORK    (workspace) COMPLEX array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */
/*          > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its */
/*               inverse could not be computed. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHETRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	for (*info = *n; *info >= 1; --(*info)) {
	    i__1 = *info + *info * a_dim1;
	    if (ipiv[*info] > 0 && (a[i__1].r == 0.f && a[i__1].i == 0.f)) {
		return 0;
	    }
/* L10: */
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    i__2 = *info + *info * a_dim1;
	    if (ipiv[*info] > 0 && (a[i__2].r == 0.f && a[i__2].i == 0.f)) {
		return 0;
	    }
/* L20: */
	}
    }
    *info = 0;

    if (upper) {

/*        Compute inv(A) from the factorization A = U*D*U'. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
L30:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L50;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    i__1 = k + k * a_dim1;
	    i__2 = k + k * a_dim1;
	    r__1 = 1.f / a[i__2].r;
	    a[i__1].r = r__1, a[i__1].i = 0.f;

/*           Compute column K of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		ccopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[a_offset], lda, &work[1], &c__1, 
			 &c_b2, &a[k * a_dim1 + 1], &c__1);
		i__1 = k + k * a_dim1;
		i__2 = k + k * a_dim1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[k * a_dim1 + 1], &
			c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = c_abs(&a[k + (k + 1) * a_dim1]);
	    i__1 = k + k * a_dim1;
	    ak = a[i__1].r / t;
	    i__1 = k + 1 + (k + 1) * a_dim1;
	    akp1 = a[i__1].r / t;
	    i__1 = k + (k + 1) * a_dim1;
	    q__1.r = a[i__1].r / t, q__1.i = a[i__1].i / t;
	    akkp1.r = q__1.r, akkp1.i = q__1.i;
	    d__ = t * (ak * akp1 - 1.f);
	    i__1 = k + k * a_dim1;
	    r__1 = akp1 / d__;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	    i__1 = k + 1 + (k + 1) * a_dim1;
	    r__1 = ak / d__;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	    i__1 = k + (k + 1) * a_dim1;
	    q__2.r = -akkp1.r, q__2.i = -akkp1.i;
	    q__1.r = q__2.r / d__, q__1.i = q__2.i / d__;
	    a[i__1].r = q__1.r, a[i__1].i = q__1.i;

/*           Compute columns K and K+1 of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		ccopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[a_offset], lda, &work[1], &c__1, 
			 &c_b2, &a[k * a_dim1 + 1], &c__1);
		i__1 = k + k * a_dim1;
		i__2 = k + k * a_dim1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[k * a_dim1 + 1], &
			c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = k + (k + 1) * a_dim1;
		i__2 = k + (k + 1) * a_dim1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &a[k * a_dim1 + 1], &c__1, &a[(k + 1) * 
			a_dim1 + 1], &c__1);
		q__1.r = a[i__2].r - q__2.r, q__1.i = a[i__2].i - q__2.i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = k - 1;
		ccopy_(&i__1, &a[(k + 1) * a_dim1 + 1], &c__1, &work[1], &
			c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[a_offset], lda, &work[1], &c__1, 
			 &c_b2, &a[(k + 1) * a_dim1 + 1], &c__1);
		i__1 = k + 1 + (k + 1) * a_dim1;
		i__2 = k + 1 + (k + 1) * a_dim1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[(k + 1) * a_dim1 + 1]
, &c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    }
	    kstep = 2;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the leading */
/*           submatrix A(1:k+1,1:k+1) */

	    i__1 = kp - 1;
	    cswap_(&i__1, &a[k * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 1], &
		    c__1);
	    i__1 = k - 1;
	    for (j = kp + 1; j <= i__1; ++j) {
		r_cnjg(&q__1, &a[j + k * a_dim1]);
		temp.r = q__1.r, temp.i = q__1.i;
		i__2 = j + k * a_dim1;
		r_cnjg(&q__1, &a[kp + j * a_dim1]);
		a[i__2].r = q__1.r, a[i__2].i = q__1.i;
		i__2 = kp + j * a_dim1;
		a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L40: */
	    }
	    i__1 = kp + k * a_dim1;
	    r_cnjg(&q__1, &a[kp + k * a_dim1]);
	    a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    i__1 = k + k * a_dim1;
	    temp.r = a[i__1].r, temp.i = a[i__1].i;
	    i__1 = k + k * a_dim1;
	    i__2 = kp + kp * a_dim1;
	    a[i__1].r = a[i__2].r, a[i__1].i = a[i__2].i;
	    i__1 = kp + kp * a_dim1;
	    a[i__1].r = temp.r, a[i__1].i = temp.i;
	    if (kstep == 2) {
		i__1 = k + (k + 1) * a_dim1;
		temp.r = a[i__1].r, temp.i = a[i__1].i;
		i__1 = k + (k + 1) * a_dim1;
		i__2 = kp + (k + 1) * a_dim1;
		a[i__1].r = a[i__2].r, a[i__1].i = a[i__2].i;
		i__1 = kp + (k + 1) * a_dim1;
		a[i__1].r = temp.r, a[i__1].i = temp.i;
	    }
	}

	k += kstep;
	goto L30;
L50:

	;
    } else {

/*        Compute inv(A) from the factorization A = L*D*L'. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = *n;
L60:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L80;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    i__1 = k + k * a_dim1;
	    i__2 = k + k * a_dim1;
	    r__1 = 1.f / a[i__2].r;
	    a[i__1].r = r__1, a[i__1].i = 0.f;

/*           Compute column K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		ccopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[k + 1 + (k + 1) * a_dim1], lda, 
			&work[1], &c__1, &c_b2, &a[k + 1 + k * a_dim1], &c__1);
		i__1 = k + k * a_dim1;
		i__2 = k + k * a_dim1;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[k + 1 + k * a_dim1], 
			&c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = c_abs(&a[k + (k - 1) * a_dim1]);
	    i__1 = k - 1 + (k - 1) * a_dim1;
	    ak = a[i__1].r / t;
	    i__1 = k + k * a_dim1;
	    akp1 = a[i__1].r / t;
	    i__1 = k + (k - 1) * a_dim1;
	    q__1.r = a[i__1].r / t, q__1.i = a[i__1].i / t;
	    akkp1.r = q__1.r, akkp1.i = q__1.i;
	    d__ = t * (ak * akp1 - 1.f);
	    i__1 = k - 1 + (k - 1) * a_dim1;
	    r__1 = akp1 / d__;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	    i__1 = k + k * a_dim1;
	    r__1 = ak / d__;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	    i__1 = k + (k - 1) * a_dim1;
	    q__2.r = -akkp1.r, q__2.i = -akkp1.i;
	    q__1.r = q__2.r / d__, q__1.i = q__2.i / d__;
	    a[i__1].r = q__1.r, a[i__1].i = q__1.i;

/*           Compute columns K-1 and K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		ccopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[k + 1 + (k + 1) * a_dim1], lda, 
			&work[1], &c__1, &c_b2, &a[k + 1 + k * a_dim1], &c__1);
		i__1 = k + k * a_dim1;
		i__2 = k + k * a_dim1;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[k + 1 + k * a_dim1], 
			&c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = k + (k - 1) * a_dim1;
		i__2 = k + (k - 1) * a_dim1;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &a[k + 1 + k * a_dim1], &c__1, &a[k + 1 
			+ (k - 1) * a_dim1], &c__1);
		q__1.r = a[i__2].r - q__2.r, q__1.i = a[i__2].i - q__2.i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
		i__1 = *n - k;
		ccopy_(&i__1, &a[k + 1 + (k - 1) * a_dim1], &c__1, &work[1], &
			c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chemv_(uplo, &i__1, &q__1, &a[k + 1 + (k + 1) * a_dim1], lda, 
			&work[1], &c__1, &c_b2, &a[k + 1 + (k - 1) * a_dim1], 
			&c__1);
		i__1 = k - 1 + (k - 1) * a_dim1;
		i__2 = k - 1 + (k - 1) * a_dim1;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &a[k + 1 + (k - 1) * 
			a_dim1], &c__1);
		r__1 = q__2.r;
		q__1.r = a[i__2].r - r__1, q__1.i = a[i__2].i;
		a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    }
	    kstep = 2;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the trailing */
/*           submatrix A(k-1:n,k-1:n) */

	    if (kp < *n) {
		i__1 = *n - kp;
		cswap_(&i__1, &a[kp + 1 + k * a_dim1], &c__1, &a[kp + 1 + kp *
			 a_dim1], &c__1);
	    }
	    i__1 = kp - 1;
	    for (j = k + 1; j <= i__1; ++j) {
		r_cnjg(&q__1, &a[j + k * a_dim1]);
		temp.r = q__1.r, temp.i = q__1.i;
		i__2 = j + k * a_dim1;
		r_cnjg(&q__1, &a[kp + j * a_dim1]);
		a[i__2].r = q__1.r, a[i__2].i = q__1.i;
		i__2 = kp + j * a_dim1;
		a[i__2].r = temp.r, a[i__2].i = temp.i;
/* L70: */
	    }
	    i__1 = kp + k * a_dim1;
	    r_cnjg(&q__1, &a[kp + k * a_dim1]);
	    a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    i__1 = k + k * a_dim1;
	    temp.r = a[i__1].r, temp.i = a[i__1].i;
	    i__1 = k + k * a_dim1;
	    i__2 = kp + kp * a_dim1;
	    a[i__1].r = a[i__2].r, a[i__1].i = a[i__2].i;
	    i__1 = kp + kp * a_dim1;
	    a[i__1].r = temp.r, a[i__1].i = temp.i;
	    if (kstep == 2) {
		i__1 = k + (k - 1) * a_dim1;
		temp.r = a[i__1].r, temp.i = a[i__1].i;
		i__1 = k + (k - 1) * a_dim1;
		i__2 = kp + (k - 1) * a_dim1;
		a[i__1].r = a[i__2].r, a[i__1].i = a[i__2].i;
		i__1 = kp + (k - 1) * a_dim1;
		a[i__1].r = temp.r, a[i__1].i = temp.i;
	    }
	}

	k -= kstep;
	goto L60;
L80:
	;
    }

    return 0;

/*     End of CHETRI */

} /* chetri_ */
