/* chptri.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int chptri_(char *uplo, integer *n, complex *ap, integer *
	ipiv, complex *work, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2;

    /* Builtin functions */
    double c_abs(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    real d__;
    integer j, k;
    real t, ak;
    integer kc, kp, kx, kpc, npp;
    real akp1;
    complex temp, akkp1;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), chpmv_(char *, integer *, complex *, 
	    complex *, complex *, integer *, complex *, complex *, integer *), cswap_(integer *, complex *, integer *, complex *, 
	    integer *);
    integer kstep;
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    integer kcnext;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHPTRI computes the inverse of a complex Hermitian indefinite matrix */
/*  A in packed storage using the factorization A = U*D*U**H or */
/*  A = L*D*L**H computed by CHPTRF. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the details of the factorization are stored */
/*          as an upper or lower triangular matrix. */
/*          = 'U':  Upper triangular, form is A = U*D*U**H; */
/*          = 'L':  Lower triangular, form is A = L*D*L**H. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input/output) COMPLEX array, dimension (N*(N+1)/2) */
/*          On entry, the block diagonal matrix D and the multipliers */
/*          used to obtain the factor U or L as computed by CHPTRF, */
/*          stored as a packed triangular matrix. */

/*          On exit, if INFO = 0, the (Hermitian) inverse of the original */
/*          matrix, stored as a packed triangular matrix. The j-th column */
/*          of inv(A) is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = inv(A)(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', */
/*             AP(i + (j-1)*(2n-j)/2) = inv(A)(i,j) for j<=i<=n. */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          Details of the interchanges and the block structure of D */
/*          as determined by CHPTRF. */

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
    --work;
    --ipiv;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHPTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	kp = *n * (*n + 1) / 2;
	for (*info = *n; *info >= 1; --(*info)) {
	    i__1 = kp;
	    if (ipiv[*info] > 0 && (ap[i__1].r == 0.f && ap[i__1].i == 0.f)) {
		return 0;
	    }
	    kp -= *info;
/* L10: */
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	kp = 1;
	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    i__2 = kp;
	    if (ipiv[*info] > 0 && (ap[i__2].r == 0.f && ap[i__2].i == 0.f)) {
		return 0;
	    }
	    kp = kp + *n - *info + 1;
/* L20: */
	}
    }
    *info = 0;

    if (upper) {

/*        Compute inv(A) from the factorization A = U*D*U'. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
	kc = 1;
L30:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L50;
	}

	kcnext = kc + k;
	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    i__1 = kc + k - 1;
	    i__2 = kc + k - 1;
	    r__1 = 1.f / ap[i__2].r;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;

/*           Compute column K of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		ccopy_(&i__1, &ap[kc], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[1], &work[1], &c__1, &c_b2, &
			ap[kc], &c__1);
		i__1 = kc + k - 1;
		i__2 = kc + k - 1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kc], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = c_abs(&ap[kcnext + k - 1]);
	    i__1 = kc + k - 1;
	    ak = ap[i__1].r / t;
	    i__1 = kcnext + k;
	    akp1 = ap[i__1].r / t;
	    i__1 = kcnext + k - 1;
	    q__1.r = ap[i__1].r / t, q__1.i = ap[i__1].i / t;
	    akkp1.r = q__1.r, akkp1.i = q__1.i;
	    d__ = t * (ak * akp1 - 1.f);
	    i__1 = kc + k - 1;
	    r__1 = akp1 / d__;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;
	    i__1 = kcnext + k;
	    r__1 = ak / d__;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;
	    i__1 = kcnext + k - 1;
	    q__2.r = -akkp1.r, q__2.i = -akkp1.i;
	    q__1.r = q__2.r / d__, q__1.i = q__2.i / d__;
	    ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;

/*           Compute columns K and K+1 of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		ccopy_(&i__1, &ap[kc], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[1], &work[1], &c__1, &c_b2, &
			ap[kc], &c__1);
		i__1 = kc + k - 1;
		i__2 = kc + k - 1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kc], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
		i__1 = kcnext + k - 1;
		i__2 = kcnext + k - 1;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &ap[kc], &c__1, &ap[kcnext], &c__1);
		q__1.r = ap[i__2].r - q__2.r, q__1.i = ap[i__2].i - q__2.i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
		i__1 = k - 1;
		ccopy_(&i__1, &ap[kcnext], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[1], &work[1], &c__1, &c_b2, &
			ap[kcnext], &c__1);
		i__1 = kcnext + k;
		i__2 = kcnext + k;
		i__3 = k - 1;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kcnext], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    }
	    kstep = 2;
	    kcnext = kcnext + k + 1;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the leading */
/*           submatrix A(1:k+1,1:k+1) */

	    kpc = (kp - 1) * kp / 2 + 1;
	    i__1 = kp - 1;
	    cswap_(&i__1, &ap[kc], &c__1, &ap[kpc], &c__1);
	    kx = kpc + kp - 1;
	    i__1 = k - 1;
	    for (j = kp + 1; j <= i__1; ++j) {
		kx = kx + j - 1;
		r_cnjg(&q__1, &ap[kc + j - 1]);
		temp.r = q__1.r, temp.i = q__1.i;
		i__2 = kc + j - 1;
		r_cnjg(&q__1, &ap[kx]);
		ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		i__2 = kx;
		ap[i__2].r = temp.r, ap[i__2].i = temp.i;
/* L40: */
	    }
	    i__1 = kc + kp - 1;
	    r_cnjg(&q__1, &ap[kc + kp - 1]);
	    ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    i__1 = kc + k - 1;
	    temp.r = ap[i__1].r, temp.i = ap[i__1].i;
	    i__1 = kc + k - 1;
	    i__2 = kpc + kp - 1;
	    ap[i__1].r = ap[i__2].r, ap[i__1].i = ap[i__2].i;
	    i__1 = kpc + kp - 1;
	    ap[i__1].r = temp.r, ap[i__1].i = temp.i;
	    if (kstep == 2) {
		i__1 = kc + k + k - 1;
		temp.r = ap[i__1].r, temp.i = ap[i__1].i;
		i__1 = kc + k + k - 1;
		i__2 = kc + k + kp - 1;
		ap[i__1].r = ap[i__2].r, ap[i__1].i = ap[i__2].i;
		i__1 = kc + k + kp - 1;
		ap[i__1].r = temp.r, ap[i__1].i = temp.i;
	    }
	}

	k += kstep;
	kc = kcnext;
	goto L30;
L50:

	;
    } else {

/*        Compute inv(A) from the factorization A = L*D*L'. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	npp = *n * (*n + 1) / 2;
	k = *n;
	kc = npp;
L60:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L80;
	}

	kcnext = kc - (*n - k + 2);
	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Invert the diagonal block. */

	    i__1 = kc;
	    i__2 = kc;
	    r__1 = 1.f / ap[i__2].r;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;

/*           Compute column K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		ccopy_(&i__1, &ap[kc + 1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[kc + *n - k + 1], &work[1], &
			c__1, &c_b2, &ap[kc + 1], &c__1);
		i__1 = kc;
		i__2 = kc;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kc + 1], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = c_abs(&ap[kcnext + 1]);
	    i__1 = kcnext;
	    ak = ap[i__1].r / t;
	    i__1 = kc;
	    akp1 = ap[i__1].r / t;
	    i__1 = kcnext + 1;
	    q__1.r = ap[i__1].r / t, q__1.i = ap[i__1].i / t;
	    akkp1.r = q__1.r, akkp1.i = q__1.i;
	    d__ = t * (ak * akp1 - 1.f);
	    i__1 = kcnext;
	    r__1 = akp1 / d__;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;
	    i__1 = kc;
	    r__1 = ak / d__;
	    ap[i__1].r = r__1, ap[i__1].i = 0.f;
	    i__1 = kcnext + 1;
	    q__2.r = -akkp1.r, q__2.i = -akkp1.i;
	    q__1.r = q__2.r / d__, q__1.i = q__2.i / d__;
	    ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;

/*           Compute columns K-1 and K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		ccopy_(&i__1, &ap[kc + 1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[kc + (*n - k + 1)], &work[1], &
			c__1, &c_b2, &ap[kc + 1], &c__1);
		i__1 = kc;
		i__2 = kc;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kc + 1], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
		i__1 = kcnext + 1;
		i__2 = kcnext + 1;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &ap[kc + 1], &c__1, &ap[kcnext + 2], &
			c__1);
		q__1.r = ap[i__2].r - q__2.r, q__1.i = ap[i__2].i - q__2.i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
		i__1 = *n - k;
		ccopy_(&i__1, &ap[kcnext + 2], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		chpmv_(uplo, &i__1, &q__1, &ap[kc + (*n - k + 1)], &work[1], &
			c__1, &c_b2, &ap[kcnext + 2], &c__1);
		i__1 = kcnext;
		i__2 = kcnext;
		i__3 = *n - k;
		cdotc_(&q__2, &i__3, &work[1], &c__1, &ap[kcnext + 2], &c__1);
		r__1 = q__2.r;
		q__1.r = ap[i__2].r - r__1, q__1.i = ap[i__2].i;
		ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    }
	    kstep = 2;
	    kcnext -= *n - k + 3;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the trailing */
/*           submatrix A(k-1:n,k-1:n) */

	    kpc = npp - (*n - kp + 1) * (*n - kp + 2) / 2 + 1;
	    if (kp < *n) {
		i__1 = *n - kp;
		cswap_(&i__1, &ap[kc + kp - k + 1], &c__1, &ap[kpc + 1], &
			c__1);
	    }
	    kx = kc + kp - k;
	    i__1 = kp - 1;
	    for (j = k + 1; j <= i__1; ++j) {
		kx = kx + *n - j + 1;
		r_cnjg(&q__1, &ap[kc + j - k]);
		temp.r = q__1.r, temp.i = q__1.i;
		i__2 = kc + j - k;
		r_cnjg(&q__1, &ap[kx]);
		ap[i__2].r = q__1.r, ap[i__2].i = q__1.i;
		i__2 = kx;
		ap[i__2].r = temp.r, ap[i__2].i = temp.i;
/* L70: */
	    }
	    i__1 = kc + kp - k;
	    r_cnjg(&q__1, &ap[kc + kp - k]);
	    ap[i__1].r = q__1.r, ap[i__1].i = q__1.i;
	    i__1 = kc;
	    temp.r = ap[i__1].r, temp.i = ap[i__1].i;
	    i__1 = kc;
	    i__2 = kpc;
	    ap[i__1].r = ap[i__2].r, ap[i__1].i = ap[i__2].i;
	    i__1 = kpc;
	    ap[i__1].r = temp.r, ap[i__1].i = temp.i;
	    if (kstep == 2) {
		i__1 = kc - *n + k - 1;
		temp.r = ap[i__1].r, temp.i = ap[i__1].i;
		i__1 = kc - *n + k - 1;
		i__2 = kc - *n + kp - 1;
		ap[i__1].r = ap[i__2].r, ap[i__1].i = ap[i__2].i;
		i__1 = kc - *n + kp - 1;
		ap[i__1].r = temp.r, ap[i__1].i = temp.i;
	    }
	}

	k -= kstep;
	kc = kcnext;
	goto L60;
L80:
	;
    }

    return 0;

/*     End of CHPTRI */

} /* chptri_ */
