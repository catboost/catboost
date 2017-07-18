/* ssptri.f -- translated by f2c (version 20061008).
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

static integer c__1 = 1;
static real c_b11 = -1.f;
static real c_b13 = 0.f;

/* Subroutine */ int ssptri_(char *uplo, integer *n, real *ap, integer *ipiv, 
	real *work, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Local variables */
    real d__;
    integer j, k;
    real t, ak;
    integer kc, kp, kx, kpc, npp;
    real akp1, temp;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    real akkp1;
    extern logical lsame_(char *, char *);
    integer kstep;
    logical upper;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *), sswap_(integer *, real *, integer *, real *, integer *
), sspmv_(char *, integer *, real *, real *, real *, integer *, 
	    real *, real *, integer *), xerbla_(char *, integer *);
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

/*  SSPTRI computes the inverse of a real symmetric indefinite matrix */
/*  A in packed storage using the factorization A = U*D*U**T or */
/*  A = L*D*L**T computed by SSPTRF. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the details of the factorization are stored */
/*          as an upper or lower triangular matrix. */
/*          = 'U':  Upper triangular, form is A = U*D*U**T; */
/*          = 'L':  Lower triangular, form is A = L*D*L**T. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input/output) REAL array, dimension (N*(N+1)/2) */
/*          On entry, the block diagonal matrix D and the multipliers */
/*          used to obtain the factor U or L as computed by SSPTRF, */
/*          stored as a packed triangular matrix. */

/*          On exit, if INFO = 0, the (symmetric) inverse of the original */
/*          matrix, stored as a packed triangular matrix. The j-th column */
/*          of inv(A) is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = inv(A)(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', */
/*             AP(i + (j-1)*(2n-j)/2) = inv(A)(i,j) for j<=i<=n. */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          Details of the interchanges and the block structure of D */
/*          as determined by SSPTRF. */

/*  WORK    (workspace) REAL array, dimension (N) */

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
	xerbla_("SSPTRI", &i__1);
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
	    if (ipiv[*info] > 0 && ap[kp] == 0.f) {
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
	    if (ipiv[*info] > 0 && ap[kp] == 0.f) {
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

	    ap[kc + k - 1] = 1.f / ap[kc + k - 1];

/*           Compute column K of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		scopy_(&i__1, &ap[kc], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		sspmv_(uplo, &i__1, &c_b11, &ap[1], &work[1], &c__1, &c_b13, &
			ap[kc], &c__1);
		i__1 = k - 1;
		ap[kc + k - 1] -= sdot_(&i__1, &work[1], &c__1, &ap[kc], &
			c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = (r__1 = ap[kcnext + k - 1], dabs(r__1));
	    ak = ap[kc + k - 1] / t;
	    akp1 = ap[kcnext + k] / t;
	    akkp1 = ap[kcnext + k - 1] / t;
	    d__ = t * (ak * akp1 - 1.f);
	    ap[kc + k - 1] = akp1 / d__;
	    ap[kcnext + k] = ak / d__;
	    ap[kcnext + k - 1] = -akkp1 / d__;

/*           Compute columns K and K+1 of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		scopy_(&i__1, &ap[kc], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		sspmv_(uplo, &i__1, &c_b11, &ap[1], &work[1], &c__1, &c_b13, &
			ap[kc], &c__1);
		i__1 = k - 1;
		ap[kc + k - 1] -= sdot_(&i__1, &work[1], &c__1, &ap[kc], &
			c__1);
		i__1 = k - 1;
		ap[kcnext + k - 1] -= sdot_(&i__1, &ap[kc], &c__1, &ap[kcnext]
, &c__1);
		i__1 = k - 1;
		scopy_(&i__1, &ap[kcnext], &c__1, &work[1], &c__1);
		i__1 = k - 1;
		sspmv_(uplo, &i__1, &c_b11, &ap[1], &work[1], &c__1, &c_b13, &
			ap[kcnext], &c__1);
		i__1 = k - 1;
		ap[kcnext + k] -= sdot_(&i__1, &work[1], &c__1, &ap[kcnext], &
			c__1);
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
	    sswap_(&i__1, &ap[kc], &c__1, &ap[kpc], &c__1);
	    kx = kpc + kp - 1;
	    i__1 = k - 1;
	    for (j = kp + 1; j <= i__1; ++j) {
		kx = kx + j - 1;
		temp = ap[kc + j - 1];
		ap[kc + j - 1] = ap[kx];
		ap[kx] = temp;
/* L40: */
	    }
	    temp = ap[kc + k - 1];
	    ap[kc + k - 1] = ap[kpc + kp - 1];
	    ap[kpc + kp - 1] = temp;
	    if (kstep == 2) {
		temp = ap[kc + k + k - 1];
		ap[kc + k + k - 1] = ap[kc + k + kp - 1];
		ap[kc + k + kp - 1] = temp;
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

	    ap[kc] = 1.f / ap[kc];

/*           Compute column K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		scopy_(&i__1, &ap[kc + 1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		sspmv_(uplo, &i__1, &c_b11, &ap[kc + *n - k + 1], &work[1], &
			c__1, &c_b13, &ap[kc + 1], &c__1);
		i__1 = *n - k;
		ap[kc] -= sdot_(&i__1, &work[1], &c__1, &ap[kc + 1], &c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block */

/*           Invert the diagonal block. */

	    t = (r__1 = ap[kcnext + 1], dabs(r__1));
	    ak = ap[kcnext] / t;
	    akp1 = ap[kc] / t;
	    akkp1 = ap[kcnext + 1] / t;
	    d__ = t * (ak * akp1 - 1.f);
	    ap[kcnext] = akp1 / d__;
	    ap[kc] = ak / d__;
	    ap[kcnext + 1] = -akkp1 / d__;

/*           Compute columns K-1 and K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		scopy_(&i__1, &ap[kc + 1], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		sspmv_(uplo, &i__1, &c_b11, &ap[kc + (*n - k + 1)], &work[1], 
			&c__1, &c_b13, &ap[kc + 1], &c__1);
		i__1 = *n - k;
		ap[kc] -= sdot_(&i__1, &work[1], &c__1, &ap[kc + 1], &c__1);
		i__1 = *n - k;
		ap[kcnext + 1] -= sdot_(&i__1, &ap[kc + 1], &c__1, &ap[kcnext 
			+ 2], &c__1);
		i__1 = *n - k;
		scopy_(&i__1, &ap[kcnext + 2], &c__1, &work[1], &c__1);
		i__1 = *n - k;
		sspmv_(uplo, &i__1, &c_b11, &ap[kc + (*n - k + 1)], &work[1], 
			&c__1, &c_b13, &ap[kcnext + 2], &c__1);
		i__1 = *n - k;
		ap[kcnext] -= sdot_(&i__1, &work[1], &c__1, &ap[kcnext + 2], &
			c__1);
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
		sswap_(&i__1, &ap[kc + kp - k + 1], &c__1, &ap[kpc + 1], &
			c__1);
	    }
	    kx = kc + kp - k;
	    i__1 = kp - 1;
	    for (j = k + 1; j <= i__1; ++j) {
		kx = kx + *n - j + 1;
		temp = ap[kc + j - k];
		ap[kc + j - k] = ap[kx];
		ap[kx] = temp;
/* L70: */
	    }
	    temp = ap[kc];
	    ap[kc] = ap[kpc];
	    ap[kpc] = temp;
	    if (kstep == 2) {
		temp = ap[kc - *n + k - 1];
		ap[kc - *n + k - 1] = ap[kc - *n + kp - 1];
		ap[kc - *n + kp - 1] = temp;
	    }
	}

	k -= kstep;
	kc = kcnext;
	goto L60;
L80:
	;
    }

    return 0;

/*     End of SSPTRI */

} /* ssptri_ */
