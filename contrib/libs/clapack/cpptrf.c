/* cpptrf.f -- translated by f2c (version 20061008).
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
static real c_b16 = -1.f;

/* Subroutine */ int cpptrf_(char *uplo, integer *n, complex *ap, integer *
	info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    real r__1;
    complex q__1, q__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer j, jc, jj;
    real ajj;
    extern /* Subroutine */ int chpr_(char *, integer *, real *, complex *, 
	    integer *, complex *);
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    logical upper;
    extern /* Subroutine */ int ctpsv_(char *, char *, char *, integer *, 
	    complex *, complex *, integer *), csscal_(
	    integer *, real *, complex *, integer *), xerbla_(char *, integer 
	    *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPPTRF computes the Cholesky factorization of a complex Hermitian */
/*  positive definite matrix A stored in packed format. */

/*  The factorization has the form */
/*     A = U**H * U,  if UPLO = 'U', or */
/*     A = L  * L**H,  if UPLO = 'L', */
/*  where U is an upper triangular matrix and L is lower triangular. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input/output) COMPLEX array, dimension (N*(N+1)/2) */
/*          On entry, the upper or lower triangle of the Hermitian matrix */
/*          A, packed columnwise in a linear array.  The j-th column of A */
/*          is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/*          See below for further details. */

/*          On exit, if INFO = 0, the triangular factor U or L from the */
/*          Cholesky factorization A = U**H*U or A = L*L**H, in the same */
/*          storage format as A. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, the leading minor of order i is not */
/*                positive definite, and the factorization could not be */
/*                completed. */

/*  Further Details */
/*  =============== */

/*  The packed storage scheme is illustrated by the following example */
/*  when N = 4, UPLO = 'U': */

/*  Two-dimensional storage of the Hermitian matrix A: */

/*     a11 a12 a13 a14 */
/*         a22 a23 a24 */
/*             a33 a34     (aij = conjg(aji)) */
/*                 a44 */

/*  Packed storage of the upper triangle of A: */

/*  AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ] */

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
	xerbla_("CPPTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (upper) {

/*        Compute the Cholesky factorization A = U'*U. */

	jj = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    jc = jj + 1;
	    jj += j;

/*           Compute elements 1:J-1 of column J. */

	    if (j > 1) {
		i__2 = j - 1;
		ctpsv_("Upper", "Conjugate transpose", "Non-unit", &i__2, &ap[
			1], &ap[jc], &c__1);
	    }

/*           Compute U(J,J) and test for non-positive-definiteness. */

	    i__2 = jj;
	    r__1 = ap[i__2].r;
	    i__3 = j - 1;
	    cdotc_(&q__2, &i__3, &ap[jc], &c__1, &ap[jc], &c__1);
	    q__1.r = r__1 - q__2.r, q__1.i = -q__2.i;
	    ajj = q__1.r;
	    if (ajj <= 0.f) {
		i__2 = jj;
		ap[i__2].r = ajj, ap[i__2].i = 0.f;
		goto L30;
	    }
	    i__2 = jj;
	    r__1 = sqrt(ajj);
	    ap[i__2].r = r__1, ap[i__2].i = 0.f;
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	jj = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness. */

	    i__2 = jj;
	    ajj = ap[i__2].r;
	    if (ajj <= 0.f) {
		i__2 = jj;
		ap[i__2].r = ajj, ap[i__2].i = 0.f;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    i__2 = jj;
	    ap[i__2].r = ajj, ap[i__2].i = 0.f;

/*           Compute elements J+1:N of column J and update the trailing */
/*           submatrix. */

	    if (j < *n) {
		i__2 = *n - j;
		r__1 = 1.f / ajj;
		csscal_(&i__2, &r__1, &ap[jj + 1], &c__1);
		i__2 = *n - j;
		chpr_("Lower", &i__2, &c_b16, &ap[jj + 1], &c__1, &ap[jj + *n 
			- j + 1]);
		jj = jj + *n - j + 1;
	    }
/* L20: */
	}
    }
    goto L40;

L30:
    *info = j;

L40:
    return 0;

/*     End of CPPTRF */

} /* cpptrf_ */
