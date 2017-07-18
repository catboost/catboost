/* dpptrf.f -- translated by f2c (version 20061008).
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
static doublereal c_b16 = -1.;

/* Subroutine */ int dpptrf_(char *uplo, integer *n, doublereal *ap, integer *
	info)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer j, jc, jj;
    doublereal ajj;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    extern /* Subroutine */ int dspr_(char *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *), dscal_(integer *, 
	    doublereal *, doublereal *, integer *);
    extern logical lsame_(char *, char *);
    logical upper;
    extern /* Subroutine */ int dtpsv_(char *, char *, char *, integer *, 
	    doublereal *, doublereal *, integer *), 
	    xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DPPTRF computes the Cholesky factorization of a real symmetric */
/*  positive definite matrix A stored in packed format. */

/*  The factorization has the form */
/*     A = U**T * U,  if UPLO = 'U', or */
/*     A = L  * L**T,  if UPLO = 'L', */
/*  where U is an upper triangular matrix and L is lower triangular. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2) */
/*          On entry, the upper or lower triangle of the symmetric matrix */
/*          A, packed columnwise in a linear array.  The j-th column of A */
/*          is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/*          See below for further details. */

/*          On exit, if INFO = 0, the triangular factor U or L from the */
/*          Cholesky factorization A = U**T*U or A = L*L**T, in the same */
/*          storage format as A. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, the leading minor of order i is not */
/*                positive definite, and the factorization could not be */
/*                completed. */

/*  Further Details */
/*  ======= ======= */

/*  The packed storage scheme is illustrated by the following example */
/*  when N = 4, UPLO = 'U': */

/*  Two-dimensional storage of the symmetric matrix A: */

/*     a11 a12 a13 a14 */
/*         a22 a23 a24 */
/*             a33 a34     (aij = aji) */
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
	xerbla_("DPPTRF", &i__1);
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
		dtpsv_("Upper", "Transpose", "Non-unit", &i__2, &ap[1], &ap[
			jc], &c__1);
	    }

/*           Compute U(J,J) and test for non-positive-definiteness. */

	    i__2 = j - 1;
	    ajj = ap[jj] - ddot_(&i__2, &ap[jc], &c__1, &ap[jc], &c__1);
	    if (ajj <= 0.) {
		ap[jj] = ajj;
		goto L30;
	    }
	    ap[jj] = sqrt(ajj);
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	jj = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness. */

	    ajj = ap[jj];
	    if (ajj <= 0.) {
		ap[jj] = ajj;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    ap[jj] = ajj;

/*           Compute elements J+1:N of column J and update the trailing */
/*           submatrix. */

	    if (j < *n) {
		i__2 = *n - j;
		d__1 = 1. / ajj;
		dscal_(&i__2, &d__1, &ap[jj + 1], &c__1);
		i__2 = *n - j;
		dspr_("Lower", &i__2, &c_b16, &ap[jj + 1], &c__1, &ap[jj + *n 
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

/*     End of DPPTRF */

} /* dpptrf_ */
