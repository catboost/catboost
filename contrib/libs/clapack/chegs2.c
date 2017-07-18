/* chegs2.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {1.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int chegs2_(integer *itype, char *uplo, integer *n, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    real r__1, r__2;
    complex q__1;

    /* Local variables */
    integer k;
    complex ct;
    real akk, bkk;
    extern /* Subroutine */ int cher2_(char *, integer *, complex *, complex *
, integer *, complex *, integer *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    logical upper;
    extern /* Subroutine */ int ctrmv_(char *, char *, char *, integer *, 
	    complex *, integer *, complex *, integer *), ctrsv_(char *, char *, char *, integer *, complex *, 
	    integer *, complex *, integer *), clacgv_(
	    integer *, complex *, integer *), csscal_(integer *, real *, 
	    complex *, integer *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHEGS2 reduces a complex Hermitian-definite generalized */
/*  eigenproblem to standard form. */

/*  If ITYPE = 1, the problem is A*x = lambda*B*x, */
/*  and A is overwritten by inv(U')*A*inv(U) or inv(L)*A*inv(L') */

/*  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/*  B*A*x = lambda*x, and A is overwritten by U*A*U` or L'*A*L. */

/*  B must have been previously factorized as U'*U or L*L' by CPOTRF. */

/*  Arguments */
/*  ========= */

/*  ITYPE   (input) INTEGER */
/*          = 1: compute inv(U')*A*inv(U) or inv(L)*A*inv(L'); */
/*          = 2 or 3: compute U*A*U' or L'*A*L. */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          Hermitian matrix A is stored, and how B has been factorized. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/*          n by n upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading n by n lower triangular part of A contains the lower */
/*          triangular part of the matrix A, and the strictly upper */
/*          triangular part of A is not referenced. */

/*          On exit, if INFO = 0, the transformed matrix, stored in the */
/*          same format as A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  B       (input) COMPLEX array, dimension (LDB,N) */
/*          The triangular factor from the Cholesky factorization of B, */
/*          as returned by CPOTRF. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHEGS2", &i__1);
	return 0;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U')*A*inv(U) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(k:n,k:n) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
/* Computing 2nd power */
		r__1 = bkk;
		akk /= r__1 * r__1;
		i__2 = k + k * a_dim1;
		a[i__2].r = akk, a[i__2].i = 0.f;
		if (k < *n) {
		    i__2 = *n - k;
		    r__1 = 1.f / bkk;
		    csscal_(&i__2, &r__1, &a[k + (k + 1) * a_dim1], lda);
		    r__1 = akk * -.5f;
		    ct.r = r__1, ct.i = 0.f;
		    i__2 = *n - k;
		    clacgv_(&i__2, &a[k + (k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    clacgv_(&i__2, &b[k + (k + 1) * b_dim1], ldb);
		    i__2 = *n - k;
		    caxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cher2_(uplo, &i__2, &q__1, &a[k + (k + 1) * a_dim1], lda, 
			    &b[k + (k + 1) * b_dim1], ldb, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    caxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    clacgv_(&i__2, &b[k + (k + 1) * b_dim1], ldb);
		    i__2 = *n - k;
		    ctrsv_(uplo, "Conjugate transpose", "Non-unit", &i__2, &b[
			    k + 1 + (k + 1) * b_dim1], ldb, &a[k + (k + 1) * 
			    a_dim1], lda);
		    i__2 = *n - k;
		    clacgv_(&i__2, &a[k + (k + 1) * a_dim1], lda);
		}
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L') */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(k:n,k:n) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
/* Computing 2nd power */
		r__1 = bkk;
		akk /= r__1 * r__1;
		i__2 = k + k * a_dim1;
		a[i__2].r = akk, a[i__2].i = 0.f;
		if (k < *n) {
		    i__2 = *n - k;
		    r__1 = 1.f / bkk;
		    csscal_(&i__2, &r__1, &a[k + 1 + k * a_dim1], &c__1);
		    r__1 = akk * -.5f;
		    ct.r = r__1, ct.i = 0.f;
		    i__2 = *n - k;
		    caxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cher2_(uplo, &i__2, &q__1, &a[k + 1 + k * a_dim1], &c__1, 
			    &b[k + 1 + k * b_dim1], &c__1, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    caxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    ctrsv_(uplo, "No transpose", "Non-unit", &i__2, &b[k + 1 
			    + (k + 1) * b_dim1], ldb, &a[k + 1 + k * a_dim1], 
			    &c__1);
		}
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U' */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(1:k,1:k) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
		i__2 = k - 1;
		ctrmv_(uplo, "No transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a[k * a_dim1 + 1], &c__1);
		r__1 = akk * .5f;
		ct.r = r__1, ct.i = 0.f;
		i__2 = k - 1;
		caxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		cher2_(uplo, &i__2, &c_b1, &a[k * a_dim1 + 1], &c__1, &b[k * 
			b_dim1 + 1], &c__1, &a[a_offset], lda);
		i__2 = k - 1;
		caxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		csscal_(&i__2, &bkk, &a[k * a_dim1 + 1], &c__1);
		i__2 = k + k * a_dim1;
/* Computing 2nd power */
		r__2 = bkk;
		r__1 = akk * (r__2 * r__2);
		a[i__2].r = r__1, a[i__2].i = 0.f;
/* L30: */
	    }
	} else {

/*           Compute L'*A*L */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(1:k,1:k) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
		i__2 = k - 1;
		clacgv_(&i__2, &a[k + a_dim1], lda);
		i__2 = k - 1;
		ctrmv_(uplo, "Conjugate transpose", "Non-unit", &i__2, &b[
			b_offset], ldb, &a[k + a_dim1], lda);
		r__1 = akk * .5f;
		ct.r = r__1, ct.i = 0.f;
		i__2 = k - 1;
		clacgv_(&i__2, &b[k + b_dim1], ldb);
		i__2 = k - 1;
		caxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		cher2_(uplo, &i__2, &c_b1, &a[k + a_dim1], lda, &b[k + b_dim1]
, ldb, &a[a_offset], lda);
		i__2 = k - 1;
		caxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		clacgv_(&i__2, &b[k + b_dim1], ldb);
		i__2 = k - 1;
		csscal_(&i__2, &bkk, &a[k + a_dim1], lda);
		i__2 = k - 1;
		clacgv_(&i__2, &a[k + a_dim1], lda);
		i__2 = k + k * a_dim1;
/* Computing 2nd power */
		r__2 = bkk;
		r__1 = akk * (r__2 * r__2);
		a[i__2].r = r__1, a[i__2].i = 0.f;
/* L40: */
	    }
	}
    }
    return 0;

/*     End of CHEGS2 */

} /* chegs2_ */
