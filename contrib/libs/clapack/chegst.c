/* chegst.f -- translated by f2c (version 20061008).
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
static complex c_b2 = {.5f,0.f};
static integer c__1 = 1;
static integer c_n1 = -1;
static real c_b18 = 1.f;

/* Subroutine */ int chegst_(integer *itype, char *uplo, integer *n, complex *
	a, integer *lda, complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    complex q__1;

    /* Local variables */
    integer k, kb, nb;
    extern /* Subroutine */ int chemm_(char *, char *, integer *, integer *, 
	    complex *, complex *, integer *, complex *, integer *, complex *, 
	    complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrmm_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *), ctrsm_(char *, char *, 
	     char *, char *, integer *, integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    logical upper;
    extern /* Subroutine */ int chegs2_(integer *, char *, integer *, complex 
	    *, integer *, complex *, integer *, integer *), cher2k_(
	    char *, char *, integer *, integer *, complex *, complex *, 
	    integer *, complex *, integer *, real *, complex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHEGST reduces a complex Hermitian-definite generalized */
/*  eigenproblem to standard form. */

/*  If ITYPE = 1, the problem is A*x = lambda*B*x, */
/*  and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H) */

/*  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/*  B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L. */

/*  B must have been previously factorized as U**H*U or L*L**H by CPOTRF. */

/*  Arguments */
/*  ========= */

/*  ITYPE   (input) INTEGER */
/*          = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H); */
/*          = 2 or 3: compute U*A*U**H or L**H*A*L. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored and B is factored as */
/*                  U**H*U; */
/*          = 'L':  Lower triangle of A is stored and B is factored as */
/*                  L*L**H. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/*          N-by-N upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading N-by-N lower triangular part of A contains the lower */
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
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

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
	xerbla_("CHEGST", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "CHEGST", uplo, n, &c_n1, &c_n1, &c_n1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	chegs2_(itype, uplo, n, &a[a_offset], lda, &b[b_offset], ldb, info);
    } else {

/*        Use blocked code */

	if (*itype == 1) {
	    if (upper) {

/*              Compute inv(U')*A*inv(U) */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the upper triangle of A(k:n,k:n) */

		    chegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			ctrsm_("Left", uplo, "Conjugate transpose", "Non-unit"
, &kb, &i__3, &c_b1, &b[k + k * b_dim1], ldb, 
				&a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			q__1.r = -.5f, q__1.i = -0.f;
			chemm_("Left", uplo, &kb, &i__3, &q__1, &a[k + k * 
				a_dim1], lda, &b[k + (k + kb) * b_dim1], ldb, 
				&c_b1, &a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			q__1.r = -1.f, q__1.i = -0.f;
			cher2k_(uplo, "Conjugate transpose", &i__3, &kb, &
				q__1, &a[k + (k + kb) * a_dim1], lda, &b[k + (
				k + kb) * b_dim1], ldb, &c_b18, &a[k + kb + (
				k + kb) * a_dim1], lda)
				;
			i__3 = *n - k - kb + 1;
			q__1.r = -.5f, q__1.i = -0.f;
			chemm_("Left", uplo, &kb, &i__3, &q__1, &a[k + k * 
				a_dim1], lda, &b[k + (k + kb) * b_dim1], ldb, 
				&c_b1, &a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			ctrsm_("Right", uplo, "No transpose", "Non-unit", &kb, 
				 &i__3, &c_b1, &b[k + kb + (k + kb) * b_dim1], 
				 ldb, &a[k + (k + kb) * a_dim1], lda);
		    }
/* L10: */
		}
	    } else {

/*              Compute inv(L)*A*inv(L') */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the lower triangle of A(k:n,k:n) */

		    chegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			ctrsm_("Right", uplo, "Conjugate transpose", "Non-un"
				"it", &i__3, &kb, &c_b1, &b[k + k * b_dim1], 
				ldb, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			q__1.r = -.5f, q__1.i = -0.f;
			chemm_("Right", uplo, &i__3, &kb, &q__1, &a[k + k * 
				a_dim1], lda, &b[k + kb + k * b_dim1], ldb, &
				c_b1, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			q__1.r = -1.f, q__1.i = -0.f;
			cher2k_(uplo, "No transpose", &i__3, &kb, &q__1, &a[k 
				+ kb + k * a_dim1], lda, &b[k + kb + k * 
				b_dim1], ldb, &c_b18, &a[k + kb + (k + kb) * 
				a_dim1], lda);
			i__3 = *n - k - kb + 1;
			q__1.r = -.5f, q__1.i = -0.f;
			chemm_("Right", uplo, &i__3, &kb, &q__1, &a[k + k * 
				a_dim1], lda, &b[k + kb + k * b_dim1], ldb, &
				c_b1, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			ctrsm_("Left", uplo, "No transpose", "Non-unit", &
				i__3, &kb, &c_b1, &b[k + kb + (k + kb) * 
				b_dim1], ldb, &a[k + kb + k * a_dim1], lda);
		    }
/* L20: */
		}
	    }
	} else {
	    if (upper) {

/*              Compute U*A*U' */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    ctrmm_("Left", uplo, "No transpose", "Non-unit", &i__3, &
			    kb, &c_b1, &b[b_offset], ldb, &a[k * a_dim1 + 1], 
			    lda);
		    i__3 = k - 1;
		    chemm_("Right", uplo, &i__3, &kb, &c_b2, &a[k + k * 
			    a_dim1], lda, &b[k * b_dim1 + 1], ldb, &c_b1, &a[
			    k * a_dim1 + 1], lda);
		    i__3 = k - 1;
		    cher2k_(uplo, "No transpose", &i__3, &kb, &c_b1, &a[k * 
			    a_dim1 + 1], lda, &b[k * b_dim1 + 1], ldb, &c_b18, 
			     &a[a_offset], lda);
		    i__3 = k - 1;
		    chemm_("Right", uplo, &i__3, &kb, &c_b2, &a[k + k * 
			    a_dim1], lda, &b[k * b_dim1 + 1], ldb, &c_b1, &a[
			    k * a_dim1 + 1], lda);
		    i__3 = k - 1;
		    ctrmm_("Right", uplo, "Conjugate transpose", "Non-unit", &
			    i__3, &kb, &c_b1, &b[k + k * b_dim1], ldb, &a[k * 
			    a_dim1 + 1], lda);
		    chegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
/* L30: */
		}
	    } else {

/*              Compute L'*A*L */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = min(i__3,nb);

/*                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    ctrmm_("Right", uplo, "No transpose", "Non-unit", &kb, &
			    i__3, &c_b1, &b[b_offset], ldb, &a[k + a_dim1], 
			    lda);
		    i__3 = k - 1;
		    chemm_("Left", uplo, &kb, &i__3, &c_b2, &a[k + k * a_dim1]
, lda, &b[k + b_dim1], ldb, &c_b1, &a[k + a_dim1], 
			     lda);
		    i__3 = k - 1;
		    cher2k_(uplo, "Conjugate transpose", &i__3, &kb, &c_b1, &
			    a[k + a_dim1], lda, &b[k + b_dim1], ldb, &c_b18, &
			    a[a_offset], lda);
		    i__3 = k - 1;
		    chemm_("Left", uplo, &kb, &i__3, &c_b2, &a[k + k * a_dim1]
, lda, &b[k + b_dim1], ldb, &c_b1, &a[k + a_dim1], 
			     lda);
		    i__3 = k - 1;
		    ctrmm_("Left", uplo, "Conjugate transpose", "Non-unit", &
			    kb, &i__3, &c_b1, &b[k + k * b_dim1], ldb, &a[k + 
			    a_dim1], lda);
		    chegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
/* L40: */
		}
	    }
	}
    }
    return 0;

/*     End of CHEGST */

} /* chegst_ */
