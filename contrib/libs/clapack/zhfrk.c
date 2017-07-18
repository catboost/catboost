/* zhfrk.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zhfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, doublereal *alpha, doublecomplex *a, integer *lda, 
	doublereal *beta, doublecomplex *c__)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublecomplex z__1;

    /* Local variables */
    integer j, n1, n2, nk, info;
    doublecomplex cbeta;
    logical normaltransr;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *), zherk_(char *, char *, integer *, 
	    integer *, doublereal *, doublecomplex *, integer *, doublereal *, 
	     doublecomplex *, integer *);
    integer nrowa;
    logical lower;
    doublecomplex calpha;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical nisodd, notrans;


/*  -- LAPACK routine (version 3.2)                                    -- */

/*  -- Contributed by Julien Langou of the Univ. of Colorado Denver    -- */
/*  -- November 2008                                                   -- */

/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Level 3 BLAS like routine for C in RFP Format. */

/*  ZHFRK performs one of the Hermitian rank--k operations */

/*     C := alpha*A*conjg( A' ) + beta*C, */

/*  or */

/*     C := alpha*conjg( A' )*A + beta*C, */

/*  where alpha and beta are real scalars, C is an n--by--n Hermitian */
/*  matrix and A is an n--by--k matrix in the first case and a k--by--n */
/*  matrix in the second case. */

/*  Arguments */
/*  ========== */

/*  TRANSR    (input) CHARACTER. */
/*          = 'N':  The Normal Form of RFP A is stored; */
/*          = 'C':  The Conjugate-transpose Form of RFP A is stored. */

/*  UPLO   - (input) CHARACTER. */
/*           On  entry,   UPLO  specifies  whether  the  upper  or  lower */
/*           triangular  part  of the  array  C  is to be  referenced  as */
/*           follows: */

/*              UPLO = 'U' or 'u'   Only the  upper triangular part of  C */
/*                                  is to be referenced. */

/*              UPLO = 'L' or 'l'   Only the  lower triangular part of  C */
/*                                  is to be referenced. */

/*           Unchanged on exit. */

/*  TRANS  - (input) CHARACTER. */
/*           On entry,  TRANS  specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   C := alpha*A*conjg( A' ) + beta*C. */

/*              TRANS = 'C' or 'c'   C := alpha*conjg( A' )*A + beta*C. */

/*           Unchanged on exit. */

/*  N      - (input) INTEGER. */
/*           On entry,  N specifies the order of the matrix C.  N must be */
/*           at least zero. */
/*           Unchanged on exit. */

/*  K      - (input) INTEGER. */
/*           On entry with  TRANS = 'N' or 'n',  K  specifies  the number */
/*           of  columns   of  the   matrix   A,   and  on   entry   with */
/*           TRANS = 'C' or 'c',  K  specifies  the number of rows of the */
/*           matrix A.  K must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - (input) DOUBLE PRECISION. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - (input) COMPLEX*16 array of DIMENSION ( LDA, ka ), where KA */
/*           is K  when TRANS = 'N' or 'n', and is N otherwise. Before */
/*           entry with TRANS = 'N' or 'n', the leading N--by--K part of */
/*           the array A must contain the matrix A, otherwise the leading */
/*           K--by--N part of the array A must contain the matrix A. */
/*           Unchanged on exit. */

/*  LDA    - (input) INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n' */
/*           then  LDA must be at least  max( 1, n ), otherwise  LDA must */
/*           be at least  max( 1, k ). */
/*           Unchanged on exit. */

/*  BETA   - (input) DOUBLE PRECISION. */
/*           On entry, BETA specifies the scalar beta. */
/*           Unchanged on exit. */

/*  C      - (input/output) COMPLEX*16 array, dimension ( N*(N+1)/2 ). */
/*           On entry, the matrix A in RFP Format. RFP Format is */
/*           described by TRANSR, UPLO and N. Note that the imaginary */
/*           parts of the diagonal elements need not be set, they are */
/*           assumed to be zero, and on exit they are set to zero. */

/*  Arguments */
/*  ========== */

/*     .. */
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
    --c__;

    /* Function Body */
    info = 0;
    normaltransr = lsame_(transr, "N");
    lower = lsame_(uplo, "L");
    notrans = lsame_(trans, "N");

    if (notrans) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }

    if (! normaltransr && ! lsame_(transr, "C")) {
	info = -1;
    } else if (! lower && ! lsame_(uplo, "U")) {
	info = -2;
    } else if (! notrans && ! lsame_(trans, "C")) {
	info = -3;
    } else if (*n < 0) {
	info = -4;
    } else if (*k < 0) {
	info = -5;
    } else if (*lda < max(1,nrowa)) {
	info = -8;
    }
    if (info != 0) {
	i__1 = -info;
	xerbla_("ZHFRK ", &i__1);
	return 0;
    }

/*     Quick return if possible. */

/*     The quick return case: ((ALPHA.EQ.0).AND.(BETA.NE.ZERO)) is not */
/*     done (it is in ZHERK for example) and left in the general case. */

    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }

    if (*alpha == 0. && *beta == 0.) {
	i__1 = *n * (*n + 1) / 2;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    c__[i__2].r = 0., c__[i__2].i = 0.;
	}
	return 0;
    }

    z__1.r = *alpha, z__1.i = 0.;
    calpha.r = z__1.r, calpha.i = z__1.i;
    z__1.r = *beta, z__1.i = 0.;
    cbeta.r = z__1.r, cbeta.i = z__1.i;

/*     C is N-by-N. */
/*     If N is odd, set NISODD = .TRUE., and N1 and N2. */
/*     If N is even, NISODD = .FALSE., and NK. */

    if (*n % 2 == 0) {
	nisodd = FALSE_;
	nk = *n / 2;
    } else {
	nisodd = TRUE_;
	if (lower) {
	    n2 = *n / 2;
	    n1 = *n - n2;
	} else {
	    n1 = *n / 2;
	    n2 = *n - n1;
	}
    }

    if (nisodd) {

/*        N is odd */

	if (normaltransr) {

/*           N is odd and TRANSR = 'N' */

	    if (lower) {

/*              N is odd, TRANSR = 'N', and UPLO = 'L' */

		if (notrans) {

/*                 N is odd, TRANSR = 'N', UPLO = 'L', and TRANS = 'N' */

		    zherk_("L", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], n);
		    zherk_("U", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[*n + 1], n);
		    zgemm_("N", "C", &n2, &n1, k, &calpha, &a[n1 + 1 + a_dim1]
, lda, &a[a_dim1 + 1], lda, &cbeta, &c__[n1 + 1], 
			    n);

		} else {

/*                 N is odd, TRANSR = 'N', UPLO = 'L', and TRANS = 'C' */

		    zherk_("L", "C", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], n);
		    zherk_("U", "C", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[*n + 1], n)
			    ;
		    zgemm_("C", "N", &n2, &n1, k, &calpha, &a[(n1 + 1) * 
			    a_dim1 + 1], lda, &a[a_dim1 + 1], lda, &cbeta, &
			    c__[n1 + 1], n);

		}

	    } else {

/*              N is odd, TRANSR = 'N', and UPLO = 'U' */

		if (notrans) {

/*                 N is odd, TRANSR = 'N', UPLO = 'U', and TRANS = 'N' */

		    zherk_("L", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 + 1], n);
		    zherk_("U", "N", &n2, k, alpha, &a[n2 + a_dim1], lda, 
			    beta, &c__[n1 + 1], n);
		    zgemm_("N", "C", &n1, &n2, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[n2 + a_dim1], lda, &cbeta, &c__[1], n);

		} else {

/*                 N is odd, TRANSR = 'N', UPLO = 'U', and TRANS = 'C' */

		    zherk_("L", "C", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 + 1], n);
		    zherk_("U", "C", &n2, k, alpha, &a[n2 * a_dim1 + 1], lda, 
			    beta, &c__[n1 + 1], n);
		    zgemm_("C", "N", &n1, &n2, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[n2 * a_dim1 + 1], lda, &cbeta, &c__[1], n);

		}

	    }

	} else {

/*           N is odd, and TRANSR = 'C' */

	    if (lower) {

/*              N is odd, TRANSR = 'C', and UPLO = 'L' */

		if (notrans) {

/*                 N is odd, TRANSR = 'C', UPLO = 'L', and TRANS = 'N' */

		    zherk_("U", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], &n1);
		    zherk_("L", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[2], &n1);
		    zgemm_("N", "C", &n1, &n2, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[n1 + 1 + a_dim1], lda, &cbeta, &c__[n1 * 
			    n1 + 1], &n1);

		} else {

/*                 N is odd, TRANSR = 'C', UPLO = 'L', and TRANS = 'C' */

		    zherk_("U", "C", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], &n1);
		    zherk_("L", "C", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[2], &n1);
		    zgemm_("C", "N", &n1, &n2, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[(n1 + 1) * a_dim1 + 1], lda, &cbeta, &c__[
			    n1 * n1 + 1], &n1);

		}

	    } else {

/*              N is odd, TRANSR = 'C', and UPLO = 'U' */

		if (notrans) {

/*                 N is odd, TRANSR = 'C', UPLO = 'U', and TRANS = 'N' */

		    zherk_("U", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 * n2 + 1], &n2);
		    zherk_("L", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[n1 * n2 + 1], &n2);
		    zgemm_("N", "C", &n2, &n1, k, &calpha, &a[n1 + 1 + a_dim1]
, lda, &a[a_dim1 + 1], lda, &cbeta, &c__[1], &n2);

		} else {

/*                 N is odd, TRANSR = 'C', UPLO = 'U', and TRANS = 'C' */

		    zherk_("U", "C", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 * n2 + 1], &n2);
		    zherk_("L", "C", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[n1 * n2 + 1], &n2);
		    zgemm_("C", "N", &n2, &n1, k, &calpha, &a[(n1 + 1) * 
			    a_dim1 + 1], lda, &a[a_dim1 + 1], lda, &cbeta, &
			    c__[1], &n2);

		}

	    }

	}

    } else {

/*        N is even */

	if (normaltransr) {

/*           N is even and TRANSR = 'N' */

	    if (lower) {

/*              N is even, TRANSR = 'N', and UPLO = 'L' */

		if (notrans) {

/*                 N is even, TRANSR = 'N', UPLO = 'L', and TRANS = 'N' */

		    i__1 = *n + 1;
		    zherk_("L", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[2], &i__1);
		    i__1 = *n + 1;
		    zherk_("U", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[1], &i__1);
		    i__1 = *n + 1;
		    zgemm_("N", "C", &nk, &nk, k, &calpha, &a[nk + 1 + a_dim1]
, lda, &a[a_dim1 + 1], lda, &cbeta, &c__[nk + 2], 
			    &i__1);

		} else {

/*                 N is even, TRANSR = 'N', UPLO = 'L', and TRANS = 'C' */

		    i__1 = *n + 1;
		    zherk_("L", "C", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[2], &i__1);
		    i__1 = *n + 1;
		    zherk_("U", "C", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[1], &i__1);
		    i__1 = *n + 1;
		    zgemm_("C", "N", &nk, &nk, k, &calpha, &a[(nk + 1) * 
			    a_dim1 + 1], lda, &a[a_dim1 + 1], lda, &cbeta, &
			    c__[nk + 2], &i__1);

		}

	    } else {

/*              N is even, TRANSR = 'N', and UPLO = 'U' */

		if (notrans) {

/*                 N is even, TRANSR = 'N', UPLO = 'U', and TRANS = 'N' */

		    i__1 = *n + 1;
		    zherk_("L", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 2], &i__1);
		    i__1 = *n + 1;
		    zherk_("U", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[nk + 1], &i__1);
		    i__1 = *n + 1;
		    zgemm_("N", "C", &nk, &nk, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[nk + 1 + a_dim1], lda, &cbeta, &c__[1], &
			    i__1);

		} else {

/*                 N is even, TRANSR = 'N', UPLO = 'U', and TRANS = 'C' */

		    i__1 = *n + 1;
		    zherk_("L", "C", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 2], &i__1);
		    i__1 = *n + 1;
		    zherk_("U", "C", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[nk + 1], &i__1);
		    i__1 = *n + 1;
		    zgemm_("C", "N", &nk, &nk, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[(nk + 1) * a_dim1 + 1], lda, &cbeta, &c__[
			    1], &i__1);

		}

	    }

	} else {

/*           N is even, and TRANSR = 'C' */

	    if (lower) {

/*              N is even, TRANSR = 'C', and UPLO = 'L' */

		if (notrans) {

/*                 N is even, TRANSR = 'C', UPLO = 'L', and TRANS = 'N' */

		    zherk_("U", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 1], &nk);
		    zherk_("L", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[1], &nk);
		    zgemm_("N", "C", &nk, &nk, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[nk + 1 + a_dim1], lda, &cbeta, &c__[(nk + 
			    1) * nk + 1], &nk);

		} else {

/*                 N is even, TRANSR = 'C', UPLO = 'L', and TRANS = 'C' */

		    zherk_("U", "C", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 1], &nk);
		    zherk_("L", "C", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[1], &nk);
		    zgemm_("C", "N", &nk, &nk, k, &calpha, &a[a_dim1 + 1], 
			    lda, &a[(nk + 1) * a_dim1 + 1], lda, &cbeta, &c__[
			    (nk + 1) * nk + 1], &nk);

		}

	    } else {

/*              N is even, TRANSR = 'C', and UPLO = 'U' */

		if (notrans) {

/*                 N is even, TRANSR = 'C', UPLO = 'U', and TRANS = 'N' */

		    zherk_("U", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk * (nk + 1) + 1], &nk);
		    zherk_("L", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[nk * nk + 1], &nk);
		    zgemm_("N", "C", &nk, &nk, k, &calpha, &a[nk + 1 + a_dim1]
, lda, &a[a_dim1 + 1], lda, &cbeta, &c__[1], &nk);

		} else {

/*                 N is even, TRANSR = 'C', UPLO = 'U', and TRANS = 'C' */

		    zherk_("U", "C", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk * (nk + 1) + 1], &nk);
		    zherk_("L", "C", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[nk * nk + 1], &nk);
		    zgemm_("C", "N", &nk, &nk, k, &calpha, &a[(nk + 1) * 
			    a_dim1 + 1], lda, &a[a_dim1 + 1], lda, &cbeta, &
			    c__[1], &nk);

		}

	    }

	}

    }

    return 0;

/*     End of ZHFRK */

} /* zhfrk_ */
