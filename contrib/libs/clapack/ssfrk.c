/* ssfrk.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ssfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, real *alpha, real *a, integer *lda, real *beta, real *
	c__)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1;

    /* Local variables */
    integer j, n1, n2, nk, info;
    logical normaltransr;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *, 
	    integer *, real *, real *, integer *, real *, integer *, real *, 
	    real *, integer *);
    integer nrowa;
    logical lower;
    extern /* Subroutine */ int ssyrk_(char *, char *, integer *, integer *, 
	    real *, real *, integer *, real *, real *, integer *), xerbla_(char *, integer *);
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

/*  SSFRK performs one of the symmetric rank--k operations */

/*     C := alpha*A*A' + beta*C, */

/*  or */

/*     C := alpha*A'*A + beta*C, */

/*  where alpha and beta are real scalars, C is an n--by--n symmetric */
/*  matrix and A is an n--by--k matrix in the first case and a k--by--n */
/*  matrix in the second case. */

/*  Arguments */
/*  ========== */

/*  TRANSR    (input) CHARACTER */
/*          = 'N':  The Normal Form of RFP A is stored; */
/*          = 'T':  The Transpose Form of RFP A is stored. */

/*  UPLO   - (input) CHARACTER */
/*           On  entry, UPLO specifies whether the upper or lower */
/*           triangular part of the array C is to be referenced as */
/*           follows: */

/*              UPLO = 'U' or 'u'   Only the upper triangular part of C */
/*                                  is to be referenced. */

/*              UPLO = 'L' or 'l'   Only the lower triangular part of C */
/*                                  is to be referenced. */

/*           Unchanged on exit. */

/*  TRANS  - (input) CHARACTER */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C. */

/*              TRANS = 'T' or 't'   C := alpha*A'*A + beta*C. */

/*           Unchanged on exit. */

/*  N      - (input) INTEGER. */
/*           On entry, N specifies the order of the matrix C. N must be */
/*           at least zero. */
/*           Unchanged on exit. */

/*  K      - (input) INTEGER. */
/*           On entry with TRANS = 'N' or 'n', K specifies the number */
/*           of  columns of the matrix A, and on entry with TRANS = 'T' */
/*           or 't', K specifies the number of rows of the matrix A. K */
/*           must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - (input) REAL. */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - (input) REAL array of DIMENSION ( LDA, ka ), where KA */
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

/*  BETA   - (input) REAL. */
/*           On entry, BETA specifies the scalar beta. */
/*           Unchanged on exit. */


/*  C      - (input/output) REAL array, dimension ( NT ); */
/*           NT = N*(N+1)/2. On entry, the symmetric matrix C in RFP */
/*           Format. RFP Format is described by TRANSR, UPLO and N. */

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

    if (! normaltransr && ! lsame_(transr, "T")) {
	info = -1;
    } else if (! lower && ! lsame_(uplo, "U")) {
	info = -2;
    } else if (! notrans && ! lsame_(trans, "T")) {
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
	xerbla_("SSFRK ", &i__1);
	return 0;
    }

/*     Quick return if possible. */

/*     The quick return case: ((ALPHA.EQ.0).AND.(BETA.NE.ZERO)) is not */
/*     done (it is in SSYRK for example) and left in the general case. */

    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
	return 0;
    }

    if (*alpha == 0.f && *beta == 0.f) {
	i__1 = *n * (*n + 1) / 2;
	for (j = 1; j <= i__1; ++j) {
	    c__[j] = 0.f;
	}
	return 0;
    }

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

		    ssyrk_("L", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], n);
		    ssyrk_("U", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[*n + 1], n);
		    sgemm_("N", "T", &n2, &n1, k, alpha, &a[n1 + 1 + a_dim1], 
			    lda, &a[a_dim1 + 1], lda, beta, &c__[n1 + 1], n);

		} else {

/*                 N is odd, TRANSR = 'N', UPLO = 'L', and TRANS = 'T' */

		    ssyrk_("L", "T", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], n);
		    ssyrk_("U", "T", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[*n + 1], n)
			    ;
		    sgemm_("T", "N", &n2, &n1, k, alpha, &a[(n1 + 1) * a_dim1 
			    + 1], lda, &a[a_dim1 + 1], lda, beta, &c__[n1 + 1]
, n);

		}

	    } else {

/*              N is odd, TRANSR = 'N', and UPLO = 'U' */

		if (notrans) {

/*                 N is odd, TRANSR = 'N', UPLO = 'U', and TRANS = 'N' */

		    ssyrk_("L", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 + 1], n);
		    ssyrk_("U", "N", &n2, k, alpha, &a[n2 + a_dim1], lda, 
			    beta, &c__[n1 + 1], n);
		    sgemm_("N", "T", &n1, &n2, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[n2 + a_dim1], lda, beta, &c__[1], n);

		} else {

/*                 N is odd, TRANSR = 'N', UPLO = 'U', and TRANS = 'T' */

		    ssyrk_("L", "T", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 + 1], n);
		    ssyrk_("U", "T", &n2, k, alpha, &a[n2 * a_dim1 + 1], lda, 
			    beta, &c__[n1 + 1], n);
		    sgemm_("T", "N", &n1, &n2, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[n2 * a_dim1 + 1], lda, beta, &c__[1], n);

		}

	    }

	} else {

/*           N is odd, and TRANSR = 'T' */

	    if (lower) {

/*              N is odd, TRANSR = 'T', and UPLO = 'L' */

		if (notrans) {

/*                 N is odd, TRANSR = 'T', UPLO = 'L', and TRANS = 'N' */

		    ssyrk_("U", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], &n1);
		    ssyrk_("L", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[2], &n1);
		    sgemm_("N", "T", &n1, &n2, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[n1 + 1 + a_dim1], lda, beta, &c__[n1 * n1 + 1], 
			     &n1);

		} else {

/*                 N is odd, TRANSR = 'T', UPLO = 'L', and TRANS = 'T' */

		    ssyrk_("U", "T", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[1], &n1);
		    ssyrk_("L", "T", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[2], &n1);
		    sgemm_("T", "N", &n1, &n2, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[(n1 + 1) * a_dim1 + 1], lda, beta, &c__[n1 * 
			    n1 + 1], &n1);

		}

	    } else {

/*              N is odd, TRANSR = 'T', and UPLO = 'U' */

		if (notrans) {

/*                 N is odd, TRANSR = 'T', UPLO = 'U', and TRANS = 'N' */

		    ssyrk_("U", "N", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 * n2 + 1], &n2);
		    ssyrk_("L", "N", &n2, k, alpha, &a[n1 + 1 + a_dim1], lda, 
			    beta, &c__[n1 * n2 + 1], &n2);
		    sgemm_("N", "T", &n2, &n1, k, alpha, &a[n1 + 1 + a_dim1], 
			    lda, &a[a_dim1 + 1], lda, beta, &c__[1], &n2);

		} else {

/*                 N is odd, TRANSR = 'T', UPLO = 'U', and TRANS = 'T' */

		    ssyrk_("U", "T", &n1, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[n2 * n2 + 1], &n2);
		    ssyrk_("L", "T", &n2, k, alpha, &a[(n1 + 1) * a_dim1 + 1], 
			     lda, beta, &c__[n1 * n2 + 1], &n2);
		    sgemm_("T", "N", &n2, &n1, k, alpha, &a[(n1 + 1) * a_dim1 
			    + 1], lda, &a[a_dim1 + 1], lda, beta, &c__[1], &
			    n2);

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
		    ssyrk_("L", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[2], &i__1);
		    i__1 = *n + 1;
		    ssyrk_("U", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[1], &i__1);
		    i__1 = *n + 1;
		    sgemm_("N", "T", &nk, &nk, k, alpha, &a[nk + 1 + a_dim1], 
			    lda, &a[a_dim1 + 1], lda, beta, &c__[nk + 2], &
			    i__1);

		} else {

/*                 N is even, TRANSR = 'N', UPLO = 'L', and TRANS = 'T' */

		    i__1 = *n + 1;
		    ssyrk_("L", "T", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[2], &i__1);
		    i__1 = *n + 1;
		    ssyrk_("U", "T", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[1], &i__1);
		    i__1 = *n + 1;
		    sgemm_("T", "N", &nk, &nk, k, alpha, &a[(nk + 1) * a_dim1 
			    + 1], lda, &a[a_dim1 + 1], lda, beta, &c__[nk + 2]
, &i__1);

		}

	    } else {

/*              N is even, TRANSR = 'N', and UPLO = 'U' */

		if (notrans) {

/*                 N is even, TRANSR = 'N', UPLO = 'U', and TRANS = 'N' */

		    i__1 = *n + 1;
		    ssyrk_("L", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 2], &i__1);
		    i__1 = *n + 1;
		    ssyrk_("U", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[nk + 1], &i__1);
		    i__1 = *n + 1;
		    sgemm_("N", "T", &nk, &nk, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[nk + 1 + a_dim1], lda, beta, &c__[1], &i__1);

		} else {

/*                 N is even, TRANSR = 'N', UPLO = 'U', and TRANS = 'T' */

		    i__1 = *n + 1;
		    ssyrk_("L", "T", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 2], &i__1);
		    i__1 = *n + 1;
		    ssyrk_("U", "T", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[nk + 1], &i__1);
		    i__1 = *n + 1;
		    sgemm_("T", "N", &nk, &nk, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[(nk + 1) * a_dim1 + 1], lda, beta, &c__[1], &
			    i__1);

		}

	    }

	} else {

/*           N is even, and TRANSR = 'T' */

	    if (lower) {

/*              N is even, TRANSR = 'T', and UPLO = 'L' */

		if (notrans) {

/*                 N is even, TRANSR = 'T', UPLO = 'L', and TRANS = 'N' */

		    ssyrk_("U", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 1], &nk);
		    ssyrk_("L", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[1], &nk);
		    sgemm_("N", "T", &nk, &nk, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[nk + 1 + a_dim1], lda, beta, &c__[(nk + 1) * 
			    nk + 1], &nk);

		} else {

/*                 N is even, TRANSR = 'T', UPLO = 'L', and TRANS = 'T' */

		    ssyrk_("U", "T", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk + 1], &nk);
		    ssyrk_("L", "T", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[1], &nk);
		    sgemm_("T", "N", &nk, &nk, k, alpha, &a[a_dim1 + 1], lda, 
			    &a[(nk + 1) * a_dim1 + 1], lda, beta, &c__[(nk + 
			    1) * nk + 1], &nk);

		}

	    } else {

/*              N is even, TRANSR = 'T', and UPLO = 'U' */

		if (notrans) {

/*                 N is even, TRANSR = 'T', UPLO = 'U', and TRANS = 'N' */

		    ssyrk_("U", "N", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk * (nk + 1) + 1], &nk);
		    ssyrk_("L", "N", &nk, k, alpha, &a[nk + 1 + a_dim1], lda, 
			    beta, &c__[nk * nk + 1], &nk);
		    sgemm_("N", "T", &nk, &nk, k, alpha, &a[nk + 1 + a_dim1], 
			    lda, &a[a_dim1 + 1], lda, beta, &c__[1], &nk);

		} else {

/*                 N is even, TRANSR = 'T', UPLO = 'U', and TRANS = 'T' */

		    ssyrk_("U", "T", &nk, k, alpha, &a[a_dim1 + 1], lda, beta, 
			     &c__[nk * (nk + 1) + 1], &nk);
		    ssyrk_("L", "T", &nk, k, alpha, &a[(nk + 1) * a_dim1 + 1], 
			     lda, beta, &c__[nk * nk + 1], &nk);
		    sgemm_("T", "N", &nk, &nk, k, alpha, &a[(nk + 1) * a_dim1 
			    + 1], lda, &a[a_dim1 + 1], lda, beta, &c__[1], &
			    nk);

		}

	    }

	}

    }

    return 0;

/*     End of SSFRK */

} /* ssfrk_ */
