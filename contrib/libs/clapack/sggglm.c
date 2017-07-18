/* sggglm.f -- translated by f2c (version 20061008).
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
static integer c_n1 = -1;
static real c_b32 = -1.f;
static real c_b34 = 1.f;

/* Subroutine */ int sggglm_(integer *n, integer *m, integer *p, real *a, 
	integer *lda, real *b, integer *ldb, real *d__, real *x, real *y, 
	real *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, nb, np, nb1, nb2, nb3, nb4, lopt;
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *, 
	    real *, integer *, real *, integer *, real *, real *, integer *), scopy_(integer *, real *, integer *, real *, integer *), 
	    xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int sggqrf_(integer *, integer *, integer *, real 
	    *, integer *, real *, real *, integer *, real *, real *, integer *
, integer *);
    integer lwkmin, lwkopt;
    logical lquery;
    extern /* Subroutine */ int sormqr_(char *, char *, integer *, integer *, 
	    integer *, real *, integer *, real *, real *, integer *, real *, 
	    integer *, integer *), sormrq_(char *, char *, 
	    integer *, integer *, integer *, real *, integer *, real *, real *
, integer *, real *, integer *, integer *), 
	    strtrs_(char *, char *, char *, integer *, integer *, real *, 
	    integer *, real *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SGGGLM solves a general Gauss-Markov linear model (GLM) problem: */

/*          minimize || y ||_2   subject to   d = A*x + B*y */
/*              x */

/*  where A is an N-by-M matrix, B is an N-by-P matrix, and d is a */
/*  given N-vector. It is assumed that M <= N <= M+P, and */

/*             rank(A) = M    and    rank( A B ) = N. */

/*  Under these assumptions, the constrained equation is always */
/*  consistent, and there is a unique solution x and a minimal 2-norm */
/*  solution y, which is obtained using a generalized QR factorization */
/*  of the matrices (A, B) given by */

/*     A = Q*(R),   B = Q*T*Z. */
/*           (0) */

/*  In particular, if matrix B is square nonsingular, then the problem */
/*  GLM is equivalent to the following weighted linear least squares */
/*  problem */

/*               minimize || inv(B)*(d-A*x) ||_2 */
/*                   x */

/*  where inv(B) denotes the inverse of B. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of rows of the matrices A and B.  N >= 0. */

/*  M       (input) INTEGER */
/*          The number of columns of the matrix A.  0 <= M <= N. */

/*  P       (input) INTEGER */
/*          The number of columns of the matrix B.  P >= N-M. */

/*  A       (input/output) REAL array, dimension (LDA,M) */
/*          On entry, the N-by-M matrix A. */
/*          On exit, the upper triangular part of the array A contains */
/*          the M-by-M upper triangular matrix R. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,N). */

/*  B       (input/output) REAL array, dimension (LDB,P) */
/*          On entry, the N-by-P matrix B. */
/*          On exit, if N <= P, the upper triangle of the subarray */
/*          B(1:N,P-N+1:P) contains the N-by-N upper triangular matrix T; */
/*          if N > P, the elements on and above the (N-P)th subdiagonal */
/*          contain the N-by-P upper trapezoidal matrix T. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  D       (input/output) REAL array, dimension (N) */
/*          On entry, D is the left hand side of the GLM equation. */
/*          On exit, D is destroyed. */

/*  X       (output) REAL array, dimension (M) */
/*  Y       (output) REAL array, dimension (P) */
/*          On exit, X and Y are the solutions of the GLM problem. */

/*  WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= max(1,N+M+P). */
/*          For optimum performance, LWORK >= M+min(N,P)+max(N,P)*NB, */
/*          where NB is an upper bound for the optimal blocksizes for */
/*          SGEQRF, SGERQF, SORMQR and SORMRQ. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          = 1:  the upper triangular factor R associated with A in the */
/*                generalized QR factorization of the pair (A, B) is */
/*                singular, so that rank(A) < M; the least squares */
/*                solution could not be computed. */
/*          = 2:  the bottom (N-M) by (N-M) part of the upper trapezoidal */
/*                factor T associated with B in the generalized QR */
/*                factorization of the pair (A, B) is singular, so that */
/*                rank( A B ) < N; the least squares solution could not */
/*                be computed. */

/*  =================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --d__;
    --x;
    --y;
    --work;

    /* Function Body */
    *info = 0;
    np = min(*n,*p);
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*m < 0 || *m > *n) {
	*info = -2;
    } else if (*p < 0 || *p < *n - *m) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }

/*     Calculate workspace */

    if (*info == 0) {
	if (*n == 0) {
	    lwkmin = 1;
	    lwkopt = 1;
	} else {
	    nb1 = ilaenv_(&c__1, "SGEQRF", " ", n, m, &c_n1, &c_n1);
	    nb2 = ilaenv_(&c__1, "SGERQF", " ", n, m, &c_n1, &c_n1);
	    nb3 = ilaenv_(&c__1, "SORMQR", " ", n, m, p, &c_n1);
	    nb4 = ilaenv_(&c__1, "SORMRQ", " ", n, m, p, &c_n1);
/* Computing MAX */
	    i__1 = max(nb1,nb2), i__1 = max(i__1,nb3);
	    nb = max(i__1,nb4);
	    lwkmin = *m + *n + *p;
	    lwkopt = *m + np + max(*n,*p) * nb;
	}
	work[1] = (real) lwkopt;

	if (*lwork < lwkmin && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGGGLM", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Compute the GQR factorization of matrices A and B: */

/*            Q'*A = ( R11 ) M,    Q'*B*Z' = ( T11   T12 ) M */
/*                   (  0  ) N-M             (  0    T22 ) N-M */
/*                      M                     M+P-N  N-M */

/*     where R11 and T22 are upper triangular, and Q and Z are */
/*     orthogonal. */

    i__1 = *lwork - *m - np;
    sggqrf_(n, m, p, &a[a_offset], lda, &work[1], &b[b_offset], ldb, &work[*m 
	    + 1], &work[*m + np + 1], &i__1, info);
    lopt = work[*m + np + 1];

/*     Update left-hand-side vector d = Q'*d = ( d1 ) M */
/*                                             ( d2 ) N-M */

    i__1 = max(1,*n);
    i__2 = *lwork - *m - np;
    sormqr_("Left", "Transpose", n, &c__1, m, &a[a_offset], lda, &work[1], &
	    d__[1], &i__1, &work[*m + np + 1], &i__2, info);
/* Computing MAX */
    i__1 = lopt, i__2 = (integer) work[*m + np + 1];
    lopt = max(i__1,i__2);

/*     Solve T22*y2 = d2 for y2 */

    if (*n > *m) {
	i__1 = *n - *m;
	i__2 = *n - *m;
	strtrs_("Upper", "No transpose", "Non unit", &i__1, &c__1, &b[*m + 1 
		+ (*m + *p - *n + 1) * b_dim1], ldb, &d__[*m + 1], &i__2, 
		info);

	if (*info > 0) {
	    *info = 1;
	    return 0;
	}

	i__1 = *n - *m;
	scopy_(&i__1, &d__[*m + 1], &c__1, &y[*m + *p - *n + 1], &c__1);
    }

/*     Set y1 = 0 */

    i__1 = *m + *p - *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	y[i__] = 0.f;
/* L10: */
    }

/*     Update d1 = d1 - T12*y2 */

    i__1 = *n - *m;
    sgemv_("No transpose", m, &i__1, &c_b32, &b[(*m + *p - *n + 1) * b_dim1 + 
	    1], ldb, &y[*m + *p - *n + 1], &c__1, &c_b34, &d__[1], &c__1);

/*     Solve triangular system: R11*x = d1 */

    if (*m > 0) {
	strtrs_("Upper", "No Transpose", "Non unit", m, &c__1, &a[a_offset], 
		lda, &d__[1], m, info);

	if (*info > 0) {
	    *info = 2;
	    return 0;
	}

/*        Copy D to X */

	scopy_(m, &d__[1], &c__1, &x[1], &c__1);
    }

/*     Backward transformation y = Z'*y */

/* Computing MAX */
    i__1 = 1, i__2 = *n - *p + 1;
    i__3 = max(1,*p);
    i__4 = *lwork - *m - np;
    sormrq_("Left", "Transpose", p, &c__1, &np, &b[max(i__1, i__2)+ b_dim1], 
	    ldb, &work[*m + 1], &y[1], &i__3, &work[*m + np + 1], &i__4, info);
/* Computing MAX */
    i__1 = lopt, i__2 = (integer) work[*m + np + 1];
    work[1] = (real) (*m + np + max(i__1,i__2));

    return 0;

/*     End of SGGGLM */

} /* sggglm_ */
