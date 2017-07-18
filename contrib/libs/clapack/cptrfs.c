/* cptrfs.f -- translated by f2c (version 20061008).
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
static complex c_b16 = {1.f,0.f};

/* Subroutine */ int cptrfs_(char *uplo, integer *n, integer *nrhs, real *d__, 
	 complex *e, real *df, complex *ef, complex *b, integer *ldb, complex 
	*x, integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1, i__2, i__3, i__4, i__5, 
	    i__6;
    real r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8, r__9, r__10, r__11, 
	    r__12;
    complex q__1, q__2, q__3;

    /* Builtin functions */
    double r_imag(complex *);
    void r_cnjg(complex *, complex *);
    double c_abs(complex *);

    /* Local variables */
    integer i__, j;
    real s;
    complex bi, cx, dx, ex;
    integer ix, nz;
    real eps, safe1, safe2;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    integer count;
    logical upper;
    extern doublereal slamch_(char *);
    real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer isamax_(integer *, real *, integer *);
    real lstres;
    extern /* Subroutine */ int cpttrs_(char *, integer *, integer *, real *, 
	    complex *, complex *, integer *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPTRFS improves the computed solution to a system of linear */
/*  equations when the coefficient matrix is Hermitian positive definite */
/*  and tridiagonal, and provides error bounds and backward error */
/*  estimates for the solution. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the superdiagonal or the subdiagonal of the */
/*          tridiagonal matrix A is stored and the form of the */
/*          factorization: */
/*          = 'U':  E is the superdiagonal of A, and A = U**H*D*U; */
/*          = 'L':  E is the subdiagonal of A, and A = L*D*L**H. */
/*          (The two forms are equivalent if A is real.) */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  D       (input) REAL array, dimension (N) */
/*          The n real diagonal elements of the tridiagonal matrix A. */

/*  E       (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) off-diagonal elements of the tridiagonal matrix A */
/*          (see UPLO). */

/*  DF      (input) REAL array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from */
/*          the factorization computed by CPTTRF. */

/*  EF      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) off-diagonal elements of the unit bidiagonal */
/*          factor U or L from the factorization computed by CPTTRF */
/*          (see UPLO). */

/*  B       (input) COMPLEX array, dimension (LDB,NRHS) */
/*          The right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (input/output) COMPLEX array, dimension (LDX,NRHS) */
/*          On entry, the solution matrix X, as computed by CPTTRS. */
/*          On exit, the improved solution matrix X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  FERR    (output) REAL array, dimension (NRHS) */
/*          The forward error bound for each solution vector */
/*          X(j) (the j-th column of the solution matrix X). */
/*          If XTRUE is the true solution corresponding to X(j), FERR(j) */
/*          is an estimated upper bound for the magnitude of the largest */
/*          element in (X(j) - XTRUE) divided by the magnitude of the */
/*          largest element in X(j). */

/*  BERR    (output) REAL array, dimension (NRHS) */
/*          The componentwise relative backward error of each solution */
/*          vector X(j) (i.e., the smallest relative change in */
/*          any element of A or B that makes X(j) an exact solution). */

/*  WORK    (workspace) COMPLEX array, dimension (N) */

/*  RWORK   (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Internal Parameters */
/*  =================== */

/*  ITMAX is the maximum number of steps of iterative refinement. */

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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    --df;
    --ef;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --ferr;
    --berr;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*ldb < max(1,*n)) {
	*info = -9;
    } else if (*ldx < max(1,*n)) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPTRFS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    ferr[j] = 0.f;
	    berr[j] = 0.f;
/* L10: */
	}
	return 0;
    }

/*     NZ = maximum number of nonzero elements in each row of A, plus 1 */

    nz = 4;
    eps = slamch_("Epsilon");
    safmin = slamch_("Safe minimum");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

/*     Do for each right hand side */

    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {

	count = 1;
	lstres = 3.f;
L20:

/*        Loop until stopping criterion is satisfied. */

/*        Compute residual R = B - A * X.  Also compute */
/*        abs(A)*abs(x) + abs(b) for use in the backward error bound. */

	if (upper) {
	    if (*n == 1) {
		i__2 = j * b_dim1 + 1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		i__2 = j * x_dim1 + 1;
		q__1.r = d__[1] * x[i__2].r, q__1.i = d__[1] * x[i__2].i;
		dx.r = q__1.r, dx.i = q__1.i;
		q__1.r = bi.r - dx.r, q__1.i = bi.i - dx.i;
		work[1].r = q__1.r, work[1].i = q__1.i;
		rwork[1] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = dx.r, dabs(r__3)) + (r__4 = 
			r_imag(&dx), dabs(r__4)));
	    } else {
		i__2 = j * b_dim1 + 1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		i__2 = j * x_dim1 + 1;
		q__1.r = d__[1] * x[i__2].r, q__1.i = d__[1] * x[i__2].i;
		dx.r = q__1.r, dx.i = q__1.i;
		i__2 = j * x_dim1 + 2;
		q__1.r = e[1].r * x[i__2].r - e[1].i * x[i__2].i, q__1.i = e[
			1].r * x[i__2].i + e[1].i * x[i__2].r;
		ex.r = q__1.r, ex.i = q__1.i;
		q__2.r = bi.r - dx.r, q__2.i = bi.i - dx.i;
		q__1.r = q__2.r - ex.r, q__1.i = q__2.i - ex.i;
		work[1].r = q__1.r, work[1].i = q__1.i;
		i__2 = j * x_dim1 + 2;
		rwork[1] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = dx.r, dabs(r__3)) + (r__4 = 
			r_imag(&dx), dabs(r__4))) + ((r__5 = e[1].r, dabs(
			r__5)) + (r__6 = r_imag(&e[1]), dabs(r__6))) * ((r__7 
			= x[i__2].r, dabs(r__7)) + (r__8 = r_imag(&x[j * 
			x_dim1 + 2]), dabs(r__8)));
		i__2 = *n - 1;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    bi.r = b[i__3].r, bi.i = b[i__3].i;
		    r_cnjg(&q__2, &e[i__ - 1]);
		    i__3 = i__ - 1 + j * x_dim1;
		    q__1.r = q__2.r * x[i__3].r - q__2.i * x[i__3].i, q__1.i =
			     q__2.r * x[i__3].i + q__2.i * x[i__3].r;
		    cx.r = q__1.r, cx.i = q__1.i;
		    i__3 = i__;
		    i__4 = i__ + j * x_dim1;
		    q__1.r = d__[i__3] * x[i__4].r, q__1.i = d__[i__3] * x[
			    i__4].i;
		    dx.r = q__1.r, dx.i = q__1.i;
		    i__3 = i__;
		    i__4 = i__ + 1 + j * x_dim1;
		    q__1.r = e[i__3].r * x[i__4].r - e[i__3].i * x[i__4].i, 
			    q__1.i = e[i__3].r * x[i__4].i + e[i__3].i * x[
			    i__4].r;
		    ex.r = q__1.r, ex.i = q__1.i;
		    i__3 = i__;
		    q__3.r = bi.r - cx.r, q__3.i = bi.i - cx.i;
		    q__2.r = q__3.r - dx.r, q__2.i = q__3.i - dx.i;
		    q__1.r = q__2.r - ex.r, q__1.i = q__2.i - ex.i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
		    i__3 = i__ - 1;
		    i__4 = i__ - 1 + j * x_dim1;
		    i__5 = i__;
		    i__6 = i__ + 1 + j * x_dim1;
		    rwork[i__] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&
			    bi), dabs(r__2)) + ((r__3 = e[i__3].r, dabs(r__3))
			     + (r__4 = r_imag(&e[i__ - 1]), dabs(r__4))) * ((
			    r__5 = x[i__4].r, dabs(r__5)) + (r__6 = r_imag(&x[
			    i__ - 1 + j * x_dim1]), dabs(r__6))) + ((r__7 = 
			    dx.r, dabs(r__7)) + (r__8 = r_imag(&dx), dabs(
			    r__8))) + ((r__9 = e[i__5].r, dabs(r__9)) + (
			    r__10 = r_imag(&e[i__]), dabs(r__10))) * ((r__11 =
			     x[i__6].r, dabs(r__11)) + (r__12 = r_imag(&x[i__ 
			    + 1 + j * x_dim1]), dabs(r__12)));
/* L30: */
		}
		i__2 = *n + j * b_dim1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		r_cnjg(&q__2, &e[*n - 1]);
		i__2 = *n - 1 + j * x_dim1;
		q__1.r = q__2.r * x[i__2].r - q__2.i * x[i__2].i, q__1.i = 
			q__2.r * x[i__2].i + q__2.i * x[i__2].r;
		cx.r = q__1.r, cx.i = q__1.i;
		i__2 = *n;
		i__3 = *n + j * x_dim1;
		q__1.r = d__[i__2] * x[i__3].r, q__1.i = d__[i__2] * x[i__3]
			.i;
		dx.r = q__1.r, dx.i = q__1.i;
		i__2 = *n;
		q__2.r = bi.r - cx.r, q__2.i = bi.i - cx.i;
		q__1.r = q__2.r - dx.r, q__1.i = q__2.i - dx.i;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		i__2 = *n - 1;
		i__3 = *n - 1 + j * x_dim1;
		rwork[*n] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = e[i__2].r, dabs(r__3)) + (r__4 
			= r_imag(&e[*n - 1]), dabs(r__4))) * ((r__5 = x[i__3]
			.r, dabs(r__5)) + (r__6 = r_imag(&x[*n - 1 + j * 
			x_dim1]), dabs(r__6))) + ((r__7 = dx.r, dabs(r__7)) + 
			(r__8 = r_imag(&dx), dabs(r__8)));
	    }
	} else {
	    if (*n == 1) {
		i__2 = j * b_dim1 + 1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		i__2 = j * x_dim1 + 1;
		q__1.r = d__[1] * x[i__2].r, q__1.i = d__[1] * x[i__2].i;
		dx.r = q__1.r, dx.i = q__1.i;
		q__1.r = bi.r - dx.r, q__1.i = bi.i - dx.i;
		work[1].r = q__1.r, work[1].i = q__1.i;
		rwork[1] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = dx.r, dabs(r__3)) + (r__4 = 
			r_imag(&dx), dabs(r__4)));
	    } else {
		i__2 = j * b_dim1 + 1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		i__2 = j * x_dim1 + 1;
		q__1.r = d__[1] * x[i__2].r, q__1.i = d__[1] * x[i__2].i;
		dx.r = q__1.r, dx.i = q__1.i;
		r_cnjg(&q__2, &e[1]);
		i__2 = j * x_dim1 + 2;
		q__1.r = q__2.r * x[i__2].r - q__2.i * x[i__2].i, q__1.i = 
			q__2.r * x[i__2].i + q__2.i * x[i__2].r;
		ex.r = q__1.r, ex.i = q__1.i;
		q__2.r = bi.r - dx.r, q__2.i = bi.i - dx.i;
		q__1.r = q__2.r - ex.r, q__1.i = q__2.i - ex.i;
		work[1].r = q__1.r, work[1].i = q__1.i;
		i__2 = j * x_dim1 + 2;
		rwork[1] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = dx.r, dabs(r__3)) + (r__4 = 
			r_imag(&dx), dabs(r__4))) + ((r__5 = e[1].r, dabs(
			r__5)) + (r__6 = r_imag(&e[1]), dabs(r__6))) * ((r__7 
			= x[i__2].r, dabs(r__7)) + (r__8 = r_imag(&x[j * 
			x_dim1 + 2]), dabs(r__8)));
		i__2 = *n - 1;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    bi.r = b[i__3].r, bi.i = b[i__3].i;
		    i__3 = i__ - 1;
		    i__4 = i__ - 1 + j * x_dim1;
		    q__1.r = e[i__3].r * x[i__4].r - e[i__3].i * x[i__4].i, 
			    q__1.i = e[i__3].r * x[i__4].i + e[i__3].i * x[
			    i__4].r;
		    cx.r = q__1.r, cx.i = q__1.i;
		    i__3 = i__;
		    i__4 = i__ + j * x_dim1;
		    q__1.r = d__[i__3] * x[i__4].r, q__1.i = d__[i__3] * x[
			    i__4].i;
		    dx.r = q__1.r, dx.i = q__1.i;
		    r_cnjg(&q__2, &e[i__]);
		    i__3 = i__ + 1 + j * x_dim1;
		    q__1.r = q__2.r * x[i__3].r - q__2.i * x[i__3].i, q__1.i =
			     q__2.r * x[i__3].i + q__2.i * x[i__3].r;
		    ex.r = q__1.r, ex.i = q__1.i;
		    i__3 = i__;
		    q__3.r = bi.r - cx.r, q__3.i = bi.i - cx.i;
		    q__2.r = q__3.r - dx.r, q__2.i = q__3.i - dx.i;
		    q__1.r = q__2.r - ex.r, q__1.i = q__2.i - ex.i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
		    i__3 = i__ - 1;
		    i__4 = i__ - 1 + j * x_dim1;
		    i__5 = i__;
		    i__6 = i__ + 1 + j * x_dim1;
		    rwork[i__] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&
			    bi), dabs(r__2)) + ((r__3 = e[i__3].r, dabs(r__3))
			     + (r__4 = r_imag(&e[i__ - 1]), dabs(r__4))) * ((
			    r__5 = x[i__4].r, dabs(r__5)) + (r__6 = r_imag(&x[
			    i__ - 1 + j * x_dim1]), dabs(r__6))) + ((r__7 = 
			    dx.r, dabs(r__7)) + (r__8 = r_imag(&dx), dabs(
			    r__8))) + ((r__9 = e[i__5].r, dabs(r__9)) + (
			    r__10 = r_imag(&e[i__]), dabs(r__10))) * ((r__11 =
			     x[i__6].r, dabs(r__11)) + (r__12 = r_imag(&x[i__ 
			    + 1 + j * x_dim1]), dabs(r__12)));
/* L40: */
		}
		i__2 = *n + j * b_dim1;
		bi.r = b[i__2].r, bi.i = b[i__2].i;
		i__2 = *n - 1;
		i__3 = *n - 1 + j * x_dim1;
		q__1.r = e[i__2].r * x[i__3].r - e[i__2].i * x[i__3].i, 
			q__1.i = e[i__2].r * x[i__3].i + e[i__2].i * x[i__3]
			.r;
		cx.r = q__1.r, cx.i = q__1.i;
		i__2 = *n;
		i__3 = *n + j * x_dim1;
		q__1.r = d__[i__2] * x[i__3].r, q__1.i = d__[i__2] * x[i__3]
			.i;
		dx.r = q__1.r, dx.i = q__1.i;
		i__2 = *n;
		q__2.r = bi.r - cx.r, q__2.i = bi.i - cx.i;
		q__1.r = q__2.r - dx.r, q__1.i = q__2.i - dx.i;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		i__2 = *n - 1;
		i__3 = *n - 1 + j * x_dim1;
		rwork[*n] = (r__1 = bi.r, dabs(r__1)) + (r__2 = r_imag(&bi), 
			dabs(r__2)) + ((r__3 = e[i__2].r, dabs(r__3)) + (r__4 
			= r_imag(&e[*n - 1]), dabs(r__4))) * ((r__5 = x[i__3]
			.r, dabs(r__5)) + (r__6 = r_imag(&x[*n - 1 + j * 
			x_dim1]), dabs(r__6))) + ((r__7 = dx.r, dabs(r__7)) + 
			(r__8 = r_imag(&dx), dabs(r__8)));
	    }
	}

/*        Compute componentwise relative backward error from formula */

/*        max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) ) */

/*        where abs(Z) is the componentwise absolute value of the matrix */
/*        or vector Z.  If the i-th component of the denominator is less */
/*        than SAFE2, then SAFE1 is added to the i-th components of the */
/*        numerator and denominator before dividing. */

	s = 0.f;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (rwork[i__] > safe2) {
/* Computing MAX */
		i__3 = i__;
		r__3 = s, r__4 = ((r__1 = work[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2))) / rwork[i__];
		s = dmax(r__3,r__4);
	    } else {
/* Computing MAX */
		i__3 = i__;
		r__3 = s, r__4 = ((r__1 = work[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + safe1) / (rwork[i__]
			 + safe1);
		s = dmax(r__3,r__4);
	    }
/* L50: */
	}
	berr[j] = s;

/*        Test stopping criterion. Continue iterating if */
/*           1) The residual BERR(J) is larger than machine epsilon, and */
/*           2) BERR(J) decreased by at least a factor of 2 during the */
/*              last iteration, and */
/*           3) At most ITMAX iterations tried. */

	if (berr[j] > eps && berr[j] * 2.f <= lstres && count <= 5) {

/*           Update solution and try again. */

	    cpttrs_(uplo, n, &c__1, &df[1], &ef[1], &work[1], n, info);
	    caxpy_(n, &c_b16, &work[1], &c__1, &x[j * x_dim1 + 1], &c__1);
	    lstres = berr[j];
	    ++count;
	    goto L20;
	}

/*        Bound error from formula */

/*        norm(X - XTRUE) / norm(X) .le. FERR = */
/*        norm( abs(inv(A))* */
/*           ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X) */

/*        where */
/*          norm(Z) is the magnitude of the largest component of Z */
/*          inv(A) is the inverse of A */
/*          abs(Z) is the componentwise absolute value of the matrix or */
/*             vector Z */
/*          NZ is the maximum number of nonzeros in any row of A, plus 1 */
/*          EPS is machine epsilon */

/*        The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B)) */
/*        is incremented by SAFE1 if the i-th component of */
/*        abs(A)*abs(X) + abs(B) is less than SAFE2. */

	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (rwork[i__] > safe2) {
		i__3 = i__;
		rwork[i__] = (r__1 = work[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + nz * eps * rwork[
			i__];
	    } else {
		i__3 = i__;
		rwork[i__] = (r__1 = work[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + nz * eps * rwork[
			i__] + safe1;
	    }
/* L60: */
	}
	ix = isamax_(n, &rwork[1], &c__1);
	ferr[j] = rwork[ix];

/*        Estimate the norm of inv(A). */

/*        Solve M(A) * x = e, where M(A) = (m(i,j)) is given by */

/*           m(i,j) =  abs(A(i,j)), i = j, */
/*           m(i,j) = -abs(A(i,j)), i .ne. j, */

/*        and e = [ 1, 1, ..., 1 ]'.  Note M(A) = M(L)*D*M(L)'. */

/*        Solve M(L) * x = e. */

	rwork[1] = 1.f;
	i__2 = *n;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    rwork[i__] = rwork[i__ - 1] * c_abs(&ef[i__ - 1]) + 1.f;
/* L70: */
	}

/*        Solve D * M(L)' * x = b. */

	rwork[*n] /= df[*n];
	for (i__ = *n - 1; i__ >= 1; --i__) {
	    rwork[i__] = rwork[i__] / df[i__] + rwork[i__ + 1] * c_abs(&ef[
		    i__]);
/* L80: */
	}

/*        Compute norm(inv(A)) = max(x(i)), 1<=i<=n. */

	ix = isamax_(n, &rwork[1], &c__1);
	ferr[j] *= (r__1 = rwork[ix], dabs(r__1));

/*        Normalize error. */

	lstres = 0.f;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    r__1 = lstres, r__2 = c_abs(&x[i__ + j * x_dim1]);
	    lstres = dmax(r__1,r__2);
/* L90: */
	}
	if (lstres != 0.f) {
	    ferr[j] /= lstres;
	}

/* L100: */
    }

    return 0;

/*     End of CPTRFS */

} /* cptrfs_ */
