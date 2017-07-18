/* cgtrfs.f -- translated by f2c (version 20061008).
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
static real c_b18 = -1.f;
static real c_b19 = 1.f;
static complex c_b26 = {1.f,0.f};

/* Subroutine */ int cgtrfs_(char *trans, integer *n, integer *nrhs, complex *
	dl, complex *d__, complex *du, complex *dlf, complex *df, complex *
	duf, complex *du2, integer *ipiv, complex *b, integer *ldb, complex *
	x, integer *ldx, real *ferr, real *berr, complex *work, real *rwork, 
	integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1, i__2, i__3, i__4, i__5, 
	    i__6, i__7, i__8, i__9;
    real r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8, r__9, r__10, r__11, 
	    r__12, r__13, r__14;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    integer i__, j;
    real s;
    integer nz;
    real eps;
    integer kase;
    real safe1, safe2;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    integer count;
    extern /* Subroutine */ int clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *), clagtm_(char *, integer *, integer *, 
	    real *, complex *, complex *, complex *, complex *, integer *, 
	    real *, complex *, integer *);
    extern doublereal slamch_(char *);
    real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical notran;
    char transn[1];
    extern /* Subroutine */ int cgttrs_(char *, integer *, integer *, complex 
	    *, complex *, complex *, complex *, integer *, complex *, integer 
	    *, integer *);
    char transt[1];
    real lstres;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     Modified to call CLACN2 in place of CLACON, 10 Feb 03, SJH. */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGTRFS improves the computed solution to a system of linear */
/*  equations when the coefficient matrix is tridiagonal, and provides */
/*  error bounds and backward error estimates for the solution. */

/*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the form of the system of equations: */
/*          = 'N':  A * X = B     (No transpose) */
/*          = 'T':  A**T * X = B  (Transpose) */
/*          = 'C':  A**H * X = B  (Conjugate transpose) */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  DL      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) subdiagonal elements of A. */

/*  D       (input) COMPLEX array, dimension (N) */
/*          The diagonal elements of A. */

/*  DU      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) superdiagonal elements of A. */

/*  DLF     (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) multipliers that define the matrix L from the */
/*          LU factorization of A as computed by CGTTRF. */

/*  DF      (input) COMPLEX array, dimension (N) */
/*          The n diagonal elements of the upper triangular matrix U from */
/*          the LU factorization of A. */

/*  DUF     (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) elements of the first superdiagonal of U. */

/*  DU2     (input) COMPLEX array, dimension (N-2) */
/*          The (n-2) elements of the second superdiagonal of U. */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          The pivot indices; for 1 <= i <= n, row i of the matrix was */
/*          interchanged with row IPIV(i).  IPIV(i) will always be either */
/*          i or i+1; IPIV(i) = i indicates a row interchange was not */
/*          required. */

/*  B       (input) COMPLEX array, dimension (LDB,NRHS) */
/*          The right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (input/output) COMPLEX array, dimension (LDX,NRHS) */
/*          On entry, the solution matrix X, as computed by CGTTRS. */
/*          On exit, the improved solution matrix X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  FERR    (output) REAL array, dimension (NRHS) */
/*          The estimated forward error bound for each solution vector */
/*          X(j) (the j-th column of the solution matrix X). */
/*          If XTRUE is the true solution corresponding to X(j), FERR(j) */
/*          is an estimated upper bound for the magnitude of the largest */
/*          element in (X(j) - XTRUE) divided by the magnitude of the */
/*          largest element in X(j).  The estimate is as reliable as */
/*          the estimate for RCOND, and is almost always a slight */
/*          overestimate of the true error. */

/*  BERR    (output) REAL array, dimension (NRHS) */
/*          The componentwise relative backward error of each solution */
/*          vector X(j) (i.e., the smallest relative change in */
/*          any element of A or B that makes X(j) an exact solution). */

/*  WORK    (workspace) COMPLEX array, dimension (2*N) */

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
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    --dlf;
    --df;
    --duf;
    --du2;
    --ipiv;
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
    notran = lsame_(trans, "N");
    if (! notran && ! lsame_(trans, "T") && ! lsame_(
	    trans, "C")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*ldb < max(1,*n)) {
	*info = -13;
    } else if (*ldx < max(1,*n)) {
	*info = -15;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGTRFS", &i__1);
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

    if (notran) {
	*(unsigned char *)transn = 'N';
	*(unsigned char *)transt = 'C';
    } else {
	*(unsigned char *)transn = 'C';
	*(unsigned char *)transt = 'N';
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

/*        Compute residual R = B - op(A) * X, */
/*        where op(A) = A, A**T, or A**H, depending on TRANS. */

	ccopy_(n, &b[j * b_dim1 + 1], &c__1, &work[1], &c__1);
	clagtm_(trans, n, &c__1, &c_b18, &dl[1], &d__[1], &du[1], &x[j * 
		x_dim1 + 1], ldx, &c_b19, &work[1], n);

/*        Compute abs(op(A))*abs(x) + abs(b) for use in the backward */
/*        error bound. */

	if (notran) {
	    if (*n == 1) {
		i__2 = j * b_dim1 + 1;
		i__3 = j * x_dim1 + 1;
		rwork[1] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&b[
			j * b_dim1 + 1]), dabs(r__2)) + ((r__3 = d__[1].r, 
			dabs(r__3)) + (r__4 = r_imag(&d__[1]), dabs(r__4))) * 
			((r__5 = x[i__3].r, dabs(r__5)) + (r__6 = r_imag(&x[j 
			* x_dim1 + 1]), dabs(r__6)));
	    } else {
		i__2 = j * b_dim1 + 1;
		i__3 = j * x_dim1 + 1;
		i__4 = j * x_dim1 + 2;
		rwork[1] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&b[
			j * b_dim1 + 1]), dabs(r__2)) + ((r__3 = d__[1].r, 
			dabs(r__3)) + (r__4 = r_imag(&d__[1]), dabs(r__4))) * 
			((r__5 = x[i__3].r, dabs(r__5)) + (r__6 = r_imag(&x[j 
			* x_dim1 + 1]), dabs(r__6))) + ((r__7 = du[1].r, dabs(
			r__7)) + (r__8 = r_imag(&du[1]), dabs(r__8))) * ((
			r__9 = x[i__4].r, dabs(r__9)) + (r__10 = r_imag(&x[j *
			 x_dim1 + 2]), dabs(r__10)));
		i__2 = *n - 1;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ - 1;
		    i__5 = i__ - 1 + j * x_dim1;
		    i__6 = i__;
		    i__7 = i__ + j * x_dim1;
		    i__8 = i__;
		    i__9 = i__ + 1 + j * x_dim1;
		    rwork[i__] = (r__1 = b[i__3].r, dabs(r__1)) + (r__2 = 
			    r_imag(&b[i__ + j * b_dim1]), dabs(r__2)) + ((
			    r__3 = dl[i__4].r, dabs(r__3)) + (r__4 = r_imag(&
			    dl[i__ - 1]), dabs(r__4))) * ((r__5 = x[i__5].r, 
			    dabs(r__5)) + (r__6 = r_imag(&x[i__ - 1 + j * 
			    x_dim1]), dabs(r__6))) + ((r__7 = d__[i__6].r, 
			    dabs(r__7)) + (r__8 = r_imag(&d__[i__]), dabs(
			    r__8))) * ((r__9 = x[i__7].r, dabs(r__9)) + (
			    r__10 = r_imag(&x[i__ + j * x_dim1]), dabs(r__10))
			    ) + ((r__11 = du[i__8].r, dabs(r__11)) + (r__12 = 
			    r_imag(&du[i__]), dabs(r__12))) * ((r__13 = x[
			    i__9].r, dabs(r__13)) + (r__14 = r_imag(&x[i__ + 
			    1 + j * x_dim1]), dabs(r__14)));
/* L30: */
		}
		i__2 = *n + j * b_dim1;
		i__3 = *n - 1;
		i__4 = *n - 1 + j * x_dim1;
		i__5 = *n;
		i__6 = *n + j * x_dim1;
		rwork[*n] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&
			b[*n + j * b_dim1]), dabs(r__2)) + ((r__3 = dl[i__3]
			.r, dabs(r__3)) + (r__4 = r_imag(&dl[*n - 1]), dabs(
			r__4))) * ((r__5 = x[i__4].r, dabs(r__5)) + (r__6 = 
			r_imag(&x[*n - 1 + j * x_dim1]), dabs(r__6))) + ((
			r__7 = d__[i__5].r, dabs(r__7)) + (r__8 = r_imag(&d__[
			*n]), dabs(r__8))) * ((r__9 = x[i__6].r, dabs(r__9)) 
			+ (r__10 = r_imag(&x[*n + j * x_dim1]), dabs(r__10)));
	    }
	} else {
	    if (*n == 1) {
		i__2 = j * b_dim1 + 1;
		i__3 = j * x_dim1 + 1;
		rwork[1] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&b[
			j * b_dim1 + 1]), dabs(r__2)) + ((r__3 = d__[1].r, 
			dabs(r__3)) + (r__4 = r_imag(&d__[1]), dabs(r__4))) * 
			((r__5 = x[i__3].r, dabs(r__5)) + (r__6 = r_imag(&x[j 
			* x_dim1 + 1]), dabs(r__6)));
	    } else {
		i__2 = j * b_dim1 + 1;
		i__3 = j * x_dim1 + 1;
		i__4 = j * x_dim1 + 2;
		rwork[1] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&b[
			j * b_dim1 + 1]), dabs(r__2)) + ((r__3 = d__[1].r, 
			dabs(r__3)) + (r__4 = r_imag(&d__[1]), dabs(r__4))) * 
			((r__5 = x[i__3].r, dabs(r__5)) + (r__6 = r_imag(&x[j 
			* x_dim1 + 1]), dabs(r__6))) + ((r__7 = dl[1].r, dabs(
			r__7)) + (r__8 = r_imag(&dl[1]), dabs(r__8))) * ((
			r__9 = x[i__4].r, dabs(r__9)) + (r__10 = r_imag(&x[j *
			 x_dim1 + 2]), dabs(r__10)));
		i__2 = *n - 1;
		for (i__ = 2; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ - 1;
		    i__5 = i__ - 1 + j * x_dim1;
		    i__6 = i__;
		    i__7 = i__ + j * x_dim1;
		    i__8 = i__;
		    i__9 = i__ + 1 + j * x_dim1;
		    rwork[i__] = (r__1 = b[i__3].r, dabs(r__1)) + (r__2 = 
			    r_imag(&b[i__ + j * b_dim1]), dabs(r__2)) + ((
			    r__3 = du[i__4].r, dabs(r__3)) + (r__4 = r_imag(&
			    du[i__ - 1]), dabs(r__4))) * ((r__5 = x[i__5].r, 
			    dabs(r__5)) + (r__6 = r_imag(&x[i__ - 1 + j * 
			    x_dim1]), dabs(r__6))) + ((r__7 = d__[i__6].r, 
			    dabs(r__7)) + (r__8 = r_imag(&d__[i__]), dabs(
			    r__8))) * ((r__9 = x[i__7].r, dabs(r__9)) + (
			    r__10 = r_imag(&x[i__ + j * x_dim1]), dabs(r__10))
			    ) + ((r__11 = dl[i__8].r, dabs(r__11)) + (r__12 = 
			    r_imag(&dl[i__]), dabs(r__12))) * ((r__13 = x[
			    i__9].r, dabs(r__13)) + (r__14 = r_imag(&x[i__ + 
			    1 + j * x_dim1]), dabs(r__14)));
/* L40: */
		}
		i__2 = *n + j * b_dim1;
		i__3 = *n - 1;
		i__4 = *n - 1 + j * x_dim1;
		i__5 = *n;
		i__6 = *n + j * x_dim1;
		rwork[*n] = (r__1 = b[i__2].r, dabs(r__1)) + (r__2 = r_imag(&
			b[*n + j * b_dim1]), dabs(r__2)) + ((r__3 = du[i__3]
			.r, dabs(r__3)) + (r__4 = r_imag(&du[*n - 1]), dabs(
			r__4))) * ((r__5 = x[i__4].r, dabs(r__5)) + (r__6 = 
			r_imag(&x[*n - 1 + j * x_dim1]), dabs(r__6))) + ((
			r__7 = d__[i__5].r, dabs(r__7)) + (r__8 = r_imag(&d__[
			*n]), dabs(r__8))) * ((r__9 = x[i__6].r, dabs(r__9)) 
			+ (r__10 = r_imag(&x[*n + j * x_dim1]), dabs(r__10)));
	    }
	}

/*        Compute componentwise relative backward error from formula */

/*        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) ) */

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

	    cgttrs_(trans, n, &c__1, &dlf[1], &df[1], &duf[1], &du2[1], &ipiv[
		    1], &work[1], n, info);
	    caxpy_(n, &c_b26, &work[1], &c__1, &x[j * x_dim1 + 1], &c__1);
	    lstres = berr[j];
	    ++count;
	    goto L20;
	}

/*        Bound error from formula */

/*        norm(X - XTRUE) / norm(X) .le. FERR = */
/*        norm( abs(inv(op(A)))* */
/*           ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X) */

/*        where */
/*          norm(Z) is the magnitude of the largest component of Z */
/*          inv(op(A)) is the inverse of op(A) */
/*          abs(Z) is the componentwise absolute value of the matrix or */
/*             vector Z */
/*          NZ is the maximum number of nonzeros in any row of A, plus 1 */
/*          EPS is machine epsilon */

/*        The i-th component of abs(R)+NZ*EPS*(abs(op(A))*abs(X)+abs(B)) */
/*        is incremented by SAFE1 if the i-th component of */
/*        abs(op(A))*abs(X) + abs(B) is less than SAFE2. */

/*        Use CLACN2 to estimate the infinity-norm of the matrix */
/*           inv(op(A)) * diag(W), */
/*        where W = abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) */

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

	kase = 0;
L70:
	clacn2_(n, &work[*n + 1], &work[1], &ferr[j], &kase, isave);
	if (kase != 0) {
	    if (kase == 1) {

/*              Multiply by diag(W)*inv(op(A)**H). */

		cgttrs_(transt, n, &c__1, &dlf[1], &df[1], &duf[1], &du2[1], &
			ipiv[1], &work[1], n, info);
		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = i__;
		    i__4 = i__;
		    i__5 = i__;
		    q__1.r = rwork[i__4] * work[i__5].r, q__1.i = rwork[i__4] 
			    * work[i__5].i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
/* L80: */
		}
	    } else {

/*              Multiply by inv(op(A))*diag(W). */

		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = i__;
		    i__4 = i__;
		    i__5 = i__;
		    q__1.r = rwork[i__4] * work[i__5].r, q__1.i = rwork[i__4] 
			    * work[i__5].i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
/* L90: */
		}
		cgttrs_(transn, n, &c__1, &dlf[1], &df[1], &duf[1], &du2[1], &
			ipiv[1], &work[1], n, info);
	    }
	    goto L70;
	}

/*        Normalize error. */

	lstres = 0.f;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__3 = i__ + j * x_dim1;
	    r__3 = lstres, r__4 = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = 
		    r_imag(&x[i__ + j * x_dim1]), dabs(r__2));
	    lstres = dmax(r__3,r__4);
/* L100: */
	}
	if (lstres != 0.f) {
	    ferr[j] /= lstres;
	}

/* L110: */
    }

    return 0;

/*     End of CGTRFS */

} /* cgtrfs_ */
