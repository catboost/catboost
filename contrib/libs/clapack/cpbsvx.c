/* cpbsvx.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, complex *ab, integer *ldab, complex *afb, integer *
	ldafb, char *equed, real *s, complex *b, integer *ldb, complex *x, 
	integer *ldx, real *rcond, real *ferr, real *berr, complex *work, 
	real *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, afb_dim1, afb_offset, b_dim1, b_offset, 
	    x_dim1, x_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2;
    complex q__1;

    /* Local variables */
    integer i__, j, j1, j2;
    real amax, smin, smax;
    extern logical lsame_(char *, char *);
    real scond, anorm;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *);
    logical equil, rcequ, upper;
    extern doublereal clanhb_(char *, char *, integer *, integer *, complex *, 
	     integer *, real *);
    extern /* Subroutine */ int claqhb_(char *, integer *, integer *, complex 
	    *, integer *, real *, real *, real *, char *), 
	    cpbcon_(char *, integer *, integer *, complex *, integer *, real *
, real *, complex *, real *, integer *);
    extern doublereal slamch_(char *);
    logical nofact;
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), xerbla_(char *, 
	    integer *), cpbequ_(char *, integer *, integer *, complex 
	    *, integer *, real *, real *, real *, integer *), cpbrfs_(
	    char *, integer *, integer *, integer *, complex *, integer *, 
	    complex *, integer *, complex *, integer *, complex *, integer *, 
	    real *, real *, complex *, real *, integer *);
    real bignum;
    extern /* Subroutine */ int cpbtrf_(char *, integer *, integer *, complex 
	    *, integer *, integer *);
    integer infequ;
    extern /* Subroutine */ int cpbtrs_(char *, integer *, integer *, integer 
	    *, complex *, integer *, complex *, integer *, integer *);
    real smlnum;


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPBSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to */
/*  compute the solution to a complex system of linear equations */
/*     A * X = B, */
/*  where A is an N-by-N Hermitian positive definite band matrix and X */
/*  and B are N-by-NRHS matrices. */

/*  Error bounds on the solution and a condition estimate are also */
/*  provided. */

/*  Description */
/*  =========== */

/*  The following steps are performed: */

/*  1. If FACT = 'E', real scaling factors are computed to equilibrate */
/*     the system: */
/*        diag(S) * A * diag(S) * inv(diag(S)) * X = diag(S) * B */
/*     Whether or not the system will be equilibrated depends on the */
/*     scaling of the matrix A, but if equilibration is used, A is */
/*     overwritten by diag(S)*A*diag(S) and B by diag(S)*B. */

/*  2. If FACT = 'N' or 'E', the Cholesky decomposition is used to */
/*     factor the matrix A (after equilibration if FACT = 'E') as */
/*        A = U**H * U,  if UPLO = 'U', or */
/*        A = L * L**H,  if UPLO = 'L', */
/*     where U is an upper triangular band matrix, and L is a lower */
/*     triangular band matrix. */

/*  3. If the leading i-by-i principal minor is not positive definite, */
/*     then the routine returns with INFO = i. Otherwise, the factored */
/*     form of A is used to estimate the condition number of the matrix */
/*     A.  If the reciprocal of the condition number is less than machine */
/*     precision, INFO = N+1 is returned as a warning, but the routine */
/*     still goes on to solve for X and compute error bounds as */
/*     described below. */

/*  4. The system of equations is solved for X using the factored form */
/*     of A. */

/*  5. Iterative refinement is applied to improve the computed solution */
/*     matrix and calculate error bounds and backward error estimates */
/*     for it. */

/*  6. If equilibration was used, the matrix X is premultiplied by */
/*     diag(S) so that it solves the original system before */
/*     equilibration. */

/*  Arguments */
/*  ========= */

/*  FACT    (input) CHARACTER*1 */
/*          Specifies whether or not the factored form of the matrix A is */
/*          supplied on entry, and if not, whether the matrix A should be */
/*          equilibrated before it is factored. */
/*          = 'F':  On entry, AFB contains the factored form of A. */
/*                  If EQUED = 'Y', the matrix A has been equilibrated */
/*                  with scaling factors given by S.  AB and AFB will not */
/*                  be modified. */
/*          = 'N':  The matrix A will be copied to AFB and factored. */
/*          = 'E':  The matrix A will be equilibrated if necessary, then */
/*                  copied to AFB and factored. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The number of linear equations, i.e., the order of the */
/*          matrix A.  N >= 0. */

/*  KD      (input) INTEGER */
/*          The number of superdiagonals of the matrix A if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'.  KD >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right-hand sides, i.e., the number of columns */
/*          of the matrices B and X.  NRHS >= 0. */

/*  AB      (input/output) COMPLEX array, dimension (LDAB,N) */
/*          On entry, the upper or lower triangle of the Hermitian band */
/*          matrix A, stored in the first KD+1 rows of the array, except */
/*          if FACT = 'F' and EQUED = 'Y', then A must contain the */
/*          equilibrated matrix diag(S)*A*diag(S).  The j-th column of A */
/*          is stored in the j-th column of the array AB as follows: */
/*          if UPLO = 'U', AB(KD+1+i-j,j) = A(i,j) for max(1,j-KD)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(N,j+KD). */
/*          See below for further details. */

/*          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by */
/*          diag(S)*A*diag(S). */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array A.  LDAB >= KD+1. */

/*  AFB     (input or output) COMPLEX array, dimension (LDAFB,N) */
/*          If FACT = 'F', then AFB is an input argument and on entry */
/*          contains the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H of the band matrix */
/*          A, in the same storage format as A (see AB).  If EQUED = 'Y', */
/*          then AFB is the factored form of the equilibrated matrix A. */

/*          If FACT = 'N', then AFB is an output argument and on exit */
/*          returns the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H. */

/*          If FACT = 'E', then AFB is an output argument and on exit */
/*          returns the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H of the equilibrated */
/*          matrix A (see the description of A for the form of the */
/*          equilibrated matrix). */

/*  LDAFB   (input) INTEGER */
/*          The leading dimension of the array AFB.  LDAFB >= KD+1. */

/*  EQUED   (input or output) CHARACTER*1 */
/*          Specifies the form of equilibration that was done. */
/*          = 'N':  No equilibration (always true if FACT = 'N'). */
/*          = 'Y':  Equilibration was done, i.e., A has been replaced by */
/*                  diag(S) * A * diag(S). */
/*          EQUED is an input argument if FACT = 'F'; otherwise, it is an */
/*          output argument. */

/*  S       (input or output) REAL array, dimension (N) */
/*          The scale factors for A; not accessed if EQUED = 'N'.  S is */
/*          an input argument if FACT = 'F'; otherwise, S is an output */
/*          argument.  If FACT = 'F' and EQUED = 'Y', each element of S */
/*          must be positive. */

/*  B       (input/output) COMPLEX array, dimension (LDB,NRHS) */
/*          On entry, the N-by-NRHS right hand side matrix B. */
/*          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y', */
/*          B is overwritten by diag(S) * B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (output) COMPLEX array, dimension (LDX,NRHS) */
/*          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to */
/*          the original system of equations.  Note that if EQUED = 'Y', */
/*          A and B are modified on exit, and the solution to the */
/*          equilibrated system is inv(diag(S))*X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  RCOND   (output) REAL */
/*          The estimate of the reciprocal condition number of the matrix */
/*          A after equilibration (if done).  If RCOND is less than the */
/*          machine precision (in particular, if RCOND = 0), the matrix */
/*          is singular to working precision.  This condition is */
/*          indicated by a return code of INFO > 0. */

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
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */
/*          > 0: if INFO = i, and i is */
/*                <= N:  the leading minor of order i of A is */
/*                       not positive definite, so the factorization */
/*                       could not be completed, and the solution has not */
/*                       been computed. RCOND = 0 is returned. */
/*                = N+1: U is nonsingular, but RCOND is less than machine */
/*                       precision, meaning that the matrix is singular */
/*                       to working precision.  Nevertheless, the */
/*                       solution and error bounds are computed because */
/*                       there are a number of situations where the */
/*                       computed solution can be more accurate than the */
/*                       value of RCOND would suggest. */

/*  Further Details */
/*  =============== */

/*  The band storage scheme is illustrated by the following example, when */
/*  N = 6, KD = 2, and UPLO = 'U': */

/*  Two-dimensional storage of the Hermitian matrix A: */

/*     a11  a12  a13 */
/*          a22  a23  a24 */
/*               a33  a34  a35 */
/*                    a44  a45  a46 */
/*                         a55  a56 */
/*     (aij=conjg(aji))         a66 */

/*  Band storage of the upper triangle of A: */

/*      *    *   a13  a24  a35  a46 */
/*      *   a12  a23  a34  a45  a56 */
/*     a11  a22  a33  a44  a55  a66 */

/*  Similarly, if UPLO = 'L' the format of A is as follows: */

/*     a11  a22  a33  a44  a55  a66 */
/*     a21  a32  a43  a54  a65   * */
/*     a31  a42  a53  a64   *    * */

/*  Array elements marked * are not used by the routine. */

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

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    afb_dim1 = *ldafb;
    afb_offset = 1 + afb_dim1;
    afb -= afb_offset;
    --s;
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
    nofact = lsame_(fact, "N");
    equil = lsame_(fact, "E");
    upper = lsame_(uplo, "U");
    if (nofact || equil) {
	*(unsigned char *)equed = 'N';
	rcequ = FALSE_;
    } else {
	rcequ = lsame_(equed, "Y");
	smlnum = slamch_("Safe minimum");
	bignum = 1.f / smlnum;
    }

/*     Test the input parameters. */

    if (! nofact && ! equil && ! lsame_(fact, "F")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*kd < 0) {
	*info = -4;
    } else if (*nrhs < 0) {
	*info = -5;
    } else if (*ldab < *kd + 1) {
	*info = -7;
    } else if (*ldafb < *kd + 1) {
	*info = -9;
    } else if (lsame_(fact, "F") && ! (rcequ || lsame_(
	    equed, "N"))) {
	*info = -10;
    } else {
	if (rcequ) {
	    smin = bignum;
	    smax = 0.f;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		r__1 = smin, r__2 = s[j];
		smin = dmin(r__1,r__2);
/* Computing MAX */
		r__1 = smax, r__2 = s[j];
		smax = dmax(r__1,r__2);
/* L10: */
	    }
	    if (smin <= 0.f) {
		*info = -11;
	    } else if (*n > 0) {
		scond = dmax(smin,smlnum) / dmin(smax,bignum);
	    } else {
		scond = 1.f;
	    }
	}
	if (*info == 0) {
	    if (*ldb < max(1,*n)) {
		*info = -13;
	    } else if (*ldx < max(1,*n)) {
		*info = -15;
	    }
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPBSVX", &i__1);
	return 0;
    }

    if (equil) {

/*        Compute row and column scalings to equilibrate the matrix A. */

	cpbequ_(uplo, n, kd, &ab[ab_offset], ldab, &s[1], &scond, &amax, &
		infequ);
	if (infequ == 0) {

/*           Equilibrate the matrix. */

	    claqhb_(uplo, n, kd, &ab[ab_offset], ldab, &s[1], &scond, &amax, 
		    equed);
	    rcequ = lsame_(equed, "Y");
	}
    }

/*     Scale the right-hand side. */

    if (rcequ) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__;
		i__5 = i__ + j * b_dim1;
		q__1.r = s[i__4] * b[i__5].r, q__1.i = s[i__4] * b[i__5].i;
		b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L20: */
	    }
/* L30: */
	}
    }

    if (nofact || equil) {

/*        Compute the Cholesky factorization A = U'*U or A = L*L'. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
		i__2 = j - *kd;
		j1 = max(i__2,1);
		i__2 = j - j1 + 1;
		ccopy_(&i__2, &ab[*kd + 1 - j + j1 + j * ab_dim1], &c__1, &
			afb[*kd + 1 - j + j1 + j * afb_dim1], &c__1);
/* L40: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		i__2 = j + *kd;
		j2 = min(i__2,*n);
		i__2 = j2 - j + 1;
		ccopy_(&i__2, &ab[j * ab_dim1 + 1], &c__1, &afb[j * afb_dim1 
			+ 1], &c__1);
/* L50: */
	    }
	}

	cpbtrf_(uplo, n, kd, &afb[afb_offset], ldafb, info);

/*        Return if INFO is non-zero. */

	if (*info > 0) {
	    *rcond = 0.f;
	    return 0;
	}
    }

/*     Compute the norm of the matrix A. */

    anorm = clanhb_("1", uplo, n, kd, &ab[ab_offset], ldab, &rwork[1]);

/*     Compute the reciprocal of the condition number of A. */

    cpbcon_(uplo, n, kd, &afb[afb_offset], ldafb, &anorm, rcond, &work[1], &
	    rwork[1], info);

/*     Compute the solution matrix X. */

    clacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    cpbtrs_(uplo, n, kd, nrhs, &afb[afb_offset], ldafb, &x[x_offset], ldx, 
	    info);

/*     Use iterative refinement to improve the computed solution and */
/*     compute error bounds and backward error estimates for it. */

    cpbrfs_(uplo, n, kd, nrhs, &ab[ab_offset], ldab, &afb[afb_offset], ldafb, 
	    &b[b_offset], ldb, &x[x_offset], ldx, &ferr[1], &berr[1], &work[1]
, &rwork[1], info);

/*     Transform the solution matrix X to a solution of the original */
/*     system. */

    if (rcequ) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * x_dim1;
		i__4 = i__;
		i__5 = i__ + j * x_dim1;
		q__1.r = s[i__4] * x[i__5].r, q__1.i = s[i__4] * x[i__5].i;
		x[i__3].r = q__1.r, x[i__3].i = q__1.i;
/* L60: */
	    }
/* L70: */
	}
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    ferr[j] /= scond;
/* L80: */
	}
    }

/*     Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < slamch_("Epsilon")) {
	*info = *n + 1;
    }

    return 0;

/*     End of CPBSVX */

} /* cpbsvx_ */
