/* zposvx.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublecomplex *a, integer *lda, doublecomplex *af, integer *
	ldaf, char *equed, doublereal *s, doublecomplex *b, integer *ldb, 
	doublecomplex *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublecomplex *work, doublereal *rwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, b_dim1, b_offset, x_dim1, 
	    x_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Local variables */
    integer i__, j;
    doublereal amax, smin, smax;
    extern logical lsame_(char *, char *);
    doublereal scond, anorm;
    logical equil, rcequ;
    extern doublereal dlamch_(char *);
    logical nofact;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;
    extern doublereal zlanhe_(char *, char *, integer *, doublecomplex *, 
	    integer *, doublereal *);
    extern /* Subroutine */ int zlaqhe_(char *, integer *, doublecomplex *, 
	    integer *, doublereal *, doublereal *, doublereal *, char *);
    integer infequ;
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), 
	    zpocon_(char *, integer *, doublecomplex *, integer *, doublereal 
	    *, doublereal *, doublecomplex *, doublereal *, integer *)
	    ;
    doublereal smlnum;
    extern /* Subroutine */ int zpoequ_(integer *, doublecomplex *, integer *, 
	     doublereal *, doublereal *, doublereal *, integer *), zporfs_(
	    char *, integer *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublereal *, doublereal *, 
	    doublecomplex *, doublereal *, integer *), zpotrf_(char *, 
	     integer *, doublecomplex *, integer *, integer *), 
	    zpotrs_(char *, integer *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, integer *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZPOSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to */
/*  compute the solution to a complex system of linear equations */
/*     A * X = B, */
/*  where A is an N-by-N Hermitian positive definite matrix and X and B */
/*  are N-by-NRHS matrices. */

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
/*        A = U**H* U,  if UPLO = 'U', or */
/*        A = L * L**H,  if UPLO = 'L', */
/*     where U is an upper triangular matrix and L is a lower triangular */
/*     matrix. */

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
/*          = 'F':  On entry, AF contains the factored form of A. */
/*                  If EQUED = 'Y', the matrix A has been equilibrated */
/*                  with scaling factors given by S.  A and AF will not */
/*                  be modified. */
/*          = 'N':  The matrix A will be copied to AF and factored. */
/*          = 'E':  The matrix A will be equilibrated if necessary, then */
/*                  copied to AF and factored. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The number of linear equations, i.e., the order of the */
/*          matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices B and X.  NRHS >= 0. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the Hermitian matrix A, except if FACT = 'F' and */
/*          EQUED = 'Y', then A must contain the equilibrated matrix */
/*          diag(S)*A*diag(S).  If UPLO = 'U', the leading */
/*          N-by-N upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading N-by-N lower triangular part of A contains the lower */
/*          triangular part of the matrix A, and the strictly upper */
/*          triangular part of A is not referenced.  A is not modified if */
/*          FACT = 'F' or 'N', or if FACT = 'E' and EQUED = 'N' on exit. */

/*          On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by */
/*          diag(S)*A*diag(S). */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  AF      (input or output) COMPLEX*16 array, dimension (LDAF,N) */
/*          If FACT = 'F', then AF is an input argument and on entry */
/*          contains the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H, in the same storage */
/*          format as A.  If EQUED .ne. 'N', then AF is the factored form */
/*          of the equilibrated matrix diag(S)*A*diag(S). */

/*          If FACT = 'N', then AF is an output argument and on exit */
/*          returns the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H of the original */
/*          matrix A. */

/*          If FACT = 'E', then AF is an output argument and on exit */
/*          returns the triangular factor U or L from the Cholesky */
/*          factorization A = U**H*U or A = L*L**H of the equilibrated */
/*          matrix A (see the description of A for the form of the */
/*          equilibrated matrix). */

/*  LDAF    (input) INTEGER */
/*          The leading dimension of the array AF.  LDAF >= max(1,N). */

/*  EQUED   (input or output) CHARACTER*1 */
/*          Specifies the form of equilibration that was done. */
/*          = 'N':  No equilibration (always true if FACT = 'N'). */
/*          = 'Y':  Equilibration was done, i.e., A has been replaced by */
/*                  diag(S) * A * diag(S). */
/*          EQUED is an input argument if FACT = 'F'; otherwise, it is an */
/*          output argument. */

/*  S       (input or output) DOUBLE PRECISION array, dimension (N) */
/*          The scale factors for A; not accessed if EQUED = 'N'.  S is */
/*          an input argument if FACT = 'F'; otherwise, S is an output */
/*          argument.  If FACT = 'F' and EQUED = 'Y', each element of S */
/*          must be positive. */

/*  B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS) */
/*          On entry, the N-by-NRHS righthand side matrix B. */
/*          On exit, if EQUED = 'N', B is not modified; if EQUED = 'Y', */
/*          B is overwritten by diag(S) * B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (output) COMPLEX*16 array, dimension (LDX,NRHS) */
/*          If INFO = 0 or INFO = N+1, the N-by-NRHS solution matrix X to */
/*          the original system of equations.  Note that if EQUED = 'Y', */
/*          A and B are modified on exit, and the solution to the */
/*          equilibrated system is inv(diag(S))*X. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(1,N). */

/*  RCOND   (output) DOUBLE PRECISION */
/*          The estimate of the reciprocal condition number of the matrix */
/*          A after equilibration (if done).  If RCOND is less than the */
/*          machine precision (in particular, if RCOND = 0), the matrix */
/*          is singular to working precision.  This condition is */
/*          indicated by a return code of INFO > 0. */

/*  FERR    (output) DOUBLE PRECISION array, dimension (NRHS) */
/*          The estimated forward error bound for each solution vector */
/*          X(j) (the j-th column of the solution matrix X). */
/*          If XTRUE is the true solution corresponding to X(j), FERR(j) */
/*          is an estimated upper bound for the magnitude of the largest */
/*          element in (X(j) - XTRUE) divided by the magnitude of the */
/*          largest element in X(j).  The estimate is as reliable as */
/*          the estimate for RCOND, and is almost always a slight */
/*          overestimate of the true error. */

/*  BERR    (output) DOUBLE PRECISION array, dimension (NRHS) */
/*          The componentwise relative backward error of each solution */
/*          vector X(j) (i.e., the smallest relative change in */
/*          any element of A or B that makes X(j) an exact solution). */

/*  WORK    (workspace) COMPLEX*16 array, dimension (2*N) */

/*  RWORK   (workspace) DOUBLE PRECISION array, dimension (N) */

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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    af_dim1 = *ldaf;
    af_offset = 1 + af_dim1;
    af -= af_offset;
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
    if (nofact || equil) {
	*(unsigned char *)equed = 'N';
	rcequ = FALSE_;
    } else {
	rcequ = lsame_(equed, "Y");
	smlnum = dlamch_("Safe minimum");
	bignum = 1. / smlnum;
    }

/*     Test the input parameters. */

    if (! nofact && ! equil && ! lsame_(fact, "F")) {
	*info = -1;
    } else if (! lsame_(uplo, "U") && ! lsame_(uplo, 
	    "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*lda < max(1,*n)) {
	*info = -6;
    } else if (*ldaf < max(1,*n)) {
	*info = -8;
    } else if (lsame_(fact, "F") && ! (rcequ || lsame_(
	    equed, "N"))) {
	*info = -9;
    } else {
	if (rcequ) {
	    smin = bignum;
	    smax = 0.;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		d__1 = smin, d__2 = s[j];
		smin = min(d__1,d__2);
/* Computing MAX */
		d__1 = smax, d__2 = s[j];
		smax = max(d__1,d__2);
/* L10: */
	    }
	    if (smin <= 0.) {
		*info = -10;
	    } else if (*n > 0) {
		scond = max(smin,smlnum) / min(smax,bignum);
	    } else {
		scond = 1.;
	    }
	}
	if (*info == 0) {
	    if (*ldb < max(1,*n)) {
		*info = -12;
	    } else if (*ldx < max(1,*n)) {
		*info = -14;
	    }
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZPOSVX", &i__1);
	return 0;
    }

    if (equil) {

/*        Compute row and column scalings to equilibrate the matrix A. */

	zpoequ_(n, &a[a_offset], lda, &s[1], &scond, &amax, &infequ);
	if (infequ == 0) {

/*           Equilibrate the matrix. */

	    zlaqhe_(uplo, n, &a[a_offset], lda, &s[1], &scond, &amax, equed);
	    rcequ = lsame_(equed, "Y");
	}
    }

/*     Scale the right hand side. */

    if (rcequ) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__;
		i__5 = i__ + j * b_dim1;
		z__1.r = s[i__4] * b[i__5].r, z__1.i = s[i__4] * b[i__5].i;
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L20: */
	    }
/* L30: */
	}
    }

    if (nofact || equil) {

/*        Compute the Cholesky factorization A = U'*U or A = L*L'. */

	zlacpy_(uplo, n, n, &a[a_offset], lda, &af[af_offset], ldaf);
	zpotrf_(uplo, n, &af[af_offset], ldaf, info);

/*        Return if INFO is non-zero. */

	if (*info > 0) {
	    *rcond = 0.;
	    return 0;
	}
    }

/*     Compute the norm of the matrix A. */

    anorm = zlanhe_("1", uplo, n, &a[a_offset], lda, &rwork[1]);

/*     Compute the reciprocal of the condition number of A. */

    zpocon_(uplo, n, &af[af_offset], ldaf, &anorm, rcond, &work[1], &rwork[1], 
	     info);

/*     Compute the solution matrix X. */

    zlacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    zpotrs_(uplo, n, nrhs, &af[af_offset], ldaf, &x[x_offset], ldx, info);

/*     Use iterative refinement to improve the computed solution and */
/*     compute error bounds and backward error estimates for it. */

    zporfs_(uplo, n, nrhs, &a[a_offset], lda, &af[af_offset], ldaf, &b[
	    b_offset], ldb, &x[x_offset], ldx, &ferr[1], &berr[1], &work[1], &
	    rwork[1], info);

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
		z__1.r = s[i__4] * x[i__5].r, z__1.i = s[i__4] * x[i__5].i;
		x[i__3].r = z__1.r, x[i__3].i = z__1.i;
/* L40: */
	    }
/* L50: */
	}
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    ferr[j] /= scond;
/* L60: */
	}
    }

/*     Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < dlamch_("Epsilon")) {
	*info = *n + 1;
    }

    return 0;

/*     End of ZPOSVX */

} /* zposvx_ */
