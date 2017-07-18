/* sposvxx.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int sposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, char *equed, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, b_dim1, b_offset, x_dim1, 
	    x_offset, err_bnds_norm_dim1, err_bnds_norm_offset, 
	    err_bnds_comp_dim1, err_bnds_comp_offset, i__1;
    real r__1, r__2;

    /* Local variables */
    integer j;
    real amax, smin, smax;
    extern doublereal sla_porpvgrw__(char *, integer *, real *, integer *, 
	    real *, integer *, real *, ftnlen);
    extern logical lsame_(char *, char *);
    real scond;
    logical equil, rcequ;
    extern doublereal slamch_(char *);
    logical nofact;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    real bignum;
    integer infequ;
    extern /* Subroutine */ int slacpy_(char *, integer *, integer *, real *, 
	    integer *, real *, integer *);
    real smlnum;
    extern /* Subroutine */ int slaqsy_(char *, integer *, real *, integer *, 
	    real *, real *, real *, char *), spotrf_(char *, 
	    integer *, real *, integer *, integer *), spotrs_(char *, 
	    integer *, integer *, real *, integer *, real *, integer *, 
	    integer *), slascl2_(integer *, integer *, real *, real *, 
	     integer *), spoequb_(integer *, real *, integer *, real *, real *
, real *, integer *), sporfsx_(char *, char *, integer *, integer 
	    *, real *, integer *, real *, integer *, real *, real *, integer *
, real *, integer *, real *, real *, integer *, real *, real *, 
	    integer *, real *, real *, integer *, integer *);


/*     -- LAPACK driver routine (version 3.2)                          -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- November 2008                                                -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     Purpose */
/*     ======= */

/*     SPOSVXX uses the Cholesky factorization A = U**T*U or A = L*L**T */
/*     to compute the solution to a real system of linear equations */
/*     A * X = B, where A is an N-by-N symmetric positive definite matrix */
/*     and X and B are N-by-NRHS matrices. */

/*     If requested, both normwise and maximum componentwise error bounds */
/*     are returned. SPOSVXX will return a solution with a tiny */
/*     guaranteed error (O(eps) where eps is the working machine */
/*     precision) unless the matrix is very ill-conditioned, in which */
/*     case a warning is returned. Relevant condition numbers also are */
/*     calculated and returned. */

/*     SPOSVXX accepts user-provided factorizations and equilibration */
/*     factors; see the definitions of the FACT and EQUED options. */
/*     Solving with refinement and using a factorization from a previous */
/*     SPOSVXX call will also produce a solution with either O(eps) */
/*     errors or warnings, but we cannot make that claim for general */
/*     user-provided factorizations and equilibration factors if they */
/*     differ from what SPOSVXX would itself produce. */

/*     Description */
/*     =========== */

/*     The following steps are performed: */

/*     1. If FACT = 'E', real scaling factors are computed to equilibrate */
/*     the system: */

/*       diag(S)*A*diag(S)     *inv(diag(S))*X = diag(S)*B */

/*     Whether or not the system will be equilibrated depends on the */
/*     scaling of the matrix A, but if equilibration is used, A is */
/*     overwritten by diag(S)*A*diag(S) and B by diag(S)*B. */

/*     2. If FACT = 'N' or 'E', the Cholesky decomposition is used to */
/*     factor the matrix A (after equilibration if FACT = 'E') as */
/*        A = U**T* U,  if UPLO = 'U', or */
/*        A = L * L**T,  if UPLO = 'L', */
/*     where U is an upper triangular matrix and L is a lower triangular */
/*     matrix. */

/*     3. If the leading i-by-i principal minor is not positive definite, */
/*     then the routine returns with INFO = i. Otherwise, the factored */
/*     form of A is used to estimate the condition number of the matrix */
/*     A (see argument RCOND).  If the reciprocal of the condition number */
/*     is less than machine precision, the routine still goes on to solve */
/*     for X and compute error bounds as described below. */

/*     4. The system of equations is solved for X using the factored form */
/*     of A. */

/*     5. By default (unless PARAMS(LA_LINRX_ITREF_I) is set to zero), */
/*     the routine will use iterative refinement to try to get a small */
/*     error and error bounds.  Refinement calculates the residual to at */
/*     least twice the working precision. */

/*     6. If equilibration was used, the matrix X is premultiplied by */
/*     diag(S) so that it solves the original system before */
/*     equilibration. */

/*     Arguments */
/*     ========= */

/*     Some optional parameters are bundled in the PARAMS array.  These */
/*     settings determine how refinement is performed, but often the */
/*     defaults are acceptable.  If the defaults are acceptable, users */
/*     can pass NPARAMS = 0 which prevents the source code from accessing */
/*     the PARAMS argument. */

/*     FACT    (input) CHARACTER*1 */
/*     Specifies whether or not the factored form of the matrix A is */
/*     supplied on entry, and if not, whether the matrix A should be */
/*     equilibrated before it is factored. */
/*       = 'F':  On entry, AF contains the factored form of A. */
/*               If EQUED is not 'N', the matrix A has been */
/*               equilibrated with scaling factors given by S. */
/*               A and AF are not modified. */
/*       = 'N':  The matrix A will be copied to AF and factored. */
/*       = 'E':  The matrix A will be equilibrated if necessary, then */
/*               copied to AF and factored. */

/*     UPLO    (input) CHARACTER*1 */
/*       = 'U':  Upper triangle of A is stored; */
/*       = 'L':  Lower triangle of A is stored. */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     NRHS    (input) INTEGER */
/*     The number of right hand sides, i.e., the number of columns */
/*     of the matrices B and X.  NRHS >= 0. */

/*     A       (input/output) REAL array, dimension (LDA,N) */
/*     On entry, the symmetric matrix A, except if FACT = 'F' and EQUED = */
/*     'Y', then A must contain the equilibrated matrix */
/*     diag(S)*A*diag(S).  If UPLO = 'U', the leading N-by-N upper */
/*     triangular part of A contains the upper triangular part of the */
/*     matrix A, and the strictly lower triangular part of A is not */
/*     referenced.  If UPLO = 'L', the leading N-by-N lower triangular */
/*     part of A contains the lower triangular part of the matrix A, and */
/*     the strictly upper triangular part of A is not referenced.  A is */
/*     not modified if FACT = 'F' or 'N', or if FACT = 'E' and EQUED = */
/*     'N' on exit. */

/*     On exit, if FACT = 'E' and EQUED = 'Y', A is overwritten by */
/*     diag(S)*A*diag(S). */

/*     LDA     (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF      (input or output) REAL array, dimension (LDAF,N) */
/*     If FACT = 'F', then AF is an input argument and on entry */
/*     contains the triangular factor U or L from the Cholesky */
/*     factorization A = U**T*U or A = L*L**T, in the same storage */
/*     format as A.  If EQUED .ne. 'N', then AF is the factored */
/*     form of the equilibrated matrix diag(S)*A*diag(S). */

/*     If FACT = 'N', then AF is an output argument and on exit */
/*     returns the triangular factor U or L from the Cholesky */
/*     factorization A = U**T*U or A = L*L**T of the original */
/*     matrix A. */

/*     If FACT = 'E', then AF is an output argument and on exit */
/*     returns the triangular factor U or L from the Cholesky */
/*     factorization A = U**T*U or A = L*L**T of the equilibrated */
/*     matrix A (see the description of A for the form of the */
/*     equilibrated matrix). */

/*     LDAF    (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     EQUED   (input or output) CHARACTER*1 */
/*     Specifies the form of equilibration that was done. */
/*       = 'N':  No equilibration (always true if FACT = 'N'). */
/*       = 'Y':  Both row and column equilibration, i.e., A has been */
/*               replaced by diag(S) * A * diag(S). */
/*     EQUED is an input argument if FACT = 'F'; otherwise, it is an */
/*     output argument. */

/*     S       (input or output) REAL array, dimension (N) */
/*     The row scale factors for A.  If EQUED = 'Y', A is multiplied on */
/*     the left and right by diag(S).  S is an input argument if FACT = */
/*     'F'; otherwise, S is an output argument.  If FACT = 'F' and EQUED */
/*     = 'Y', each element of S must be positive.  If S is output, each */
/*     element of S is a power of the radix. If S is input, each element */
/*     of S should be a power of the radix to ensure a reliable solution */
/*     and error estimates. Scaling by powers of the radix does not cause */
/*     rounding errors unless the result underflows or overflows. */
/*     Rounding errors during scaling lead to refining with a matrix that */
/*     is not equivalent to the input matrix, producing error estimates */
/*     that may not be reliable. */

/*     B       (input/output) REAL array, dimension (LDB,NRHS) */
/*     On entry, the N-by-NRHS right hand side matrix B. */
/*     On exit, */
/*     if EQUED = 'N', B is not modified; */
/*     if EQUED = 'Y', B is overwritten by diag(S)*B; */

/*     LDB     (input) INTEGER */
/*     The leading dimension of the array B.  LDB >= max(1,N). */

/*     X       (output) REAL array, dimension (LDX,NRHS) */
/*     If INFO = 0, the N-by-NRHS solution matrix X to the original */
/*     system of equations.  Note that A and B are modified on exit if */
/*     EQUED .ne. 'N', and the solution to the equilibrated system is */
/*     inv(diag(S))*X. */

/*     LDX     (input) INTEGER */
/*     The leading dimension of the array X.  LDX >= max(1,N). */

/*     RCOND   (output) REAL */
/*     Reciprocal scaled condition number.  This is an estimate of the */
/*     reciprocal Skeel condition number of the matrix A after */
/*     equilibration (if done).  If this is less than the machine */
/*     precision (in particular, if it is zero), the matrix is singular */
/*     to working precision.  Note that the error may still be small even */
/*     if this number is very small and the matrix appears ill- */
/*     conditioned. */

/*     RPVGRW  (output) REAL */
/*     Reciprocal pivot growth.  On exit, this contains the reciprocal */
/*     pivot growth factor norm(A)/norm(U). The "max absolute element" */
/*     norm is used.  If this is much less than 1, then the stability of */
/*     the LU factorization of the (equilibrated) matrix A could be poor. */
/*     This also means that the solution X, estimated condition numbers, */
/*     and error bounds could be unreliable. If factorization fails with */
/*     0<INFO<=N, then this contains the reciprocal pivot growth factor */
/*     for the leading INFO columns of A. */

/*     BERR    (output) REAL array, dimension (NRHS) */
/*     Componentwise relative backward error.  This is the */
/*     componentwise relative backward error of each solution vector X(j) */
/*     (i.e., the smallest relative change in any element of A or B that */
/*     makes X(j) an exact solution). */

/*     N_ERR_BNDS (input) INTEGER */
/*     Number of error bounds to return for each right hand side */
/*     and each type (normwise or componentwise).  See ERR_BNDS_NORM and */
/*     ERR_BNDS_COMP below. */

/*     ERR_BNDS_NORM  (output) REAL array, dimension (NRHS, N_ERR_BNDS) */
/*     For each right-hand side, this array contains information about */
/*     various error bounds and condition numbers corresponding to the */
/*     normwise relative error, which is defined as follows: */

/*     Normwise relative error in the ith solution vector: */
/*             max_j (abs(XTRUE(j,i) - X(j,i))) */
/*            ------------------------------ */
/*                  max_j abs(X(j,i)) */

/*     The array is indexed by the type of error information as described */
/*     below. There currently are up to three pieces of information */
/*     returned. */

/*     The first index in ERR_BNDS_NORM(i,:) corresponds to the ith */
/*     right-hand side. */

/*     The second index in ERR_BNDS_NORM(:,err) contains the following */
/*     three fields: */
/*     err = 1 "Trust/don't trust" boolean. Trust the answer if the */
/*              reciprocal condition number is less than the threshold */
/*              sqrt(n) * slamch('Epsilon'). */

/*     err = 2 "Guaranteed" error bound: The estimated forward error, */
/*              almost certainly within a factor of 10 of the true error */
/*              so long as the next entry is greater than the threshold */
/*              sqrt(n) * slamch('Epsilon'). This error bound should only */
/*              be trusted if the previous boolean is true. */

/*     err = 3  Reciprocal condition number: Estimated normwise */
/*              reciprocal condition number.  Compared with the threshold */
/*              sqrt(n) * slamch('Epsilon') to determine if the error */
/*              estimate is "guaranteed". These reciprocal condition */
/*              numbers are 1 / (norm(Z^{-1},inf) * norm(Z,inf)) for some */
/*              appropriately scaled matrix Z. */
/*              Let Z = S*A, where S scales each row by a power of the */
/*              radix so all absolute row sums of Z are approximately 1. */

/*     See Lapack Working Note 165 for further details and extra */
/*     cautions. */

/*     ERR_BNDS_COMP  (output) REAL array, dimension (NRHS, N_ERR_BNDS) */
/*     For each right-hand side, this array contains information about */
/*     various error bounds and condition numbers corresponding to the */
/*     componentwise relative error, which is defined as follows: */

/*     Componentwise relative error in the ith solution vector: */
/*                    abs(XTRUE(j,i) - X(j,i)) */
/*             max_j ---------------------- */
/*                         abs(X(j,i)) */

/*     The array is indexed by the right-hand side i (on which the */
/*     componentwise relative error depends), and the type of error */
/*     information as described below. There currently are up to three */
/*     pieces of information returned for each right-hand side. If */
/*     componentwise accuracy is not requested (PARAMS(3) = 0.0), then */
/*     ERR_BNDS_COMP is not accessed.  If N_ERR_BNDS .LT. 3, then at most */
/*     the first (:,N_ERR_BNDS) entries are returned. */

/*     The first index in ERR_BNDS_COMP(i,:) corresponds to the ith */
/*     right-hand side. */

/*     The second index in ERR_BNDS_COMP(:,err) contains the following */
/*     three fields: */
/*     err = 1 "Trust/don't trust" boolean. Trust the answer if the */
/*              reciprocal condition number is less than the threshold */
/*              sqrt(n) * slamch('Epsilon'). */

/*     err = 2 "Guaranteed" error bound: The estimated forward error, */
/*              almost certainly within a factor of 10 of the true error */
/*              so long as the next entry is greater than the threshold */
/*              sqrt(n) * slamch('Epsilon'). This error bound should only */
/*              be trusted if the previous boolean is true. */

/*     err = 3  Reciprocal condition number: Estimated componentwise */
/*              reciprocal condition number.  Compared with the threshold */
/*              sqrt(n) * slamch('Epsilon') to determine if the error */
/*              estimate is "guaranteed". These reciprocal condition */
/*              numbers are 1 / (norm(Z^{-1},inf) * norm(Z,inf)) for some */
/*              appropriately scaled matrix Z. */
/*              Let Z = S*(A*diag(x)), where x is the solution for the */
/*              current right-hand side and S scales each row of */
/*              A*diag(x) by a power of the radix so all absolute row */
/*              sums of Z are approximately 1. */

/*     See Lapack Working Note 165 for further details and extra */
/*     cautions. */

/*     NPARAMS (input) INTEGER */
/*     Specifies the number of parameters set in PARAMS.  If .LE. 0, the */
/*     PARAMS array is never referenced and default values are used. */

/*     PARAMS  (input / output) REAL array, dimension NPARAMS */
/*     Specifies algorithm parameters.  If an entry is .LT. 0.0, then */
/*     that entry will be filled with default value used for that */
/*     parameter.  Only positions up to NPARAMS are accessed; defaults */
/*     are used for higher-numbered parameters. */

/*       PARAMS(LA_LINRX_ITREF_I = 1) : Whether to perform iterative */
/*            refinement or not. */
/*         Default: 1.0 */
/*            = 0.0 : No refinement is performed, and no error bounds are */
/*                    computed. */
/*            = 1.0 : Use the double-precision refinement algorithm, */
/*                    possibly with doubled-single computations if the */
/*                    compilation environment does not support DOUBLE */
/*                    PRECISION. */
/*              (other values are reserved for future use) */

/*       PARAMS(LA_LINRX_ITHRESH_I = 2) : Maximum number of residual */
/*            computations allowed for refinement. */
/*         Default: 10 */
/*         Aggressive: Set to 100 to permit convergence using approximate */
/*                     factorizations or factorizations other than LU. If */
/*                     the factorization uses a technique other than */
/*                     Gaussian elimination, the guarantees in */
/*                     err_bnds_norm and err_bnds_comp may no longer be */
/*                     trustworthy. */

/*       PARAMS(LA_LINRX_CWISE_I = 3) : Flag determining if the code */
/*            will attempt to find a solution with small componentwise */
/*            relative error in the double-precision algorithm.  Positive */
/*            is true, 0.0 is false. */
/*         Default: 1.0 (attempt componentwise convergence) */

/*     WORK    (workspace) REAL array, dimension (4*N) */

/*     IWORK   (workspace) INTEGER array, dimension (N) */

/*     INFO    (output) INTEGER */
/*       = 0:  Successful exit. The solution to every right-hand side is */
/*         guaranteed. */
/*       < 0:  If INFO = -i, the i-th argument had an illegal value */
/*       > 0 and <= N:  U(INFO,INFO) is exactly zero.  The factorization */
/*         has been completed, but the factor U is exactly singular, so */
/*         the solution and error bounds could not be computed. RCOND = 0 */
/*         is returned. */
/*       = N+J: The solution corresponding to the Jth right-hand side is */
/*         not guaranteed. The solutions corresponding to other right- */
/*         hand sides K with K > J may not be guaranteed as well, but */
/*         only the first such right-hand side is reported. If a small */
/*         componentwise error is not requested (PARAMS(3) = 0.0) then */
/*         the Jth right-hand side is the first with a normwise error */
/*         bound that is not guaranteed (the smallest J such */
/*         that ERR_BNDS_NORM(J,1) = 0.0). By default (PARAMS(3) = 1.0) */
/*         the Jth right-hand side is the first with either a normwise or */
/*         componentwise error bound that is not guaranteed (the smallest */
/*         J such that either ERR_BNDS_NORM(J,1) = 0.0 or */
/*         ERR_BNDS_COMP(J,1) = 0.0). See the definition of */
/*         ERR_BNDS_NORM(:,1) and ERR_BNDS_COMP(:,1). To get information */
/*         about all of the right-hand sides check ERR_BNDS_NORM or */
/*         ERR_BNDS_COMP. */

/*     ================================================================== */

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
    err_bnds_comp_dim1 = *nrhs;
    err_bnds_comp_offset = 1 + err_bnds_comp_dim1;
    err_bnds_comp__ -= err_bnds_comp_offset;
    err_bnds_norm_dim1 = *nrhs;
    err_bnds_norm_offset = 1 + err_bnds_norm_dim1;
    err_bnds_norm__ -= err_bnds_norm_offset;
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
    --berr;
    --params;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    nofact = lsame_(fact, "N");
    equil = lsame_(fact, "E");
    smlnum = slamch_("Safe minimum");
    bignum = 1.f / smlnum;
    if (nofact || equil) {
	*(unsigned char *)equed = 'N';
	rcequ = FALSE_;
    } else {
	rcequ = lsame_(equed, "Y");
    }

/*     Default is failure.  If an input parameter is wrong or */
/*     factorization fails, make everything look horrible.  Only the */
/*     pivot growth is set here, the rest is initialized in SPORFSX. */

    *rpvgrw = 0.f;

/*     Test the input parameters.  PARAMS is not tested until SPORFSX. */

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
		*info = -10;
	    } else if (*n > 0) {
		scond = dmax(smin,smlnum) / dmin(smax,bignum);
	    } else {
		scond = 1.f;
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
	xerbla_("SPOSVXX", &i__1);
	return 0;
    }

    if (equil) {

/*     Compute row and column scalings to equilibrate the matrix A. */

	spoequb_(n, &a[a_offset], lda, &s[1], &scond, &amax, &infequ);
	if (infequ == 0) {

/*     Equilibrate the matrix. */

	    slaqsy_(uplo, n, &a[a_offset], lda, &s[1], &scond, &amax, equed);
	    rcequ = lsame_(equed, "Y");
	}
    }

/*     Scale the right-hand side. */

    if (rcequ) {
	slascl2_(n, nrhs, &s[1], &b[b_offset], ldb);
    }

    if (nofact || equil) {

/*        Compute the LU factorization of A. */

	slacpy_(uplo, n, n, &a[a_offset], lda, &af[af_offset], ldaf);
	spotrf_(uplo, n, &af[af_offset], ldaf, info);

/*        Return if INFO is non-zero. */

	if (*info != 0) {

/*           Pivot in column INFO is exactly 0 */
/*           Compute the reciprocal pivot growth factor of the */
/*           leading rank-deficient INFO columns of A. */

	    *rpvgrw = sla_porpvgrw__(uplo, info, &a[a_offset], lda, &af[
		    af_offset], ldaf, &work[1], (ftnlen)1);
	    return 0;
	}
    }

/*     Compute the reciprocal growth factor RPVGRW. */

    *rpvgrw = sla_porpvgrw__(uplo, n, &a[a_offset], lda, &af[af_offset], ldaf,
	     &work[1], (ftnlen)1);

/*     Compute the solution matrix X. */

    slacpy_("Full", n, nrhs, &b[b_offset], ldb, &x[x_offset], ldx);
    spotrs_(uplo, n, nrhs, &af[af_offset], ldaf, &x[x_offset], ldx, info);

/*     Use iterative refinement to improve the computed solution and */
/*     compute error bounds and backward error estimates for it. */

    sporfsx_(uplo, equed, n, nrhs, &a[a_offset], lda, &af[af_offset], ldaf, &
	    s[1], &b[b_offset], ldb, &x[x_offset], ldx, rcond, &berr[1], 
	    n_err_bnds__, &err_bnds_norm__[err_bnds_norm_offset], &
	    err_bnds_comp__[err_bnds_comp_offset], nparams, &params[1], &work[
	    1], &iwork[1], info);

/*     Scale solutions. */

    if (rcequ) {
	slascl2_(n, nrhs, &s[1], &x[x_offset], ldx);
    }

    return 0;

/*     End of SPOSVXX */

} /* sposvxx_ */
