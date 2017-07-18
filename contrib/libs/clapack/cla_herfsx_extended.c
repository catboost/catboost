/* cla_herfsx_extended.f -- translated by f2c (version 20061008).
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
static complex c_b11 = {-1.f,0.f};
static complex c_b12 = {1.f,0.f};
static real c_b33 = 1.f;

/* Subroutine */ int cla_herfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, complex *a, integer *lda, complex *af, 
	integer *ldaf, integer *ipiv, logical *colequ, real *c__, complex *b, 
	integer *ldb, complex *y, integer *ldy, real *berr_out__, integer *
	n_norms__, real *err_bnds_norm__, real *err_bnds_comp__, complex *res,
	 real *ayb, complex *dy, complex *y_tail__, real *rcond, integer *
	ithresh, real *rthresh, real *dz_ub__, logical *ignore_cwise__, 
	integer *info, ftnlen uplo_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, b_dim1, b_offset, y_dim1, 
	    y_offset, err_bnds_norm_dim1, err_bnds_norm_offset, 
	    err_bnds_comp_dim1, err_bnds_comp_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    real dxratmax, dzratmax;
    integer i__, j;
    extern /* Subroutine */ int cla_heamv__(integer *, integer *, real *, 
	    complex *, integer *, complex *, integer *, real *, real *, 
	    integer *);
    logical incr_prec__;
    real prev_dz_z__, yk, final_dx_x__;
    extern /* Subroutine */ int cla_wwaddw__(integer *, complex *, complex *, 
	    complex *);
    real final_dz_z__, prevnormdx;
    integer cnt;
    real dyk, eps, incr_thresh__, dx_x__, dz_z__;
    extern /* Subroutine */ int cla_lin_berr__(integer *, integer *, integer *
	    , complex *, real *, real *);
    real ymin;
    extern /* Subroutine */ int blas_chemv_x__(integer *, integer *, complex *
	    , complex *, integer *, complex *, integer *, complex *, complex *
	    , integer *, integer *);
    integer y_prec_state__, uplo2;
    extern /* Subroutine */ int blas_chemv2_x__(integer *, integer *, complex 
	    *, complex *, integer *, complex *, complex *, integer *, complex 
	    *, complex *, integer *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int chemv_(char *, integer *, complex *, complex *
, integer *, complex *, integer *, complex *, complex *, integer *
), ccopy_(integer *, complex *, integer *, complex *, 
	    integer *);
    real dxrat, dzrat;
    extern /* Subroutine */ int caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *);
    real normx, normy;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int chetrs_(char *, integer *, integer *, complex 
	    *, integer *, integer *, complex *, integer *, integer *);
    real normdx, hugeval;
    extern integer ilauplo_(char *);
    integer x_state__, z_state__;


/*     -- LAPACK routine (version 3.2.1)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- April 2009                                                   -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLA_HERFSX_EXTENDED improves the computed solution to a system of */
/*  linear equations by performing extra-precise iterative refinement */
/*  and provides error bounds and backward error estimates for the solution. */
/*  This subroutine is called by CHERFSX to perform iterative refinement. */
/*  In addition to normwise error bound, the code provides maximum */
/*  componentwise error bound if possible. See comments for ERR_BNDS_NORM */
/*  and ERR_BNDS_COMP for details of the error bounds. Note that this */
/*  subroutine is only resonsible for setting the second fields of */
/*  ERR_BNDS_NORM and ERR_BNDS_COMP. */

/*  Arguments */
/*  ========= */

/*     PREC_TYPE      (input) INTEGER */
/*     Specifies the intermediate precision to be used in refinement. */
/*     The value is defined by ILAPREC(P) where P is a CHARACTER and */
/*     P    = 'S':  Single */
/*          = 'D':  Double */
/*          = 'I':  Indigenous */
/*          = 'X', 'E':  Extra */

/*     UPLO    (input) CHARACTER*1 */
/*       = 'U':  Upper triangle of A is stored; */
/*       = 'L':  Lower triangle of A is stored. */

/*     N              (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     NRHS           (input) INTEGER */
/*     The number of right-hand-sides, i.e., the number of columns of the */
/*     matrix B. */

/*     A              (input) COMPLEX array, dimension (LDA,N) */
/*     On entry, the N-by-N matrix A. */

/*     LDA            (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF             (input) COMPLEX array, dimension (LDAF,N) */
/*     The block diagonal matrix D and the multipliers used to */
/*     obtain the factor U or L as computed by CHETRF. */

/*     LDAF           (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     IPIV           (input) INTEGER array, dimension (N) */
/*     Details of the interchanges and the block structure of D */
/*     as determined by CHETRF. */

/*     COLEQU         (input) LOGICAL */
/*     If .TRUE. then column equilibration was done to A before calling */
/*     this routine. This is needed to compute the solution and error */
/*     bounds correctly. */

/*     C              (input) REAL array, dimension (N) */
/*     The column scale factors for A. If COLEQU = .FALSE., C */
/*     is not accessed. If C is input, each element of C should be a power */
/*     of the radix to ensure a reliable solution and error estimates. */
/*     Scaling by powers of the radix does not cause rounding errors unless */
/*     the result underflows or overflows. Rounding errors during scaling */
/*     lead to refining with a matrix that is not equivalent to the */
/*     input matrix, producing error estimates that may not be */
/*     reliable. */

/*     B              (input) COMPLEX array, dimension (LDB,NRHS) */
/*     The right-hand-side matrix B. */

/*     LDB            (input) INTEGER */
/*     The leading dimension of the array B.  LDB >= max(1,N). */

/*     Y              (input/output) COMPLEX array, dimension */
/*                    (LDY,NRHS) */
/*     On entry, the solution matrix X, as computed by CHETRS. */
/*     On exit, the improved solution matrix Y. */

/*     LDY            (input) INTEGER */
/*     The leading dimension of the array Y.  LDY >= max(1,N). */

/*     BERR_OUT       (output) REAL array, dimension (NRHS) */
/*     On exit, BERR_OUT(j) contains the componentwise relative backward */
/*     error for right-hand-side j from the formula */
/*         max(i) ( abs(RES(i)) / ( abs(op(A_s))*abs(Y) + abs(B_s) )(i) ) */
/*     where abs(Z) is the componentwise absolute value of the matrix */
/*     or vector Z. This is computed by CLA_LIN_BERR. */

/*     N_NORMS        (input) INTEGER */
/*     Determines which error bounds to return (see ERR_BNDS_NORM */
/*     and ERR_BNDS_COMP). */
/*     If N_NORMS >= 1 return normwise error bounds. */
/*     If N_NORMS >= 2 return componentwise error bounds. */

/*     ERR_BNDS_NORM  (input/output) REAL array, dimension */
/*                    (NRHS, N_ERR_BNDS) */
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

/*     This subroutine is only responsible for setting the second field */
/*     above. */
/*     See Lapack Working Note 165 for further details and extra */
/*     cautions. */

/*     ERR_BNDS_COMP  (input/output) REAL array, dimension */
/*                    (NRHS, N_ERR_BNDS) */
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

/*     This subroutine is only responsible for setting the second field */
/*     above. */
/*     See Lapack Working Note 165 for further details and extra */
/*     cautions. */

/*     RES            (input) COMPLEX array, dimension (N) */
/*     Workspace to hold the intermediate residual. */

/*     AYB            (input) REAL array, dimension (N) */
/*     Workspace. */

/*     DY             (input) COMPLEX array, dimension (N) */
/*     Workspace to hold the intermediate solution. */

/*     Y_TAIL         (input) COMPLEX array, dimension (N) */
/*     Workspace to hold the trailing bits of the intermediate solution. */

/*     RCOND          (input) REAL */
/*     Reciprocal scaled condition number.  This is an estimate of the */
/*     reciprocal Skeel condition number of the matrix A after */
/*     equilibration (if done).  If this is less than the machine */
/*     precision (in particular, if it is zero), the matrix is singular */
/*     to working precision.  Note that the error may still be small even */
/*     if this number is very small and the matrix appears ill- */
/*     conditioned. */

/*     ITHRESH        (input) INTEGER */
/*     The maximum number of residual computations allowed for */
/*     refinement. The default is 10. For 'aggressive' set to 100 to */
/*     permit convergence using approximate factorizations or */
/*     factorizations other than LU. If the factorization uses a */
/*     technique other than Gaussian elimination, the guarantees in */
/*     ERR_BNDS_NORM and ERR_BNDS_COMP may no longer be trustworthy. */

/*     RTHRESH        (input) REAL */
/*     Determines when to stop refinement if the error estimate stops */
/*     decreasing. Refinement will stop when the next solution no longer */
/*     satisfies norm(dx_{i+1}) < RTHRESH * norm(dx_i) where norm(Z) is */
/*     the infinity norm of Z. RTHRESH satisfies 0 < RTHRESH <= 1. The */
/*     default value is 0.5. For 'aggressive' set to 0.9 to permit */
/*     convergence on extremely ill-conditioned matrices. See LAWN 165 */
/*     for more details. */

/*     DZ_UB          (input) REAL */
/*     Determines when to start considering componentwise convergence. */
/*     Componentwise convergence is only considered after each component */
/*     of the solution Y is stable, which we definte as the relative */
/*     change in each component being less than DZ_UB. The default value */
/*     is 0.25, requiring the first bit to be stable. See LAWN 165 for */
/*     more details. */

/*     IGNORE_CWISE   (input) LOGICAL */
/*     If .TRUE. then ignore componentwise convergence. Default value */
/*     is .FALSE.. */

/*     INFO           (output) INTEGER */
/*       = 0:  Successful exit. */
/*       < 0:  if INFO = -i, the ith argument to CHETRS had an illegal */
/*             value */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
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
    --ipiv;
    --c__;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;
    --berr_out__;
    --res;
    --ayb;
    --dy;
    --y_tail__;

    /* Function Body */
    if (*info != 0) {
	return 0;
    }
    eps = slamch_("Epsilon");
    hugeval = slamch_("Overflow");
/*     Force HUGEVAL to Inf */
    hugeval *= hugeval;
/*     Using HUGEVAL may lead to spurious underflows. */
    incr_thresh__ = (real) (*n) * eps;
    if (lsame_(uplo, "L")) {
	uplo2 = ilauplo_("L");
    } else {
	uplo2 = ilauplo_("U");
    }
    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {
	y_prec_state__ = 1;
	if (y_prec_state__ == 2) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__;
		y_tail__[i__3].r = 0.f, y_tail__[i__3].i = 0.f;
	    }
	}
	dxrat = 0.f;
	dxratmax = 0.f;
	dzrat = 0.f;
	dzratmax = 0.f;
	final_dx_x__ = hugeval;
	final_dz_z__ = hugeval;
	prevnormdx = hugeval;
	prev_dz_z__ = hugeval;
	dz_z__ = hugeval;
	dx_x__ = hugeval;
	x_state__ = 1;
	z_state__ = 0;
	incr_prec__ = FALSE_;
	i__2 = *ithresh;
	for (cnt = 1; cnt <= i__2; ++cnt) {

/*         Compute residual RES = B_s - op(A_s) * Y, */
/*             op(A) = A, A**T, or A**H depending on TRANS (and type). */

	    ccopy_(n, &b[j * b_dim1 + 1], &c__1, &res[1], &c__1);
	    if (y_prec_state__ == 0) {
		chemv_(uplo, n, &c_b11, &a[a_offset], lda, &y[j * y_dim1 + 1], 
			 &c__1, &c_b12, &res[1], &c__1);
	    } else if (y_prec_state__ == 1) {
		blas_chemv_x__(&uplo2, n, &c_b11, &a[a_offset], lda, &y[j * 
			y_dim1 + 1], &c__1, &c_b12, &res[1], &c__1, 
			prec_type__);
	    } else {
		blas_chemv2_x__(&uplo2, n, &c_b11, &a[a_offset], lda, &y[j * 
			y_dim1 + 1], &y_tail__[1], &c__1, &c_b12, &res[1], &
			c__1, prec_type__);
	    }
/*         XXX: RES is no longer needed. */
	    ccopy_(n, &res[1], &c__1, &dy[1], &c__1);
	    chetrs_(uplo, n, nrhs, &af[af_offset], ldaf, &ipiv[1], &dy[1], n, 
		    info);

/*         Calculate relative changes DX_X, DZ_Z and ratios DXRAT, DZRAT. */

	    normx = 0.f;
	    normy = 0.f;
	    normdx = 0.f;
	    dz_z__ = 0.f;
	    ymin = hugeval;
	    i__3 = *n;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		i__4 = i__ + j * y_dim1;
		yk = (r__1 = y[i__4].r, dabs(r__1)) + (r__2 = r_imag(&y[i__ + 
			j * y_dim1]), dabs(r__2));
		i__4 = i__;
		dyk = (r__1 = dy[i__4].r, dabs(r__1)) + (r__2 = r_imag(&dy[
			i__]), dabs(r__2));
		if (yk != 0.f) {
/* Computing MAX */
		    r__1 = dz_z__, r__2 = dyk / yk;
		    dz_z__ = dmax(r__1,r__2);
		} else if (dyk != 0.f) {
		    dz_z__ = hugeval;
		}
		ymin = dmin(ymin,yk);
		normy = dmax(normy,yk);
		if (*colequ) {
/* Computing MAX */
		    r__1 = normx, r__2 = yk * c__[i__];
		    normx = dmax(r__1,r__2);
/* Computing MAX */
		    r__1 = normdx, r__2 = dyk * c__[i__];
		    normdx = dmax(r__1,r__2);
		} else {
		    normx = normy;
		    normdx = dmax(normdx,dyk);
		}
	    }
	    if (normx != 0.f) {
		dx_x__ = normdx / normx;
	    } else if (normdx == 0.f) {
		dx_x__ = 0.f;
	    } else {
		dx_x__ = hugeval;
	    }
	    dxrat = normdx / prevnormdx;
	    dzrat = dz_z__ / prev_dz_z__;

/*         Check termination criteria. */

	    if (ymin * *rcond < incr_thresh__ * normy && y_prec_state__ < 2) {
		incr_prec__ = TRUE_;
	    }
	    if (x_state__ == 3 && dxrat <= *rthresh) {
		x_state__ = 1;
	    }
	    if (x_state__ == 1) {
		if (dx_x__ <= eps) {
		    x_state__ = 2;
		} else if (dxrat > *rthresh) {
		    if (y_prec_state__ != 2) {
			incr_prec__ = TRUE_;
		    } else {
			x_state__ = 3;
		    }
		} else {
		    if (dxrat > dxratmax) {
			dxratmax = dxrat;
		    }
		}
		if (x_state__ > 1) {
		    final_dx_x__ = dx_x__;
		}
	    }
	    if (z_state__ == 0 && dz_z__ <= *dz_ub__) {
		z_state__ = 1;
	    }
	    if (z_state__ == 3 && dzrat <= *rthresh) {
		z_state__ = 1;
	    }
	    if (z_state__ == 1) {
		if (dz_z__ <= eps) {
		    z_state__ = 2;
		} else if (dz_z__ > *dz_ub__) {
		    z_state__ = 0;
		    dzratmax = 0.f;
		    final_dz_z__ = hugeval;
		} else if (dzrat > *rthresh) {
		    if (y_prec_state__ != 2) {
			incr_prec__ = TRUE_;
		    } else {
			z_state__ = 3;
		    }
		} else {
		    if (dzrat > dzratmax) {
			dzratmax = dzrat;
		    }
		}
		if (z_state__ > 1) {
		    final_dz_z__ = dz_z__;
		}
	    }
	    if (x_state__ != 1 && (*ignore_cwise__ || z_state__ != 1)) {
		goto L666;
	    }
	    if (incr_prec__) {
		incr_prec__ = FALSE_;
		++y_prec_state__;
		i__3 = *n;
		for (i__ = 1; i__ <= i__3; ++i__) {
		    i__4 = i__;
		    y_tail__[i__4].r = 0.f, y_tail__[i__4].i = 0.f;
		}
	    }
	    prevnormdx = normdx;
	    prev_dz_z__ = dz_z__;

/*           Update soluton. */

	    if (y_prec_state__ < 2) {
		caxpy_(n, &c_b12, &dy[1], &c__1, &y[j * y_dim1 + 1], &c__1);
	    } else {
		cla_wwaddw__(n, &y[j * y_dim1 + 1], &y_tail__[1], &dy[1]);
	    }
	}
/*        Target of "IF (Z_STOP .AND. X_STOP)".  Sun's f77 won't EXIT. */
L666:

/*     Set final_* when cnt hits ithresh. */

	if (x_state__ == 1) {
	    final_dx_x__ = dx_x__;
	}
	if (z_state__ == 1) {
	    final_dz_z__ = dz_z__;
	}

/*     Compute error bounds. */

	if (*n_norms__ >= 1) {
	    err_bnds_norm__[j + (err_bnds_norm_dim1 << 1)] = final_dx_x__ / (
		    1 - dxratmax);
	}
	if (*n_norms__ >= 2) {
	    err_bnds_comp__[j + (err_bnds_comp_dim1 << 1)] = final_dz_z__ / (
		    1 - dzratmax);
	}

/*     Compute componentwise relative backward error from formula */
/*         max(i) ( abs(R(i)) / ( abs(op(A_s))*abs(Y) + abs(B_s) )(i) ) */
/*     where abs(Z) is the componentwise absolute value of the matrix */
/*     or vector Z. */

/*         Compute residual RES = B_s - op(A_s) * Y, */
/*             op(A) = A, A**T, or A**H depending on TRANS (and type). */

	ccopy_(n, &b[j * b_dim1 + 1], &c__1, &res[1], &c__1);
	chemv_(uplo, n, &c_b11, &a[a_offset], lda, &y[j * y_dim1 + 1], &c__1, 
		&c_b12, &res[1], &c__1);
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * b_dim1;
	    ayb[i__] = (r__1 = b[i__3].r, dabs(r__1)) + (r__2 = r_imag(&b[i__ 
		    + j * b_dim1]), dabs(r__2));
	}

/*     Compute abs(op(A_s))*abs(Y) + abs(B_s). */

	cla_heamv__(&uplo2, n, &c_b33, &a[a_offset], lda, &y[j * y_dim1 + 1], 
		&c__1, &c_b33, &ayb[1], &c__1);
	cla_lin_berr__(n, n, &c__1, &res[1], &ayb[1], &berr_out__[j]);

/*     End of loop for each RHS. */

    }

    return 0;
} /* cla_herfsx_extended__ */
