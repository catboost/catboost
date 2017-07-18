/* dla_gercond.f -- translated by f2c (version 20061008).
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

doublereal dla_gercond__(char *trans, integer *n, doublereal *a, integer *lda,
	 doublereal *af, integer *ldaf, integer *ipiv, integer *cmode, 
	doublereal *c__, integer *info, doublereal *work, integer *iwork, 
	ftnlen trans_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, i__1, i__2;
    doublereal ret_val, d__1;

    /* Local variables */
    integer i__, j;
    doublereal tmp;
    integer kase;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern /* Subroutine */ int dlacn2_(integer *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, integer *), xerbla_(char *, 
	    integer *);
    doublereal ainvnm;
    extern /* Subroutine */ int dgetrs_(char *, integer *, integer *, 
	    doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *);
    logical notrans;


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

/*     DLA_GERCOND estimates the Skeel condition number of op(A) * op2(C) */
/*     where op2 is determined by CMODE as follows */
/*     CMODE =  1    op2(C) = C */
/*     CMODE =  0    op2(C) = I */
/*     CMODE = -1    op2(C) = inv(C) */
/*     The Skeel condition number cond(A) = norminf( |inv(A)||A| ) */
/*     is computed by computing scaling factors R such that */
/*     diag(R)*A*op2(C) is row equilibrated and computing the standard */
/*     infinity-norm condition number. */

/*  Arguments */
/*  ========== */

/*     TRANS   (input) CHARACTER*1 */
/*     Specifies the form of the system of equations: */
/*       = 'N':  A * X = B     (No transpose) */
/*       = 'T':  A**T * X = B  (Transpose) */
/*       = 'C':  A**H * X = B  (Conjugate Transpose = Transpose) */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     A       (input) DOUBLE PRECISION array, dimension (LDA,N) */
/*     On entry, the N-by-N matrix A. */

/*     LDA     (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF      (input) DOUBLE PRECISION array, dimension (LDAF,N) */
/*     The factors L and U from the factorization */
/*     A = P*L*U as computed by DGETRF. */

/*     LDAF    (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     IPIV    (input) INTEGER array, dimension (N) */
/*     The pivot indices from the factorization A = P*L*U */
/*     as computed by DGETRF; row i of the matrix was interchanged */
/*     with row IPIV(i). */

/*     CMODE   (input) INTEGER */
/*     Determines op2(C) in the formula op(A) * op2(C) as follows: */
/*     CMODE =  1    op2(C) = C */
/*     CMODE =  0    op2(C) = I */
/*     CMODE = -1    op2(C) = inv(C) */

/*     C       (input) DOUBLE PRECISION array, dimension (N) */
/*     The vector C in the formula op(A) * op2(C). */

/*     INFO    (output) INTEGER */
/*       = 0:  Successful exit. */
/*     i > 0:  The ith argument is invalid. */

/*     WORK    (input) DOUBLE PRECISION array, dimension (3*N). */
/*     Workspace. */

/*     IWORK   (input) INTEGER array, dimension (N). */
/*     Workspace. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
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
    --ipiv;
    --c__;
    --work;
    --iwork;

    /* Function Body */
    ret_val = 0.;

    *info = 0;
    notrans = lsame_(trans, "N");
    if (! notrans && ! lsame_(trans, "T") && ! lsame_(
	    trans, "C")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*ldaf < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLA_GERCOND", &i__1);
	return ret_val;
    }
    if (*n == 0) {
	ret_val = 1.;
	return ret_val;
    }

/*     Compute the equilibration matrix R such that */
/*     inv(R)*A*C has unit 1-norm. */

    if (notrans) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.;
	    if (*cmode == 1) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[i__ + j * a_dim1] * c__[j], abs(d__1));
		}
	    } else if (*cmode == 0) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[i__ + j * a_dim1], abs(d__1));
		}
	    } else {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[i__ + j * a_dim1] / c__[j], abs(d__1));
		}
	    }
	    work[(*n << 1) + i__] = tmp;
	}
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.;
	    if (*cmode == 1) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[j + i__ * a_dim1] * c__[j], abs(d__1));
		}
	    } else if (*cmode == 0) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[j + i__ * a_dim1], abs(d__1));
		}
	    } else {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (d__1 = a[j + i__ * a_dim1] / c__[j], abs(d__1));
		}
	    }
	    work[(*n << 1) + i__] = tmp;
	}
    }

/*     Estimate the norm of inv(op(A)). */

    ainvnm = 0.;
    kase = 0;
L10:
    dlacn2_(n, &work[*n + 1], &work[1], &iwork[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (kase == 2) {

/*           Multiply by R. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] *= work[(*n << 1) + i__];
	    }
	    if (notrans) {
		dgetrs_("No transpose", n, &c__1, &af[af_offset], ldaf, &ipiv[
			1], &work[1], n, info);
	    } else {
		dgetrs_("Transpose", n, &c__1, &af[af_offset], ldaf, &ipiv[1], 
			 &work[1], n, info);
	    }

/*           Multiply by inv(C). */

	    if (*cmode == 1) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] /= c__[i__];
		}
	    } else if (*cmode == -1) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] *= c__[i__];
		}
	    }
	} else {

/*           Multiply by inv(C'). */

	    if (*cmode == 1) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] /= c__[i__];
		}
	    } else if (*cmode == -1) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] *= c__[i__];
		}
	    }
	    if (notrans) {
		dgetrs_("Transpose", n, &c__1, &af[af_offset], ldaf, &ipiv[1], 
			 &work[1], n, info);
	    } else {
		dgetrs_("No transpose", n, &c__1, &af[af_offset], ldaf, &ipiv[
			1], &work[1], n, info);
	    }

/*           Multiply by R. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] *= work[(*n << 1) + i__];
	    }
	}
	goto L10;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.) {
	ret_val = 1. / ainvnm;
    }

    return ret_val;

} /* dla_gercond__ */
