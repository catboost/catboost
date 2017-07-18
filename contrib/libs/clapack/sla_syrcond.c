/* sla_syrcond.f -- translated by f2c (version 20061008).
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

doublereal sla_syrcond__(char *uplo, integer *n, real *a, integer *lda, real *
	af, integer *ldaf, integer *ipiv, integer *cmode, real *c__, integer *
	info, real *work, integer *iwork, ftnlen uplo_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, i__1, i__2;
    real ret_val, r__1;

    /* Local variables */
    integer i__, j;
    logical up;
    real tmp;
    integer kase;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern /* Subroutine */ int slacn2_(integer *, real *, real *, integer *, 
	    real *, integer *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    real ainvnm;
    char normin[1];
    real smlnum;
    extern /* Subroutine */ int ssytrs_(char *, integer *, integer *, real *, 
	    integer *, integer *, real *, integer *, integer *);


/*     -- LAPACK routine (version 3.2.1)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- April 2009                                                   -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments */
/*     .. */

/*  Purpose */
/*  ======= */

/*     SLA_SYRCOND estimates the Skeel condition number of  op(A) * op2(C) */
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

/*     UPLO    (input) CHARACTER*1 */
/*       = 'U':  Upper triangle of A is stored; */
/*       = 'L':  Lower triangle of A is stored. */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     A       (input) REAL array, dimension (LDA,N) */
/*     On entry, the N-by-N matrix A. */

/*     LDA     (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF      (input) REAL array, dimension (LDAF,N) */
/*     The block diagonal matrix D and the multipliers used to */
/*     obtain the factor U or L as computed by SSYTRF. */

/*     LDAF    (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     IPIV    (input) INTEGER array, dimension (N) */
/*     Details of the interchanges and the block structure of D */
/*     as determined by SSYTRF. */

/*     CMODE   (input) INTEGER */
/*     Determines op2(C) in the formula op(A) * op2(C) as follows: */
/*     CMODE =  1    op2(C) = C */
/*     CMODE =  0    op2(C) = I */
/*     CMODE = -1    op2(C) = inv(C) */

/*     C       (input) REAL array, dimension (N) */
/*     The vector C in the formula op(A) * op2(C). */

/*     INFO    (output) INTEGER */
/*       = 0:  Successful exit. */
/*     i > 0:  The ith argument is invalid. */

/*     WORK    (input) REAL array, dimension (3*N). */
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
    ret_val = 0.f;

    *info = 0;
    if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLA_SYRCOND", &i__1);
	return ret_val;
    }
    if (*n == 0) {
	ret_val = 1.f;
	return ret_val;
    }
    up = FALSE_;
    if (lsame_(uplo, "U")) {
	up = TRUE_;
    }

/*     Compute the equilibration matrix R such that */
/*     inv(R)*A*C has unit 1-norm. */

    if (up) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.f;
	    if (*cmode == 1) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1] * c__[j], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1] * c__[j], dabs(r__1));
		}
	    } else if (*cmode == 0) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		}
	    } else {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1] / c__[j], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1] / c__[j], dabs(r__1));
		}
	    }
	    work[(*n << 1) + i__] = tmp;
	}
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.f;
	    if (*cmode == 1) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1] * c__[j], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1] * c__[j], dabs(r__1));
		}
	    } else if (*cmode == 0) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1], dabs(r__1));
		}
	    } else {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[i__ + j * a_dim1] / c__[j], dabs(r__1));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    tmp += (r__1 = a[j + i__ * a_dim1] / c__[j], dabs(r__1));
		}
	    }
	    work[(*n << 1) + i__] = tmp;
	}
    }

/*     Estimate the norm of inv(op(A)). */

    smlnum = slamch_("Safe minimum");
    ainvnm = 0.f;
    *(unsigned char *)normin = 'N';
    kase = 0;
L10:
    slacn2_(n, &work[*n + 1], &work[1], &iwork[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (kase == 2) {

/*           Multiply by R. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] *= work[(*n << 1) + i__];
	    }
	    if (up) {
		ssytrs_("U", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    } else {
		ssytrs_("L", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
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
	    if (up) {
		ssytrs_("U", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    } else {
		ssytrs_("L", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
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

    if (ainvnm != 0.f) {
	ret_val = 1.f / ainvnm;
    }

    return ret_val;

} /* sla_syrcond__ */
