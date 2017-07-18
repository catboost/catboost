/* cla_hercond_c.f -- translated by f2c (version 20061008).
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

doublereal cla_hercond_c__(char *uplo, integer *n, complex *a, integer *lda, 
	complex *af, integer *ldaf, integer *ipiv, real *c__, logical *capply,
	 integer *info, complex *work, real *rwork, ftnlen uplo_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, i__1, i__2, i__3, i__4;
    real ret_val, r__1, r__2;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    integer i__, j;
    logical up;
    real tmp;
    integer kase;
    extern logical lsame_(char *, char *);
    integer isave[3];
    real anorm;
    extern /* Subroutine */ int clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *), xerbla_(char *, integer *);
    real ainvnm;
    extern /* Subroutine */ int chetrs_(char *, integer *, integer *, complex 
	    *, integer *, integer *, complex *, integer *, integer *);


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

/*     CLA_HERCOND_C computes the infinity norm condition number of */
/*     op(A) * inv(diag(C)) where C is a REAL vector. */

/*  Arguments */
/*  ========= */

/*     UPLO    (input) CHARACTER*1 */
/*       = 'U':  Upper triangle of A is stored; */
/*       = 'L':  Lower triangle of A is stored. */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     A       (input) COMPLEX array, dimension (LDA,N) */
/*     On entry, the N-by-N matrix A */

/*     LDA     (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF      (input) COMPLEX array, dimension (LDAF,N) */
/*     The block diagonal matrix D and the multipliers used to */
/*     obtain the factor U or L as computed by CHETRF. */

/*     LDAF    (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     IPIV    (input) INTEGER array, dimension (N) */
/*     Details of the interchanges and the block structure of D */
/*     as determined by CHETRF. */

/*     C       (input) REAL array, dimension (N) */
/*     The vector C in the formula op(A) * inv(diag(C)). */

/*     CAPPLY  (input) LOGICAL */
/*     If .TRUE. then access the vector C in the formula above. */

/*     INFO    (output) INTEGER */
/*       = 0:  Successful exit. */
/*     i > 0:  The ith argument is invalid. */

/*     WORK    (input) COMPLEX array, dimension (2*N). */
/*     Workspace. */

/*     RWORK   (input) REAL array, dimension (N). */
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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
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
    --rwork;

    /* Function Body */
    ret_val = 0.f;

    *info = 0;
    if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLA_HERCOND_C", &i__1);
	return ret_val;
    }
    up = FALSE_;
    if (lsame_(uplo, "U")) {
	up = TRUE_;
    }

/*     Compute norm of op(A)*op2(C). */

    anorm = 0.f;
    if (up) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.f;
	    if (*capply) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j + i__ * a_dim1;
		    tmp += ((r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
			    a[j + i__ * a_dim1]), dabs(r__2))) / c__[j];
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    i__3 = i__ + j * a_dim1;
		    tmp += ((r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
			    a[i__ + j * a_dim1]), dabs(r__2))) / c__[j];
		}
	    } else {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = j + i__ * a_dim1;
		    tmp += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[
			    j + i__ * a_dim1]), dabs(r__2));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    i__3 = i__ + j * a_dim1;
		    tmp += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[
			    i__ + j * a_dim1]), dabs(r__2));
		}
	    }
	    rwork[i__] = tmp;
	    anorm = dmax(anorm,tmp);
	}
    } else {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tmp = 0.f;
	    if (*capply) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = i__ + j * a_dim1;
		    tmp += ((r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
			    a[i__ + j * a_dim1]), dabs(r__2))) / c__[j];
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    i__3 = j + i__ * a_dim1;
		    tmp += ((r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
			    a[j + i__ * a_dim1]), dabs(r__2))) / c__[j];
		}
	    } else {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    i__3 = i__ + j * a_dim1;
		    tmp += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[
			    i__ + j * a_dim1]), dabs(r__2));
		}
		i__2 = *n;
		for (j = i__ + 1; j <= i__2; ++j) {
		    i__3 = j + i__ * a_dim1;
		    tmp += (r__1 = a[i__3].r, dabs(r__1)) + (r__2 = r_imag(&a[
			    j + i__ * a_dim1]), dabs(r__2));
		}
	    }
	    rwork[i__] = tmp;
	    anorm = dmax(anorm,tmp);
	}
    }

/*     Quick return if possible. */

    if (*n == 0) {
	ret_val = 1.f;
	return ret_val;
    } else if (anorm == 0.f) {
	return ret_val;
    }

/*     Estimate the norm of inv(op(A)). */

    ainvnm = 0.f;

    kase = 0;
L10:
    clacn2_(n, &work[*n + 1], &work[1], &ainvnm, &kase, isave);
    if (kase != 0) {
	if (kase == 2) {

/*           Multiply by R. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		i__3 = i__;
		i__4 = i__;
		q__1.r = rwork[i__4] * work[i__3].r, q__1.i = rwork[i__4] * 
			work[i__3].i;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
	    }

	    if (up) {
		chetrs_("U", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    } else {
		chetrs_("L", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    }

/*           Multiply by inv(C). */

	    if (*capply) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    i__4 = i__;
		    q__1.r = c__[i__4] * work[i__3].r, q__1.i = c__[i__4] * 
			    work[i__3].i;
		    work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		}
	    }
	} else {

/*           Multiply by inv(C'). */

	    if (*capply) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    i__3 = i__;
		    i__4 = i__;
		    q__1.r = c__[i__4] * work[i__3].r, q__1.i = c__[i__4] * 
			    work[i__3].i;
		    work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		}
	    }

	    if (up) {
		chetrs_("U", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    } else {
		chetrs_("L", n, &c__1, &af[af_offset], ldaf, &ipiv[1], &work[
			1], n, info);
	    }

/*           Multiply by R. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		i__3 = i__;
		i__4 = i__;
		q__1.r = rwork[i__4] * work[i__3].r, q__1.i = rwork[i__4] * 
			work[i__3].i;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
	    }
	}
	goto L10;
    }

/*     Compute the estimate of the reciprocal condition number. */

    if (ainvnm != 0.f) {
	ret_val = 1.f / ainvnm;
    }

    return ret_val;

} /* cla_hercond_c__ */
