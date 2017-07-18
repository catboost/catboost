/* ctbrfs.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ctbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, complex *ab, integer *ldab, complex *b, 
	integer *ldb, complex *x, integer *ldx, real *ferr, real *berr, 
	complex *work, real *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, b_dim1, b_offset, x_dim1, x_offset, i__1, 
	    i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    integer i__, j, k;
    real s, xk;
    integer nz;
    real eps;
    integer kase;
    real safe1, safe2;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern /* Subroutine */ int ctbmv_(char *, char *, char *, integer *, 
	    integer *, complex *, integer *, complex *, integer *), ccopy_(integer *, complex *, integer *, complex *
, integer *), ctbsv_(char *, char *, char *, integer *, integer *, 
	     complex *, integer *, complex *, integer *), caxpy_(integer *, complex *, complex *, integer *, 
	    complex *, integer *);
    logical upper;
    extern /* Subroutine */ int clacn2_(integer *, complex *, complex *, real 
	    *, integer *, integer *);
    extern doublereal slamch_(char *);
    real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical notran;
    char transn[1], transt[1];
    logical nounit;
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

/*  CTBRFS provides error bounds and backward error estimates for the */
/*  solution to a system of linear equations with a triangular band */
/*  coefficient matrix. */

/*  The solution matrix X must be computed by CTBTRS or some other */
/*  means before entering this routine.  CTBRFS does not do iterative */
/*  refinement because doing so cannot improve the backward error. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  A is upper triangular; */
/*          = 'L':  A is lower triangular. */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the form of the system of equations: */
/*          = 'N':  A * X = B     (No transpose) */
/*          = 'T':  A**T * X = B  (Transpose) */
/*          = 'C':  A**H * X = B  (Conjugate transpose) */

/*  DIAG    (input) CHARACTER*1 */
/*          = 'N':  A is non-unit triangular; */
/*          = 'U':  A is unit triangular. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  KD      (input) INTEGER */
/*          The number of superdiagonals or subdiagonals of the */
/*          triangular band matrix A.  KD >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices B and X.  NRHS >= 0. */

/*  AB      (input) COMPLEX array, dimension (LDAB,N) */
/*          The upper or lower triangular band matrix A, stored in the */
/*          first kd+1 rows of the array. The j-th column of A is stored */
/*          in the j-th column of the array AB as follows: */
/*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd). */
/*          If DIAG = 'U', the diagonal elements of A are not referenced */
/*          and are assumed to be 1. */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KD+1. */

/*  B       (input) COMPLEX array, dimension (LDB,NRHS) */
/*          The right hand side matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  X       (input) COMPLEX array, dimension (LDX,NRHS) */
/*          The solution matrix X. */

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
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
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
    notran = lsame_(trans, "N");
    nounit = lsame_(diag, "N");

    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T") && ! 
	    lsame_(trans, "C")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*kd < 0) {
	*info = -5;
    } else if (*nrhs < 0) {
	*info = -6;
    } else if (*ldab < *kd + 1) {
	*info = -8;
    } else if (*ldb < max(1,*n)) {
	*info = -10;
    } else if (*ldx < max(1,*n)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTBRFS", &i__1);
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

    nz = *kd + 2;
    eps = slamch_("Epsilon");
    safmin = slamch_("Safe minimum");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

/*     Do for each right hand side */

    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {

/*        Compute residual R = B - op(A) * X, */
/*        where op(A) = A, A**T, or A**H, depending on TRANS. */

	ccopy_(n, &x[j * x_dim1 + 1], &c__1, &work[1], &c__1);
	ctbmv_(uplo, trans, diag, n, kd, &ab[ab_offset], ldab, &work[1], &
		c__1);
	q__1.r = -1.f, q__1.i = -0.f;
	caxpy_(n, &q__1, &b[j * b_dim1 + 1], &c__1, &work[1], &c__1);

/*        Compute componentwise relative backward error from formula */

/*        max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) ) */

/*        where abs(Z) is the componentwise absolute value of the matrix */
/*        or vector Z.  If the i-th component of the denominator is less */
/*        than SAFE2, then SAFE1 is added to the i-th components of the */
/*        numerator and denominator before dividing. */

	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * b_dim1;
	    rwork[i__] = (r__1 = b[i__3].r, dabs(r__1)) + (r__2 = r_imag(&b[
		    i__ + j * b_dim1]), dabs(r__2));
/* L20: */
	}

	if (notran) {

/*           Compute abs(A)*abs(X) + abs(B). */

	    if (upper) {
		if (nounit) {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = k + j * x_dim1;
			xk = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MAX */
			i__3 = 1, i__4 = k - *kd;
			i__5 = k;
			for (i__ = max(i__3,i__4); i__ <= i__5; ++i__) {
			    i__3 = *kd + 1 + i__ - k + k * ab_dim1;
			    rwork[i__] += ((r__1 = ab[i__3].r, dabs(r__1)) + (
				    r__2 = r_imag(&ab[*kd + 1 + i__ - k + k * 
				    ab_dim1]), dabs(r__2))) * xk;
/* L30: */
			}
/* L40: */
		    }
		} else {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__5 = k + j * x_dim1;
			xk = (r__1 = x[i__5].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MAX */
			i__5 = 1, i__3 = k - *kd;
			i__4 = k - 1;
			for (i__ = max(i__5,i__3); i__ <= i__4; ++i__) {
			    i__5 = *kd + 1 + i__ - k + k * ab_dim1;
			    rwork[i__] += ((r__1 = ab[i__5].r, dabs(r__1)) + (
				    r__2 = r_imag(&ab[*kd + 1 + i__ - k + k * 
				    ab_dim1]), dabs(r__2))) * xk;
/* L50: */
			}
			rwork[k] += xk;
/* L60: */
		    }
		}
	    } else {
		if (nounit) {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__4 = k + j * x_dim1;
			xk = (r__1 = x[i__4].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MIN */
			i__5 = *n, i__3 = k + *kd;
			i__4 = min(i__5,i__3);
			for (i__ = k; i__ <= i__4; ++i__) {
			    i__5 = i__ + 1 - k + k * ab_dim1;
			    rwork[i__] += ((r__1 = ab[i__5].r, dabs(r__1)) + (
				    r__2 = r_imag(&ab[i__ + 1 - k + k * 
				    ab_dim1]), dabs(r__2))) * xk;
/* L70: */
			}
/* L80: */
		    }
		} else {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__4 = k + j * x_dim1;
			xk = (r__1 = x[i__4].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MIN */
			i__5 = *n, i__3 = k + *kd;
			i__4 = min(i__5,i__3);
			for (i__ = k + 1; i__ <= i__4; ++i__) {
			    i__5 = i__ + 1 - k + k * ab_dim1;
			    rwork[i__] += ((r__1 = ab[i__5].r, dabs(r__1)) + (
				    r__2 = r_imag(&ab[i__ + 1 - k + k * 
				    ab_dim1]), dabs(r__2))) * xk;
/* L90: */
			}
			rwork[k] += xk;
/* L100: */
		    }
		}
	    }
	} else {

/*           Compute abs(A**H)*abs(X) + abs(B). */

	    if (upper) {
		if (nounit) {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			s = 0.f;
/* Computing MAX */
			i__4 = 1, i__5 = k - *kd;
			i__3 = k;
			for (i__ = max(i__4,i__5); i__ <= i__3; ++i__) {
			    i__4 = *kd + 1 + i__ - k + k * ab_dim1;
			    i__5 = i__ + j * x_dim1;
			    s += ((r__1 = ab[i__4].r, dabs(r__1)) + (r__2 = 
				    r_imag(&ab[*kd + 1 + i__ - k + k * 
				    ab_dim1]), dabs(r__2))) * ((r__3 = x[i__5]
				    .r, dabs(r__3)) + (r__4 = r_imag(&x[i__ + 
				    j * x_dim1]), dabs(r__4)));
/* L110: */
			}
			rwork[k] += s;
/* L120: */
		    }
		} else {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__3 = k + j * x_dim1;
			s = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MAX */
			i__3 = 1, i__4 = k - *kd;
			i__5 = k - 1;
			for (i__ = max(i__3,i__4); i__ <= i__5; ++i__) {
			    i__3 = *kd + 1 + i__ - k + k * ab_dim1;
			    i__4 = i__ + j * x_dim1;
			    s += ((r__1 = ab[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&ab[*kd + 1 + i__ - k + k * 
				    ab_dim1]), dabs(r__2))) * ((r__3 = x[i__4]
				    .r, dabs(r__3)) + (r__4 = r_imag(&x[i__ + 
				    j * x_dim1]), dabs(r__4)));
/* L130: */
			}
			rwork[k] += s;
/* L140: */
		    }
		}
	    } else {
		if (nounit) {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			s = 0.f;
/* Computing MIN */
			i__3 = *n, i__4 = k + *kd;
			i__5 = min(i__3,i__4);
			for (i__ = k; i__ <= i__5; ++i__) {
			    i__3 = i__ + 1 - k + k * ab_dim1;
			    i__4 = i__ + j * x_dim1;
			    s += ((r__1 = ab[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&ab[i__ + 1 - k + k * ab_dim1]), 
				    dabs(r__2))) * ((r__3 = x[i__4].r, dabs(
				    r__3)) + (r__4 = r_imag(&x[i__ + j * 
				    x_dim1]), dabs(r__4)));
/* L150: */
			}
			rwork[k] += s;
/* L160: */
		    }
		} else {
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			i__5 = k + j * x_dim1;
			s = (r__1 = x[i__5].r, dabs(r__1)) + (r__2 = r_imag(&
				x[k + j * x_dim1]), dabs(r__2));
/* Computing MIN */
			i__3 = *n, i__4 = k + *kd;
			i__5 = min(i__3,i__4);
			for (i__ = k + 1; i__ <= i__5; ++i__) {
			    i__3 = i__ + 1 - k + k * ab_dim1;
			    i__4 = i__ + j * x_dim1;
			    s += ((r__1 = ab[i__3].r, dabs(r__1)) + (r__2 = 
				    r_imag(&ab[i__ + 1 - k + k * ab_dim1]), 
				    dabs(r__2))) * ((r__3 = x[i__4].r, dabs(
				    r__3)) + (r__4 = r_imag(&x[i__ + j * 
				    x_dim1]), dabs(r__4)));
/* L170: */
			}
			rwork[k] += s;
/* L180: */
		    }
		}
	    }
	}
	s = 0.f;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (rwork[i__] > safe2) {
/* Computing MAX */
		i__5 = i__;
		r__3 = s, r__4 = ((r__1 = work[i__5].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2))) / rwork[i__];
		s = dmax(r__3,r__4);
	    } else {
/* Computing MAX */
		i__5 = i__;
		r__3 = s, r__4 = ((r__1 = work[i__5].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + safe1) / (rwork[i__]
			 + safe1);
		s = dmax(r__3,r__4);
	    }
/* L190: */
	}
	berr[j] = s;

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
		i__5 = i__;
		rwork[i__] = (r__1 = work[i__5].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + nz * eps * rwork[
			i__];
	    } else {
		i__5 = i__;
		rwork[i__] = (r__1 = work[i__5].r, dabs(r__1)) + (r__2 = 
			r_imag(&work[i__]), dabs(r__2)) + nz * eps * rwork[
			i__] + safe1;
	    }
/* L200: */
	}

	kase = 0;
L210:
	clacn2_(n, &work[*n + 1], &work[1], &ferr[j], &kase, isave);
	if (kase != 0) {
	    if (kase == 1) {

/*              Multiply by diag(W)*inv(op(A)**H). */

		ctbsv_(uplo, transt, diag, n, kd, &ab[ab_offset], ldab, &work[
			1], &c__1);
		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__5 = i__;
		    i__3 = i__;
		    i__4 = i__;
		    q__1.r = rwork[i__3] * work[i__4].r, q__1.i = rwork[i__3] 
			    * work[i__4].i;
		    work[i__5].r = q__1.r, work[i__5].i = q__1.i;
/* L220: */
		}
	    } else {

/*              Multiply by inv(op(A))*diag(W). */

		i__2 = *n;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__5 = i__;
		    i__3 = i__;
		    i__4 = i__;
		    q__1.r = rwork[i__3] * work[i__4].r, q__1.i = rwork[i__3] 
			    * work[i__4].i;
		    work[i__5].r = q__1.r, work[i__5].i = q__1.i;
/* L230: */
		}
		ctbsv_(uplo, transn, diag, n, kd, &ab[ab_offset], ldab, &work[
			1], &c__1);
	    }
	    goto L210;
	}

/*        Normalize error. */

	lstres = 0.f;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__5 = i__ + j * x_dim1;
	    r__3 = lstres, r__4 = (r__1 = x[i__5].r, dabs(r__1)) + (r__2 = 
		    r_imag(&x[i__ + j * x_dim1]), dabs(r__2));
	    lstres = dmax(r__3,r__4);
/* L240: */
	}
	if (lstres != 0.f) {
	    ferr[j] /= lstres;
	}

/* L250: */
    }

    return 0;

/*     End of CTBRFS */

} /* ctbrfs_ */
