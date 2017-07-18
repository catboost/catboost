/* dspgst.f -- translated by f2c (version 20061008).
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
static doublereal c_b9 = -1.;
static doublereal c_b11 = 1.;

/* Subroutine */ int dspgst_(integer *itype, char *uplo, integer *n, 
	doublereal *ap, doublereal *bp, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1;

    /* Local variables */
    integer j, k, j1, k1, jj, kk;
    doublereal ct, ajj;
    integer j1j1;
    doublereal akk;
    integer k1k1;
    doublereal bjj, bkk;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    extern /* Subroutine */ int dspr2_(char *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *), dscal_(integer *, doublereal *, doublereal *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *), dspmv_(char *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *, 
	     doublereal *, integer *);
    logical upper;
    extern /* Subroutine */ int dtpmv_(char *, char *, char *, integer *, 
	    doublereal *, doublereal *, integer *), 
	    dtpsv_(char *, char *, char *, integer *, doublereal *, 
	    doublereal *, integer *), xerbla_(char *, 
	    integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DSPGST reduces a real symmetric-definite generalized eigenproblem */
/*  to standard form, using packed storage. */

/*  If ITYPE = 1, the problem is A*x = lambda*B*x, */
/*  and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T) */

/*  If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/*  B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L. */

/*  B must have been previously factorized as U**T*U or L*L**T by DPPTRF. */

/*  Arguments */
/*  ========= */

/*  ITYPE   (input) INTEGER */
/*          = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T); */
/*          = 2 or 3: compute U*A*U**T or L**T*A*L. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored and B is factored as */
/*                  U**T*U; */
/*          = 'L':  Lower triangle of A is stored and B is factored as */
/*                  L*L**T. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  AP      (input/output) DOUBLE PRECISION array, dimension (N*(N+1)/2) */
/*          On entry, the upper or lower triangle of the symmetric matrix */
/*          A, packed columnwise in a linear array.  The j-th column of A */
/*          is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */

/*          On exit, if INFO = 0, the transformed matrix, stored in the */
/*          same format as A. */

/*  BP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2) */
/*          The triangular factor from the Cholesky factorization of B, */
/*          stored in the same format as A, as returned by DPPTRF. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --bp;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSPGST", &i__1);
	return 0;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U')*A*inv(U) */

/*           J1 and JJ are the indices of A(1,j) and A(j,j) */

	    jj = 0;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		j1 = jj + 1;
		jj += j;

/*              Compute the j-th column of the upper triangle of A */

		bjj = bp[jj];
		dtpsv_(uplo, "Transpose", "Nonunit", &j, &bp[1], &ap[j1], &
			c__1);
		i__2 = j - 1;
		dspmv_(uplo, &i__2, &c_b9, &ap[1], &bp[j1], &c__1, &c_b11, &
			ap[j1], &c__1);
		i__2 = j - 1;
		d__1 = 1. / bjj;
		dscal_(&i__2, &d__1, &ap[j1], &c__1);
		i__2 = j - 1;
		ap[jj] = (ap[jj] - ddot_(&i__2, &ap[j1], &c__1, &bp[j1], &
			c__1)) / bjj;
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L') */

/*           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1) */

	    kk = 1;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		k1k1 = kk + *n - k + 1;

/*              Update the lower triangle of A(k:n,k:n) */

		akk = ap[kk];
		bkk = bp[kk];
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		ap[kk] = akk;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    dscal_(&i__2, &d__1, &ap[kk + 1], &c__1);
		    ct = akk * -.5;
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &bp[kk + 1], &c__1, &ap[kk + 1], &c__1)
			    ;
		    i__2 = *n - k;
		    dspr2_(uplo, &i__2, &c_b9, &ap[kk + 1], &c__1, &bp[kk + 1]
, &c__1, &ap[k1k1]);
		    i__2 = *n - k;
		    daxpy_(&i__2, &ct, &bp[kk + 1], &c__1, &ap[kk + 1], &c__1)
			    ;
		    i__2 = *n - k;
		    dtpsv_(uplo, "No transpose", "Non-unit", &i__2, &bp[k1k1], 
			     &ap[kk + 1], &c__1);
		}
		kk = k1k1;
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U' */

/*           K1 and KK are the indices of A(1,k) and A(k,k) */

	    kk = 0;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		k1 = kk + 1;
		kk += k;

/*              Update the upper triangle of A(1:k,1:k) */

		akk = ap[kk];
		bkk = bp[kk];
		i__2 = k - 1;
		dtpmv_(uplo, "No transpose", "Non-unit", &i__2, &bp[1], &ap[
			k1], &c__1);
		ct = akk * .5;
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &bp[k1], &c__1, &ap[k1], &c__1);
		i__2 = k - 1;
		dspr2_(uplo, &i__2, &c_b11, &ap[k1], &c__1, &bp[k1], &c__1, &
			ap[1]);
		i__2 = k - 1;
		daxpy_(&i__2, &ct, &bp[k1], &c__1, &ap[k1], &c__1);
		i__2 = k - 1;
		dscal_(&i__2, &bkk, &ap[k1], &c__1);
/* Computing 2nd power */
		d__1 = bkk;
		ap[kk] = akk * (d__1 * d__1);
/* L30: */
	    }
	} else {

/*           Compute L'*A*L */

/*           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1) */

	    jj = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		j1j1 = jj + *n - j + 1;

/*              Compute the j-th column of the lower triangle of A */

		ajj = ap[jj];
		bjj = bp[jj];
		i__2 = *n - j;
		ap[jj] = ajj * bjj + ddot_(&i__2, &ap[jj + 1], &c__1, &bp[jj 
			+ 1], &c__1);
		i__2 = *n - j;
		dscal_(&i__2, &bjj, &ap[jj + 1], &c__1);
		i__2 = *n - j;
		dspmv_(uplo, &i__2, &c_b11, &ap[j1j1], &bp[jj + 1], &c__1, &
			c_b11, &ap[jj + 1], &c__1);
		i__2 = *n - j + 1;
		dtpmv_(uplo, "Transpose", "Non-unit", &i__2, &bp[jj], &ap[jj], 
			 &c__1);
		jj = j1j1;
/* L40: */
	    }
	}
    }
    return 0;

/*     End of DSPGST */

} /* dspgst_ */
