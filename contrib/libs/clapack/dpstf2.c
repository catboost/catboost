/* dpstf2.f -- translated by f2c (version 20061008).
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
static doublereal c_b16 = -1.;
static doublereal c_b18 = 1.;

/* Subroutine */ int dpstf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *piv, integer *rank, doublereal *tol, doublereal *work, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, maxlocval;
    doublereal ajj;
    integer pvt;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *);
    doublereal dtemp;
    integer itemp;
    extern /* Subroutine */ int dswap_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    doublereal dstop;
    logical upper;
    extern doublereal dlamch_(char *);
    extern logical disnan_(doublereal *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer dmaxloc_(doublereal *, integer *);


/*  -- LAPACK PROTOTYPE routine (version 3.2) -- */
/*     Craig Lucas, University of Manchester / NAG Ltd. */
/*     October, 2008 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DPSTF2 computes the Cholesky factorization with complete */
/*  pivoting of a real symmetric positive semidefinite matrix A. */

/*  The factorization has the form */
/*     P' * A * P = U' * U ,  if UPLO = 'U', */
/*     P' * A * P = L  * L',  if UPLO = 'L', */
/*  where U is an upper triangular matrix and L is lower triangular, and */
/*  P is stored as vector PIV. */

/*  This algorithm does not attempt to check that A is positive */
/*  semidefinite. This version of the algorithm calls level 2 BLAS. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          symmetric matrix A is stored. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the symmetric matrix A.  If UPLO = 'U', the leading */
/*          n by n upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading n by n lower triangular part of A contains the lower */
/*          triangular part of the matrix A, and the strictly upper */
/*          triangular part of A is not referenced. */

/*          On exit, if INFO = 0, the factor U or L from the Cholesky */
/*          factorization as above. */

/*  PIV     (output) INTEGER array, dimension (N) */
/*          PIV is such that the nonzero entries are P( PIV(K), K ) = 1. */

/*  RANK    (output) INTEGER */
/*          The rank of A given by the number of steps the algorithm */
/*          completed. */

/*  TOL     (input) DOUBLE PRECISION */
/*          User defined tolerance. If TOL < 0, then N*U*MAX( A( K,K ) ) */
/*          will be used. The algorithm terminates at the (K-1)st step */
/*          if the pivot <= TOL. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  WORK    DOUBLE PRECISION array, dimension (2*N) */
/*          Work space. */

/*  INFO    (output) INTEGER */
/*          < 0: If INFO = -K, the K-th argument had an illegal value, */
/*          = 0: algorithm completed successfully, and */
/*          > 0: the matrix A is either rank deficient with computed rank */
/*               as returned in RANK, or is indefinite.  See Section 7 of */
/*               LAPACK Working Note #161 for further information. */

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

/*     Test the input parameters */

    /* Parameter adjustments */
    --work;
    --piv;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPSTF2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Initialize PIV */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	piv[i__] = i__;
/* L100: */
    }

/*     Compute stopping value */

    pvt = 1;
    ajj = a[pvt + pvt * a_dim1];
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if (a[i__ + i__ * a_dim1] > ajj) {
	    pvt = i__;
	    ajj = a[pvt + pvt * a_dim1];
	}
    }
    if (ajj == 0. || disnan_(&ajj)) {
	*rank = 0;
	*info = 1;
	goto L170;
    }

/*     Compute stopping value if not supplied */

    if (*tol < 0.) {
	dstop = *n * dlamch_("Epsilon") * ajj;
    } else {
	dstop = *tol;
    }

/*     Set first half of WORK to zero, holds dot products */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = 0.;
/* L110: */
    }

    if (upper) {

/*        Compute the Cholesky factorization P' * A * P = U' * U */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*        Find pivot, test for exit, else swap rows and columns */
/*        Update dot products, compute possible pivots which are */
/*        stored in the second half of WORK */

	    i__2 = *n;
	    for (i__ = j; i__ <= i__2; ++i__) {

		if (j > 1) {
/* Computing 2nd power */
		    d__1 = a[j - 1 + i__ * a_dim1];
		    work[i__] += d__1 * d__1;
		}
		work[*n + i__] = a[i__ + i__ * a_dim1] - work[i__];

/* L120: */
	    }

	    if (j > 1) {
		maxlocval = (*n << 1) - (*n + j) + 1;
		itemp = dmaxloc_(&work[*n + j], &maxlocval);
		pvt = itemp + j - 1;
		ajj = work[*n + pvt];
		if (ajj <= dstop || disnan_(&ajj)) {
		    a[j + j * a_dim1] = ajj;
		    goto L160;
		}
	    }

	    if (j != pvt) {

/*              Pivot OK, so can now swap pivot rows and columns */

		a[pvt + pvt * a_dim1] = a[j + j * a_dim1];
		i__2 = j - 1;
		dswap_(&i__2, &a[j * a_dim1 + 1], &c__1, &a[pvt * a_dim1 + 1], 
			 &c__1);
		if (pvt < *n) {
		    i__2 = *n - pvt;
		    dswap_(&i__2, &a[j + (pvt + 1) * a_dim1], lda, &a[pvt + (
			    pvt + 1) * a_dim1], lda);
		}
		i__2 = pvt - j - 1;
		dswap_(&i__2, &a[j + (j + 1) * a_dim1], lda, &a[j + 1 + pvt * 
			a_dim1], &c__1);

/*              Swap dot products and PIV */

		dtemp = work[j];
		work[j] = work[pvt];
		work[pvt] = dtemp;
		itemp = piv[pvt];
		piv[pvt] = piv[j];
		piv[j] = itemp;
	    }

	    ajj = sqrt(ajj);
	    a[j + j * a_dim1] = ajj;

/*           Compute elements J+1:N of row J */

	    if (j < *n) {
		i__2 = j - 1;
		i__3 = *n - j;
		dgemv_("Trans", &i__2, &i__3, &c_b16, &a[(j + 1) * a_dim1 + 1]
, lda, &a[j * a_dim1 + 1], &c__1, &c_b18, &a[j + (j + 
			1) * a_dim1], lda);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		dscal_(&i__2, &d__1, &a[j + (j + 1) * a_dim1], lda);
	    }

/* L130: */
	}

    } else {

/*        Compute the Cholesky factorization P' * A * P = L * L' */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {

/*        Find pivot, test for exit, else swap rows and columns */
/*        Update dot products, compute possible pivots which are */
/*        stored in the second half of WORK */

	    i__2 = *n;
	    for (i__ = j; i__ <= i__2; ++i__) {

		if (j > 1) {
/* Computing 2nd power */
		    d__1 = a[i__ + (j - 1) * a_dim1];
		    work[i__] += d__1 * d__1;
		}
		work[*n + i__] = a[i__ + i__ * a_dim1] - work[i__];

/* L140: */
	    }

	    if (j > 1) {
		maxlocval = (*n << 1) - (*n + j) + 1;
		itemp = dmaxloc_(&work[*n + j], &maxlocval);
		pvt = itemp + j - 1;
		ajj = work[*n + pvt];
		if (ajj <= dstop || disnan_(&ajj)) {
		    a[j + j * a_dim1] = ajj;
		    goto L160;
		}
	    }

	    if (j != pvt) {

/*              Pivot OK, so can now swap pivot rows and columns */

		a[pvt + pvt * a_dim1] = a[j + j * a_dim1];
		i__2 = j - 1;
		dswap_(&i__2, &a[j + a_dim1], lda, &a[pvt + a_dim1], lda);
		if (pvt < *n) {
		    i__2 = *n - pvt;
		    dswap_(&i__2, &a[pvt + 1 + j * a_dim1], &c__1, &a[pvt + 1 
			    + pvt * a_dim1], &c__1);
		}
		i__2 = pvt - j - 1;
		dswap_(&i__2, &a[j + 1 + j * a_dim1], &c__1, &a[pvt + (j + 1) 
			* a_dim1], lda);

/*              Swap dot products and PIV */

		dtemp = work[j];
		work[j] = work[pvt];
		work[pvt] = dtemp;
		itemp = piv[pvt];
		piv[pvt] = piv[j];
		piv[j] = itemp;
	    }

	    ajj = sqrt(ajj);
	    a[j + j * a_dim1] = ajj;

/*           Compute elements J+1:N of column J */

	    if (j < *n) {
		i__2 = *n - j;
		i__3 = j - 1;
		dgemv_("No Trans", &i__2, &i__3, &c_b16, &a[j + 1 + a_dim1], 
			lda, &a[j + a_dim1], lda, &c_b18, &a[j + 1 + j * 
			a_dim1], &c__1);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		dscal_(&i__2, &d__1, &a[j + 1 + j * a_dim1], &c__1);
	    }

/* L150: */
	}

    }

/*     Ran to completion, A has full rank */

    *rank = *n;

    goto L170;
L160:

/*     Rank is number of steps completed.  Set INFO = 1 to signal */
/*     that the factorization cannot be used to solve a system. */

    *rank = j - 1;
    *info = 1;

L170:
    return 0;

/*     End of DPSTF2 */

} /* dpstf2_ */
