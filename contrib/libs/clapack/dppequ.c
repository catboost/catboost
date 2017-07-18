/* dppequ.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dppequ_(char *uplo, integer *n, doublereal *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, jj;
    doublereal smin;
    extern logical lsame_(char *, char *);
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DPPEQU computes row and column scalings intended to equilibrate a */
/*  symmetric positive definite matrix A in packed storage and reduce */
/*  its condition number (with respect to the two-norm).  S contains the */
/*  scale factors, S(i)=1/sqrt(A(i,i)), chosen so that the scaled matrix */
/*  B with elements B(i,j)=S(i)*A(i,j)*S(j) has ones on the diagonal. */
/*  This choice of S puts the condition number of B within a factor N of */
/*  the smallest possible condition number over all possible diagonal */
/*  scalings. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input) DOUBLE PRECISION array, dimension (N*(N+1)/2) */
/*          The upper or lower triangle of the symmetric matrix A, packed */
/*          columnwise in a linear array.  The j-th column of A is stored */
/*          in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */

/*  S       (output) DOUBLE PRECISION array, dimension (N) */
/*          If INFO = 0, S contains the scale factors for A. */

/*  SCOND   (output) DOUBLE PRECISION */
/*          If INFO = 0, S contains the ratio of the smallest S(i) to */
/*          the largest S(i).  If SCOND >= 0.1 and AMAX is neither too */
/*          large nor too small, it is not worth scaling by S. */

/*  AMAX    (output) DOUBLE PRECISION */
/*          Absolute value of largest matrix element.  If AMAX is very */
/*          close to overflow or very close to underflow, the matrix */
/*          should be scaled. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, the i-th diagonal element is nonpositive. */

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

/*     Test the input parameters. */

    /* Parameter adjustments */
    --s;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPPEQU", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	*scond = 1.;
	*amax = 0.;
	return 0;
    }

/*     Initialize SMIN and AMAX. */

    s[1] = ap[1];
    smin = s[1];
    *amax = s[1];

    if (upper) {

/*        UPLO = 'U':  Upper triangle of A is stored. */
/*        Find the minimum and maximum diagonal elements. */

	jj = 1;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    jj += i__;
	    s[i__] = ap[jj];
/* Computing MIN */
	    d__1 = smin, d__2 = s[i__];
	    smin = min(d__1,d__2);
/* Computing MAX */
	    d__1 = *amax, d__2 = s[i__];
	    *amax = max(d__1,d__2);
/* L10: */
	}

    } else {

/*        UPLO = 'L':  Lower triangle of A is stored. */
/*        Find the minimum and maximum diagonal elements. */

	jj = 1;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    jj = jj + *n - i__ + 2;
	    s[i__] = ap[jj];
/* Computing MIN */
	    d__1 = smin, d__2 = s[i__];
	    smin = min(d__1,d__2);
/* Computing MAX */
	    d__1 = *amax, d__2 = s[i__];
	    *amax = max(d__1,d__2);
/* L20: */
	}
    }

    if (smin <= 0.) {

/*        Find the first non-positive diagonal element and return. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (s[i__] <= 0.) {
		*info = i__;
		return 0;
	    }
/* L30: */
	}
    } else {

/*        Set the scale factors to the reciprocals */
/*        of the diagonal elements. */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    s[i__] = 1. / sqrt(s[i__]);
/* L40: */
	}

/*        Compute SCOND = min(S(I)) / max(S(I)) */

	*scond = sqrt(smin) / sqrt(*amax);
    }
    return 0;

/*     End of DPPEQU */

} /* dppequ_ */
