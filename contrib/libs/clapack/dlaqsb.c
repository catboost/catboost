/* dlaqsb.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dlaqsb_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *s, doublereal *scond, doublereal *amax, 
	 char *equed)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, j;
    doublereal cj, large;
    extern logical lsame_(char *, char *);
    doublereal small;
    extern doublereal dlamch_(char *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAQSB equilibrates a symmetric band matrix A using the scaling */
/*  factors in the vector S. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          symmetric matrix A is stored. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  KD      (input) INTEGER */
/*          The number of super-diagonals of the matrix A if UPLO = 'U', */
/*          or the number of sub-diagonals if UPLO = 'L'.  KD >= 0. */

/*  AB      (input/output) DOUBLE PRECISION array, dimension (LDAB,N) */
/*          On entry, the upper or lower triangle of the symmetric band */
/*          matrix A, stored in the first KD+1 rows of the array.  The */
/*          j-th column of A is stored in the j-th column of the array AB */
/*          as follows: */
/*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd). */

/*          On exit, if INFO = 0, the triangular factor U or L from the */
/*          Cholesky factorization A = U'*U or A = L*L' of the band */
/*          matrix A, in the same storage format as A. */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KD+1. */

/*  S       (input) DOUBLE PRECISION array, dimension (N) */
/*          The scale factors for A. */

/*  SCOND   (input) DOUBLE PRECISION */
/*          Ratio of the smallest S(i) to the largest S(i). */

/*  AMAX    (input) DOUBLE PRECISION */
/*          Absolute value of largest matrix entry. */

/*  EQUED   (output) CHARACTER*1 */
/*          Specifies whether or not equilibration was done. */
/*          = 'N':  No equilibration. */
/*          = 'Y':  Equilibration was done, i.e., A has been replaced by */
/*                  diag(S) * A * diag(S). */

/*  Internal Parameters */
/*  =================== */

/*  THRESH is a threshold value used to decide if scaling should be done */
/*  based on the ratio of the scaling factors.  If SCOND < THRESH, */
/*  scaling is done. */

/*  LARGE and SMALL are threshold values used to decide if scaling should */
/*  be done based on the absolute size of the largest matrix element. */
/*  If AMAX > LARGE or AMAX < SMALL, scaling is done. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --s;

    /* Function Body */
    if (*n <= 0) {
	*(unsigned char *)equed = 'N';
	return 0;
    }

/*     Initialize LARGE and SMALL. */

    small = dlamch_("Safe minimum") / dlamch_("Precision");
    large = 1. / small;

    if (*scond >= .1 && *amax >= small && *amax <= large) {

/*        No equilibration */

	*(unsigned char *)equed = 'N';
    } else {

/*        Replace A by diag(S) * A * diag(S). */

	if (lsame_(uplo, "U")) {

/*           Upper triangle of A is stored in band format. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		cj = s[j];
/* Computing MAX */
		i__2 = 1, i__3 = j - *kd;
		i__4 = j;
		for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
		    ab[*kd + 1 + i__ - j + j * ab_dim1] = cj * s[i__] * ab[*
			    kd + 1 + i__ - j + j * ab_dim1];
/* L10: */
		}
/* L20: */
	    }
	} else {

/*           Lower triangle of A is stored. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		cj = s[j];
/* Computing MIN */
		i__2 = *n, i__3 = j + *kd;
		i__4 = min(i__2,i__3);
		for (i__ = j; i__ <= i__4; ++i__) {
		    ab[i__ + 1 - j + j * ab_dim1] = cj * s[i__] * ab[i__ + 1 
			    - j + j * ab_dim1];
/* L30: */
		}
/* L40: */
	    }
	}
	*(unsigned char *)equed = 'Y';
    }

    return 0;

/*     End of DLAQSB */

} /* dlaqsb_ */
