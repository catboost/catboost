/* dlat2s.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dlat2s_(char *uplo, integer *n, doublereal *a, integer *
	lda, real *sa, integer *ldsa, integer *info)
{
    /* System generated locals */
    integer sa_dim1, sa_offset, a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__, j;
    doublereal rmax;
    extern logical lsame_(char *, char *);
    logical upper;
    extern doublereal slamch_(char *);


/*  -- LAPACK PROTOTYPE auxiliary routine (version 3.1.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     May 2007 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAT2S converts a DOUBLE PRECISION triangular matrix, SA, to a SINGLE */
/*  PRECISION triangular matrix, A. */

/*  RMAX is the overflow for the SINGLE PRECISION arithmetic */
/*  DLAS2S checks that all the entries of A are between -RMAX and */
/*  RMAX. If not the convertion is aborted and a flag is raised. */

/*  This is an auxiliary routine so there is no argument checking. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  A is upper triangular; */
/*          = 'L':  A is lower triangular. */

/*  N       (input) INTEGER */
/*          The number of rows and columns of the matrix A.  N >= 0. */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the N-by-N triangular coefficient matrix A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  SA      (output) REAL array, dimension (LDSA,N) */
/*          Only the UPLO part of SA is referenced.  On exit, if INFO=0, */
/*          the N-by-N coefficient matrix SA; if INFO>0, the content of */
/*          the UPLO part of SA is unspecified. */

/*  LDSA    (input) INTEGER */
/*          The leading dimension of the array SA.  LDSA >= max(1,M). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          = 1:  an entry of the matrix A is greater than the SINGLE */
/*                PRECISION overflow threshold, in this case, the content */
/*                of the UPLO part of SA in exit is unspecified. */

/*  ========= */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    sa_dim1 = *ldsa;
    sa_offset = 1 + sa_dim1;
    sa -= sa_offset;

    /* Function Body */
    rmax = slamch_("O");
    upper = lsame_(uplo, "U");
    if (upper) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		if (a[i__ + j * a_dim1] < -rmax || a[i__ + j * a_dim1] > rmax)
			 {
		    *info = 1;
		    goto L50;
		}
		sa[i__ + j * sa_dim1] = a[i__ + j * a_dim1];
/* L10: */
	    }
/* L20: */
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = j; i__ <= i__2; ++i__) {
		if (a[i__ + j * a_dim1] < -rmax || a[i__ + j * a_dim1] > rmax)
			 {
		    *info = 1;
		    goto L50;
		}
		sa[i__ + j * sa_dim1] = a[i__ + j * a_dim1];
/* L30: */
	    }
/* L40: */
	}
    }
L50:

    return 0;

/*     End of DLAT2S */

} /* dlat2s_ */
