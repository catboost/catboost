/* zlaqge.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlaqge_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, 
	doublereal *colcnd, doublereal *amax, char *equed)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1;
    doublecomplex z__1;

    /* Local variables */
    integer i__, j;
    doublereal cj, large, small;
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

/*  ZLAQGE equilibrates a general M by N matrix A using the row and */
/*  column scaling factors in the vectors R and C. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the M by N matrix A. */
/*          On exit, the equilibrated matrix.  See EQUED for the form of */
/*          the equilibrated matrix. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(M,1). */

/*  R       (input) DOUBLE PRECISION array, dimension (M) */
/*          The row scale factors for A. */

/*  C       (input) DOUBLE PRECISION array, dimension (N) */
/*          The column scale factors for A. */

/*  ROWCND  (input) DOUBLE PRECISION */
/*          Ratio of the smallest R(i) to the largest R(i). */

/*  COLCND  (input) DOUBLE PRECISION */
/*          Ratio of the smallest C(i) to the largest C(i). */

/*  AMAX    (input) DOUBLE PRECISION */
/*          Absolute value of largest matrix entry. */

/*  EQUED   (output) CHARACTER*1 */
/*          Specifies the form of equilibration that was done. */
/*          = 'N':  No equilibration */
/*          = 'R':  Row equilibration, i.e., A has been premultiplied by */
/*                  diag(R). */
/*          = 'C':  Column equilibration, i.e., A has been postmultiplied */
/*                  by diag(C). */
/*          = 'B':  Both row and column equilibration, i.e., A has been */
/*                  replaced by diag(R) * A * diag(C). */

/*  Internal Parameters */
/*  =================== */

/*  THRESH is a threshold value used to decide if row or column scaling */
/*  should be done based on the ratio of the row or column scaling */
/*  factors.  If ROWCND < THRESH, row scaling is done, and if */
/*  COLCND < THRESH, column scaling is done. */

/*  LARGE and SMALL are threshold values used to decide if row scaling */
/*  should be done based on the absolute size of the largest matrix */
/*  element.  If AMAX > LARGE or AMAX < SMALL, row scaling is done. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --r__;
    --c__;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	*(unsigned char *)equed = 'N';
	return 0;
    }

/*     Initialize LARGE and SMALL. */

    small = dlamch_("Safe minimum") / dlamch_("Precision");
    large = 1. / small;

    if (*rowcnd >= .1 && *amax >= small && *amax <= large) {

/*        No row scaling */

	if (*colcnd >= .1) {

/*           No column scaling */

	    *(unsigned char *)equed = 'N';
	} else {

/*           Column scaling */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		cj = c__[j];
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * a_dim1;
		    i__4 = i__ + j * a_dim1;
		    z__1.r = cj * a[i__4].r, z__1.i = cj * a[i__4].i;
		    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L10: */
		}
/* L20: */
	    }
	    *(unsigned char *)equed = 'C';
	}
    } else if (*colcnd >= .1) {

/*        Row scaling, no column scaling */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		i__4 = i__;
		i__5 = i__ + j * a_dim1;
		z__1.r = r__[i__4] * a[i__5].r, z__1.i = r__[i__4] * a[i__5]
			.i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L30: */
	    }
/* L40: */
	}
	*(unsigned char *)equed = 'R';
    } else {

/*        Row and column scaling */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    cj = c__[j];
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		d__1 = cj * r__[i__];
		i__4 = i__ + j * a_dim1;
		z__1.r = d__1 * a[i__4].r, z__1.i = d__1 * a[i__4].i;
		a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L50: */
	    }
/* L60: */
	}
	*(unsigned char *)equed = 'B';
    }

    return 0;

/*     End of ZLAQGE */

} /* zlaqge_ */
