/* zlag2c.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlag2c_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, complex *sa, integer *ldsa, integer *info)
{
    /* System generated locals */
    integer sa_dim1, sa_offset, a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    integer i__, j;
    doublereal rmax;
    extern doublereal slamch_(char *);


/*  -- LAPACK PROTOTYPE auxiliary routine (version 3.1.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     August 2007 */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLAG2C converts a COMPLEX*16 matrix, SA, to a COMPLEX matrix, A. */

/*  RMAX is the overflow for the SINGLE PRECISION arithmetic */
/*  ZLAG2C checks that all the entries of A are between -RMAX and */
/*  RMAX. If not the convertion is aborted and a flag is raised. */

/*  This is an auxiliary routine so there is no argument checking. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of lines of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the M-by-N coefficient matrix A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  SA      (output) COMPLEX array, dimension (LDSA,N) */
/*          On exit, if INFO=0, the M-by-N coefficient matrix SA; if */
/*          INFO>0, the content of SA is unspecified. */

/*  LDSA    (input) INTEGER */
/*          The leading dimension of the array SA.  LDSA >= max(1,M). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          = 1:  an entry of the matrix A is greater than the SINGLE */
/*                PRECISION overflow threshold, in this case, the content */
/*                of SA in exit is unspecified. */

/*  ========= */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
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
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    i__4 = i__ + j * a_dim1;
	    if (a[i__3].r < -rmax || a[i__4].r > rmax || d_imag(&a[i__ + j * 
		    a_dim1]) < -rmax || d_imag(&a[i__ + j * a_dim1]) > rmax) {
		*info = 1;
		goto L30;
	    }
	    i__3 = i__ + j * sa_dim1;
	    i__4 = i__ + j * a_dim1;
	    sa[i__3].r = a[i__4].r, sa[i__3].i = a[i__4].i;
/* L10: */
	}
/* L20: */
    }
    *info = 0;
L30:
    return 0;

/*     End of ZLAG2C */

} /* zlag2c_ */
