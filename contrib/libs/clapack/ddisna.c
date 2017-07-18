/* ddisna.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ddisna_(char *job, integer *m, integer *n, doublereal *
	d__, doublereal *sep, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    integer i__, k;
    doublereal eps;
    logical decr, left, incr, sing, eigen;
    extern logical lsame_(char *, char *);
    doublereal anorm;
    logical right;
    extern doublereal dlamch_(char *);
    doublereal oldgap, safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal newgap, thresh;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DDISNA computes the reciprocal condition numbers for the eigenvectors */
/*  of a real symmetric or complex Hermitian matrix or for the left or */
/*  right singular vectors of a general m-by-n matrix. The reciprocal */
/*  condition number is the 'gap' between the corresponding eigenvalue or */
/*  singular value and the nearest other one. */

/*  The bound on the error, measured by angle in radians, in the I-th */
/*  computed vector is given by */

/*         DLAMCH( 'E' ) * ( ANORM / SEP( I ) ) */

/*  where ANORM = 2-norm(A) = max( abs( D(j) ) ).  SEP(I) is not allowed */
/*  to be smaller than DLAMCH( 'E' )*ANORM in order to limit the size of */
/*  the error bound. */

/*  DDISNA may also be used to compute error bounds for eigenvectors of */
/*  the generalized symmetric definite eigenproblem. */

/*  Arguments */
/*  ========= */

/*  JOB     (input) CHARACTER*1 */
/*          Specifies for which problem the reciprocal condition numbers */
/*          should be computed: */
/*          = 'E':  the eigenvectors of a symmetric/Hermitian matrix; */
/*          = 'L':  the left singular vectors of a general matrix; */
/*          = 'R':  the right singular vectors of a general matrix. */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix. M >= 0. */

/*  N       (input) INTEGER */
/*          If JOB = 'L' or 'R', the number of columns of the matrix, */
/*          in which case N >= 0. Ignored if JOB = 'E'. */

/*  D       (input) DOUBLE PRECISION array, dimension (M) if JOB = 'E' */
/*                              dimension (min(M,N)) if JOB = 'L' or 'R' */
/*          The eigenvalues (if JOB = 'E') or singular values (if JOB = */
/*          'L' or 'R') of the matrix, in either increasing or decreasing */
/*          order. If singular values, they must be non-negative. */

/*  SEP     (output) DOUBLE PRECISION array, dimension (M) if JOB = 'E' */
/*                               dimension (min(M,N)) if JOB = 'L' or 'R' */
/*          The reciprocal condition numbers of the vectors. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    --sep;
    --d__;

    /* Function Body */
    *info = 0;
    eigen = lsame_(job, "E");
    left = lsame_(job, "L");
    right = lsame_(job, "R");
    sing = left || right;
    if (eigen) {
	k = *m;
    } else if (sing) {
	k = min(*m,*n);
    }
    if (! eigen && ! sing) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (k < 0) {
	*info = -3;
    } else {
	incr = TRUE_;
	decr = TRUE_;
	i__1 = k - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (incr) {
		incr = incr && d__[i__] <= d__[i__ + 1];
	    }
	    if (decr) {
		decr = decr && d__[i__] >= d__[i__ + 1];
	    }
/* L10: */
	}
	if (sing && k > 0) {
	    if (incr) {
		incr = incr && 0. <= d__[1];
	    }
	    if (decr) {
		decr = decr && d__[k] >= 0.;
	    }
	}
	if (! (incr || decr)) {
	    *info = -4;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DDISNA", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (k == 0) {
	return 0;
    }

/*     Compute reciprocal condition numbers */

    if (k == 1) {
	sep[1] = dlamch_("O");
    } else {
	oldgap = (d__1 = d__[2] - d__[1], abs(d__1));
	sep[1] = oldgap;
	i__1 = k - 1;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    newgap = (d__1 = d__[i__ + 1] - d__[i__], abs(d__1));
	    sep[i__] = min(oldgap,newgap);
	    oldgap = newgap;
/* L20: */
	}
	sep[k] = oldgap;
    }
    if (sing) {
	if (left && *m > *n || right && *m < *n) {
	    if (incr) {
		sep[1] = min(sep[1],d__[1]);
	    }
	    if (decr) {
/* Computing MIN */
		d__1 = sep[k], d__2 = d__[k];
		sep[k] = min(d__1,d__2);
	    }
	}
    }

/*     Ensure that reciprocal condition numbers are not less than */
/*     threshold, in order to limit the size of the error bound */

    eps = dlamch_("E");
    safmin = dlamch_("S");
/* Computing MAX */
    d__2 = abs(d__[1]), d__3 = (d__1 = d__[k], abs(d__1));
    anorm = max(d__2,d__3);
    if (anorm == 0.) {
	thresh = eps;
    } else {
/* Computing MAX */
	d__1 = eps * anorm;
	thresh = max(d__1,safmin);
    }
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	d__1 = sep[i__];
	sep[i__] = max(d__1,thresh);
/* L30: */
    }

    return 0;

/*     End of DDISNA */

} /* ddisna_ */
