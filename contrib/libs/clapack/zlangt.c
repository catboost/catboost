/* zlangt.f -- translated by f2c (version 20061008).
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

doublereal zlangt_(char *norm, integer *n, doublecomplex *dl, doublecomplex *
	d__, doublecomplex *du)
{
    /* System generated locals */
    integer i__1;
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double z_abs(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    integer i__;
    doublereal sum, scale;
    extern logical lsame_(char *, char *);
    doublereal anorm;
    extern /* Subroutine */ int zlassq_(integer *, doublecomplex *, integer *, 
	     doublereal *, doublereal *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLANGT  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the  element of  largest absolute value  of a */
/*  complex tridiagonal matrix A. */

/*  Description */
/*  =========== */

/*  ZLANGT returns the value */

/*     ZLANGT = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
/*              ( */
/*              ( norm1(A),         NORM = '1', 'O' or 'o' */
/*              ( */
/*              ( normI(A),         NORM = 'I' or 'i' */
/*              ( */
/*              ( normF(A),         NORM = 'F', 'f', 'E' or 'e' */

/*  where  norm1  denotes the  one norm of a matrix (maximum column sum), */
/*  normI  denotes the  infinity norm  of a matrix  (maximum row sum) and */
/*  normF  denotes the  Frobenius norm of a matrix (square root of sum of */
/*  squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm. */

/*  Arguments */
/*  ========= */

/*  NORM    (input) CHARACTER*1 */
/*          Specifies the value to be returned in ZLANGT as described */
/*          above. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0.  When N = 0, ZLANGT is */
/*          set to zero. */

/*  DL      (input) COMPLEX*16 array, dimension (N-1) */
/*          The (n-1) sub-diagonal elements of A. */

/*  D       (input) COMPLEX*16 array, dimension (N) */
/*          The diagonal elements of A. */

/*  DU      (input) COMPLEX*16 array, dimension (N-1) */
/*          The (n-1) super-diagonal elements of A. */

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

    /* Parameter adjustments */
    --du;
    --d__;
    --dl;

    /* Function Body */
    if (*n <= 0) {
	anorm = 0.;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	anorm = z_abs(&d__[*n]);
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__1 = anorm, d__2 = z_abs(&dl[i__]);
	    anorm = max(d__1,d__2);
/* Computing MAX */
	    d__1 = anorm, d__2 = z_abs(&d__[i__]);
	    anorm = max(d__1,d__2);
/* Computing MAX */
	    d__1 = anorm, d__2 = z_abs(&du[i__]);
	    anorm = max(d__1,d__2);
/* L10: */
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {

/*        Find norm1(A). */

	if (*n == 1) {
	    anorm = z_abs(&d__[1]);
	} else {
/* Computing MAX */
	    d__1 = z_abs(&d__[1]) + z_abs(&dl[1]), d__2 = z_abs(&d__[*n]) + 
		    z_abs(&du[*n - 1]);
	    anorm = max(d__1,d__2);
	    i__1 = *n - 1;
	    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MAX */
		d__1 = anorm, d__2 = z_abs(&d__[i__]) + z_abs(&dl[i__]) + 
			z_abs(&du[i__ - 1]);
		anorm = max(d__1,d__2);
/* L20: */
	    }
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	if (*n == 1) {
	    anorm = z_abs(&d__[1]);
	} else {
/* Computing MAX */
	    d__1 = z_abs(&d__[1]) + z_abs(&du[1]), d__2 = z_abs(&d__[*n]) + 
		    z_abs(&dl[*n - 1]);
	    anorm = max(d__1,d__2);
	    i__1 = *n - 1;
	    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MAX */
		d__1 = anorm, d__2 = z_abs(&d__[i__]) + z_abs(&du[i__]) + 
			z_abs(&dl[i__ - 1]);
		anorm = max(d__1,d__2);
/* L30: */
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.;
	sum = 1.;
	zlassq_(n, &d__[1], &c__1, &scale, &sum);
	if (*n > 1) {
	    i__1 = *n - 1;
	    zlassq_(&i__1, &dl[1], &c__1, &scale, &sum);
	    i__1 = *n - 1;
	    zlassq_(&i__1, &du[1], &c__1, &scale, &sum);
	}
	anorm = scale * sqrt(sum);
    }

    ret_val = anorm;
    return ret_val;

/*     End of ZLANGT */

} /* zlangt_ */
