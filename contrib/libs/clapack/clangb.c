/* clangb.f -- translated by f2c (version 20061008).
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

doublereal clangb_(char *norm, integer *n, integer *kl, integer *ku, complex *
	ab, integer *ldab, real *work)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    real ret_val, r__1, r__2;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

    /* Local variables */
    integer i__, j, k, l;
    real sum, scale;
    extern logical lsame_(char *, char *);
    real value;
    extern /* Subroutine */ int classq_(integer *, complex *, integer *, real 
	    *, real *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLANGB  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the element of  largest absolute value  of an */
/*  n by n band matrix  A,  with kl sub-diagonals and ku super-diagonals. */

/*  Description */
/*  =========== */

/*  CLANGB returns the value */

/*     CLANGB = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
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
/*          Specifies the value to be returned in CLANGB as described */
/*          above. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0.  When N = 0, CLANGB is */
/*          set to zero. */

/*  KL      (input) INTEGER */
/*          The number of sub-diagonals of the matrix A.  KL >= 0. */

/*  KU      (input) INTEGER */
/*          The number of super-diagonals of the matrix A.  KU >= 0. */

/*  AB      (input) COMPLEX array, dimension (LDAB,N) */
/*          The band matrix A, stored in rows 1 to KL+KU+1.  The j-th */
/*          column of A is stored in the j-th column of the array AB as */
/*          follows: */
/*          AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(n,j+kl). */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KL+KU+1. */

/*  WORK    (workspace) REAL array, dimension (MAX(1,LWORK)), */
/*          where LWORK >= N when NORM = 'I'; otherwise, WORK is not */
/*          referenced. */

/* ===================================================================== */

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
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --work;

    /* Function Body */
    if (*n == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    i__2 = *ku + 2 - j;
/* Computing MIN */
	    i__4 = *n + *ku + 1 - j, i__5 = *kl + *ku + 1;
	    i__3 = min(i__4,i__5);
	    for (i__ = max(i__2,1); i__ <= i__3; ++i__) {
/* Computing MAX */
		r__1 = value, r__2 = c_abs(&ab[i__ + j * ab_dim1]);
		value = dmax(r__1,r__2);
/* L10: */
	    }
/* L20: */
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {

/*        Find norm1(A). */

	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.f;
/* Computing MAX */
	    i__3 = *ku + 2 - j;
/* Computing MIN */
	    i__4 = *n + *ku + 1 - j, i__5 = *kl + *ku + 1;
	    i__2 = min(i__4,i__5);
	    for (i__ = max(i__3,1); i__ <= i__2; ++i__) {
		sum += c_abs(&ab[i__ + j * ab_dim1]);
/* L30: */
	    }
	    value = dmax(value,sum);
/* L40: */
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.f;
/* L50: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    k = *ku + 1 - j;
/* Computing MAX */
	    i__2 = 1, i__3 = j - *ku;
/* Computing MIN */
	    i__5 = *n, i__6 = j + *kl;
	    i__4 = min(i__5,i__6);
	    for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
		work[i__] += c_abs(&ab[k + i__ + j * ab_dim1]);
/* L60: */
	    }
/* L70: */
	}
	value = 0.f;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__1 = value, r__2 = work[i__];
	    value = dmax(r__1,r__2);
/* L80: */
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    i__4 = 1, i__2 = j - *ku;
	    l = max(i__4,i__2);
	    k = *ku + 1 - j + l;
/* Computing MIN */
	    i__2 = *n, i__3 = j + *kl;
	    i__4 = min(i__2,i__3) - l + 1;
	    classq_(&i__4, &ab[k + j * ab_dim1], &c__1, &scale, &sum);
/* L90: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANGB */

} /* clangb_ */
