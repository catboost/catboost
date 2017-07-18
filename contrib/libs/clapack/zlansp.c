/* zlansp.f -- translated by f2c (version 20061008).
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

doublereal zlansp_(char *norm, char *uplo, integer *n, doublecomplex *ap, 
	doublereal *work)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double z_abs(doublecomplex *), d_imag(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    integer i__, j, k;
    doublereal sum, absa, scale;
    extern logical lsame_(char *, char *);
    doublereal value;
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

/*  ZLANSP  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the  element of  largest absolute value  of a */
/*  complex symmetric matrix A,  supplied in packed form. */

/*  Description */
/*  =========== */

/*  ZLANSP returns the value */

/*     ZLANSP = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
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
/*          Specifies the value to be returned in ZLANSP as described */
/*          above. */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          symmetric matrix A is supplied. */
/*          = 'U':  Upper triangular part of A is supplied */
/*          = 'L':  Lower triangular part of A is supplied */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0.  When N = 0, ZLANSP is */
/*          set to zero. */

/*  AP      (input) COMPLEX*16 array, dimension (N*(N+1)/2) */
/*          The upper or lower triangle of the symmetric matrix A, packed */
/*          columnwise in a linear array.  The j-th column of A is stored */
/*          in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK)), */
/*          where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise, */
/*          WORK is not referenced. */

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
    --work;
    --ap;

    /* Function Body */
    if (*n == 0) {
	value = 0.;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.;
	if (lsame_(uplo, "U")) {
	    k = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = k + j - 1;
		for (i__ = k; i__ <= i__2; ++i__) {
/* Computing MAX */
		    d__1 = value, d__2 = z_abs(&ap[i__]);
		    value = max(d__1,d__2);
/* L10: */
		}
		k += j;
/* L20: */
	    }
	} else {
	    k = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = k + *n - j;
		for (i__ = k; i__ <= i__2; ++i__) {
/* Computing MAX */
		    d__1 = value, d__2 = z_abs(&ap[i__]);
		    value = max(d__1,d__2);
/* L30: */
		}
		k = k + *n - j + 1;
/* L40: */
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*        Find normI(A) ( = norm1(A), since A is symmetric). */

	value = 0.;
	k = 1;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = z_abs(&ap[k]);
		    sum += absa;
		    work[i__] += absa;
		    ++k;
/* L50: */
		}
		work[j] = sum + z_abs(&ap[k]);
		++k;
/* L60: */
	    }
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		d__1 = value, d__2 = work[i__];
		value = max(d__1,d__2);
/* L70: */
	    }
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] = 0.;
/* L80: */
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = work[j] + z_abs(&ap[k]);
		++k;
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = z_abs(&ap[k]);
		    sum += absa;
		    work[i__] += absa;
		    ++k;
/* L90: */
		}
		value = max(value,sum);
/* L100: */
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.;
	sum = 1.;
	k = 2;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		i__2 = j - 1;
		zlassq_(&i__2, &ap[k], &c__1, &scale, &sum);
		k += j;
/* L110: */
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		zlassq_(&i__2, &ap[k], &c__1, &scale, &sum);
		k = k + *n - j + 1;
/* L120: */
	    }
	}
	sum *= 2;
	k = 1;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = k;
	    if (ap[i__2].r != 0.) {
		i__2 = k;
		absa = (d__1 = ap[i__2].r, abs(d__1));
		if (scale < absa) {
/* Computing 2nd power */
		    d__1 = scale / absa;
		    sum = sum * (d__1 * d__1) + 1.;
		    scale = absa;
		} else {
/* Computing 2nd power */
		    d__1 = absa / scale;
		    sum += d__1 * d__1;
		}
	    }
	    if (d_imag(&ap[k]) != 0.) {
		absa = (d__1 = d_imag(&ap[k]), abs(d__1));
		if (scale < absa) {
/* Computing 2nd power */
		    d__1 = scale / absa;
		    sum = sum * (d__1 * d__1) + 1.;
		    scale = absa;
		} else {
/* Computing 2nd power */
		    d__1 = absa / scale;
		    sum += d__1 * d__1;
		}
	    }
	    if (lsame_(uplo, "U")) {
		k = k + i__ + 1;
	    } else {
		k = k + *n - i__ + 1;
	    }
/* L130: */
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of ZLANSP */

} /* zlansp_ */
