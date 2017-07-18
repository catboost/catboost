/* clantp.f -- translated by f2c (version 20061008).
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

doublereal clantp_(char *norm, char *uplo, char *diag, integer *n, complex *
	ap, real *work)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val, r__1, r__2;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

    /* Local variables */
    integer i__, j, k;
    real sum, scale;
    logical udiag;
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

/*  CLANTP  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the  element of  largest absolute value  of a */
/*  triangular matrix A, supplied in packed form. */

/*  Description */
/*  =========== */

/*  CLANTP returns the value */

/*     CLANTP = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
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
/*          Specifies the value to be returned in CLANTP as described */
/*          above. */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the matrix A is upper or lower triangular. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  DIAG    (input) CHARACTER*1 */
/*          Specifies whether or not the matrix A is unit triangular. */
/*          = 'N':  Non-unit triangular */
/*          = 'U':  Unit triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0.  When N = 0, CLANTP is */
/*          set to zero. */

/*  AP      (input) COMPLEX array, dimension (N*(N+1)/2) */
/*          The upper or lower triangular matrix A, packed columnwise in */
/*          a linear array.  The j-th column of A is stored in the array */
/*          AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/*          Note that when DIAG = 'U', the elements of the array AP */
/*          corresponding to the diagonal elements of the matrix A are */
/*          not referenced, but are assumed to be one. */

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
    --work;
    --ap;

    /* Function Body */
    if (*n == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	k = 1;
	if (lsame_(diag, "U")) {
	    value = 1.f;
	    if (lsame_(uplo, "U")) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = k + j - 2;
		    for (i__ = k; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&ap[i__]);
			value = dmax(r__1,r__2);
/* L10: */
		    }
		    k += j;
/* L20: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = k + *n - j;
		    for (i__ = k + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&ap[i__]);
			value = dmax(r__1,r__2);
/* L30: */
		    }
		    k = k + *n - j + 1;
/* L40: */
		}
	    }
	} else {
	    value = 0.f;
	    if (lsame_(uplo, "U")) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = k + j - 1;
		    for (i__ = k; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&ap[i__]);
			value = dmax(r__1,r__2);
/* L50: */
		    }
		    k += j;
/* L60: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = k + *n - j;
		    for (i__ = k; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&ap[i__]);
			value = dmax(r__1,r__2);
/* L70: */
		    }
		    k = k + *n - j + 1;
/* L80: */
		}
	    }
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {

/*        Find norm1(A). */

	value = 0.f;
	k = 1;
	udiag = lsame_(diag, "U");
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (udiag) {
		    sum = 1.f;
		    i__2 = k + j - 2;
		    for (i__ = k; i__ <= i__2; ++i__) {
			sum += c_abs(&ap[i__]);
/* L90: */
		    }
		} else {
		    sum = 0.f;
		    i__2 = k + j - 1;
		    for (i__ = k; i__ <= i__2; ++i__) {
			sum += c_abs(&ap[i__]);
/* L100: */
		    }
		}
		k += j;
		value = dmax(value,sum);
/* L110: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (udiag) {
		    sum = 1.f;
		    i__2 = k + *n - j;
		    for (i__ = k + 1; i__ <= i__2; ++i__) {
			sum += c_abs(&ap[i__]);
/* L120: */
		    }
		} else {
		    sum = 0.f;
		    i__2 = k + *n - j;
		    for (i__ = k; i__ <= i__2; ++i__) {
			sum += c_abs(&ap[i__]);
/* L130: */
		    }
		}
		k = k + *n - j + 1;
		value = dmax(value,sum);
/* L140: */
	    }
	}
    } else if (lsame_(norm, "I")) {

/*        Find normI(A). */

	k = 1;
	if (lsame_(uplo, "U")) {
	    if (lsame_(diag, "U")) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] = 1.f;
/* L150: */
		}
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			work[i__] += c_abs(&ap[k]);
			++k;
/* L160: */
		    }
		    ++k;
/* L170: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] = 0.f;
/* L180: */
		}
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			work[i__] += c_abs(&ap[k]);
			++k;
/* L190: */
		    }
/* L200: */
		}
	    }
	} else {
	    if (lsame_(diag, "U")) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] = 1.f;
/* L210: */
		}
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    ++k;
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			work[i__] += c_abs(&ap[k]);
			++k;
/* L220: */
		    }
/* L230: */
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    work[i__] = 0.f;
/* L240: */
		}
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			work[i__] += c_abs(&ap[k]);
			++k;
/* L250: */
		    }
/* L260: */
		}
	    }
	}
	value = 0.f;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__1 = value, r__2 = work[i__];
	    value = dmax(r__1,r__2);
/* L270: */
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	if (lsame_(uplo, "U")) {
	    if (lsame_(diag, "U")) {
		scale = 1.f;
		sum = (real) (*n);
		k = 2;
		i__1 = *n;
		for (j = 2; j <= i__1; ++j) {
		    i__2 = j - 1;
		    classq_(&i__2, &ap[k], &c__1, &scale, &sum);
		    k += j;
/* L280: */
		}
	    } else {
		scale = 0.f;
		sum = 1.f;
		k = 1;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    classq_(&j, &ap[k], &c__1, &scale, &sum);
		    k += j;
/* L290: */
		}
	    }
	} else {
	    if (lsame_(diag, "U")) {
		scale = 1.f;
		sum = (real) (*n);
		k = 2;
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n - j;
		    classq_(&i__2, &ap[k], &c__1, &scale, &sum);
		    k = k + *n - j + 1;
/* L300: */
		}
	    } else {
		scale = 0.f;
		sum = 1.f;
		k = 1;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n - j + 1;
		    classq_(&i__2, &ap[k], &c__1, &scale, &sum);
		    k = k + *n - j + 1;
/* L310: */
		}
	    }
	}
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANTP */

} /* clantp_ */
