/* slansf.f -- translated by f2c (version 20061008).
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

doublereal slansf_(char *norm, char *transr, char *uplo, integer *n, real *a, 
	real *work)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, k, l;
    real s;
    integer n1;
    real aa;
    integer lda, ifm, noe, ilu;
    real scale;
    extern logical lsame_(char *, char *);
    real value;
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *, 
	    real *);


/*  -- LAPACK routine (version 3.2)                                    -- */

/*  -- Contributed by Fred Gustavson of the IBM Watson Research Center -- */
/*  -- November 2008                                                   -- */

/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLANSF returns the value of the one norm, or the Frobenius norm, or */
/*  the infinity norm, or the element of largest absolute value of a */
/*  real symmetric matrix A in RFP format. */

/*  Description */
/*  =========== */

/*  SLANSF returns the value */

/*     SLANSF = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
/*              ( */
/*              ( norm1(A),         NORM = '1', 'O' or 'o' */
/*              ( */
/*              ( normI(A),         NORM = 'I' or 'i' */
/*              ( */
/*              ( normF(A),         NORM = 'F', 'f', 'E' or 'e' */

/*  where  norm1  denotes the  one norm of a matrix (maximum column sum), */
/*  normI  denotes the  infinity norm  of a matrix  (maximum row sum) and */
/*  normF  denotes the  Frobenius norm of a matrix (square root of sum of */
/*  squares).  Note that  max(abs(A(i,j)))  is not a  matrix norm. */

/*  Arguments */
/*  ========= */

/*  NORM    (input) CHARACTER */
/*          Specifies the value to be returned in SLANSF as described */
/*          above. */

/*  TRANSR  (input) CHARACTER */
/*          Specifies whether the RFP format of A is normal or */
/*          transposed format. */
/*          = 'N':  RFP format is Normal; */
/*          = 'T':  RFP format is Transpose. */

/*  UPLO    (input) CHARACTER */
/*           On entry, UPLO specifies whether the RFP matrix A came from */
/*           an upper or lower triangular matrix as follows: */
/*           = 'U': RFP A came from an upper triangular matrix; */
/*           = 'L': RFP A came from a lower triangular matrix. */

/*  N       (input) INTEGER */
/*          The order of the matrix A. N >= 0. When N = 0, SLANSF is */
/*          set to zero. */

/*  A       (input) REAL array, dimension ( N*(N+1)/2 ); */
/*          On entry, the upper (if UPLO = 'U') or lower (if UPLO = 'L') */
/*          part of the symmetric matrix A stored in RFP format. See the */
/*          "Notes" below for more details. */
/*          Unchanged on exit. */

/*  WORK    (workspace) REAL array, dimension (MAX(1,LWORK)), */
/*          where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise, */
/*          WORK is not referenced. */

/*  Notes */
/*  ===== */

/*  We first consider Rectangular Full Packed (RFP) Format when N is */
/*  even. We give an example where N = 6. */

/*      AP is Upper             AP is Lower */

/*   00 01 02 03 04 05       00 */
/*      11 12 13 14 15       10 11 */
/*         22 23 24 25       20 21 22 */
/*            33 34 35       30 31 32 33 */
/*               44 45       40 41 42 43 44 */
/*                  55       50 51 52 53 54 55 */


/*  Let TRANSR = 'N'. RFP holds AP as follows: */
/*  For UPLO = 'U' the upper trapezoid A(0:5,0:2) consists of the last */
/*  three columns of AP upper. The lower triangle A(4:6,0:2) consists of */
/*  the transpose of the first three columns of AP upper. */
/*  For UPLO = 'L' the lower trapezoid A(1:6,0:2) consists of the first */
/*  three columns of AP lower. The upper triangle A(0:2,0:2) consists of */
/*  the transpose of the last three columns of AP lower. */
/*  This covers the case N even and TRANSR = 'N'. */

/*         RFP A                   RFP A */

/*        03 04 05                33 43 53 */
/*        13 14 15                00 44 54 */
/*        23 24 25                10 11 55 */
/*        33 34 35                20 21 22 */
/*        00 44 45                30 31 32 */
/*        01 11 55                40 41 42 */
/*        02 12 22                50 51 52 */

/*  Now let TRANSR = 'T'. RFP A in both UPLO cases is just the */
/*  transpose of RFP A above. One therefore gets: */


/*           RFP A                   RFP A */

/*     03 13 23 33 00 01 02    33 00 10 20 30 40 50 */
/*     04 14 24 34 44 11 12    43 44 11 21 31 41 51 */
/*     05 15 25 35 45 55 22    53 54 55 22 32 42 52 */


/*  We first consider Rectangular Full Packed (RFP) Format when N is */
/*  odd. We give an example where N = 5. */

/*     AP is Upper                 AP is Lower */

/*   00 01 02 03 04              00 */
/*      11 12 13 14              10 11 */
/*         22 23 24              20 21 22 */
/*            33 34              30 31 32 33 */
/*               44              40 41 42 43 44 */


/*  Let TRANSR = 'N'. RFP holds AP as follows: */
/*  For UPLO = 'U' the upper trapezoid A(0:4,0:2) consists of the last */
/*  three columns of AP upper. The lower triangle A(3:4,0:1) consists of */
/*  the transpose of the first two columns of AP upper. */
/*  For UPLO = 'L' the lower trapezoid A(0:4,0:2) consists of the first */
/*  three columns of AP lower. The upper triangle A(0:1,1:2) consists of */
/*  the transpose of the last two columns of AP lower. */
/*  This covers the case N odd and TRANSR = 'N'. */

/*         RFP A                   RFP A */

/*        02 03 04                00 33 43 */
/*        12 13 14                10 11 44 */
/*        22 23 24                20 21 22 */
/*        00 33 34                30 31 32 */
/*        01 11 44                40 41 42 */

/*  Now let TRANSR = 'T'. RFP A in both UPLO cases is just the */
/*  transpose of RFP A above. One therefore gets: */

/*           RFP A                   RFP A */

/*     02 12 22 00 01             00 10 20 30 40 50 */
/*     03 13 23 33 11             33 11 21 31 41 51 */
/*     04 14 24 34 44             43 44 22 32 42 52 */

/*  Reference */
/*  ========= */

/*  ===================================================================== */

/*     .. */
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

    if (*n == 0) {
	ret_val = 0.f;
	return ret_val;
    }

/*     set noe = 1 if n is odd. if n is even set noe=0 */

    noe = 1;
    if (*n % 2 == 0) {
	noe = 0;
    }

/*     set ifm = 0 when form='T or 't' and 1 otherwise */

    ifm = 1;
    if (lsame_(transr, "T")) {
	ifm = 0;
    }

/*     set ilu = 0 when uplo='U or 'u' and 1 otherwise */

    ilu = 1;
    if (lsame_(uplo, "U")) {
	ilu = 0;
    }

/*     set lda = (n+1)/2 when ifm = 0 */
/*     set lda = n when ifm = 1 and noe = 1 */
/*     set lda = n+1 when ifm = 1 and noe = 0 */

    if (ifm == 1) {
	if (noe == 1) {
	    lda = *n;
	} else {
/*           noe=0 */
	    lda = *n + 1;
	}
    } else {
/*        ifm=0 */
	lda = (*n + 1) / 2;
    }

    if (lsame_(norm, "M")) {

/*       Find max(abs(A(i,j))). */

	k = (*n + 1) / 2;
	value = 0.f;
	if (noe == 1) {
/*           n is odd */
	    if (ifm == 1) {
/*           A is n by k */
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__2 = value, r__3 = (r__1 = a[i__ + j * lda], dabs(
				r__1));
			value = dmax(r__2,r__3);
		    }
		}
	    } else {
/*              xpose case; A is k by n */
		i__1 = *n - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = k - 1;
		    for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__2 = value, r__3 = (r__1 = a[i__ + j * lda], dabs(
				r__1));
			value = dmax(r__2,r__3);
		    }
		}
	    }
	} else {
/*           n is even */
	    if (ifm == 1) {
/*              A is n+1 by k */
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__2 = value, r__3 = (r__1 = a[i__ + j * lda], dabs(
				r__1));
			value = dmax(r__2,r__3);
		    }
		}
	    } else {
/*              xpose case; A is k by n+1 */
		i__1 = *n;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = k - 1;
		    for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			r__2 = value, r__3 = (r__1 = a[i__ + j * lda], dabs(
				r__1));
			value = dmax(r__2,r__3);
		    }
		}
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*        Find normI(A) ( = norm1(A), since A is symmetric). */

	if (ifm == 1) {
	    k = *n / 2;
	    if (noe == 1) {
/*              n is odd */
		if (ilu == 0) {
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k + j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(i,j+k) */
			    s += aa;
			    work[i__] += aa;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j+k,j+k) */
			work[j + k] = s + aa;
			if (i__ == k + k) {
			    goto L10;
			}
			++i__;
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j,j) */
			work[j] += aa;
			s = 0.f;
			i__2 = k - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(l,j) */
			    s += aa;
			    work[l] += aa;
			}
			work[j] += s;
		    }
L10:
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu = 1 */
		    ++k;
/*                 k=(n+1)/2 for n odd and ilu=1 */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    for (j = k - 1; j >= 0; --j) {
			s = 0.f;
			i__1 = j - 2;
			for (i__ = 0; i__ <= i__1; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(j+k,i+k) */
			    s += aa;
			    work[i__ + k] += aa;
			}
			if (j > 0) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(j+k,j+k) */
			    s += aa;
			    work[i__ + k] += s;
/*                       i=j */
			    ++i__;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j,j) */
			work[j] = aa;
			s = 0.f;
			i__1 = *n - 1;
			for (l = j + 1; l <= i__1; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(l,j) */
			    s += aa;
			    work[l] += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		}
	    } else {
/*              n is even */
		if (ilu == 0) {
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k + j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(i,j+k) */
			    s += aa;
			    work[i__] += aa;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j+k,j+k) */
			work[j + k] = s + aa;
			++i__;
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j,j) */
			work[j] += aa;
			s = 0.f;
			i__2 = k - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(l,j) */
			    s += aa;
			    work[l] += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu = 1 */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    for (j = k - 1; j >= 0; --j) {
			s = 0.f;
			i__1 = j - 1;
			for (i__ = 0; i__ <= i__1; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(j+k,i+k) */
			    s += aa;
			    work[i__ + k] += aa;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j+k,j+k) */
			s += aa;
			work[i__ + k] += s;
/*                    i=j */
			++i__;
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    -> A(j,j) */
			work[j] = aa;
			s = 0.f;
			i__1 = *n - 1;
			for (l = j + 1; l <= i__1; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       -> A(l,j) */
			    s += aa;
			    work[l] += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		}
	    }
	} else {
/*           ifm=0 */
	    k = *n / 2;
	    if (noe == 1) {
/*              n is odd */
		if (ilu == 0) {
		    n1 = k;
/*                 n/2 */
		    ++k;
/*                 k is the row size and lda */
		    i__1 = *n - 1;
		    for (i__ = n1; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = n1 - 1;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,n1+i) */
			    work[i__ + n1] += aa;
			    s += aa;
			}
			work[j] = s;
		    }
/*                 j=n1=k-1 is special */
		    s = (r__1 = a[j * lda], dabs(r__1));
/*                 A(k-1,k-1) */
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(k-1,i+n1) */
			work[i__ + n1] += aa;
			s += aa;
		    }
		    work[j] += s;
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			s = 0.f;
			i__2 = j - k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(i,j-k) */
			    work[i__] += aa;
			    s += aa;
			}
/*                    i=j-k */
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(j-k,j-k) */
			s += aa;
			work[j - k] += s;
			++i__;
			s = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(j,j) */
			i__2 = *n - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,l) */
			    work[l] += aa;
			    s += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu=1 */
		    ++k;
/*                 k=(n+1)/2 for n odd and ilu=1 */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
/*                    process */
			s = 0.f;
			i__2 = j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,i) */
			    work[i__] += aa;
			    s += aa;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    i=j so process of A(j,j) */
			s += aa;
			work[j] = s;
/*                    is initialised here */
			++i__;
/*                    i=j process A(j+k,j+k) */
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
			s = aa;
			i__2 = *n - 1;
			for (l = k + j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(l,k+j) */
			    s += aa;
			    work[l] += aa;
			}
			work[k + j] += s;
		    }
/*                 j=k-1 is special :process col A(k-1,0:k-1) */
		    s = 0.f;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(k,i) */
			work[i__] += aa;
			s += aa;
		    }
/*                 i=k-1 */
		    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                 A(k-1,k-1) */
		    s += aa;
		    work[i__] = s;
/*                 done with col j=k+1 */
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
/*                    process col j of A = A(j,0:k-1) */
			s = 0.f;
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,i) */
			    work[i__] += aa;
			    s += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		}
	    } else {
/*              n is even */
		if (ilu == 0) {
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,i+k) */
			    work[i__ + k] += aa;
			    s += aa;
			}
			work[j] = s;
		    }
/*                 j=k */
		    aa = (r__1 = a[j * lda], dabs(r__1));
/*                 A(k,k) */
		    s = aa;
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(k,k+i) */
			work[i__ + k] += aa;
			s += aa;
		    }
		    work[j] += s;
		    i__1 = *n - 1;
		    for (j = k + 1; j <= i__1; ++j) {
			s = 0.f;
			i__2 = j - 2 - k;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(i,j-k-1) */
			    work[i__] += aa;
			    s += aa;
			}
/*                     i=j-1-k */
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(j-k-1,j-k-1) */
			s += aa;
			work[j - k - 1] += s;
			++i__;
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(j,j) */
			s = aa;
			i__2 = *n - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j,l) */
			    work[l] += aa;
			    s += aa;
			}
			work[j] += s;
		    }
/*                 j=n */
		    s = 0.f;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(i,k-1) */
			work[i__] += aa;
			s += aa;
		    }
/*                 i=k-1 */
		    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                 A(k-1,k-1) */
		    s += aa;
		    work[i__] += s;
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu=1 */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
/*                 j=0 is special :process col A(k:n-1,k) */
		    s = dabs(a[0]);
/*                 A(k,k) */
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__], dabs(r__1));
/*                    A(k+i,k) */
			work[i__ + k] += aa;
			s += aa;
		    }
		    work[k] += s;
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
/*                    process */
			s = 0.f;
			i__2 = j - 2;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j-1,i) */
			    work[i__] += aa;
			    s += aa;
			}
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    i=j-1 so process of A(j-1,j-1) */
			s += aa;
			work[j - 1] = s;
/*                    is initialised here */
			++i__;
/*                    i=j process A(j+k,j+k) */
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
			s = aa;
			i__2 = *n - 1;
			for (l = k + j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(l,k+j) */
			    s += aa;
			    work[l] += aa;
			}
			work[k + j] += s;
		    }
/*                 j=k is special :process col A(k,0:k-1) */
		    s = 0.f;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                    A(k,i) */
			work[i__] += aa;
			s += aa;
		    }
/*                 i=k-1 */
		    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                 A(k-1,k-1) */
		    s += aa;
		    work[i__] = s;
/*                 done with col j=k+1 */
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
/*                    process col j-1 of A = A(j-1,0:k-1) */
			s = 0.f;
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = (r__1 = a[i__ + j * lda], dabs(r__1));
/*                       A(j-1,i) */
			    work[i__] += aa;
			    s += aa;
			}
			work[j - 1] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		}
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*       Find normF(A). */

	k = (*n + 1) / 2;
	scale = 0.f;
	s = 1.f;
	if (noe == 1) {
/*           n is odd */
	    if (ifm == 1) {
/*              A is normal */
		if (ilu == 0) {
/*                 A is upper */
		    i__1 = k - 3;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 2;
			slassq_(&i__2, &a[k + j + 1 + j * lda], &c__1, &scale, 
				 &s);
/*                    L at A(k,0) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j - 1;
			slassq_(&i__2, &a[j * lda], &c__1, &scale, &s);
/*                    trap U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = k - 1;
		    i__2 = lda + 1;
		    slassq_(&i__1, &a[k], &i__2, &scale, &s);
/*                 tri L at A(k,0) */
		    i__1 = lda + 1;
		    slassq_(&k, &a[k - 1], &i__1, &scale, &s);
/*                 tri U at A(k-1,0) */
		} else {
/*                 ilu=1 & A is lower */
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = *n - j - 1;
			slassq_(&i__2, &a[j + 1 + j * lda], &c__1, &scale, &s)
				;
/*                    trap L at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			slassq_(&j, &a[(j + 1) * lda], &c__1, &scale, &s);
/*                    U at A(0,1) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, a, &i__1, &scale, &s);
/*                 tri L at A(0,0) */
		    i__1 = k - 1;
		    i__2 = lda + 1;
		    slassq_(&i__1, &a[lda], &i__2, &scale, &s);
/*                 tri U at A(0,1) */
		}
	    } else {
/*              A is xpose */
		if (ilu == 0) {
/*                 A' is upper */
		    i__1 = k - 2;
		    for (j = 1; j <= i__1; ++j) {
			slassq_(&j, &a[(k + j) * lda], &c__1, &scale, &s);
/*                    U at A(0,k) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			slassq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k-1 rect. at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			slassq_(&i__2, &a[j + 1 + (j + k - 1) * lda], &c__1, &
				scale, &s);
/*                    L at A(0,k-1) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = k - 1;
		    i__2 = lda + 1;
		    slassq_(&i__1, &a[k * lda], &i__2, &scale, &s);
/*                 tri U at A(0,k) */
		    i__1 = lda + 1;
		    slassq_(&k, &a[(k - 1) * lda], &i__1, &scale, &s);
/*                 tri L at A(0,k-1) */
		} else {
/*                 A' is lower */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			slassq_(&j, &a[j * lda], &c__1, &scale, &s);
/*                    U at A(0,0) */
		    }
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			slassq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k-1 rect. at A(0,k) */
		    }
		    i__1 = k - 3;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 2;
			slassq_(&i__2, &a[j + 2 + j * lda], &c__1, &scale, &s)
				;
/*                    L at A(1,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, a, &i__1, &scale, &s);
/*                 tri U at A(0,0) */
		    i__1 = k - 1;
		    i__2 = lda + 1;
		    slassq_(&i__1, &a[1], &i__2, &scale, &s);
/*                 tri L at A(1,0) */
		}
	    }
	} else {
/*           n is even */
	    if (ifm == 1) {
/*              A is normal */
		if (ilu == 0) {
/*                 A is upper */
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			slassq_(&i__2, &a[k + j + 2 + j * lda], &c__1, &scale, 
				 &s);
/*                    L at A(k+1,0) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j;
			slassq_(&i__2, &a[j * lda], &c__1, &scale, &s);
/*                    trap U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, &a[k + 1], &i__1, &scale, &s);
/*                 tri L at A(k+1,0) */
		    i__1 = lda + 1;
		    slassq_(&k, &a[k], &i__1, &scale, &s);
/*                 tri U at A(k,0) */
		} else {
/*                 ilu=1 & A is lower */
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = *n - j - 1;
			slassq_(&i__2, &a[j + 2 + j * lda], &c__1, &scale, &s)
				;
/*                    trap L at A(1,0) */
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			slassq_(&j, &a[j * lda], &c__1, &scale, &s);
/*                    U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, &a[1], &i__1, &scale, &s);
/*                 tri L at A(1,0) */
		    i__1 = lda + 1;
		    slassq_(&k, a, &i__1, &scale, &s);
/*                 tri U at A(0,0) */
		}
	    } else {
/*              A is xpose */
		if (ilu == 0) {
/*                 A' is upper */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			slassq_(&j, &a[(k + 1 + j) * lda], &c__1, &scale, &s);
/*                    U at A(0,k+1) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			slassq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k rect. at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			slassq_(&i__2, &a[j + 1 + (j + k) * lda], &c__1, &
				scale, &s);
/*                    L at A(0,k) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, &a[(k + 1) * lda], &i__1, &scale, &s);
/*                 tri U at A(0,k+1) */
		    i__1 = lda + 1;
		    slassq_(&k, &a[k * lda], &i__1, &scale, &s);
/*                 tri L at A(0,k) */
		} else {
/*                 A' is lower */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			slassq_(&j, &a[(j + 1) * lda], &c__1, &scale, &s);
/*                    U at A(0,1) */
		    }
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			slassq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k rect. at A(0,k+1) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			slassq_(&i__2, &a[j + 1 + j * lda], &c__1, &scale, &s)
				;
/*                    L at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    i__1 = lda + 1;
		    slassq_(&k, &a[lda], &i__1, &scale, &s);
/*                 tri L at A(0,1) */
		    i__1 = lda + 1;
		    slassq_(&k, a, &i__1, &scale, &s);
/*                 tri U at A(0,0) */
		}
	    }
	}
	value = scale * sqrt(s);
    }

    ret_val = value;
    return ret_val;

/*     End of SLANSF */

} /* slansf_ */
