/* clanhf.f -- translated by f2c (version 20061008).
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

doublereal clanhf_(char *norm, char *transr, char *uplo, integer *n, complex *
	a, real *work)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double c_abs(complex *), sqrt(doublereal);

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
    extern /* Subroutine */ int classq_(integer *, complex *, integer *, real 
	    *, real *);


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

/*  CLANHF  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the  element of  largest absolute value  of a */
/*  complex Hermitian matrix A in RFP format. */

/*  Description */
/*  =========== */

/*  CLANHF returns the value */

/*     CLANHF = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
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

/*  NORM      (input) CHARACTER */
/*            Specifies the value to be returned in CLANHF as described */
/*            above. */

/*  TRANSR    (input) CHARACTER */
/*            Specifies whether the RFP format of A is normal or */
/*            conjugate-transposed format. */
/*            = 'N':  RFP format is Normal */
/*            = 'C':  RFP format is Conjugate-transposed */

/*  UPLO      (input) CHARACTER */
/*            On entry, UPLO specifies whether the RFP matrix A came from */
/*            an upper or lower triangular matrix as follows: */

/*            UPLO = 'U' or 'u' RFP A came from an upper triangular */
/*            matrix */

/*            UPLO = 'L' or 'l' RFP A came from a  lower triangular */
/*            matrix */

/*  N         (input) INTEGER */
/*            The order of the matrix A.  N >= 0.  When N = 0, CLANHF is */
/*            set to zero. */

/*   A        (input) COMPLEX*16 array, dimension ( N*(N+1)/2 ); */
/*            On entry, the matrix A in RFP Format. */
/*            RFP Format is described by TRANSR, UPLO and N as follows: */
/*            If TRANSR='N' then RFP A is (0:N,0:K-1) when N is even; */
/*            K=N/2. RFP A is (0:N-1,0:K) when N is odd; K=N/2. If */
/*            TRANSR = 'C' then RFP is the Conjugate-transpose of RFP A */
/*            as defined when TRANSR = 'N'. The contents of RFP A are */
/*            defined by UPLO as follows: If UPLO = 'U' the RFP A */
/*            contains the ( N*(N+1)/2 ) elements of upper packed A */
/*            either in normal or conjugate-transpose Format. If */
/*            UPLO = 'L' the RFP A contains the ( N*(N+1) /2 ) elements */
/*            of lower packed A either in normal or conjugate-transpose */
/*            Format. The LDA of RFP A is (N+1)/2 when TRANSR = 'C'. When */
/*            TRANSR is 'N' the LDA is N+1 when N is even and is N when */
/*            is odd. See the Note below for more details. */
/*            Unchanged on exit. */

/*  WORK      (workspace) REAL array, dimension (LWORK), */
/*            where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise, */
/*            WORK is not referenced. */

/*  Note: */
/*  ===== */

/*  We first consider Standard Packed Format when N is even. */
/*  We give an example where N = 6. */

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
/*  conjugate-transpose of the first three columns of AP upper. */
/*  For UPLO = 'L' the lower trapezoid A(1:6,0:2) consists of the first */
/*  three columns of AP lower. The upper triangle A(0:2,0:2) consists of */
/*  conjugate-transpose of the last three columns of AP lower. */
/*  To denote conjugate we place -- above the element. This covers the */
/*  case N even and TRANSR = 'N'. */

/*         RFP A                   RFP A */

/*                                -- -- -- */
/*        03 04 05                33 43 53 */
/*                                   -- -- */
/*        13 14 15                00 44 54 */
/*                                      -- */
/*        23 24 25                10 11 55 */

/*        33 34 35                20 21 22 */
/*        -- */
/*        00 44 45                30 31 32 */
/*        -- -- */
/*        01 11 55                40 41 42 */
/*        -- -- -- */
/*        02 12 22                50 51 52 */

/*  Now let TRANSR = 'C'. RFP A in both UPLO cases is just the conjugate- */
/*  transpose of RFP A above. One therefore gets: */


/*           RFP A                   RFP A */

/*     -- -- -- --                -- -- -- -- -- -- */
/*     03 13 23 33 00 01 02    33 00 10 20 30 40 50 */
/*     -- -- -- -- --                -- -- -- -- -- */
/*     04 14 24 34 44 11 12    43 44 11 21 31 41 51 */
/*     -- -- -- -- -- --                -- -- -- -- */
/*     05 15 25 35 45 55 22    53 54 55 22 32 42 52 */


/*  We next  consider Standard Packed Format when N is odd. */
/*  We give an example where N = 5. */

/*     AP is Upper                 AP is Lower */

/*   00 01 02 03 04              00 */
/*      11 12 13 14              10 11 */
/*         22 23 24              20 21 22 */
/*            33 34              30 31 32 33 */
/*               44              40 41 42 43 44 */


/*  Let TRANSR = 'N'. RFP holds AP as follows: */
/*  For UPLO = 'U' the upper trapezoid A(0:4,0:2) consists of the last */
/*  three columns of AP upper. The lower triangle A(3:4,0:1) consists of */
/*  conjugate-transpose of the first two   columns of AP upper. */
/*  For UPLO = 'L' the lower trapezoid A(0:4,0:2) consists of the first */
/*  three columns of AP lower. The upper triangle A(0:1,1:2) consists of */
/*  conjugate-transpose of the last two   columns of AP lower. */
/*  To denote conjugate we place -- above the element. This covers the */
/*  case N odd  and TRANSR = 'N'. */

/*         RFP A                   RFP A */

/*                                   -- -- */
/*        02 03 04                00 33 43 */
/*                                      -- */
/*        12 13 14                10 11 44 */

/*        22 23 24                20 21 22 */
/*        -- */
/*        00 33 34                30 31 32 */
/*        -- -- */
/*        01 11 44                40 41 42 */

/*  Now let TRANSR = 'C'. RFP A in both UPLO cases is just the conjugate- */
/*  transpose of RFP A above. One therefore gets: */


/*           RFP A                   RFP A */

/*     -- -- --                   -- -- -- -- -- -- */
/*     02 12 22 00 01             00 10 20 30 40 50 */
/*     -- -- -- --                   -- -- -- -- -- */
/*     03 13 23 33 11             33 11 21 31 41 51 */
/*     -- -- -- -- --                   -- -- -- -- */
/*     04 14 24 34 44             43 44 22 32 42 52 */

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

    if (*n == 0) {
	ret_val = 0.f;
	return ret_val;
    }

/*     set noe = 1 if n is odd. if n is even set noe=0 */

    noe = 1;
    if (*n % 2 == 0) {
	noe = 0;
    }

/*     set ifm = 0 when form='C' or 'c' and 1 otherwise */

    ifm = 1;
    if (lsame_(transr, "C")) {
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
/*           n is odd & n = k + k - 1 */
	    if (ifm == 1) {
/*              A is n by k */
		if (ilu == 1) {
/*                 uplo ='L' */
		    j = 0;
/*                 -> L(0,0) */
/* Computing MAX */
		    i__1 = j + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = *n - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			i__2 = j - 2;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j - 1;
/*                    L(k+j,k+j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j;
/*                    -> L(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = *n - 1;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		} else {
/*                 uplo = 'U' */
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j - 2;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = k + j - 1;
/*                    -> U(i,i) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			++i__;
/*                    =k+j; i -> U(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = *n - 1;
			for (i__ = k + j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    i__1 = *n - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
/*                    j=k-1 */
		    }
/*                 i=n-1 -> U(n-1,n-1) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		}
	    } else {
/*              xpose case; A is k by n */
		if (ilu == 1) {
/*                 uplo ='L' */
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j;
/*                    L(i,i) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j + 1;
/*                    L(j+k,j+k) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = k - 1;
			for (i__ = j + 2; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    j = k - 1;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__ = k - 1;
/*                 -> L(i,i) is at A(i,j) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		} else {
/*                 uplo = 'U' */
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    j = k - 1;
/*                 -> U(j,j) is at A(0,j) */
/* Computing MAX */
		    i__1 = j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			i__2 = j - k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j - k;
/*                    -> U(i,i) at A(i,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j - k + 1;
/*                    U(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = k - 1;
			for (i__ = j - k + 2; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		}
	    }
	} else {
/*           n is even & k = n/2 */
	    if (ifm == 1) {
/*              A is n+1 by k */
		if (ilu == 1) {
/*                 uplo ='L' */
		    j = 0;
/*                 -> L(k,k) & j=1 -> L(0,0) */
/* Computing MAX */
		    i__1 = j + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
/* Computing MAX */
		    i__1 = j + 1 + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = *n;
		    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			i__2 = j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j;
/*                    L(k+j,k+j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j + 1;
/*                    -> L(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = *n;
			for (i__ = j + 2; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		} else {
/*                 uplo = 'U' */
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = k + j;
/*                    -> U(i,i) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			++i__;
/*                    =k+j+1; i -> U(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = *n;
			for (i__ = k + j + 2; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    i__1 = *n - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
/*                    j=k-1 */
		    }
/*                 i=n-1 -> U(n-1,n-1) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__ = *n;
/*                 -> U(k-1,k-1) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		}
	    } else {
/*              xpose case; A is k by n+1 */
		if (ilu == 1) {
/*                 uplo ='L' */
		    j = 0;
/*                 -> L(k,k) at A(0,0) */
/* Computing MAX */
		    i__1 = j + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			i__2 = j - 2;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j - 1;
/*                    L(i,i) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j;
/*                    L(j+k,j+k) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = k - 1;
			for (i__ = j + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    j = k;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__ = k - 1;
/*                 -> L(i,i) is at A(i,j) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		} else {
/*                 uplo = 'U' */
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    j = k;
/*                 -> U(j,j) is at A(0,j) */
/* Computing MAX */
		    i__1 = j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__1 = *n - 1;
		    for (j = k + 1; j <= i__1; ++j) {
			i__2 = j - k - 2;
			for (i__ = 0; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
			i__ = j - k - 1;
/*                    -> U(i,i) at A(i,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__ = j - k;
/*                    U(j,j) */
/* Computing MAX */
			i__2 = i__ + j * lda;
			r__2 = value, r__3 = (r__1 = a[i__2].r, dabs(r__1));
			value = dmax(r__2,r__3);
			i__2 = k - 1;
			for (i__ = j - k + 1; i__ <= i__2; ++i__) {
/* Computing MAX */
			    r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			    value = dmax(r__1,r__2);
			}
		    }
		    j = *n;
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
/* Computing MAX */
			r__1 = value, r__2 = c_abs(&a[i__ + j * lda]);
			value = dmax(r__1,r__2);
		    }
		    i__ = k - 1;
/*                 U(k,k) at A(i,j) */
/* Computing MAX */
		    i__1 = i__ + j * lda;
		    r__2 = value, r__3 = (r__1 = a[i__1].r, dabs(r__1));
		    value = dmax(r__2,r__3);
		}
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*       Find normI(A) ( = norm1(A), since A is Hermitian). */

	if (ifm == 1) {
/*           A is 'N' */
	    k = *n / 2;
	    if (noe == 1) {
/*              n is odd & A is n by (n+1)/2 */
		if (ilu == 0) {
/*                 uplo = 'U' */
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k + j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = c_abs(&a[i__ + j * lda]);
/*                       -> A(i,j+k) */
			    s += aa;
			    work[i__] += aa;
			}
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    -> A(j+k,j+k) */
			work[j + k] = s + aa;
			if (i__ == k + k) {
			    goto L10;
			}
			++i__;
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    -> A(j,j) */
			work[j] += aa;
			s = 0.f;
			i__2 = k - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
/*                 ilu = 1 & uplo = 'L' */
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       -> A(j+k,i+k) */
			    s += aa;
			    work[i__ + k] += aa;
			}
			if (j > 0) {
			    i__1 = i__ + j * lda;
			    aa = (r__1 = a[i__1].r, dabs(r__1));
/*                       -> A(j+k,j+k) */
			    s += aa;
			    work[i__ + k] += s;
/*                       i=j */
			    ++i__;
			}
			i__1 = i__ + j * lda;
			aa = (r__1 = a[i__1].r, dabs(r__1));
/*                    -> A(j,j) */
			work[j] = aa;
			s = 0.f;
			i__1 = *n - 1;
			for (l = j + 1; l <= i__1; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
/*              n is even & A is n+1 by k = n/2 */
		if (ilu == 0) {
/*                 uplo = 'U' */
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k + j - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = c_abs(&a[i__ + j * lda]);
/*                       -> A(i,j+k) */
			    s += aa;
			    work[i__] += aa;
			}
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    -> A(j+k,j+k) */
			work[j + k] = s + aa;
			++i__;
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    -> A(j,j) */
			work[j] += aa;
			s = 0.f;
			i__2 = k - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
/*                       -> A(l,j) */
			    s += aa;
			    work[l] += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu = 1 & uplo = 'L' */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    for (j = k - 1; j >= 0; --j) {
			s = 0.f;
			i__1 = j - 1;
			for (i__ = 0; i__ <= i__1; ++i__) {
			    aa = c_abs(&a[i__ + j * lda]);
/*                       -> A(j+k,i+k) */
			    s += aa;
			    work[i__ + k] += aa;
			}
			i__1 = i__ + j * lda;
			aa = (r__1 = a[i__1].r, dabs(r__1));
/*                    -> A(j+k,j+k) */
			s += aa;
			work[i__ + k] += s;
/*                    i=j */
			++i__;
			i__1 = i__ + j * lda;
			aa = (r__1 = a[i__1].r, dabs(r__1));
/*                    -> A(j,j) */
			work[j] = aa;
			s = 0.f;
			i__1 = *n - 1;
			for (l = j + 1; l <= i__1; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
/*              n is odd & A is (n+1)/2 by n */
		if (ilu == 0) {
/*                 uplo = 'U' */
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(j,n1+i) */
			    work[i__ + n1] += aa;
			    s += aa;
			}
			work[j] = s;
		    }
/*                 j=n1=k-1 is special */
		    i__1 = j * lda;
		    s = (r__1 = a[i__1].r, dabs(r__1));
/*                 A(k-1,k-1) */
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = c_abs(&a[i__ + j * lda]);
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(i,j-k) */
			    work[i__] += aa;
			    s += aa;
			}
/*                    i=j-k */
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    A(j-k,j-k) */
			s += aa;
			work[j - k] += s;
			++i__;
			i__2 = i__ + j * lda;
			s = (r__1 = a[i__2].r, dabs(r__1));
/*                    A(j,j) */
			i__2 = *n - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(j,l) */
			    work[l] += aa;
			    s += aa;
			}
			work[j] += s;
		    }
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu=1 & uplo = 'L' */
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(j,i) */
			    work[i__] += aa;
			    s += aa;
			}
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    i=j so process of A(j,j) */
			s += aa;
			work[j] = s;
/*                    is initialised here */
			++i__;
/*                    i=j process A(j+k,j+k) */
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
			s = aa;
			i__2 = *n - 1;
			for (l = k + j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
			aa = c_abs(&a[i__ + j * lda]);
/*                    A(k,i) */
			work[i__] += aa;
			s += aa;
		    }
/*                 i=k-1 */
		    i__1 = i__ + j * lda;
		    aa = (r__1 = a[i__1].r, dabs(r__1));
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
			    aa = c_abs(&a[i__ + j * lda]);
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
/*              n is even & A is k=n/2 by n+1 */
		if (ilu == 0) {
/*                 uplo = 'U' */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			s = 0.f;
			i__2 = k - 1;
			for (i__ = 0; i__ <= i__2; ++i__) {
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(j,i+k) */
			    work[i__ + k] += aa;
			    s += aa;
			}
			work[j] = s;
		    }
/*                 j=k */
		    i__1 = j * lda;
		    aa = (r__1 = a[i__1].r, dabs(r__1));
/*                 A(k,k) */
		    s = aa;
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = c_abs(&a[i__ + j * lda]);
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(i,j-k-1) */
			    work[i__] += aa;
			    s += aa;
			}
/*                    i=j-1-k */
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    A(j-k-1,j-k-1) */
			s += aa;
			work[j - k - 1] += s;
			++i__;
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    A(j,j) */
			s = aa;
			i__2 = *n - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
			aa = c_abs(&a[i__ + j * lda]);
/*                    A(i,k-1) */
			work[i__] += aa;
			s += aa;
		    }
/*                 i=k-1 */
		    i__1 = i__ + j * lda;
		    aa = (r__1 = a[i__1].r, dabs(r__1));
/*                 A(k-1,k-1) */
		    s += aa;
		    work[i__] += s;
		    i__ = isamax_(n, work, &c__1);
		    value = work[i__ - 1];
		} else {
/*                 ilu=1 & uplo = 'L' */
		    i__1 = *n - 1;
		    for (i__ = k; i__ <= i__1; ++i__) {
			work[i__] = 0.f;
		    }
/*                 j=0 is special :process col A(k:n-1,k) */
		    s = (r__1 = a[0].r, dabs(r__1));
/*                 A(k,k) */
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			aa = c_abs(&a[i__]);
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
			    aa = c_abs(&a[i__ + j * lda]);
/*                       A(j-1,i) */
			    work[i__] += aa;
			    s += aa;
			}
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
/*                    i=j-1 so process of A(j-1,j-1) */
			s += aa;
			work[j - 1] = s;
/*                    is initialised here */
			++i__;
/*                    i=j process A(j+k,j+k) */
			i__2 = i__ + j * lda;
			aa = (r__1 = a[i__2].r, dabs(r__1));
			s = aa;
			i__2 = *n - 1;
			for (l = k + j + 1; l <= i__2; ++l) {
			    ++i__;
			    aa = c_abs(&a[i__ + j * lda]);
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
			aa = c_abs(&a[i__ + j * lda]);
/*                    A(k,i) */
			work[i__] += aa;
			s += aa;
		    }

/*                 i=k-1 */
		    i__1 = i__ + j * lda;
		    aa = (r__1 = a[i__1].r, dabs(r__1));
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
			    aa = c_abs(&a[i__ + j * lda]);
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
/*              A is normal & A is n by k */
		if (ilu == 0) {
/*                 A is upper */
		    i__1 = k - 3;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 2;
			classq_(&i__2, &a[k + j + 1 + j * lda], &c__1, &scale, 
				 &s);
/*                    L at A(k,0) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j - 1;
			classq_(&i__2, &a[j * lda], &c__1, &scale, &s);
/*                    trap U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = k - 1;
/*                 -> U(k,k) at A(k-1,0) */
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    U(k+i,k+i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    U(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
		    i__1 = l;
		    aa = a[i__1].r;
/*                 U(n-1,n-1) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		} else {
/*                 ilu=1 & A is lower */
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = *n - j - 1;
			classq_(&i__2, &a[j + 1 + j * lda], &c__1, &scale, &s)
				;
/*                    trap L at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[(j + 1) * lda], &c__1, &scale, &s);
/*                    U at A(0,1) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    aa = a[0].r;
/*                 L(0,0) at A(0,0) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		    l = lda;
/*                 -> L(k,k) at A(0,1) */
		    i__1 = k - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    L(k-1+i,k-1+i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    L(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
		}
	    } else {
/*              A is xpose & A is k by n */
		if (ilu == 0) {
/*                 A' is upper */
		    i__1 = k - 2;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[(k + j) * lda], &c__1, &scale, &s);
/*                    U at A(0,k) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			classq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k-1 rect. at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			classq_(&i__2, &a[j + 1 + (j + k - 1) * lda], &c__1, &
				scale, &s);
/*                    L at A(0,k-1) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = k * lda - lda;
/*                 -> U(k-1,k-1) at A(0,k-1) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 U(k-1,k-1) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		    l += lda;
/*                 -> U(0,0) at A(0,k) */
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			i__2 = l;
			aa = a[i__2].r;
/*                    -> U(j-k,j-k) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    -> U(j,j) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
		} else {
/*                 A' is lower */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[j * lda], &c__1, &scale, &s);
/*                    U at A(0,0) */
		    }
		    i__1 = *n - 1;
		    for (j = k; j <= i__1; ++j) {
			classq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                    k by k-1 rect. at A(0,k) */
		    }
		    i__1 = k - 3;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 2;
			classq_(&i__2, &a[j + 2 + j * lda], &c__1, &scale, &s)
				;
/*                    L at A(1,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = 0;
/*                 -> L(0,0) at A(0,0) */
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    L(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    L(k+i,k+i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
/*                 L-> k-1 + (k-1)*lda or L(k-1,k-1) at A(k-1,k-1) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 L(k-1,k-1) at A(k-1,k-1) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
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
			classq_(&i__2, &a[k + j + 2 + j * lda], &c__1, &scale, 
				 &s);
/*                 L at A(k+1,0) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k + j;
			classq_(&i__2, &a[j * lda], &c__1, &scale, &s);
/*                 trap U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = k;
/*                 -> U(k,k) at A(k,0) */
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    U(k+i,k+i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    U(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
		} else {
/*                 ilu=1 & A is lower */
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = *n - j - 1;
			classq_(&i__2, &a[j + 2 + j * lda], &c__1, &scale, &s)
				;
/*                    trap L at A(1,0) */
		    }
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[j * lda], &c__1, &scale, &s);
/*                    U at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = 0;
/*                 -> L(k,k) at A(0,0) */
		    i__1 = k - 1;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    L(k-1+i,k-1+i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    L(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
		}
	    } else {
/*              A is xpose */
		if (ilu == 0) {
/*                 A' is upper */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[(k + 1 + j) * lda], &c__1, &scale, &s);
/*                 U at A(0,k+1) */
		    }
		    i__1 = k - 1;
		    for (j = 0; j <= i__1; ++j) {
			classq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                 k by k rect. at A(0,0) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			classq_(&i__2, &a[j + 1 + (j + k) * lda], &c__1, &
				scale, &s);
/*                 L at A(0,k) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = k * lda;
/*                 -> U(k,k) at A(0,k) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 U(k,k) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		    l += lda;
/*                 -> U(0,0) at A(0,k+1) */
		    i__1 = *n - 1;
		    for (j = k + 1; j <= i__1; ++j) {
			i__2 = l;
			aa = a[i__2].r;
/*                    -> U(j-k-1,j-k-1) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    -> U(j,j) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
/*                 L=k-1+n*lda */
/*                 -> U(k-1,k-1) at A(k-1,n) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 U(k,k) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		} else {
/*                 A' is lower */
		    i__1 = k - 1;
		    for (j = 1; j <= i__1; ++j) {
			classq_(&j, &a[(j + 1) * lda], &c__1, &scale, &s);
/*                 U at A(0,1) */
		    }
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			classq_(&k, &a[j * lda], &c__1, &scale, &s);
/*                 k by k rect. at A(0,k+1) */
		    }
		    i__1 = k - 2;
		    for (j = 0; j <= i__1; ++j) {
			i__2 = k - j - 1;
			classq_(&i__2, &a[j + 1 + j * lda], &c__1, &scale, &s)
				;
/*                 L at A(0,0) */
		    }
		    s += s;
/*                 double s for the off diagonal elements */
		    l = 0;
/*                 -> L(k,k) at A(0,0) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 L(k,k) at A(0,0) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		    l = lda;
/*                 -> L(0,0) at A(0,1) */
		    i__1 = k - 2;
		    for (i__ = 0; i__ <= i__1; ++i__) {
			i__2 = l;
			aa = a[i__2].r;
/*                    L(i,i) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			i__2 = l + 1;
			aa = a[i__2].r;
/*                    L(k+i+1,k+i+1) */
			if (aa != 0.f) {
			    if (scale < aa) {
/* Computing 2nd power */
				r__1 = scale / aa;
				s = s * (r__1 * r__1) + 1.f;
				scale = aa;
			    } else {
/* Computing 2nd power */
				r__1 = aa / scale;
				s += r__1 * r__1;
			    }
			}
			l = l + lda + 1;
		    }
/*                 L-> k - 1 + k*lda or L(k-1,k-1) at A(k-1,k) */
		    i__1 = l;
		    aa = a[i__1].r;
/*                 L(k-1,k-1) at A(k-1,k) */
		    if (aa != 0.f) {
			if (scale < aa) {
/* Computing 2nd power */
			    r__1 = scale / aa;
			    s = s * (r__1 * r__1) + 1.f;
			    scale = aa;
			} else {
/* Computing 2nd power */
			    r__1 = aa / scale;
			    s += r__1 * r__1;
			}
		    }
		}
	    }
	}
	value = scale * sqrt(s);
    }

    ret_val = value;
    return ret_val;

/*     End of CLANHF */

} /* clanhf_ */
