/* ztpttf.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ztpttf_(char *transr, char *uplo, integer *n, 
	doublecomplex *ap, doublecomplex *arf, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j, k, n1, n2, ij, jp, js, nt, lda, ijp;
    logical normaltransr;
    extern logical lsame_(char *, char *);
    logical lower;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical nisodd;


/*  -- LAPACK routine (version 3.2)                                    -- */

/*  -- Contributed by Fred Gustavson of the IBM Watson Research Center -- */
/*  -- November 2008                                                   -- */

/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */

/*  Purpose */
/*  ======= */

/*  ZTPTTF copies a triangular matrix A from standard packed format (TP) */
/*  to rectangular full packed format (TF). */

/*  Arguments */
/*  ========= */

/*  TRANSR   (input) CHARACTER */
/*          = 'N':  ARF in Normal format is wanted; */
/*          = 'C':  ARF in Conjugate-transpose format is wanted. */

/*  UPLO    (input) CHARACTER */
/*          = 'U':  A is upper triangular; */
/*          = 'L':  A is lower triangular. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input) COMPLEX*16 array, dimension ( N*(N+1)/2 ), */
/*          On entry, the upper or lower triangular matrix A, packed */
/*          columnwise in a linear array. The j-th column of A is stored */
/*          in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */

/*  ARF     (output) COMPLEX*16 array, dimension ( N*(N+1)/2 ), */
/*          On exit, the upper or lower triangular matrix A stored in */
/*          RFP format. For a further discussion see Notes below. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Notes: */
/*  ====== */

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

/*     Test the input parameters. */

    *info = 0;
    normaltransr = lsame_(transr, "N");
    lower = lsame_(uplo, "L");
    if (! normaltransr && ! lsame_(transr, "C")) {
	*info = -1;
    } else if (! lower && ! lsame_(uplo, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTPTTF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (normaltransr) {
	    arf[0].r = ap[0].r, arf[0].i = ap[0].i;
	} else {
	    d_cnjg(&z__1, ap);
	    arf[0].r = z__1.r, arf[0].i = z__1.i;
	}
	return 0;
    }

/*     Size of array ARF(0:NT-1) */

    nt = *n * (*n + 1) / 2;

/*     Set N1 and N2 depending on LOWER */

    if (lower) {
	n2 = *n / 2;
	n1 = *n - n2;
    } else {
	n1 = *n / 2;
	n2 = *n - n1;
    }

/*     If N is odd, set NISODD = .TRUE. */
/*     If N is even, set K = N/2 and NISODD = .FALSE. */

/*     set lda of ARF^C; ARF^C is (0:(N+1)/2-1,0:N-noe) */
/*     where noe = 0 if n is even, noe = 1 if n is odd */

    if (*n % 2 == 0) {
	k = *n / 2;
	nisodd = FALSE_;
	lda = *n + 1;
    } else {
	nisodd = TRUE_;
	lda = *n;
    }

/*     ARF^C has lda rows and n+1-noe cols */

    if (! normaltransr) {
	lda = (*n + 1) / 2;
    }

/*     start execution: there are eight cases */

    if (nisodd) {

/*        N is odd */

	if (normaltransr) {

/*           N is odd and TRANSR = 'N' */

	    if (lower) {

/*             SRPA for LOWER, NORMAL and N is odd ( a(0:n-1,0:n1-1) ) */
/*             T1 -> a(0,0), T2 -> a(0,1), S -> a(n1,0) */
/*             T1 -> a(0), T2 -> a(n), S -> a(n1); lda = n */

		ijp = 0;
		jp = 0;
		i__1 = n2;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			ij = i__ + jp;
			i__3 = ij;
			i__4 = ijp;
			arf[i__3].r = ap[i__4].r, arf[i__3].i = ap[i__4].i;
			++ijp;
		    }
		    jp += lda;
		}
		i__1 = n2 - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = n2;
		    for (j = i__ + 1; j <= i__2; ++j) {
			ij = i__ + j * lda;
			i__3 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__3].r = z__1.r, arf[i__3].i = z__1.i;
			++ijp;
		    }
		}

	    } else {

/*             SRPA for UPPER, NORMAL and N is odd ( a(0:n-1,0:n2-1) */
/*             T1 -> a(n1+1,0), T2 -> a(n1,0), S -> a(0,0) */
/*             T1 -> a(n2), T2 -> a(n1), S -> a(0) */

		ijp = 0;
		i__1 = n1 - 1;
		for (j = 0; j <= i__1; ++j) {
		    ij = n2 + j;
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__3].r = z__1.r, arf[i__3].i = z__1.i;
			++ijp;
			ij += lda;
		    }
		}
		js = 0;
		i__1 = *n - 1;
		for (j = n1; j <= i__1; ++j) {
		    ij = js;
		    i__2 = js + j;
		    for (ij = js; ij <= i__2; ++ij) {
			i__3 = ij;
			i__4 = ijp;
			arf[i__3].r = ap[i__4].r, arf[i__3].i = ap[i__4].i;
			++ijp;
		    }
		    js += lda;
		}

	    }

	} else {

/*           N is odd and TRANSR = 'C' */

	    if (lower) {

/*              SRPA for LOWER, TRANSPOSE and N is odd */
/*              T1 -> A(0,0) , T2 -> A(1,0) , S -> A(0,n1) */
/*              T1 -> a(0+0) , T2 -> a(1+0) , S -> a(0+n1*n1); lda=n1 */

		ijp = 0;
		i__1 = n2;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = *n * lda - 1;
		    i__3 = lda;
		    for (ij = i__ * (lda + 1); i__3 < 0 ? ij >= i__2 : ij <= 
			    i__2; ij += i__3) {
			i__4 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__4].r = z__1.r, arf[i__4].i = z__1.i;
			++ijp;
		    }
		}
		js = 1;
		i__1 = n2 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + n2 - j - 1;
		    for (ij = js; ij <= i__3; ++ij) {
			i__2 = ij;
			i__4 = ijp;
			arf[i__2].r = ap[i__4].r, arf[i__2].i = ap[i__4].i;
			++ijp;
		    }
		    js = js + lda + 1;
		}

	    } else {

/*              SRPA for UPPER, TRANSPOSE and N is odd */
/*              T1 -> A(0,n1+1), T2 -> A(0,n1), S -> A(0,0) */
/*              T1 -> a(n2*n2), T2 -> a(n1*n2), S -> a(0); lda = n2 */

		ijp = 0;
		js = n2 * lda;
		i__1 = n1 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + j;
		    for (ij = js; ij <= i__3; ++ij) {
			i__2 = ij;
			i__4 = ijp;
			arf[i__2].r = ap[i__4].r, arf[i__2].i = ap[i__4].i;
			++ijp;
		    }
		    js += lda;
		}
		i__1 = n1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__3 = i__ + (n1 + i__) * lda;
		    i__2 = lda;
		    for (ij = i__; i__2 < 0 ? ij >= i__3 : ij <= i__3; ij += 
			    i__2) {
			i__4 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__4].r = z__1.r, arf[i__4].i = z__1.i;
			++ijp;
		    }
		}

	    }

	}

    } else {

/*        N is even */

	if (normaltransr) {

/*           N is even and TRANSR = 'N' */

	    if (lower) {

/*              SRPA for LOWER, NORMAL, and N is even ( a(0:n,0:k-1) ) */
/*              T1 -> a(1,0), T2 -> a(0,0), S -> a(k+1,0) */
/*              T1 -> a(1), T2 -> a(0), S -> a(k+1) */

		ijp = 0;
		jp = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			ij = i__ + 1 + jp;
			i__3 = ij;
			i__4 = ijp;
			arf[i__3].r = ap[i__4].r, arf[i__3].i = ap[i__4].i;
			++ijp;
		    }
		    jp += lda;
		}
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = k - 1;
		    for (j = i__; j <= i__2; ++j) {
			ij = i__ + j * lda;
			i__3 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__3].r = z__1.r, arf[i__3].i = z__1.i;
			++ijp;
		    }
		}

	    } else {

/*              SRPA for UPPER, NORMAL, and N is even ( a(0:n,0:k-1) ) */
/*              T1 -> a(k+1,0) ,  T2 -> a(k,0),   S -> a(0,0) */
/*              T1 -> a(k+1), T2 -> a(k), S -> a(0) */

		ijp = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    ij = k + 1 + j;
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__3].r = z__1.r, arf[i__3].i = z__1.i;
			++ijp;
			ij += lda;
		    }
		}
		js = 0;
		i__1 = *n - 1;
		for (j = k; j <= i__1; ++j) {
		    ij = js;
		    i__2 = js + j;
		    for (ij = js; ij <= i__2; ++ij) {
			i__3 = ij;
			i__4 = ijp;
			arf[i__3].r = ap[i__4].r, arf[i__3].i = ap[i__4].i;
			++ijp;
		    }
		    js += lda;
		}

	    }

	} else {

/*           N is even and TRANSR = 'C' */

	    if (lower) {

/*              SRPA for LOWER, TRANSPOSE and N is even (see paper) */
/*              T1 -> B(0,1), T2 -> B(0,0), S -> B(0,k+1) */
/*              T1 -> a(0+k), T2 -> a(0+0), S -> a(0+k*(k+1)); lda=k */

		ijp = 0;
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = (*n + 1) * lda - 1;
		    i__3 = lda;
		    for (ij = i__ + (i__ + 1) * lda; i__3 < 0 ? ij >= i__2 : 
			    ij <= i__2; ij += i__3) {
			i__4 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__4].r = z__1.r, arf[i__4].i = z__1.i;
			++ijp;
		    }
		}
		js = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + k - j - 1;
		    for (ij = js; ij <= i__3; ++ij) {
			i__2 = ij;
			i__4 = ijp;
			arf[i__2].r = ap[i__4].r, arf[i__2].i = ap[i__4].i;
			++ijp;
		    }
		    js = js + lda + 1;
		}

	    } else {

/*              SRPA for UPPER, TRANSPOSE and N is even (see paper) */
/*              T1 -> B(0,k+1),     T2 -> B(0,k),   S -> B(0,0) */
/*              T1 -> a(0+k*(k+1)), T2 -> a(0+k*k), S -> a(0+0)); lda=k */

		ijp = 0;
		js = (k + 1) * lda;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__3 = js + j;
		    for (ij = js; ij <= i__3; ++ij) {
			i__2 = ij;
			i__4 = ijp;
			arf[i__2].r = ap[i__4].r, arf[i__2].i = ap[i__4].i;
			++ijp;
		    }
		    js += lda;
		}
		i__1 = k - 1;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__3 = i__ + (k + i__) * lda;
		    i__2 = lda;
		    for (ij = i__; i__2 < 0 ? ij >= i__3 : ij <= i__3; ij += 
			    i__2) {
			i__4 = ij;
			d_cnjg(&z__1, &ap[ijp]);
			arf[i__4].r = z__1.r, arf[i__4].i = z__1.i;
			++ijp;
		    }
		}

	    }

	}

    }

    return 0;

/*     End of ZTPTTF */

} /* ztpttf_ */
