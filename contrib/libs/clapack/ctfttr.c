/* ctfttr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ctfttr_(char *transr, char *uplo, integer *n, complex *
	arf, complex *a, integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j, k, l, n1, n2, ij, nt, nx2, np1x2;
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

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTFTTR copies a triangular matrix A from rectangular full packed */
/*  format (TF) to standard full format (TR). */

/*  Arguments */
/*  ========= */

/*  TRANSR   (input) CHARACTER */
/*          = 'N':  ARF is in Normal format; */
/*          = 'C':  ARF is in Conjugate-transpose format; */

/*  UPLO    (input) CHARACTER */
/*          = 'U':  A is upper triangular; */
/*          = 'L':  A is lower triangular. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  ARF     (input) COMPLEX array, dimension ( N*(N+1)/2 ), */
/*          On entry, the upper or lower triangular matrix A stored in */
/*          RFP format. For a further discussion see Notes below. */

/*  A       (output) COMPLEX array, dimension ( LDA, N ) */
/*          On exit, the triangular matrix A.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of the array A contains */
/*          the upper triangular matrix, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading N-by-N lower triangular part of the array A contains */
/*          the lower triangular matrix, and the strictly upper */
/*          triangular part of A is not referenced. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

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

    /* Parameter adjustments */
    a_dim1 = *lda - 1 - 0 + 1;
    a_offset = 0 + a_dim1 * 0;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    normaltransr = lsame_(transr, "N");
    lower = lsame_(uplo, "L");
    if (! normaltransr && ! lsame_(transr, "C")) {
	*info = -1;
    } else if (! lower && ! lsame_(uplo, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTFTTR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 1) {
	if (*n == 1) {
	    if (normaltransr) {
		a[0].r = arf[0].r, a[0].i = arf[0].i;
	    } else {
		r_cnjg(&q__1, arf);
		a[0].r = q__1.r, a[0].i = q__1.i;
	    }
	}
	return 0;
    }

/*     Size of array ARF(1:2,0:nt-1) */

    nt = *n * (*n + 1) / 2;

/*     set N1 and N2 depending on LOWER: for N even N1=N2=K */

    if (lower) {
	n2 = *n / 2;
	n1 = *n - n2;
    } else {
	n1 = *n / 2;
	n2 = *n - n1;
    }

/*     If N is odd, set NISODD = .TRUE., LDA=N+1 and A is (N+1)--by--K2. */
/*     If N is even, set K = N/2 and NISODD = .FALSE., LDA=N and A is */
/*     N--by--(N+1)/2. */

    if (*n % 2 == 0) {
	k = *n / 2;
	nisodd = FALSE_;
	if (! lower) {
	    np1x2 = *n + *n + 2;
	}
    } else {
	nisodd = TRUE_;
	if (! lower) {
	    nx2 = *n + *n;
	}
    }

    if (nisodd) {

/*        N is odd */

	if (normaltransr) {

/*           N is odd and TRANSR = 'N' */

	    if (lower) {

/*             SRPA for LOWER, NORMAL and N is odd ( a(0:n-1,0:n1-1) ) */
/*             T1 -> a(0,0), T2 -> a(0,1), S -> a(n1,0) */
/*             T1 -> a(0), T2 -> a(n), S -> a(n1); lda=n */

		ij = 0;
		i__1 = n2;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = n2 + j;
		    for (i__ = n1; i__ <= i__2; ++i__) {
			i__3 = n2 + j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		}

	    } else {

/*             SRPA for UPPER, NORMAL and N is odd ( a(0:n-1,0:n2-1) */
/*             T1 -> a(n1+1,0), T2 -> a(n1,0), S -> a(0,0) */
/*             T1 -> a(n2), T2 -> a(n1), S -> a(0); lda=n */

		ij = nt - *n;
		i__1 = n1;
		for (j = *n - 1; j >= i__1; --j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		    i__2 = n1 - 1;
		    for (l = j - n1; l <= i__2; ++l) {
			i__3 = j - n1 + l * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    ij -= nx2;
		}

	    }

	} else {

/*           N is odd and TRANSR = 'C' */

	    if (lower) {

/*              SRPA for LOWER, TRANSPOSE and N is odd */
/*              T1 -> A(0,0) , T2 -> A(1,0) , S -> A(0,n1) */
/*              T1 -> A(0+0) , T2 -> A(1+0) , S -> A(0+n1*n1); lda=n1 */

		ij = 0;
		i__1 = n2 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (i__ = n1 + j; i__ <= i__2; ++i__) {
			i__3 = i__ + (n1 + j) * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		}
		i__1 = *n - 1;
		for (j = n2; j <= i__1; ++j) {
		    i__2 = n1 - 1;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		}

	    } else {

/*              SRPA for UPPER, TRANSPOSE and N is odd */
/*              T1 -> A(0,n1+1), T2 -> A(0,n1), S -> A(0,0) */
/*              T1 -> A(n2*n2), T2 -> A(n1*n2), S -> A(0); lda = n2 */

		ij = 0;
		i__1 = n1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = n1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		}
		i__1 = n1 - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (l = n2 + j; l <= i__2; ++l) {
			i__3 = n2 + j + l * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
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
/*              T1 -> a(1), T2 -> a(0), S -> a(k+1); lda=n+1 */

		ij = 0;
		i__1 = k - 1;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = k + j;
		    for (i__ = k; i__ <= i__2; ++i__) {
			i__3 = k + j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (i__ = j; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		}

	    } else {

/*              SRPA for UPPER, NORMAL, and N is even ( a(0:n,0:k-1) ) */
/*              T1 -> a(k+1,0) ,  T2 -> a(k,0),   S -> a(0,0) */
/*              T1 -> a(k+1), T2 -> a(k), S -> a(0); lda=n+1 */

		ij = nt - *n - 1;
		i__1 = k;
		for (j = *n - 1; j >= i__1; --j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		    i__2 = k - 1;
		    for (l = j - k; l <= i__2; ++l) {
			i__3 = j - k + l * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    ij -= np1x2;
		}

	    }

	} else {

/*           N is even and TRANSR = 'C' */

	    if (lower) {

/*              SRPA for LOWER, TRANSPOSE and N is even (see paper, A=B) */
/*              T1 -> A(0,1) , T2 -> A(0,0) , S -> A(0,k+1) : */
/*              T1 -> A(0+k) , T2 -> A(0+0) , S -> A(0+k*(k+1)); lda=k */

		ij = 0;
		j = k;
		i__1 = *n - 1;
		for (i__ = k; i__ <= i__1; ++i__) {
		    i__2 = i__ + j * a_dim1;
		    i__3 = ij;
		    a[i__2].r = arf[i__3].r, a[i__2].i = arf[i__3].i;
		    ++ij;
		}
		i__1 = k - 2;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (i__ = k + 1 + j; i__ <= i__2; ++i__) {
			i__3 = i__ + (k + 1 + j) * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		}
		i__1 = *n - 1;
		for (j = k - 1; j <= i__1; ++j) {
		    i__2 = k - 1;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		}

	    } else {

/*              SRPA for UPPER, TRANSPOSE and N is even (see paper, A=B) */
/*              T1 -> A(0,k+1) , T2 -> A(0,k) , S -> A(0,0) */
/*              T1 -> A(0+k*(k+1)) , T2 -> A(0+k*k) , S -> A(0+0)); lda=k */

		ij = 0;
		i__1 = k;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = *n - 1;
		    for (i__ = k; i__ <= i__2; ++i__) {
			i__3 = j + i__ * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		}
		i__1 = k - 2;
		for (j = 0; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 0; i__ <= i__2; ++i__) {
			i__3 = i__ + j * a_dim1;
			i__4 = ij;
			a[i__3].r = arf[i__4].r, a[i__3].i = arf[i__4].i;
			++ij;
		    }
		    i__2 = *n - 1;
		    for (l = k + 1 + j; l <= i__2; ++l) {
			i__3 = k + 1 + j + l * a_dim1;
			r_cnjg(&q__1, &arf[ij]);
			a[i__3].r = q__1.r, a[i__3].i = q__1.i;
			++ij;
		    }
		}

/*              Note that here J = K-1 */

		i__1 = j;
		for (i__ = 0; i__ <= i__1; ++i__) {
		    i__2 = i__ + j * a_dim1;
		    i__3 = ij;
		    a[i__2].r = arf[i__3].r, a[i__2].i = arf[i__3].i;
		    ++ij;
		}

	    }

	}

    }

    return 0;

/*     End of CTFTTR */

} /* ctfttr_ */
