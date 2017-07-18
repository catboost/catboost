/* ztfsm.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {1.,0.};

/* Subroutine */ int ztfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, doublecomplex *alpha, 
	doublecomplex *a, doublecomplex *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3;
    doublecomplex z__1;

    /* Local variables */
    integer i__, j, k, m1, m2, n1, n2, info;
    logical normaltransr, lside;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *);
    logical lower;
    extern /* Subroutine */ int ztrsm_(char *, char *, char *, char *, 
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *, 
	     doublecomplex *, integer *), 
	    xerbla_(char *, integer *);
    logical misodd, nisodd, notrans;


/*  -- LAPACK routine (version 3.2.1)                                    -- */

/*  -- Contributed by Fred Gustavson of the IBM Watson Research Center -- */
/*  -- April 2009                                                      -- */

/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Level 3 BLAS like routine for A in RFP Format. */

/*  ZTFSM  solves the matrix equation */

/*     op( A )*X = alpha*B  or  X*op( A ) = alpha*B */

/*  where alpha is a scalar, X and B are m by n matrices, A is a unit, or */
/*  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of */

/*     op( A ) = A   or   op( A ) = conjg( A' ). */

/*  A is in Rectangular Full Packed (RFP) Format. */

/*  The matrix X is overwritten on B. */

/*  Arguments */
/*  ========== */

/*  TRANSR - (input) CHARACTER */
/*          = 'N':  The Normal Form of RFP A is stored; */
/*          = 'C':  The Conjugate-transpose Form of RFP A is stored. */

/*  SIDE   - (input) CHARACTER */
/*           On entry, SIDE specifies whether op( A ) appears on the left */
/*           or right of X as follows: */

/*              SIDE = 'L' or 'l'   op( A )*X = alpha*B. */

/*              SIDE = 'R' or 'r'   X*op( A ) = alpha*B. */

/*           Unchanged on exit. */

/*  UPLO   - (input) CHARACTER */
/*           On entry, UPLO specifies whether the RFP matrix A came from */
/*           an upper or lower triangular matrix as follows: */
/*           UPLO = 'U' or 'u' RFP A came from an upper triangular matrix */
/*           UPLO = 'L' or 'l' RFP A came from a  lower triangular matrix */

/*           Unchanged on exit. */

/*  TRANS  - (input) CHARACTER */
/*           On entry, TRANS  specifies the form of op( A ) to be used */
/*           in the matrix multiplication as follows: */

/*              TRANS  = 'N' or 'n'   op( A ) = A. */

/*              TRANS  = 'C' or 'c'   op( A ) = conjg( A' ). */

/*           Unchanged on exit. */

/*  DIAG   - (input) CHARACTER */
/*           On entry, DIAG specifies whether or not RFP A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  M      - (input) INTEGER. */
/*           On entry, M specifies the number of rows of B. M must be at */
/*           least zero. */
/*           Unchanged on exit. */

/*  N      - (input) INTEGER. */
/*           On entry, N specifies the number of columns of B.  N must be */
/*           at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - (input) COMPLEX*16. */
/*           On entry,  ALPHA specifies the scalar  alpha. When  alpha is */
/*           zero then  A is not referenced and  B need not be set before */
/*           entry. */
/*           Unchanged on exit. */

/*  A      - (input) COMPLEX*16 array, dimension ( N*(N+1)/2 ); */
/*           NT = N*(N+1)/2. On entry, the matrix A in RFP Format. */
/*           RFP Format is described by TRANSR, UPLO and N as follows: */
/*           If TRANSR='N' then RFP A is (0:N,0:K-1) when N is even; */
/*           K=N/2. RFP A is (0:N-1,0:K) when N is odd; K=N/2. If */
/*           TRANSR = 'C' then RFP is the Conjugate-transpose of RFP A as */
/*           defined when TRANSR = 'N'. The contents of RFP A are defined */
/*           by UPLO as follows: If UPLO = 'U' the RFP A contains the NT */
/*           elements of upper packed A either in normal or */
/*           conjugate-transpose Format. If UPLO = 'L' the RFP A contains */
/*           the NT elements of lower packed A either in normal or */
/*           conjugate-transpose Format. The LDA of RFP A is (N+1)/2 when */
/*           TRANSR = 'C'. When TRANSR is 'N' the LDA is N+1 when N is */
/*           even and is N when is odd. */
/*           See the Note below for more details. Unchanged on exit. */

/*  B      - (input/ouptut) COMPLEX*16 array,  DIMENSION ( LDB, N) */
/*           Before entry,  the leading  m by n part of the array  B must */
/*           contain  the  right-hand  side  matrix  B,  and  on exit  is */
/*           overwritten by the solution matrix  X. */

/*  LDB    - (input) INTEGER. */
/*           On entry, LDB specifies the first dimension of B as declared */
/*           in  the  calling  (sub)  program.   LDB  must  be  at  least */
/*           max( 1, m ). */
/*           Unchanged on exit. */

/*  Further Details */
/*  =============== */

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

/*     Test the input parameters. */

    /* Parameter adjustments */
    b_dim1 = *ldb - 1 - 0 + 1;
    b_offset = 0 + b_dim1 * 0;
    b -= b_offset;

    /* Function Body */
    info = 0;
    normaltransr = lsame_(transr, "N");
    lside = lsame_(side, "L");
    lower = lsame_(uplo, "L");
    notrans = lsame_(trans, "N");
    if (! normaltransr && ! lsame_(transr, "C")) {
	info = -1;
    } else if (! lside && ! lsame_(side, "R")) {
	info = -2;
    } else if (! lower && ! lsame_(uplo, "U")) {
	info = -3;
    } else if (! notrans && ! lsame_(trans, "C")) {
	info = -4;
    } else if (! lsame_(diag, "N") && ! lsame_(diag, 
	    "U")) {
	info = -5;
    } else if (*m < 0) {
	info = -6;
    } else if (*n < 0) {
	info = -7;
    } else if (*ldb < max(1,*m)) {
	info = -11;
    }
    if (info != 0) {
	i__1 = -info;
	xerbla_("ZTFSM ", &i__1);
	return 0;
    }

/*     Quick return when ( (N.EQ.0).OR.(M.EQ.0) ) */

    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Quick return when ALPHA.EQ.(0D+0,0D+0) */

    if (alpha->r == 0. && alpha->i == 0.) {
	i__1 = *n - 1;
	for (j = 0; j <= i__1; ++j) {
	    i__2 = *m - 1;
	    for (i__ = 0; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		b[i__3].r = 0., b[i__3].i = 0.;
/* L10: */
	    }
/* L20: */
	}
	return 0;
    }

    if (lside) {

/*        SIDE = 'L' */

/*        A is M-by-M. */
/*        If M is odd, set NISODD = .TRUE., and M1 and M2. */
/*        If M is even, NISODD = .FALSE., and M. */

	if (*m % 2 == 0) {
	    misodd = FALSE_;
	    k = *m / 2;
	} else {
	    misodd = TRUE_;
	    if (lower) {
		m2 = *m / 2;
		m1 = *m - m2;
	    } else {
		m1 = *m / 2;
		m2 = *m - m1;
	    }
	}

	if (misodd) {

/*           SIDE = 'L' and N is odd */

	    if (normaltransr) {

/*              SIDE = 'L', N is odd, and TRANSR = 'N' */

		if (lower) {

/*                 SIDE  ='L', N is odd, TRANSR = 'N', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='L', N is odd, TRANSR = 'N', UPLO = 'L', and */
/*                    TRANS = 'N' */

			if (*m == 1) {
			    ztrsm_("L", "L", "N", diag, &m1, n, alpha, a, m, &
				    b[b_offset], ldb);
			} else {
			    ztrsm_("L", "L", "N", diag, &m1, n, alpha, a, m, &
				    b[b_offset], ldb);
			    z__1.r = -1., z__1.i = -0.;
			    zgemm_("N", "N", &m2, n, &m1, &z__1, &a[m1], m, &
				    b[b_offset], ldb, alpha, &b[m1], ldb);
			    ztrsm_("L", "U", "C", diag, &m2, n, &c_b1, &a[*m], 
				     m, &b[m1], ldb);
			}

		    } else {

/*                    SIDE  ='L', N is odd, TRANSR = 'N', UPLO = 'L', and */
/*                    TRANS = 'C' */

			if (*m == 1) {
			    ztrsm_("L", "L", "C", diag, &m1, n, alpha, a, m, &
				    b[b_offset], ldb);
			} else {
			    ztrsm_("L", "U", "N", diag, &m2, n, alpha, &a[*m], 
				     m, &b[m1], ldb);
			    z__1.r = -1., z__1.i = -0.;
			    zgemm_("C", "N", &m1, n, &m2, &z__1, &a[m1], m, &
				    b[m1], ldb, alpha, &b[b_offset], ldb);
			    ztrsm_("L", "L", "C", diag, &m1, n, &c_b1, a, m, &
				    b[b_offset], ldb);
			}

		    }

		} else {

/*                 SIDE  ='L', N is odd, TRANSR = 'N', and UPLO = 'U' */

		    if (! notrans) {

/*                    SIDE  ='L', N is odd, TRANSR = 'N', UPLO = 'U', and */
/*                    TRANS = 'N' */

			ztrsm_("L", "L", "N", diag, &m1, n, alpha, &a[m2], m, 
				&b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("C", "N", &m2, n, &m1, &z__1, a, m, &b[
				b_offset], ldb, alpha, &b[m1], ldb);
			ztrsm_("L", "U", "C", diag, &m2, n, &c_b1, &a[m1], m, 
				&b[m1], ldb);

		    } else {

/*                    SIDE  ='L', N is odd, TRANSR = 'N', UPLO = 'U', and */
/*                    TRANS = 'C' */

			ztrsm_("L", "U", "N", diag, &m2, n, alpha, &a[m1], m, 
				&b[m1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", &m1, n, &m2, &z__1, a, m, &b[m1], 
				ldb, alpha, &b[b_offset], ldb);
			ztrsm_("L", "L", "C", diag, &m1, n, &c_b1, &a[m2], m, 
				&b[b_offset], ldb);

		    }

		}

	    } else {

/*              SIDE = 'L', N is odd, and TRANSR = 'C' */

		if (lower) {

/*                 SIDE  ='L', N is odd, TRANSR = 'C', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='L', N is odd, TRANSR = 'C', UPLO = 'L', and */
/*                    TRANS = 'N' */

			if (*m == 1) {
			    ztrsm_("L", "U", "C", diag, &m1, n, alpha, a, &m1, 
				     &b[b_offset], ldb);
			} else {
			    ztrsm_("L", "U", "C", diag, &m1, n, alpha, a, &m1, 
				     &b[b_offset], ldb);
			    z__1.r = -1., z__1.i = -0.;
			    zgemm_("C", "N", &m2, n, &m1, &z__1, &a[m1 * m1], 
				    &m1, &b[b_offset], ldb, alpha, &b[m1], 
				    ldb);
			    ztrsm_("L", "L", "N", diag, &m2, n, &c_b1, &a[1], 
				    &m1, &b[m1], ldb);
			}

		    } else {

/*                    SIDE  ='L', N is odd, TRANSR = 'C', UPLO = 'L', and */
/*                    TRANS = 'C' */

			if (*m == 1) {
			    ztrsm_("L", "U", "N", diag, &m1, n, alpha, a, &m1, 
				     &b[b_offset], ldb);
			} else {
			    ztrsm_("L", "L", "C", diag, &m2, n, alpha, &a[1], 
				    &m1, &b[m1], ldb);
			    z__1.r = -1., z__1.i = -0.;
			    zgemm_("N", "N", &m1, n, &m2, &z__1, &a[m1 * m1], 
				    &m1, &b[m1], ldb, alpha, &b[b_offset], 
				    ldb);
			    ztrsm_("L", "U", "N", diag, &m1, n, &c_b1, a, &m1, 
				     &b[b_offset], ldb);
			}

		    }

		} else {

/*                 SIDE  ='L', N is odd, TRANSR = 'C', and UPLO = 'U' */

		    if (! notrans) {

/*                    SIDE  ='L', N is odd, TRANSR = 'C', UPLO = 'U', and */
/*                    TRANS = 'N' */

			ztrsm_("L", "U", "C", diag, &m1, n, alpha, &a[m2 * m2]
, &m2, &b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", &m2, n, &m1, &z__1, a, &m2, &b[
				b_offset], ldb, alpha, &b[m1], ldb);
			ztrsm_("L", "L", "N", diag, &m2, n, &c_b1, &a[m1 * m2]
, &m2, &b[m1], ldb);

		    } else {

/*                    SIDE  ='L', N is odd, TRANSR = 'C', UPLO = 'U', and */
/*                    TRANS = 'C' */

			ztrsm_("L", "L", "C", diag, &m2, n, alpha, &a[m1 * m2]
, &m2, &b[m1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("C", "N", &m1, n, &m2, &z__1, a, &m2, &b[m1], 
				ldb, alpha, &b[b_offset], ldb);
			ztrsm_("L", "U", "N", diag, &m1, n, &c_b1, &a[m2 * m2]
, &m2, &b[b_offset], ldb);

		    }

		}

	    }

	} else {

/*           SIDE = 'L' and N is even */

	    if (normaltransr) {

/*              SIDE = 'L', N is even, and TRANSR = 'N' */

		if (lower) {

/*                 SIDE  ='L', N is even, TRANSR = 'N', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='L', N is even, TRANSR = 'N', UPLO = 'L', */
/*                    and TRANS = 'N' */

			i__1 = *m + 1;
			ztrsm_("L", "L", "N", diag, &k, n, alpha, &a[1], &
				i__1, &b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *m + 1;
			zgemm_("N", "N", &k, n, &k, &z__1, &a[k + 1], &i__1, &
				b[b_offset], ldb, alpha, &b[k], ldb);
			i__1 = *m + 1;
			ztrsm_("L", "U", "C", diag, &k, n, &c_b1, a, &i__1, &
				b[k], ldb);

		    } else {

/*                    SIDE  ='L', N is even, TRANSR = 'N', UPLO = 'L', */
/*                    and TRANS = 'C' */

			i__1 = *m + 1;
			ztrsm_("L", "U", "N", diag, &k, n, alpha, a, &i__1, &
				b[k], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *m + 1;
			zgemm_("C", "N", &k, n, &k, &z__1, &a[k + 1], &i__1, &
				b[k], ldb, alpha, &b[b_offset], ldb);
			i__1 = *m + 1;
			ztrsm_("L", "L", "C", diag, &k, n, &c_b1, &a[1], &
				i__1, &b[b_offset], ldb);

		    }

		} else {

/*                 SIDE  ='L', N is even, TRANSR = 'N', and UPLO = 'U' */

		    if (! notrans) {

/*                    SIDE  ='L', N is even, TRANSR = 'N', UPLO = 'U', */
/*                    and TRANS = 'N' */

			i__1 = *m + 1;
			ztrsm_("L", "L", "N", diag, &k, n, alpha, &a[k + 1], &
				i__1, &b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *m + 1;
			zgemm_("C", "N", &k, n, &k, &z__1, a, &i__1, &b[
				b_offset], ldb, alpha, &b[k], ldb);
			i__1 = *m + 1;
			ztrsm_("L", "U", "C", diag, &k, n, &c_b1, &a[k], &
				i__1, &b[k], ldb);

		    } else {

/*                    SIDE  ='L', N is even, TRANSR = 'N', UPLO = 'U', */
/*                    and TRANS = 'C' */
			i__1 = *m + 1;
			ztrsm_("L", "U", "N", diag, &k, n, alpha, &a[k], &
				i__1, &b[k], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *m + 1;
			zgemm_("N", "N", &k, n, &k, &z__1, a, &i__1, &b[k], 
				ldb, alpha, &b[b_offset], ldb);
			i__1 = *m + 1;
			ztrsm_("L", "L", "C", diag, &k, n, &c_b1, &a[k + 1], &
				i__1, &b[b_offset], ldb);

		    }

		}

	    } else {

/*              SIDE = 'L', N is even, and TRANSR = 'C' */

		if (lower) {

/*                 SIDE  ='L', N is even, TRANSR = 'C', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='L', N is even, TRANSR = 'C', UPLO = 'L', */
/*                    and TRANS = 'N' */

			ztrsm_("L", "U", "C", diag, &k, n, alpha, &a[k], &k, &
				b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("C", "N", &k, n, &k, &z__1, &a[k * (k + 1)], &
				k, &b[b_offset], ldb, alpha, &b[k], ldb);
			ztrsm_("L", "L", "N", diag, &k, n, &c_b1, a, &k, &b[k]
, ldb);

		    } else {

/*                    SIDE  ='L', N is even, TRANSR = 'C', UPLO = 'L', */
/*                    and TRANS = 'C' */

			ztrsm_("L", "L", "C", diag, &k, n, alpha, a, &k, &b[k]
, ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", &k, n, &k, &z__1, &a[k * (k + 1)], &
				k, &b[k], ldb, alpha, &b[b_offset], ldb);
			ztrsm_("L", "U", "N", diag, &k, n, &c_b1, &a[k], &k, &
				b[b_offset], ldb);

		    }

		} else {

/*                 SIDE  ='L', N is even, TRANSR = 'C', and UPLO = 'U' */

		    if (! notrans) {

/*                    SIDE  ='L', N is even, TRANSR = 'C', UPLO = 'U', */
/*                    and TRANS = 'N' */

			ztrsm_("L", "U", "C", diag, &k, n, alpha, &a[k * (k + 
				1)], &k, &b[b_offset], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", &k, n, &k, &z__1, a, &k, &b[b_offset]
, ldb, alpha, &b[k], ldb);
			ztrsm_("L", "L", "N", diag, &k, n, &c_b1, &a[k * k], &
				k, &b[k], ldb);

		    } else {

/*                    SIDE  ='L', N is even, TRANSR = 'C', UPLO = 'U', */
/*                    and TRANS = 'C' */

			ztrsm_("L", "L", "C", diag, &k, n, alpha, &a[k * k], &
				k, &b[k], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("C", "N", &k, n, &k, &z__1, a, &k, &b[k], ldb, 
				alpha, &b[b_offset], ldb);
			ztrsm_("L", "U", "N", diag, &k, n, &c_b1, &a[k * (k + 
				1)], &k, &b[b_offset], ldb);

		    }

		}

	    }

	}

    } else {

/*        SIDE = 'R' */

/*        A is N-by-N. */
/*        If N is odd, set NISODD = .TRUE., and N1 and N2. */
/*        If N is even, NISODD = .FALSE., and K. */

	if (*n % 2 == 0) {
	    nisodd = FALSE_;
	    k = *n / 2;
	} else {
	    nisodd = TRUE_;
	    if (lower) {
		n2 = *n / 2;
		n1 = *n - n2;
	    } else {
		n1 = *n / 2;
		n2 = *n - n1;
	    }
	}

	if (nisodd) {

/*           SIDE = 'R' and N is odd */

	    if (normaltransr) {

/*              SIDE = 'R', N is odd, and TRANSR = 'N' */

		if (lower) {

/*                 SIDE  ='R', N is odd, TRANSR = 'N', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='R', N is odd, TRANSR = 'N', UPLO = 'L', and */
/*                    TRANS = 'N' */

			ztrsm_("R", "U", "C", diag, m, &n2, alpha, &a[*n], n, 
				&b[n1 * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &n1, &n2, &z__1, &b[n1 * b_dim1], 
				ldb, &a[n1], n, alpha, b, ldb);
			ztrsm_("R", "L", "N", diag, m, &n1, &c_b1, a, n, b, 
				ldb);

		    } else {

/*                    SIDE  ='R', N is odd, TRANSR = 'N', UPLO = 'L', and */
/*                    TRANS = 'C' */

			ztrsm_("R", "L", "C", diag, m, &n1, alpha, a, n, b, 
				ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &n2, &n1, &z__1, b, ldb, &a[n1], 
				n, alpha, &b[n1 * b_dim1], ldb);
			ztrsm_("R", "U", "N", diag, m, &n2, &c_b1, &a[*n], n, 
				&b[n1 * b_dim1], ldb);

		    }

		} else {

/*                 SIDE  ='R', N is odd, TRANSR = 'N', and UPLO = 'U' */

		    if (notrans) {

/*                    SIDE  ='R', N is odd, TRANSR = 'N', UPLO = 'U', and */
/*                    TRANS = 'N' */

			ztrsm_("R", "L", "C", diag, m, &n1, alpha, &a[n2], n, 
				b, ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &n2, &n1, &z__1, b, ldb, a, n, 
				alpha, &b[n1 * b_dim1], ldb);
			ztrsm_("R", "U", "N", diag, m, &n2, &c_b1, &a[n1], n, 
				&b[n1 * b_dim1], ldb);

		    } else {

/*                    SIDE  ='R', N is odd, TRANSR = 'N', UPLO = 'U', and */
/*                    TRANS = 'C' */

			ztrsm_("R", "U", "C", diag, m, &n2, alpha, &a[n1], n, 
				&b[n1 * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &n1, &n2, &z__1, &b[n1 * b_dim1], 
				ldb, a, n, alpha, b, ldb);
			ztrsm_("R", "L", "N", diag, m, &n1, &c_b1, &a[n2], n, 
				b, ldb);

		    }

		}

	    } else {

/*              SIDE = 'R', N is odd, and TRANSR = 'C' */

		if (lower) {

/*                 SIDE  ='R', N is odd, TRANSR = 'C', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='R', N is odd, TRANSR = 'C', UPLO = 'L', and */
/*                    TRANS = 'N' */

			ztrsm_("R", "L", "N", diag, m, &n2, alpha, &a[1], &n1, 
				 &b[n1 * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &n1, &n2, &z__1, &b[n1 * b_dim1], 
				ldb, &a[n1 * n1], &n1, alpha, b, ldb);
			ztrsm_("R", "U", "C", diag, m, &n1, &c_b1, a, &n1, b, 
				ldb);

		    } else {

/*                    SIDE  ='R', N is odd, TRANSR = 'C', UPLO = 'L', and */
/*                    TRANS = 'C' */

			ztrsm_("R", "U", "N", diag, m, &n1, alpha, a, &n1, b, 
				ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &n2, &n1, &z__1, b, ldb, &a[n1 * 
				n1], &n1, alpha, &b[n1 * b_dim1], ldb);
			ztrsm_("R", "L", "C", diag, m, &n2, &c_b1, &a[1], &n1, 
				 &b[n1 * b_dim1], ldb);

		    }

		} else {

/*                 SIDE  ='R', N is odd, TRANSR = 'C', and UPLO = 'U' */

		    if (notrans) {

/*                    SIDE  ='R', N is odd, TRANSR = 'C', UPLO = 'U', and */
/*                    TRANS = 'N' */

			ztrsm_("R", "U", "N", diag, m, &n1, alpha, &a[n2 * n2]
, &n2, b, ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &n2, &n1, &z__1, b, ldb, a, &n2, 
				alpha, &b[n1 * b_dim1], ldb);
			ztrsm_("R", "L", "C", diag, m, &n2, &c_b1, &a[n1 * n2]
, &n2, &b[n1 * b_dim1], ldb);

		    } else {

/*                    SIDE  ='R', N is odd, TRANSR = 'C', UPLO = 'U', and */
/*                    TRANS = 'C' */

			ztrsm_("R", "L", "N", diag, m, &n2, alpha, &a[n1 * n2]
, &n2, &b[n1 * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &n1, &n2, &z__1, &b[n1 * b_dim1], 
				ldb, a, &n2, alpha, b, ldb);
			ztrsm_("R", "U", "C", diag, m, &n1, &c_b1, &a[n2 * n2]
, &n2, b, ldb);

		    }

		}

	    }

	} else {

/*           SIDE = 'R' and N is even */

	    if (normaltransr) {

/*              SIDE = 'R', N is even, and TRANSR = 'N' */

		if (lower) {

/*                 SIDE  ='R', N is even, TRANSR = 'N', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='R', N is even, TRANSR = 'N', UPLO = 'L', */
/*                    and TRANS = 'N' */

			i__1 = *n + 1;
			ztrsm_("R", "U", "C", diag, m, &k, alpha, a, &i__1, &
				b[k * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *n + 1;
			zgemm_("N", "N", m, &k, &k, &z__1, &b[k * b_dim1], 
				ldb, &a[k + 1], &i__1, alpha, b, ldb);
			i__1 = *n + 1;
			ztrsm_("R", "L", "N", diag, m, &k, &c_b1, &a[1], &
				i__1, b, ldb);

		    } else {

/*                    SIDE  ='R', N is even, TRANSR = 'N', UPLO = 'L', */
/*                    and TRANS = 'C' */

			i__1 = *n + 1;
			ztrsm_("R", "L", "C", diag, m, &k, alpha, &a[1], &
				i__1, b, ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *n + 1;
			zgemm_("N", "C", m, &k, &k, &z__1, b, ldb, &a[k + 1], 
				&i__1, alpha, &b[k * b_dim1], ldb);
			i__1 = *n + 1;
			ztrsm_("R", "U", "N", diag, m, &k, &c_b1, a, &i__1, &
				b[k * b_dim1], ldb);

		    }

		} else {

/*                 SIDE  ='R', N is even, TRANSR = 'N', and UPLO = 'U' */

		    if (notrans) {

/*                    SIDE  ='R', N is even, TRANSR = 'N', UPLO = 'U', */
/*                    and TRANS = 'N' */

			i__1 = *n + 1;
			ztrsm_("R", "L", "C", diag, m, &k, alpha, &a[k + 1], &
				i__1, b, ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *n + 1;
			zgemm_("N", "N", m, &k, &k, &z__1, b, ldb, a, &i__1, 
				alpha, &b[k * b_dim1], ldb);
			i__1 = *n + 1;
			ztrsm_("R", "U", "N", diag, m, &k, &c_b1, &a[k], &
				i__1, &b[k * b_dim1], ldb);

		    } else {

/*                    SIDE  ='R', N is even, TRANSR = 'N', UPLO = 'U', */
/*                    and TRANS = 'C' */

			i__1 = *n + 1;
			ztrsm_("R", "U", "C", diag, m, &k, alpha, &a[k], &
				i__1, &b[k * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			i__1 = *n + 1;
			zgemm_("N", "C", m, &k, &k, &z__1, &b[k * b_dim1], 
				ldb, a, &i__1, alpha, b, ldb);
			i__1 = *n + 1;
			ztrsm_("R", "L", "N", diag, m, &k, &c_b1, &a[k + 1], &
				i__1, b, ldb);

		    }

		}

	    } else {

/*              SIDE = 'R', N is even, and TRANSR = 'C' */

		if (lower) {

/*                 SIDE  ='R', N is even, TRANSR = 'C', and UPLO = 'L' */

		    if (notrans) {

/*                    SIDE  ='R', N is even, TRANSR = 'C', UPLO = 'L', */
/*                    and TRANS = 'N' */

			ztrsm_("R", "L", "N", diag, m, &k, alpha, a, &k, &b[k 
				* b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &k, &k, &z__1, &b[k * b_dim1], 
				ldb, &a[(k + 1) * k], &k, alpha, b, ldb);
			ztrsm_("R", "U", "C", diag, m, &k, &c_b1, &a[k], &k, 
				b, ldb);

		    } else {

/*                    SIDE  ='R', N is even, TRANSR = 'C', UPLO = 'L', */
/*                    and TRANS = 'C' */

			ztrsm_("R", "U", "N", diag, m, &k, alpha, &a[k], &k, 
				b, ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &k, &k, &z__1, b, ldb, &a[(k + 1) 
				* k], &k, alpha, &b[k * b_dim1], ldb);
			ztrsm_("R", "L", "C", diag, m, &k, &c_b1, a, &k, &b[k 
				* b_dim1], ldb);

		    }

		} else {

/*                 SIDE  ='R', N is even, TRANSR = 'C', and UPLO = 'U' */

		    if (notrans) {

/*                    SIDE  ='R', N is even, TRANSR = 'C', UPLO = 'U', */
/*                    and TRANS = 'N' */

			ztrsm_("R", "U", "N", diag, m, &k, alpha, &a[(k + 1) *
				 k], &k, b, ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "C", m, &k, &k, &z__1, b, ldb, a, &k, 
				alpha, &b[k * b_dim1], ldb);
			ztrsm_("R", "L", "C", diag, m, &k, &c_b1, &a[k * k], &
				k, &b[k * b_dim1], ldb);

		    } else {

/*                    SIDE  ='R', N is even, TRANSR = 'C', UPLO = 'U', */
/*                    and TRANS = 'C' */

			ztrsm_("R", "L", "N", diag, m, &k, alpha, &a[k * k], &
				k, &b[k * b_dim1], ldb);
			z__1.r = -1., z__1.i = -0.;
			zgemm_("N", "N", m, &k, &k, &z__1, &b[k * b_dim1], 
				ldb, a, &k, alpha, b, ldb);
			ztrsm_("R", "U", "C", diag, m, &k, &c_b1, &a[(k + 1) *
				 k], &k, b, ldb);

		    }

		}

	    }

	}
    }

    return 0;

/*     End of ZTFSM */

} /* ztfsm_ */
