/* slatrz.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slatrz_(integer *m, integer *n, integer *l, real *a, 
	integer *lda, real *tau, real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer i__;
    extern /* Subroutine */ int slarz_(char *, integer *, integer *, integer *
, real *, integer *, real *, real *, integer *, real *), 
	    slarfp_(integer *, real *, real *, integer *, real *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLATRZ factors the M-by-(M+L) real upper trapezoidal matrix */
/*  [ A1 A2 ] = [ A(1:M,1:M) A(1:M,N-L+1:N) ] as ( R  0 ) * Z, by means */
/*  of orthogonal transformations.  Z is an (M+L)-by-(M+L) orthogonal */
/*  matrix and, R and A1 are M-by-M upper triangular matrices. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  L       (input) INTEGER */
/*          The number of columns of the matrix A containing the */
/*          meaningful part of the Householder vectors. N-M >= L >= 0. */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          On entry, the leading M-by-N upper trapezoidal part of the */
/*          array A must contain the matrix to be factorized. */
/*          On exit, the leading M-by-M upper triangular part of A */
/*          contains the upper triangular matrix R, and elements N-L+1 to */
/*          N of the first M rows of A, with the array TAU, represent the */
/*          orthogonal matrix Z as a product of M elementary reflectors. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) REAL array, dimension (M) */
/*          The scalar factors of the elementary reflectors. */

/*  WORK    (workspace) REAL array, dimension (M) */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA */

/*  The factorization is obtained by Householder's method.  The kth */
/*  transformation matrix, Z( k ), which is used to introduce zeros into */
/*  the ( m - k + 1 )th row of A, is given in the form */

/*     Z( k ) = ( I     0   ), */
/*              ( 0  T( k ) ) */

/*  where */

/*     T( k ) = I - tau*u( k )*u( k )',   u( k ) = (   1    ), */
/*                                                 (   0    ) */
/*                                                 ( z( k ) ) */

/*  tau is a scalar and z( k ) is an l element vector. tau and z( k ) */
/*  are chosen to annihilate the elements of the kth row of A2. */

/*  The scalar tau is returned in the kth element of TAU and the vector */
/*  u( k ) in the kth row of A2, such that the elements of z( k ) are */
/*  in  a( k, l + 1 ), ..., a( k, n ). The elements of R are returned in */
/*  the upper triangular part of A1. */

/*  Z is given by */

/*     Z =  Z( 1 ) * Z( 2 ) * ... * Z( m ). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

/*     Quick return if possible */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    if (*m == 0) {
	return 0;
    } else if (*m == *n) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tau[i__] = 0.f;
/* L10: */
	}
	return 0;
    }

    for (i__ = *m; i__ >= 1; --i__) {

/*        Generate elementary reflector H(i) to annihilate */
/*        [ A(i,i) A(i,n-l+1:n) ] */

	i__1 = *l + 1;
	slarfp_(&i__1, &a[i__ + i__ * a_dim1], &a[i__ + (*n - *l + 1) * 
		a_dim1], lda, &tau[i__]);

/*        Apply H(i) to A(1:i-1,i:n) from the right */

	i__1 = i__ - 1;
	i__2 = *n - i__ + 1;
	slarz_("Right", &i__1, &i__2, l, &a[i__ + (*n - *l + 1) * a_dim1], 
		lda, &tau[i__], &a[i__ * a_dim1 + 1], lda, &work[1]);

/* L20: */
    }

    return 0;

/*     End of SLATRZ */

} /* slatrz_ */
