/* zlatrz.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlatrz_(integer *m, integer *n, integer *l, 
	doublecomplex *a, integer *lda, doublecomplex *tau, doublecomplex *
	work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublecomplex z__1;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__;
    doublecomplex alpha;
    extern /* Subroutine */ int zlarz_(char *, integer *, integer *, integer *
, doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, doublecomplex *), zlacgv_(integer *, 
	    doublecomplex *, integer *), zlarfp_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLATRZ factors the M-by-(M+L) complex upper trapezoidal matrix */
/*  [ A1 A2 ] = [ A(1:M,1:M) A(1:M,N-L+1:N) ] as ( R  0 ) * Z by means */
/*  of unitary transformations, where  Z is an (M+L)-by-(M+L) unitary */
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

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the leading M-by-N upper trapezoidal part of the */
/*          array A must contain the matrix to be factorized. */
/*          On exit, the leading M-by-M upper triangular part of A */
/*          contains the upper triangular matrix R, and elements N-L+1 to */
/*          N of the first M rows of A, with the array TAU, represent the */
/*          unitary matrix Z as a product of M elementary reflectors. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) COMPLEX*16 array, dimension (M) */
/*          The scalar factors of the elementary reflectors. */

/*  WORK    (workspace) COMPLEX*16 array, dimension (M) */

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
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

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
	    i__2 = i__;
	    tau[i__2].r = 0., tau[i__2].i = 0.;
/* L10: */
	}
	return 0;
    }

    for (i__ = *m; i__ >= 1; --i__) {

/*        Generate elementary reflector H(i) to annihilate */
/*        [ A(i,i) A(i,n-l+1:n) ] */

	zlacgv_(l, &a[i__ + (*n - *l + 1) * a_dim1], lda);
	d_cnjg(&z__1, &a[i__ + i__ * a_dim1]);
	alpha.r = z__1.r, alpha.i = z__1.i;
	i__1 = *l + 1;
	zlarfp_(&i__1, &alpha, &a[i__ + (*n - *l + 1) * a_dim1], lda, &tau[
		i__]);
	i__1 = i__;
	d_cnjg(&z__1, &tau[i__]);
	tau[i__1].r = z__1.r, tau[i__1].i = z__1.i;

/*        Apply H(i) to A(1:i-1,i:n) from the right */

	i__1 = i__ - 1;
	i__2 = *n - i__ + 1;
	d_cnjg(&z__1, &tau[i__]);
	zlarz_("Right", &i__1, &i__2, l, &a[i__ + (*n - *l + 1) * a_dim1], 
		lda, &z__1, &a[i__ * a_dim1 + 1], lda, &work[1]);
	i__1 = i__ + i__ * a_dim1;
	d_cnjg(&z__1, &alpha);
	a[i__1].r = z__1.r, a[i__1].i = z__1.i;

/* L20: */
    }

    return 0;

/*     End of ZLATRZ */

} /* zlatrz_ */
