/* ctzrqf.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {1.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int ctzrqf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    complex q__1, q__2;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, k, m1;
    extern /* Subroutine */ int cgerc_(integer *, integer *, complex *, 
	    complex *, integer *, complex *, integer *, complex *, integer *);
    complex alpha;
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
, complex *, integer *, complex *, integer *, complex *, complex *
, integer *), ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), caxpy_(integer *, complex *, complex *, 
	    integer *, complex *, integer *), clacgv_(integer *, complex *, 
	    integer *), clarfp_(integer *, complex *, complex *, integer *, 
	    complex *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This routine is deprecated and has been replaced by routine CTZRZF. */

/*  CTZRQF reduces the M-by-N ( M<=N ) complex upper trapezoidal matrix A */
/*  to upper triangular form by means of unitary transformations. */

/*  The upper trapezoidal matrix A is factored as */

/*     A = ( R  0 ) * Z, */

/*  where Z is an N-by-N unitary matrix and R is an M-by-M upper */
/*  triangular matrix. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= M. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the leading M-by-N upper trapezoidal part of the */
/*          array A must contain the matrix to be factorized. */
/*          On exit, the leading M-by-M upper triangular part of A */
/*          contains the upper triangular matrix R, and elements M+1 to */
/*          N of the first M rows of A, with the array TAU, represent the */
/*          unitary matrix Z as a product of M elementary reflectors. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) COMPLEX array, dimension (M) */
/*          The scalar factors of the elementary reflectors. */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The  factorization is obtained by Householder's method.  The kth */
/*  transformation matrix, Z( k ), whose conjugate transpose is used to */
/*  introduce zeros into the (m - k + 1)th row of A, is given in the form */

/*     Z( k ) = ( I     0   ), */
/*              ( 0  T( k ) ) */

/*  where */

/*     T( k ) = I - tau*u( k )*u( k )',   u( k ) = (   1    ), */
/*                                                 (   0    ) */
/*                                                 ( z( k ) ) */

/*  tau is a scalar and z( k ) is an ( n - m ) element vector. */
/*  tau and z( k ) are chosen to annihilate the elements of the kth row */
/*  of X. */

/*  The scalar tau is returned in the kth element of TAU and the vector */
/*  u( k ) in the kth row of A, such that the elements of z( k ) are */
/*  in  a( k, m + 1 ), ..., a( k, n ). The elements of R are returned in */
/*  the upper triangular part of A. */

/*  Z is given by */

/*     Z =  Z( 1 ) * Z( 2 ) * ... * Z( m ). */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTZRQF", &i__1);
	return 0;
    }

/*     Perform the factorization. */

    if (*m == 0) {
	return 0;
    }
    if (*m == *n) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    tau[i__2].r = 0.f, tau[i__2].i = 0.f;
/* L10: */
	}
    } else {
/* Computing MIN */
	i__1 = *m + 1;
	m1 = min(i__1,*n);
	for (k = *m; k >= 1; --k) {

/*           Use a Householder reflection to zero the kth row of A. */
/*           First set up the reflection. */

	    i__1 = k + k * a_dim1;
	    r_cnjg(&q__1, &a[k + k * a_dim1]);
	    a[i__1].r = q__1.r, a[i__1].i = q__1.i;
	    i__1 = *n - *m;
	    clacgv_(&i__1, &a[k + m1 * a_dim1], lda);
	    i__1 = k + k * a_dim1;
	    alpha.r = a[i__1].r, alpha.i = a[i__1].i;
	    i__1 = *n - *m + 1;
	    clarfp_(&i__1, &alpha, &a[k + m1 * a_dim1], lda, &tau[k]);
	    i__1 = k + k * a_dim1;
	    a[i__1].r = alpha.r, a[i__1].i = alpha.i;
	    i__1 = k;
	    r_cnjg(&q__1, &tau[k]);
	    tau[i__1].r = q__1.r, tau[i__1].i = q__1.i;

	    i__1 = k;
	    if ((tau[i__1].r != 0.f || tau[i__1].i != 0.f) && k > 1) {

/*              We now perform the operation  A := A*P( k )'. */

/*              Use the first ( k - 1 ) elements of TAU to store  a( k ), */
/*              where  a( k ) consists of the first ( k - 1 ) elements of */
/*              the  kth column  of  A.  Also  let  B  denote  the  first */
/*              ( k - 1 ) rows of the last ( n - m ) columns of A. */

		i__1 = k - 1;
		ccopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &tau[1], &c__1);

/*              Form   w = a( k ) + B*z( k )  in TAU. */

		i__1 = k - 1;
		i__2 = *n - *m;
		cgemv_("No transpose", &i__1, &i__2, &c_b1, &a[m1 * a_dim1 + 
			1], lda, &a[k + m1 * a_dim1], lda, &c_b1, &tau[1], &
			c__1);

/*              Now form  a( k ) := a( k ) - conjg(tau)*w */
/*              and       B      := B      - conjg(tau)*w*z( k )'. */

		i__1 = k - 1;
		r_cnjg(&q__2, &tau[k]);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		caxpy_(&i__1, &q__1, &tau[1], &c__1, &a[k * a_dim1 + 1], &
			c__1);
		i__1 = k - 1;
		i__2 = *n - *m;
		r_cnjg(&q__2, &tau[k]);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		cgerc_(&i__1, &i__2, &q__1, &tau[1], &c__1, &a[k + m1 * 
			a_dim1], lda, &a[m1 * a_dim1 + 1], lda);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of CTZRQF */

} /* ctzrqf_ */
