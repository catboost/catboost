/* stzrqf.f -- translated by f2c (version 20061008).
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
static real c_b8 = 1.f;

/* Subroutine */ int stzrqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real r__1;

    /* Local variables */
    integer i__, k, m1;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *, 
	    integer *, real *, integer *, real *, integer *), sgemv_(char *, 
	    integer *, integer *, real *, real *, integer *, real *, integer *
, real *, real *, integer *), scopy_(integer *, real *, 
	    integer *, real *, integer *), saxpy_(integer *, real *, real *, 
	    integer *, real *, integer *), xerbla_(char *, integer *),
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

/*  This routine is deprecated and has been replaced by routine STZRZF. */

/*  STZRQF reduces the M-by-N ( M<=N ) real upper trapezoidal matrix A */
/*  to upper triangular form by means of orthogonal transformations. */

/*  The upper trapezoidal matrix A is factored as */

/*     A = ( R  0 ) * Z, */

/*  where Z is an N-by-N orthogonal matrix and R is an M-by-M upper */
/*  triangular matrix. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= M. */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          On entry, the leading M-by-N upper trapezoidal part of the */
/*          array A must contain the matrix to be factorized. */
/*          On exit, the leading M-by-M upper triangular part of A */
/*          contains the upper triangular matrix R, and elements M+1 to */
/*          N of the first M rows of A, with the array TAU, represent the */
/*          orthogonal matrix Z as a product of M elementary reflectors. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) REAL array, dimension (M) */
/*          The scalar factors of the elementary reflectors. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The factorization is obtained by Householder's method.  The kth */
/*  transformation matrix, Z( k ), which is used to introduce zeros into */
/*  the ( m - k + 1 )th row of A, is given in the form */

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

/*  ===================================================================== */

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
	xerbla_("STZRQF", &i__1);
	return 0;
    }

/*     Perform the factorization. */

    if (*m == 0) {
	return 0;
    }
    if (*m == *n) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    tau[i__] = 0.f;
/* L10: */
	}
    } else {
/* Computing MIN */
	i__1 = *m + 1;
	m1 = min(i__1,*n);
	for (k = *m; k >= 1; --k) {

/*           Use a Householder reflection to zero the kth row of A. */
/*           First set up the reflection. */

	    i__1 = *n - *m + 1;
	    slarfp_(&i__1, &a[k + k * a_dim1], &a[k + m1 * a_dim1], lda, &tau[
		    k]);

	    if (tau[k] != 0.f && k > 1) {

/*              We now perform the operation  A := A*P( k ). */

/*              Use the first ( k - 1 ) elements of TAU to store  a( k ), */
/*              where  a( k ) consists of the first ( k - 1 ) elements of */
/*              the  kth column  of  A.  Also  let  B  denote  the  first */
/*              ( k - 1 ) rows of the last ( n - m ) columns of A. */

		i__1 = k - 1;
		scopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &tau[1], &c__1);

/*              Form   w = a( k ) + B*z( k )  in TAU. */

		i__1 = k - 1;
		i__2 = *n - *m;
		sgemv_("No transpose", &i__1, &i__2, &c_b8, &a[m1 * a_dim1 + 
			1], lda, &a[k + m1 * a_dim1], lda, &c_b8, &tau[1], &
			c__1);

/*              Now form  a( k ) := a( k ) - tau*w */
/*              and       B      := B      - tau*w*z( k )'. */

		i__1 = k - 1;
		r__1 = -tau[k];
		saxpy_(&i__1, &r__1, &tau[1], &c__1, &a[k * a_dim1 + 1], &
			c__1);
		i__1 = k - 1;
		i__2 = *n - *m;
		r__1 = -tau[k];
		sger_(&i__1, &i__2, &r__1, &tau[1], &c__1, &a[k + m1 * a_dim1]
, lda, &a[m1 * a_dim1 + 1], lda);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of STZRQF */

} /* stzrqf_ */
