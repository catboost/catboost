/* ctzrzf.f -- translated by f2c (version 20061008).
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
static integer c_n1 = -1;
static integer c__3 = 3;
static integer c__2 = 2;

/* Subroutine */ int ctzrzf_(integer *m, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    integer i__, m1, ib, nb, ki, kk, mu, nx, iws, nbmin;
    extern /* Subroutine */ int clarzb_(char *, char *, char *, char *, 
	    integer *, integer *, integer *, integer *, complex *, integer *, 
	    complex *, integer *, complex *, integer *, complex *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int clarzt_(char *, char *, integer *, integer *, 
	    complex *, integer *, complex *, complex *, integer *), clatrz_(integer *, integer *, integer *, complex *, 
	    integer *, complex *, complex *);
    integer ldwork, lwkopt;
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTZRZF reduces the M-by-N ( M<=N ) complex upper trapezoidal matrix A */
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

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK.  LWORK >= max(1,M). */
/*          For optimum performance LWORK >= M*NB, where NB is */
/*          the optimal blocksize. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

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
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -7;
    }

    if (*info == 0) {
	if (*m == 0 || *m == *n) {
	    lwkopt = 1;
	} else {

/*           Determine the block size. */

	    nb = ilaenv_(&c__1, "CGERQF", " ", m, n, &c_n1, &c_n1);
	    lwkopt = *m * nb;
	}
	work[1].r = (real) lwkopt, work[1].i = 0.f;

	if (*lwork < max(1,*m) && ! lquery) {
	    *info = -7;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTZRZF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0) {
	return 0;
    } else if (*m == *n) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    tau[i__2].r = 0.f, tau[i__2].i = 0.f;
/* L10: */
	}
	return 0;
    }

    nbmin = 2;
    nx = 1;
    iws = *m;
    if (nb > 1 && nb < *m) {

/*        Determine when to cross over from blocked to unblocked code. */

/* Computing MAX */
	i__1 = 0, i__2 = ilaenv_(&c__3, "CGERQF", " ", m, n, &c_n1, &c_n1);
	nx = max(i__1,i__2);
	if (nx < *m) {

/*           Determine if workspace is large enough for blocked code. */

	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {

/*              Not enough workspace to use optimal NB:  reduce NB and */
/*              determine the minimum value of NB. */

		nb = *lwork / ldwork;
/* Computing MAX */
		i__1 = 2, i__2 = ilaenv_(&c__2, "CGERQF", " ", m, n, &c_n1, &
			c_n1);
		nbmin = max(i__1,i__2);
	    }
	}
    }

    if (nb >= nbmin && nb < *m && nx < *m) {

/*        Use blocked code initially. */
/*        The last kk rows are handled by the block method. */

/* Computing MIN */
	i__1 = *m + 1;
	m1 = min(i__1,*n);
	ki = (*m - nx - 1) / nb * nb;
/* Computing MIN */
	i__1 = *m, i__2 = ki + nb;
	kk = min(i__1,i__2);

	i__1 = *m - kk + 1;
	i__2 = -nb;
	for (i__ = *m - kk + ki + 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; 
		i__ += i__2) {
/* Computing MIN */
	    i__3 = *m - i__ + 1;
	    ib = min(i__3,nb);

/*           Compute the TZ factorization of the current block */
/*           A(i:i+ib-1,i:n) */

	    i__3 = *n - i__ + 1;
	    i__4 = *n - *m;
	    clatrz_(&ib, &i__3, &i__4, &a[i__ + i__ * a_dim1], lda, &tau[i__], 
		     &work[1]);
	    if (i__ > 1) {

/*              Form the triangular factor of the block reflector */
/*              H = H(i+ib-1) . . . H(i+1) H(i) */

		i__3 = *n - *m;
		clarzt_("Backward", "Rowwise", &i__3, &ib, &a[i__ + m1 * 
			a_dim1], lda, &tau[i__], &work[1], &ldwork);

/*              Apply H to A(1:i-1,i:n) from the right */

		i__3 = i__ - 1;
		i__4 = *n - i__ + 1;
		i__5 = *n - *m;
		clarzb_("Right", "No transpose", "Backward", "Rowwise", &i__3, 
			 &i__4, &ib, &i__5, &a[i__ + m1 * a_dim1], lda, &work[
			1], &ldwork, &a[i__ * a_dim1 + 1], lda, &work[ib + 1], 
			 &ldwork)
			;
	    }
/* L20: */
	}
	mu = i__ + nb - 1;
    } else {
	mu = *m;
    }

/*     Use unblocked code to factor the last or only block */

    if (mu > 0) {
	i__2 = *n - *m;
	clatrz_(&mu, n, &i__2, &a[a_offset], lda, &tau[1], &work[1]);
    }

    work[1].r = (real) lwkopt, work[1].i = 0.f;

    return 0;

/*     End of CTZRZF */

} /* ctzrzf_ */
