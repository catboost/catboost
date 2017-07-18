/* sopgtr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int sopgtr_(char *uplo, integer *n, real *ap, real *tau, 
	real *q, integer *ldq, real *work, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, ij;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical upper;
    extern /* Subroutine */ int sorg2l_(integer *, integer *, integer *, real 
	    *, integer *, real *, real *, integer *), sorg2r_(integer *, 
	    integer *, integer *, real *, integer *, real *, real *, integer *
), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SOPGTR generates a real orthogonal matrix Q which is defined as the */
/*  product of n-1 elementary reflectors H(i) of order n, as returned by */
/*  SSPTRD using packed storage: */

/*  if UPLO = 'U', Q = H(n-1) . . . H(2) H(1), */

/*  if UPLO = 'L', Q = H(1) H(2) . . . H(n-1). */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U': Upper triangular packed storage used in previous */
/*                 call to SSPTRD; */
/*          = 'L': Lower triangular packed storage used in previous */
/*                 call to SSPTRD. */

/*  N       (input) INTEGER */
/*          The order of the matrix Q. N >= 0. */

/*  AP      (input) REAL array, dimension (N*(N+1)/2) */
/*          The vectors which define the elementary reflectors, as */
/*          returned by SSPTRD. */

/*  TAU     (input) REAL array, dimension (N-1) */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i), as returned by SSPTRD. */

/*  Q       (output) REAL array, dimension (LDQ,N) */
/*          The N-by-N orthogonal matrix Q. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. LDQ >= max(1,N). */

/*  WORK    (workspace) REAL array, dimension (N-1) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

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

/*     Test the input arguments */

    /* Parameter adjustments */
    --ap;
    --tau;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldq < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SOPGTR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (upper) {

/*        Q was determined by a call to SSPTRD with UPLO = 'U' */

/*        Unpack the vectors which define the elementary reflectors and */
/*        set the last row and column of Q equal to those of the unit */
/*        matrix */

	ij = 2;
	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		q[i__ + j * q_dim1] = ap[ij];
		++ij;
/* L10: */
	    }
	    ij += 2;
	    q[*n + j * q_dim1] = 0.f;
/* L20: */
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    q[i__ + *n * q_dim1] = 0.f;
/* L30: */
	}
	q[*n + *n * q_dim1] = 1.f;

/*        Generate Q(1:n-1,1:n-1) */

	i__1 = *n - 1;
	i__2 = *n - 1;
	i__3 = *n - 1;
	sorg2l_(&i__1, &i__2, &i__3, &q[q_offset], ldq, &tau[1], &work[1], &
		iinfo);

    } else {

/*        Q was determined by a call to SSPTRD with UPLO = 'L'. */

/*        Unpack the vectors which define the elementary reflectors and */
/*        set the first row and column of Q equal to those of the unit */
/*        matrix */

	q[q_dim1 + 1] = 1.f;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    q[i__ + q_dim1] = 0.f;
/* L40: */
	}
	ij = 3;
	i__1 = *n;
	for (j = 2; j <= i__1; ++j) {
	    q[j * q_dim1 + 1] = 0.f;
	    i__2 = *n;
	    for (i__ = j + 1; i__ <= i__2; ++i__) {
		q[i__ + j * q_dim1] = ap[ij];
		++ij;
/* L50: */
	    }
	    ij += 2;
/* L60: */
	}
	if (*n > 1) {

/*           Generate Q(2:n,2:n) */

	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    sorg2r_(&i__1, &i__2, &i__3, &q[(q_dim1 << 1) + 2], ldq, &tau[1], 
		    &work[1], &iinfo);
	}
    }
    return 0;

/*     End of SOPGTR */

} /* sopgtr_ */
