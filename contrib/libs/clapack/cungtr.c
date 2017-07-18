/* cungtr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cungtr_(char *uplo, integer *n, complex *a, integer *lda, 
	 complex *tau, complex *work, integer *lwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, j, nb;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int cungql_(integer *, integer *, integer *, 
	    complex *, integer *, complex *, complex *, integer *, integer *),
	     cungqr_(integer *, integer *, integer *, complex *, integer *, 
	    complex *, complex *, integer *, integer *);
    integer lwkopt;
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

/*  CUNGTR generates a complex unitary matrix Q which is defined as the */
/*  product of n-1 elementary reflectors of order N, as returned by */
/*  CHETRD: */

/*  if UPLO = 'U', Q = H(n-1) . . . H(2) H(1), */

/*  if UPLO = 'L', Q = H(1) H(2) . . . H(n-1). */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U': Upper triangle of A contains elementary reflectors */
/*                 from CHETRD; */
/*          = 'L': Lower triangle of A contains elementary reflectors */
/*                 from CHETRD. */

/*  N       (input) INTEGER */
/*          The order of the matrix Q. N >= 0. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the vectors which define the elementary reflectors, */
/*          as returned by CHETRD. */
/*          On exit, the N-by-N unitary matrix Q. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= N. */

/*  TAU     (input) COMPLEX array, dimension (N-1) */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i), as returned by CHETRD. */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= N-1. */
/*          For optimum performance LWORK >= (N-1)*NB, where NB is */
/*          the optimal blocksize. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__2 = *n - 1;
	if (*lwork < max(i__1,i__2) && ! lquery) {
	    *info = -7;
	}
    }

    if (*info == 0) {
	if (upper) {
	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    nb = ilaenv_(&c__1, "CUNGQL", " ", &i__1, &i__2, &i__3, &c_n1);
	} else {
	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    nb = ilaenv_(&c__1, "CUNGQR", " ", &i__1, &i__2, &i__3, &c_n1);
	}
/* Computing MAX */
	i__1 = 1, i__2 = *n - 1;
	lwkopt = max(i__1,i__2) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUNGTR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
	return 0;
    }

    if (upper) {

/*        Q was determined by a call to CHETRD with UPLO = 'U' */

/*        Shift the vectors which define the elementary reflectors one */
/*        column to the left, and set the last row and column of Q to */
/*        those of the unit matrix */

	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * a_dim1;
		i__4 = i__ + (j + 1) * a_dim1;
		a[i__3].r = a[i__4].r, a[i__3].i = a[i__4].i;
/* L10: */
	    }
	    i__2 = *n + j * a_dim1;
	    a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L20: */
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__ + *n * a_dim1;
	    a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L30: */
	}
	i__1 = *n + *n * a_dim1;
	a[i__1].r = 1.f, a[i__1].i = 0.f;

/*        Generate Q(1:n-1,1:n-1) */

	i__1 = *n - 1;
	i__2 = *n - 1;
	i__3 = *n - 1;
	cungql_(&i__1, &i__2, &i__3, &a[a_offset], lda, &tau[1], &work[1], 
		lwork, &iinfo);

    } else {

/*        Q was determined by a call to CHETRD with UPLO = 'L'. */

/*        Shift the vectors which define the elementary reflectors one */
/*        column to the right, and set the first row and column of Q to */
/*        those of the unit matrix */

	for (j = *n; j >= 2; --j) {
	    i__1 = j * a_dim1 + 1;
	    a[i__1].r = 0.f, a[i__1].i = 0.f;
	    i__1 = *n;
	    for (i__ = j + 1; i__ <= i__1; ++i__) {
		i__2 = i__ + j * a_dim1;
		i__3 = i__ + (j - 1) * a_dim1;
		a[i__2].r = a[i__3].r, a[i__2].i = a[i__3].i;
/* L40: */
	    }
/* L50: */
	}
	i__1 = a_dim1 + 1;
	a[i__1].r = 1.f, a[i__1].i = 0.f;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    i__2 = i__ + a_dim1;
	    a[i__2].r = 0.f, a[i__2].i = 0.f;
/* L60: */
	}
	if (*n > 1) {

/*           Generate Q(2:n,2:n) */

	    i__1 = *n - 1;
	    i__2 = *n - 1;
	    i__3 = *n - 1;
	    cungqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[1], 
		    &work[1], lwork, &iinfo);
	}
    }
    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNGTR */

} /* cungtr_ */
