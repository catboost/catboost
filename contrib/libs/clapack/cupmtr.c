/* cupmtr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cupmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, complex *ap, complex *tau, complex *c__, integer *ldc, 
	complex *work, integer *info)
{
    /* System generated locals */
    integer c_dim1, c_offset, i__1, i__2, i__3;
    complex q__1;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, i1, i2, i3, ic, jc, ii, mi, ni, nq;
    complex aii;
    logical left;
    complex taui;
    extern /* Subroutine */ int clarf_(char *, integer *, integer *, complex *
, integer *, complex *, complex *, integer *, complex *);
    extern logical lsame_(char *, char *);
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    logical notran, forwrd;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CUPMTR overwrites the general complex M-by-N matrix C with */

/*                  SIDE = 'L'     SIDE = 'R' */
/*  TRANS = 'N':      Q * C          C * Q */
/*  TRANS = 'C':      Q**H * C       C * Q**H */

/*  where Q is a complex unitary matrix of order nq, with nq = m if */
/*  SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of */
/*  nq-1 elementary reflectors, as returned by CHPTRD using packed */
/*  storage: */

/*  if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1); */

/*  if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1). */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'L': apply Q or Q**H from the Left; */
/*          = 'R': apply Q or Q**H from the Right. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U': Upper triangular packed storage used in previous */
/*                 call to CHPTRD; */
/*          = 'L': Lower triangular packed storage used in previous */
/*                 call to CHPTRD. */

/*  TRANS   (input) CHARACTER*1 */
/*          = 'N':  No transpose, apply Q; */
/*          = 'C':  Conjugate transpose, apply Q**H. */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix C. M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix C. N >= 0. */

/*  AP      (input) COMPLEX array, dimension */
/*                               (M*(M+1)/2) if SIDE = 'L' */
/*                               (N*(N+1)/2) if SIDE = 'R' */
/*          The vectors which define the elementary reflectors, as */
/*          returned by CHPTRD.  AP is modified by the routine but */
/*          restored on exit. */

/*  TAU     (input) COMPLEX array, dimension (M-1) if SIDE = 'L' */
/*                                     or (N-1) if SIDE = 'R' */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i), as returned by CHPTRD. */

/*  C       (input/output) COMPLEX array, dimension (LDC,N) */
/*          On entry, the M-by-N matrix C. */
/*          On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the array C. LDC >= max(1,M). */

/*  WORK    (workspace) COMPLEX array, dimension */
/*                                   (N) if SIDE = 'L' */
/*                                   (M) if SIDE = 'R' */

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
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    upper = lsame_(uplo, "U");

/*     NQ is the order of Q */

    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (! notran && ! lsame_(trans, "C")) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*ldc < max(1,*m)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CUPMTR", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

    if (upper) {

/*        Q was determined by a call to CHPTRD with UPLO = 'U' */

	forwrd = left && notran || ! left && ! notran;

	if (forwrd) {
	    i1 = 1;
	    i2 = nq - 1;
	    i3 = 1;
	    ii = 2;
	} else {
	    i1 = nq - 1;
	    i2 = 1;
	    i3 = -1;
	    ii = nq * (nq + 1) / 2 - 1;
	}

	if (left) {
	    ni = *n;
	} else {
	    mi = *m;
	}

	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    if (left) {

/*              H(i) or H(i)' is applied to C(1:i,1:n) */

		mi = i__;
	    } else {

/*              H(i) or H(i)' is applied to C(1:m,1:i) */

		ni = i__;
	    }

/*           Apply H(i) or H(i)' */

	    if (notran) {
		i__3 = i__;
		taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	    } else {
		r_cnjg(&q__1, &tau[i__]);
		taui.r = q__1.r, taui.i = q__1.i;
	    }
	    i__3 = ii;
	    aii.r = ap[i__3].r, aii.i = ap[i__3].i;
	    i__3 = ii;
	    ap[i__3].r = 1.f, ap[i__3].i = 0.f;
	    clarf_(side, &mi, &ni, &ap[ii - i__ + 1], &c__1, &taui, &c__[
		    c_offset], ldc, &work[1]);
	    i__3 = ii;
	    ap[i__3].r = aii.r, ap[i__3].i = aii.i;

	    if (forwrd) {
		ii = ii + i__ + 2;
	    } else {
		ii = ii - i__ - 1;
	    }
/* L10: */
	}
    } else {

/*        Q was determined by a call to CHPTRD with UPLO = 'L'. */

	forwrd = left && ! notran || ! left && notran;

	if (forwrd) {
	    i1 = 1;
	    i2 = nq - 1;
	    i3 = 1;
	    ii = 2;
	} else {
	    i1 = nq - 1;
	    i2 = 1;
	    i3 = -1;
	    ii = nq * (nq + 1) / 2 - 1;
	}

	if (left) {
	    ni = *n;
	    jc = 1;
	} else {
	    mi = *m;
	    ic = 1;
	}

	i__2 = i2;
	i__1 = i3;
	for (i__ = i1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
	    i__3 = ii;
	    aii.r = ap[i__3].r, aii.i = ap[i__3].i;
	    i__3 = ii;
	    ap[i__3].r = 1.f, ap[i__3].i = 0.f;
	    if (left) {

/*              H(i) or H(i)' is applied to C(i+1:m,1:n) */

		mi = *m - i__;
		ic = i__ + 1;
	    } else {

/*              H(i) or H(i)' is applied to C(1:m,i+1:n) */

		ni = *n - i__;
		jc = i__ + 1;
	    }

/*           Apply H(i) or H(i)' */

	    if (notran) {
		i__3 = i__;
		taui.r = tau[i__3].r, taui.i = tau[i__3].i;
	    } else {
		r_cnjg(&q__1, &tau[i__]);
		taui.r = q__1.r, taui.i = q__1.i;
	    }
	    clarf_(side, &mi, &ni, &ap[ii], &c__1, &taui, &c__[ic + jc * 
		    c_dim1], ldc, &work[1]);
	    i__3 = ii;
	    ap[i__3].r = aii.r, ap[i__3].i = aii.i;

	    if (forwrd) {
		ii = ii + nq - i__ + 1;
	    } else {
		ii = ii - nq + i__ - 2;
	    }
/* L20: */
	}
    }
    return 0;

/*     End of CUPMTR */

} /* cupmtr_ */
