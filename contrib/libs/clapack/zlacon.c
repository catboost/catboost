/* zlacon.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlacon_(integer *n, doublecomplex *v, doublecomplex *x, 
	doublereal *est, integer *kase)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Builtin functions */
    double z_abs(doublecomplex *), d_imag(doublecomplex *);

    /* Local variables */
    static integer i__, j, iter;
    static doublereal temp;
    static integer jump;
    static doublereal absxi;
    static integer jlast;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *);
    extern integer izmax1_(integer *, doublecomplex *, integer *);
    extern doublereal dzsum1_(integer *, doublecomplex *, integer *), dlamch_(
	    char *);
    static doublereal safmin, altsgn, estold;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLACON estimates the 1-norm of a square, complex matrix A. */
/*  Reverse communication is used for evaluating matrix-vector products. */

/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         The order of the matrix.  N >= 1. */

/*  V      (workspace) COMPLEX*16 array, dimension (N) */
/*         On the final return, V = A*W,  where  EST = norm(V)/norm(W) */
/*         (W is not returned). */

/*  X      (input/output) COMPLEX*16 array, dimension (N) */
/*         On an intermediate return, X should be overwritten by */
/*               A * X,   if KASE=1, */
/*               A' * X,  if KASE=2, */
/*         where A' is the conjugate transpose of A, and ZLACON must be */
/*         re-called with all the other parameters unchanged. */

/*  EST    (input/output) DOUBLE PRECISION */
/*         On entry with KASE = 1 or 2 and JUMP = 3, EST should be */
/*         unchanged from the previous call to ZLACON. */
/*         On exit, EST is an estimate (a lower bound) for norm(A). */

/*  KASE   (input/output) INTEGER */
/*         On the initial call to ZLACON, KASE should be 0. */
/*         On an intermediate return, KASE will be 1 or 2, indicating */
/*         whether X should be overwritten by A * X  or A' * X. */
/*         On the final return from ZLACON, KASE will again be 0. */

/*  Further Details */
/*  ======= ======= */

/*  Contributed by Nick Higham, University of Manchester. */
/*  Originally named CONEST, dated March 16, 1988. */

/*  Reference: N.J. Higham, "FORTRAN codes for estimating the one-norm of */
/*  a real or complex matrix, with applications to condition estimation", */
/*  ACM Trans. Math. Soft., vol. 14, no. 4, pp. 381-396, December 1988. */

/*  Last modified:  April, 1999 */

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
/*     .. Save statement .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;
    --v;

    /* Function Body */
    safmin = dlamch_("Safe minimum");
    if (*kase == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = i__;
	    d__1 = 1. / (doublereal) (*n);
	    z__1.r = d__1, z__1.i = 0.;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L10: */
	}
	*kase = 1;
	jump = 1;
	return 0;
    }

    switch (jump) {
	case 1:  goto L20;
	case 2:  goto L40;
	case 3:  goto L70;
	case 4:  goto L90;
	case 5:  goto L120;
    }

/*     ................ ENTRY   (JUMP = 1) */
/*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X. */

L20:
    if (*n == 1) {
	v[1].r = x[1].r, v[1].i = x[1].i;
	*est = z_abs(&v[1]);
/*        ... QUIT */
	goto L130;
    }
    *est = dzsum1_(n, &x[1], &c__1);

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	absxi = z_abs(&x[i__]);
	if (absxi > safmin) {
	    i__2 = i__;
	    i__3 = i__;
	    d__1 = x[i__3].r / absxi;
	    d__2 = d_imag(&x[i__]) / absxi;
	    z__1.r = d__1, z__1.i = d__2;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	} else {
	    i__2 = i__;
	    x[i__2].r = 1., x[i__2].i = 0.;
	}
/* L30: */
    }
    *kase = 2;
    jump = 2;
    return 0;

/*     ................ ENTRY   (JUMP = 2) */
/*     FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L40:
    j = izmax1_(n, &x[1], &c__1);
    iter = 2;

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */

L50:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	x[i__2].r = 0., x[i__2].i = 0.;
/* L60: */
    }
    i__1 = j;
    x[i__1].r = 1., x[i__1].i = 0.;
    *kase = 1;
    jump = 3;
    return 0;

/*     ................ ENTRY   (JUMP = 3) */
/*     X HAS BEEN OVERWRITTEN BY A*X. */

L70:
    zcopy_(n, &x[1], &c__1, &v[1], &c__1);
    estold = *est;
    *est = dzsum1_(n, &v[1], &c__1);

/*     TEST FOR CYCLING. */
    if (*est <= estold) {
	goto L100;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	absxi = z_abs(&x[i__]);
	if (absxi > safmin) {
	    i__2 = i__;
	    i__3 = i__;
	    d__1 = x[i__3].r / absxi;
	    d__2 = d_imag(&x[i__]) / absxi;
	    z__1.r = d__1, z__1.i = d__2;
	    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	} else {
	    i__2 = i__;
	    x[i__2].r = 1., x[i__2].i = 0.;
	}
/* L80: */
    }
    *kase = 2;
    jump = 4;
    return 0;

/*     ................ ENTRY   (JUMP = 4) */
/*     X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L90:
    jlast = j;
    j = izmax1_(n, &x[1], &c__1);
    if (z_abs(&x[jlast]) != z_abs(&x[j]) && iter < 5) {
	++iter;
	goto L50;
    }

/*     ITERATION COMPLETE.  FINAL STAGE. */

L100:
    altsgn = 1.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__;
	d__1 = altsgn * ((doublereal) (i__ - 1) / (doublereal) (*n - 1) + 1.);
	z__1.r = d__1, z__1.i = 0.;
	x[i__2].r = z__1.r, x[i__2].i = z__1.i;
	altsgn = -altsgn;
/* L110: */
    }
    *kase = 1;
    jump = 5;
    return 0;

/*     ................ ENTRY   (JUMP = 5) */
/*     X HAS BEEN OVERWRITTEN BY A*X. */

L120:
    temp = dzsum1_(n, &x[1], &c__1) / (doublereal) (*n * 3) * 2.;
    if (temp > *est) {
	zcopy_(n, &x[1], &c__1, &v[1], &c__1);
	*est = temp;
    }

L130:
    *kase = 0;
    return 0;

/*     End of ZLACON */

} /* zlacon_ */
