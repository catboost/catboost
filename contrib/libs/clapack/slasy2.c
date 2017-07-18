/* slasy2.f -- translated by f2c (version 20061008).
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

static integer c__4 = 4;
static integer c__1 = 1;
static integer c__16 = 16;
static integer c__0 = 0;

/* Subroutine */ int slasy2_(logical *ltranl, logical *ltranr, integer *isgn, 
	integer *n1, integer *n2, real *tl, integer *ldtl, real *tr, integer *
	ldtr, real *b, integer *ldb, real *scale, real *x, integer *ldx, real 
	*xnorm, integer *info)
{
    /* Initialized data */

    static integer locu12[4] = { 3,4,1,2 };
    static integer locl21[4] = { 2,1,4,3 };
    static integer locu22[4] = { 4,3,2,1 };
    static logical xswpiv[4] = { FALSE_,FALSE_,TRUE_,TRUE_ };
    static logical bswpiv[4] = { FALSE_,TRUE_,FALSE_,TRUE_ };

    /* System generated locals */
    integer b_dim1, b_offset, tl_dim1, tl_offset, tr_dim1, tr_offset, x_dim1, 
	    x_offset;
    real r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8;

    /* Local variables */
    integer i__, j, k;
    real x2[2], l21, u11, u12;
    integer ip, jp;
    real u22, t16[16]	/* was [4][4] */, gam, bet, eps, sgn, tmp[4], tau1, 
	    btmp[4], smin;
    integer ipiv;
    real temp;
    integer jpiv[4];
    real xmax;
    integer ipsv, jpsv;
    logical bswap;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *), sswap_(integer *, real *, integer *, real *, integer *
);
    logical xswap;
    extern doublereal slamch_(char *);
    extern integer isamax_(integer *, real *, integer *);
    real smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLASY2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in */

/*         op(TL)*X + ISGN*X*op(TR) = SCALE*B, */

/*  where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or */
/*  -1.  op(T) = T or T', where T' denotes the transpose of T. */

/*  Arguments */
/*  ========= */

/*  LTRANL  (input) LOGICAL */
/*          On entry, LTRANL specifies the op(TL): */
/*             = .FALSE., op(TL) = TL, */
/*             = .TRUE., op(TL) = TL'. */

/*  LTRANR  (input) LOGICAL */
/*          On entry, LTRANR specifies the op(TR): */
/*            = .FALSE., op(TR) = TR, */
/*            = .TRUE., op(TR) = TR'. */

/*  ISGN    (input) INTEGER */
/*          On entry, ISGN specifies the sign of the equation */
/*          as described before. ISGN may only be 1 or -1. */

/*  N1      (input) INTEGER */
/*          On entry, N1 specifies the order of matrix TL. */
/*          N1 may only be 0, 1 or 2. */

/*  N2      (input) INTEGER */
/*          On entry, N2 specifies the order of matrix TR. */
/*          N2 may only be 0, 1 or 2. */

/*  TL      (input) REAL array, dimension (LDTL,2) */
/*          On entry, TL contains an N1 by N1 matrix. */

/*  LDTL    (input) INTEGER */
/*          The leading dimension of the matrix TL. LDTL >= max(1,N1). */

/*  TR      (input) REAL array, dimension (LDTR,2) */
/*          On entry, TR contains an N2 by N2 matrix. */

/*  LDTR    (input) INTEGER */
/*          The leading dimension of the matrix TR. LDTR >= max(1,N2). */

/*  B       (input) REAL array, dimension (LDB,2) */
/*          On entry, the N1 by N2 matrix B contains the right-hand */
/*          side of the equation. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the matrix B. LDB >= max(1,N1). */

/*  SCALE   (output) REAL */
/*          On exit, SCALE contains the scale factor. SCALE is chosen */
/*          less than or equal to 1 to prevent the solution overflowing. */

/*  X       (output) REAL array, dimension (LDX,2) */
/*          On exit, X contains the N1 by N2 solution. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the matrix X. LDX >= max(1,N1). */

/*  XNORM   (output) REAL */
/*          On exit, XNORM is the infinity-norm of the solution. */

/*  INFO    (output) INTEGER */
/*          On exit, INFO is set to */
/*             0: successful exit. */
/*             1: TL and TR have too close eigenvalues, so TL or */
/*                TR is perturbed to get a nonsingular equation. */
/*          NOTE: In the interests of speed, this routine does not */
/*                check the inputs for errors. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Data statements .. */
    /* Parameter adjustments */
    tl_dim1 = *ldtl;
    tl_offset = 1 + tl_dim1;
    tl -= tl_offset;
    tr_dim1 = *ldtr;
    tr_offset = 1 + tr_dim1;
    tr -= tr_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;

    /* Function Body */
/*     .. */
/*     .. Executable Statements .. */

/*     Do not check the input parameters for errors */

    *info = 0;

/*     Quick return if possible */

    if (*n1 == 0 || *n2 == 0) {
	return 0;
    }

/*     Set constants to control overflow */

    eps = slamch_("P");
    smlnum = slamch_("S") / eps;
    sgn = (real) (*isgn);

    k = *n1 + *n1 + *n2 - 2;
    switch (k) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L50;
    }

/*     1 by 1: TL11*X + SGN*X*TR11 = B11 */

L10:
    tau1 = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    bet = dabs(tau1);
    if (bet <= smlnum) {
	tau1 = smlnum;
	bet = smlnum;
	*info = 1;
    }

    *scale = 1.f;
    gam = (r__1 = b[b_dim1 + 1], dabs(r__1));
    if (smlnum * gam > bet) {
	*scale = 1.f / gam;
    }

    x[x_dim1 + 1] = b[b_dim1 + 1] * *scale / tau1;
    *xnorm = (r__1 = x[x_dim1 + 1], dabs(r__1));
    return 0;

/*     1 by 2: */
/*     TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12]  = [B11 B12] */
/*                                       [TR21 TR22] */

L20:

/* Computing MAX */
/* Computing MAX */
    r__7 = (r__1 = tl[tl_dim1 + 1], dabs(r__1)), r__8 = (r__2 = tr[tr_dim1 + 
	    1], dabs(r__2)), r__7 = max(r__7,r__8), r__8 = (r__3 = tr[(
	    tr_dim1 << 1) + 1], dabs(r__3)), r__7 = max(r__7,r__8), r__8 = (
	    r__4 = tr[tr_dim1 + 2], dabs(r__4)), r__7 = max(r__7,r__8), r__8 =
	     (r__5 = tr[(tr_dim1 << 1) + 2], dabs(r__5));
    r__6 = eps * dmax(r__7,r__8);
    smin = dmax(r__6,smlnum);
    tmp[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    tmp[3] = tl[tl_dim1 + 1] + sgn * tr[(tr_dim1 << 1) + 2];
    if (*ltranr) {
	tmp[1] = sgn * tr[tr_dim1 + 2];
	tmp[2] = sgn * tr[(tr_dim1 << 1) + 1];
    } else {
	tmp[1] = sgn * tr[(tr_dim1 << 1) + 1];
	tmp[2] = sgn * tr[tr_dim1 + 2];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[(b_dim1 << 1) + 1];
    goto L40;

/*     2 by 1: */
/*          op[TL11 TL12]*[X11] + ISGN* [X11]*TR11  = [B11] */
/*            [TL21 TL22] [X21]         [X21]         [B21] */

L30:
/* Computing MAX */
/* Computing MAX */
    r__7 = (r__1 = tr[tr_dim1 + 1], dabs(r__1)), r__8 = (r__2 = tl[tl_dim1 + 
	    1], dabs(r__2)), r__7 = max(r__7,r__8), r__8 = (r__3 = tl[(
	    tl_dim1 << 1) + 1], dabs(r__3)), r__7 = max(r__7,r__8), r__8 = (
	    r__4 = tl[tl_dim1 + 2], dabs(r__4)), r__7 = max(r__7,r__8), r__8 =
	     (r__5 = tl[(tl_dim1 << 1) + 2], dabs(r__5));
    r__6 = eps * dmax(r__7,r__8);
    smin = dmax(r__6,smlnum);
    tmp[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    tmp[3] = tl[(tl_dim1 << 1) + 2] + sgn * tr[tr_dim1 + 1];
    if (*ltranl) {
	tmp[1] = tl[(tl_dim1 << 1) + 1];
	tmp[2] = tl[tl_dim1 + 2];
    } else {
	tmp[1] = tl[tl_dim1 + 2];
	tmp[2] = tl[(tl_dim1 << 1) + 1];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[b_dim1 + 2];
L40:

/*     Solve 2 by 2 system using complete pivoting. */
/*     Set pivots less than SMIN to SMIN. */

    ipiv = isamax_(&c__4, tmp, &c__1);
    u11 = tmp[ipiv - 1];
    if (dabs(u11) <= smin) {
	*info = 1;
	u11 = smin;
    }
    u12 = tmp[locu12[ipiv - 1] - 1];
    l21 = tmp[locl21[ipiv - 1] - 1] / u11;
    u22 = tmp[locu22[ipiv - 1] - 1] - u12 * l21;
    xswap = xswpiv[ipiv - 1];
    bswap = bswpiv[ipiv - 1];
    if (dabs(u22) <= smin) {
	*info = 1;
	u22 = smin;
    }
    if (bswap) {
	temp = btmp[1];
	btmp[1] = btmp[0] - l21 * temp;
	btmp[0] = temp;
    } else {
	btmp[1] -= l21 * btmp[0];
    }
    *scale = 1.f;
    if (smlnum * 2.f * dabs(btmp[1]) > dabs(u22) || smlnum * 2.f * dabs(btmp[
	    0]) > dabs(u11)) {
/* Computing MAX */
	r__1 = dabs(btmp[0]), r__2 = dabs(btmp[1]);
	*scale = .5f / dmax(r__1,r__2);
	btmp[0] *= *scale;
	btmp[1] *= *scale;
    }
    x2[1] = btmp[1] / u22;
    x2[0] = btmp[0] / u11 - u12 / u11 * x2[1];
    if (xswap) {
	temp = x2[1];
	x2[1] = x2[0];
	x2[0] = temp;
    }
    x[x_dim1 + 1] = x2[0];
    if (*n1 == 1) {
	x[(x_dim1 << 1) + 1] = x2[1];
	*xnorm = (r__1 = x[x_dim1 + 1], dabs(r__1)) + (r__2 = x[(x_dim1 << 1) 
		+ 1], dabs(r__2));
    } else {
	x[x_dim1 + 2] = x2[1];
/* Computing MAX */
	r__3 = (r__1 = x[x_dim1 + 1], dabs(r__1)), r__4 = (r__2 = x[x_dim1 + 
		2], dabs(r__2));
	*xnorm = dmax(r__3,r__4);
    }
    return 0;

/*     2 by 2: */
/*     op[TL11 TL12]*[X11 X12] +ISGN* [X11 X12]*op[TR11 TR12] = [B11 B12] */
/*       [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22] */

/*     Solve equivalent 4 by 4 system using complete pivoting. */
/*     Set pivots less than SMIN to SMIN. */

L50:
/* Computing MAX */
    r__5 = (r__1 = tr[tr_dim1 + 1], dabs(r__1)), r__6 = (r__2 = tr[(tr_dim1 <<
	     1) + 1], dabs(r__2)), r__5 = max(r__5,r__6), r__6 = (r__3 = tr[
	    tr_dim1 + 2], dabs(r__3)), r__5 = max(r__5,r__6), r__6 = (r__4 = 
	    tr[(tr_dim1 << 1) + 2], dabs(r__4));
    smin = dmax(r__5,r__6);
/* Computing MAX */
    r__5 = smin, r__6 = (r__1 = tl[tl_dim1 + 1], dabs(r__1)), r__5 = max(r__5,
	    r__6), r__6 = (r__2 = tl[(tl_dim1 << 1) + 1], dabs(r__2)), r__5 = 
	    max(r__5,r__6), r__6 = (r__3 = tl[tl_dim1 + 2], dabs(r__3)), r__5 
	    = max(r__5,r__6), r__6 = (r__4 = tl[(tl_dim1 << 1) + 2], dabs(
	    r__4));
    smin = dmax(r__5,r__6);
/* Computing MAX */
    r__1 = eps * smin;
    smin = dmax(r__1,smlnum);
    btmp[0] = 0.f;
    scopy_(&c__16, btmp, &c__0, t16, &c__1);
    t16[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    t16[5] = tl[(tl_dim1 << 1) + 2] + sgn * tr[tr_dim1 + 1];
    t16[10] = tl[tl_dim1 + 1] + sgn * tr[(tr_dim1 << 1) + 2];
    t16[15] = tl[(tl_dim1 << 1) + 2] + sgn * tr[(tr_dim1 << 1) + 2];
    if (*ltranl) {
	t16[4] = tl[tl_dim1 + 2];
	t16[1] = tl[(tl_dim1 << 1) + 1];
	t16[14] = tl[tl_dim1 + 2];
	t16[11] = tl[(tl_dim1 << 1) + 1];
    } else {
	t16[4] = tl[(tl_dim1 << 1) + 1];
	t16[1] = tl[tl_dim1 + 2];
	t16[14] = tl[(tl_dim1 << 1) + 1];
	t16[11] = tl[tl_dim1 + 2];
    }
    if (*ltranr) {
	t16[8] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[13] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[2] = sgn * tr[tr_dim1 + 2];
	t16[7] = sgn * tr[tr_dim1 + 2];
    } else {
	t16[8] = sgn * tr[tr_dim1 + 2];
	t16[13] = sgn * tr[tr_dim1 + 2];
	t16[2] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[7] = sgn * tr[(tr_dim1 << 1) + 1];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[b_dim1 + 2];
    btmp[2] = b[(b_dim1 << 1) + 1];
    btmp[3] = b[(b_dim1 << 1) + 2];

/*     Perform elimination */

    for (i__ = 1; i__ <= 3; ++i__) {
	xmax = 0.f;
	for (ip = i__; ip <= 4; ++ip) {
	    for (jp = i__; jp <= 4; ++jp) {
		if ((r__1 = t16[ip + (jp << 2) - 5], dabs(r__1)) >= xmax) {
		    xmax = (r__1 = t16[ip + (jp << 2) - 5], dabs(r__1));
		    ipsv = ip;
		    jpsv = jp;
		}
/* L60: */
	    }
/* L70: */
	}
	if (ipsv != i__) {
	    sswap_(&c__4, &t16[ipsv - 1], &c__4, &t16[i__ - 1], &c__4);
	    temp = btmp[i__ - 1];
	    btmp[i__ - 1] = btmp[ipsv - 1];
	    btmp[ipsv - 1] = temp;
	}
	if (jpsv != i__) {
	    sswap_(&c__4, &t16[(jpsv << 2) - 4], &c__1, &t16[(i__ << 2) - 4], 
		    &c__1);
	}
	jpiv[i__ - 1] = jpsv;
	if ((r__1 = t16[i__ + (i__ << 2) - 5], dabs(r__1)) < smin) {
	    *info = 1;
	    t16[i__ + (i__ << 2) - 5] = smin;
	}
	for (j = i__ + 1; j <= 4; ++j) {
	    t16[j + (i__ << 2) - 5] /= t16[i__ + (i__ << 2) - 5];
	    btmp[j - 1] -= t16[j + (i__ << 2) - 5] * btmp[i__ - 1];
	    for (k = i__ + 1; k <= 4; ++k) {
		t16[j + (k << 2) - 5] -= t16[j + (i__ << 2) - 5] * t16[i__ + (
			k << 2) - 5];
/* L80: */
	    }
/* L90: */
	}
/* L100: */
    }
    if (dabs(t16[15]) < smin) {
	t16[15] = smin;
    }
    *scale = 1.f;
    if (smlnum * 8.f * dabs(btmp[0]) > dabs(t16[0]) || smlnum * 8.f * dabs(
	    btmp[1]) > dabs(t16[5]) || smlnum * 8.f * dabs(btmp[2]) > dabs(
	    t16[10]) || smlnum * 8.f * dabs(btmp[3]) > dabs(t16[15])) {
/* Computing MAX */
	r__1 = dabs(btmp[0]), r__2 = dabs(btmp[1]), r__1 = max(r__1,r__2), 
		r__2 = dabs(btmp[2]), r__1 = max(r__1,r__2), r__2 = dabs(btmp[
		3]);
	*scale = .125f / dmax(r__1,r__2);
	btmp[0] *= *scale;
	btmp[1] *= *scale;
	btmp[2] *= *scale;
	btmp[3] *= *scale;
    }
    for (i__ = 1; i__ <= 4; ++i__) {
	k = 5 - i__;
	temp = 1.f / t16[k + (k << 2) - 5];
	tmp[k - 1] = btmp[k - 1] * temp;
	for (j = k + 1; j <= 4; ++j) {
	    tmp[k - 1] -= temp * t16[k + (j << 2) - 5] * tmp[j - 1];
/* L110: */
	}
/* L120: */
    }
    for (i__ = 1; i__ <= 3; ++i__) {
	if (jpiv[4 - i__ - 1] != 4 - i__) {
	    temp = tmp[4 - i__ - 1];
	    tmp[4 - i__ - 1] = tmp[jpiv[4 - i__ - 1] - 1];
	    tmp[jpiv[4 - i__ - 1] - 1] = temp;
	}
/* L130: */
    }
    x[x_dim1 + 1] = tmp[0];
    x[x_dim1 + 2] = tmp[1];
    x[(x_dim1 << 1) + 1] = tmp[2];
    x[(x_dim1 << 1) + 2] = tmp[3];
/* Computing MAX */
    r__1 = dabs(tmp[0]) + dabs(tmp[2]), r__2 = dabs(tmp[1]) + dabs(tmp[3]);
    *xnorm = dmax(r__1,r__2);
    return 0;

/*     End of SLASY2 */

} /* slasy2_ */
