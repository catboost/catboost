/* zlargv.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlargv_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublereal *c__, integer *incc)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, d__10;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    double log(doublereal), pow_di(doublereal *, integer *), d_imag(
	    doublecomplex *), sqrt(doublereal);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    doublereal d__;
    doublecomplex f, g;
    integer i__, j;
    doublecomplex r__;
    doublereal f2, g2;
    integer ic;
    doublereal di;
    doublecomplex ff;
    doublereal cs, dr;
    doublecomplex fs, gs;
    integer ix, iy;
    doublecomplex sn;
    doublereal f2s, g2s, eps, scale;
    integer count;
    doublereal safmn2;
    extern doublereal dlapy2_(doublereal *, doublereal *);
    doublereal safmx2;
    extern doublereal dlamch_(char *);
    doublereal safmin;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLARGV generates a vector of complex plane rotations with real */
/*  cosines, determined by elements of the complex vectors x and y. */
/*  For i = 1,2,...,n */

/*     (        c(i)   s(i) ) ( x(i) ) = ( r(i) ) */
/*     ( -conjg(s(i))  c(i) ) ( y(i) ) = (   0  ) */

/*     where c(i)**2 + ABS(s(i))**2 = 1 */

/*  The following conventions are used (these are the same as in ZLARTG, */
/*  but differ from the BLAS1 routine ZROTG): */
/*     If y(i)=0, then c(i)=1 and s(i)=0. */
/*     If x(i)=0, then c(i)=0 and s(i) is chosen so that r(i) is real. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of plane rotations to be generated. */

/*  X       (input/output) COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/*          On entry, the vector x. */
/*          On exit, x(i) is overwritten by r(i), for i = 1,...,n. */

/*  INCX    (input) INTEGER */
/*          The increment between elements of X. INCX > 0. */

/*  Y       (input/output) COMPLEX*16 array, dimension (1+(N-1)*INCY) */
/*          On entry, the vector y. */
/*          On exit, the sines of the plane rotations. */

/*  INCY    (input) INTEGER */
/*          The increment between elements of Y. INCY > 0. */

/*  C       (output) DOUBLE PRECISION array, dimension (1+(N-1)*INCC) */
/*          The cosines of the plane rotations. */

/*  INCC    (input) INTEGER */
/*          The increment between elements of C. INCC > 0. */

/*  Further Details */
/*  ======= ======= */

/*  6-6-96 - Modified with a new algorithm by W. Kahan and J. Demmel */

/*  This version has a few statements commented out for thread safety */
/*  (machine parameters are computed on each entry). 10 feb 03, SJH. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     LOGICAL            FIRST */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     SAVE               FIRST, SAFMX2, SAFMIN, SAFMN2 */
/*     .. */
/*     .. Data statements .. */
/*     DATA               FIRST / .TRUE. / */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     IF( FIRST ) THEN */
/*        FIRST = .FALSE. */
    /* Parameter adjustments */
    --c__;
    --y;
    --x;

    /* Function Body */
    safmin = dlamch_("S");
    eps = dlamch_("E");
    d__1 = dlamch_("B");
    i__1 = (integer) (log(safmin / eps) / log(dlamch_("B")) / 2.);
    safmn2 = pow_di(&d__1, &i__1);
    safmx2 = 1. / safmn2;
/*     END IF */
    ix = 1;
    iy = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	f.r = x[i__2].r, f.i = x[i__2].i;
	i__2 = iy;
	g.r = y[i__2].r, g.i = y[i__2].i;

/*        Use identical algorithm as in ZLARTG */

/* Computing MAX */
/* Computing MAX */
	d__7 = (d__1 = f.r, abs(d__1)), d__8 = (d__2 = d_imag(&f), abs(d__2));
/* Computing MAX */
	d__9 = (d__3 = g.r, abs(d__3)), d__10 = (d__4 = d_imag(&g), abs(d__4))
		;
	d__5 = max(d__7,d__8), d__6 = max(d__9,d__10);
	scale = max(d__5,d__6);
	fs.r = f.r, fs.i = f.i;
	gs.r = g.r, gs.i = g.i;
	count = 0;
	if (scale >= safmx2) {
L10:
	    ++count;
	    z__1.r = safmn2 * fs.r, z__1.i = safmn2 * fs.i;
	    fs.r = z__1.r, fs.i = z__1.i;
	    z__1.r = safmn2 * gs.r, z__1.i = safmn2 * gs.i;
	    gs.r = z__1.r, gs.i = z__1.i;
	    scale *= safmn2;
	    if (scale >= safmx2) {
		goto L10;
	    }
	} else if (scale <= safmn2) {
	    if (g.r == 0. && g.i == 0.) {
		cs = 1.;
		sn.r = 0., sn.i = 0.;
		r__.r = f.r, r__.i = f.i;
		goto L50;
	    }
L20:
	    --count;
	    z__1.r = safmx2 * fs.r, z__1.i = safmx2 * fs.i;
	    fs.r = z__1.r, fs.i = z__1.i;
	    z__1.r = safmx2 * gs.r, z__1.i = safmx2 * gs.i;
	    gs.r = z__1.r, gs.i = z__1.i;
	    scale *= safmx2;
	    if (scale <= safmn2) {
		goto L20;
	    }
	}
/* Computing 2nd power */
	d__1 = fs.r;
/* Computing 2nd power */
	d__2 = d_imag(&fs);
	f2 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
	d__1 = gs.r;
/* Computing 2nd power */
	d__2 = d_imag(&gs);
	g2 = d__1 * d__1 + d__2 * d__2;
	if (f2 <= max(g2,1.) * safmin) {

/*           This is a rare case: F is very small. */

	    if (f.r == 0. && f.i == 0.) {
		cs = 0.;
		d__2 = g.r;
		d__3 = d_imag(&g);
		d__1 = dlapy2_(&d__2, &d__3);
		r__.r = d__1, r__.i = 0.;
/*              Do complex/real division explicitly with two real */
/*              divisions */
		d__1 = gs.r;
		d__2 = d_imag(&gs);
		d__ = dlapy2_(&d__1, &d__2);
		d__1 = gs.r / d__;
		d__2 = -d_imag(&gs) / d__;
		z__1.r = d__1, z__1.i = d__2;
		sn.r = z__1.r, sn.i = z__1.i;
		goto L50;
	    }
	    d__1 = fs.r;
	    d__2 = d_imag(&fs);
	    f2s = dlapy2_(&d__1, &d__2);
/*           G2 and G2S are accurate */
/*           G2 is at least SAFMIN, and G2S is at least SAFMN2 */
	    g2s = sqrt(g2);
/*           Error in CS from underflow in F2S is at most */
/*           UNFL / SAFMN2 .lt. sqrt(UNFL*EPS) .lt. EPS */
/*           If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN, */
/*           and so CS .lt. sqrt(SAFMIN) */
/*           If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN */
/*           and so CS .lt. sqrt(SAFMIN)/SAFMN2 = sqrt(EPS) */
/*           Therefore, CS = F2S/G2S / sqrt( 1 + (F2S/G2S)**2 ) = F2S/G2S */
	    cs = f2s / g2s;
/*           Make sure abs(FF) = 1 */
/*           Do complex/real division explicitly with 2 real divisions */
/* Computing MAX */
	    d__3 = (d__1 = f.r, abs(d__1)), d__4 = (d__2 = d_imag(&f), abs(
		    d__2));
	    if (max(d__3,d__4) > 1.) {
		d__1 = f.r;
		d__2 = d_imag(&f);
		d__ = dlapy2_(&d__1, &d__2);
		d__1 = f.r / d__;
		d__2 = d_imag(&f) / d__;
		z__1.r = d__1, z__1.i = d__2;
		ff.r = z__1.r, ff.i = z__1.i;
	    } else {
		dr = safmx2 * f.r;
		di = safmx2 * d_imag(&f);
		d__ = dlapy2_(&dr, &di);
		d__1 = dr / d__;
		d__2 = di / d__;
		z__1.r = d__1, z__1.i = d__2;
		ff.r = z__1.r, ff.i = z__1.i;
	    }
	    d__1 = gs.r / g2s;
	    d__2 = -d_imag(&gs) / g2s;
	    z__2.r = d__1, z__2.i = d__2;
	    z__1.r = ff.r * z__2.r - ff.i * z__2.i, z__1.i = ff.r * z__2.i + 
		    ff.i * z__2.r;
	    sn.r = z__1.r, sn.i = z__1.i;
	    z__2.r = cs * f.r, z__2.i = cs * f.i;
	    z__3.r = sn.r * g.r - sn.i * g.i, z__3.i = sn.r * g.i + sn.i * 
		    g.r;
	    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	    r__.r = z__1.r, r__.i = z__1.i;
	} else {

/*           This is the most common case. */
/*           Neither F2 nor F2/G2 are less than SAFMIN */
/*           F2S cannot overflow, and it is accurate */

	    f2s = sqrt(g2 / f2 + 1.);
/*           Do the F2S(real)*FS(complex) multiply with two real */
/*           multiplies */
	    d__1 = f2s * fs.r;
	    d__2 = f2s * d_imag(&fs);
	    z__1.r = d__1, z__1.i = d__2;
	    r__.r = z__1.r, r__.i = z__1.i;
	    cs = 1. / f2s;
	    d__ = f2 + g2;
/*           Do complex/real division explicitly with two real divisions */
	    d__1 = r__.r / d__;
	    d__2 = d_imag(&r__) / d__;
	    z__1.r = d__1, z__1.i = d__2;
	    sn.r = z__1.r, sn.i = z__1.i;
	    d_cnjg(&z__2, &gs);
	    z__1.r = sn.r * z__2.r - sn.i * z__2.i, z__1.i = sn.r * z__2.i + 
		    sn.i * z__2.r;
	    sn.r = z__1.r, sn.i = z__1.i;
	    if (count != 0) {
		if (count > 0) {
		    i__2 = count;
		    for (j = 1; j <= i__2; ++j) {
			z__1.r = safmx2 * r__.r, z__1.i = safmx2 * r__.i;
			r__.r = z__1.r, r__.i = z__1.i;
/* L30: */
		    }
		} else {
		    i__2 = -count;
		    for (j = 1; j <= i__2; ++j) {
			z__1.r = safmn2 * r__.r, z__1.i = safmn2 * r__.i;
			r__.r = z__1.r, r__.i = z__1.i;
/* L40: */
		    }
		}
	    }
	}
L50:
	c__[ic] = cs;
	i__2 = iy;
	y[i__2].r = sn.r, y[i__2].i = sn.i;
	i__2 = ix;
	x[i__2].r = r__.r, x[i__2].i = r__.i;
	ic += *incc;
	iy += *incy;
	ix += *incx;
/* L60: */
    }
    return 0;

/*     End of ZLARGV */

} /* zlargv_ */
