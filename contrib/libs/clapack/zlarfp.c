/* zlarfp.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b5 = {1.,0.};

/* Subroutine */ int zlarfp_(integer *n, doublecomplex *alpha, doublecomplex *
	x, integer *incx, doublecomplex *tau)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double d_imag(doublecomplex *), d_sign(doublereal *, doublereal *);

    /* Local variables */
    integer j, knt;
    doublereal beta, alphi, alphr;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *);
    doublereal xnorm;
    extern doublereal dlapy2_(doublereal *, doublereal *), dlapy3_(doublereal 
	    *, doublereal *, doublereal *), dznrm2_(integer *, doublecomplex *
, integer *), dlamch_(char *);
    doublereal safmin;
    extern /* Subroutine */ int zdscal_(integer *, doublereal *, 
	    doublecomplex *, integer *);
    doublereal rsafmn;
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *, 
	     doublecomplex *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLARFP generates a complex elementary reflector H of order n, such */
/*  that */

/*        H' * ( alpha ) = ( beta ),   H' * H = I. */
/*             (   x   )   (   0  ) */

/*  where alpha and beta are scalars, beta is real and non-negative, and */
/*  x is an (n-1)-element complex vector.  H is represented in the form */

/*        H = I - tau * ( 1 ) * ( 1 v' ) , */
/*                      ( v ) */

/*  where tau is a complex scalar and v is a complex (n-1)-element */
/*  vector. Note that H is not hermitian. */

/*  If the elements of x are all zero and alpha is real, then tau = 0 */
/*  and H is taken to be the unit matrix. */

/*  Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 . */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the elementary reflector. */

/*  ALPHA   (input/output) COMPLEX*16 */
/*          On entry, the value alpha. */
/*          On exit, it is overwritten with the value beta. */

/*  X       (input/output) COMPLEX*16 array, dimension */
/*                         (1+(N-2)*abs(INCX)) */
/*          On entry, the vector x. */
/*          On exit, it is overwritten with the vector v. */

/*  INCX    (input) INTEGER */
/*          The increment between elements of X. INCX > 0. */

/*  TAU     (output) COMPLEX*16 */
/*          The value tau. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 0) {
	tau->r = 0., tau->i = 0.;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = dznrm2_(&i__1, &x[1], incx);
    alphr = alpha->r;
    alphi = d_imag(alpha);

    if (xnorm == 0. && alphi == 0.) {

/*        H  =  [1-alpha/abs(alpha) 0; 0 I], sign chosen so ALPHA >= 0. */

	if (alphi == 0.) {
	    if (alphr >= 0.) {
/*              When TAU.eq.ZERO, the vector is special-cased to be */
/*              all zeros in the application routines.  We do not need */
/*              to clear it. */
		tau->r = 0., tau->i = 0.;
	    } else {
/*              However, the application routines rely on explicit */
/*              zero checks when TAU.ne.ZERO, and we must clear X. */
		tau->r = 2., tau->i = 0.;
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = (j - 1) * *incx + 1;
		    x[i__2].r = 0., x[i__2].i = 0.;
		}
		z__1.r = -alpha->r, z__1.i = -alpha->i;
		alpha->r = z__1.r, alpha->i = z__1.i;
	    }
	} else {
/*           Only "reflecting" the diagonal entry to be real and non-negative. */
	    xnorm = dlapy2_(&alphr, &alphi);
	    d__1 = 1. - alphr / xnorm;
	    d__2 = -alphi / xnorm;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = (j - 1) * *incx + 1;
		x[i__2].r = 0., x[i__2].i = 0.;
	    }
	    alpha->r = xnorm, alpha->i = 0.;
	}
    } else {

/*        general case */

	d__1 = dlapy3_(&alphr, &alphi, &xnorm);
	beta = d_sign(&d__1, &alphr);
	safmin = dlamch_("S") / dlamch_("E");
	rsafmn = 1. / safmin;

	knt = 0;
	if (abs(beta) < safmin) {

/*           XNORM, BETA may be inaccurate; scale X and recompute them */

L10:
	    ++knt;
	    i__1 = *n - 1;
	    zdscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    alphi *= rsafmn;
	    alphr *= rsafmn;
	    if (abs(beta) < safmin) {
		goto L10;
	    }

/*           New BETA is at most 1, at least SAFMIN */

	    i__1 = *n - 1;
	    xnorm = dznrm2_(&i__1, &x[1], incx);
	    z__1.r = alphr, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	    d__1 = dlapy3_(&alphr, &alphi, &xnorm);
	    beta = d_sign(&d__1, &alphr);
	}
	z__1.r = alpha->r + beta, z__1.i = alpha->i;
	alpha->r = z__1.r, alpha->i = z__1.i;
	if (beta < 0.) {
	    beta = -beta;
	    z__2.r = -alpha->r, z__2.i = -alpha->i;
	    z__1.r = z__2.r / beta, z__1.i = z__2.i / beta;
	    tau->r = z__1.r, tau->i = z__1.i;
	} else {
	    alphr = alphi * (alphi / alpha->r);
	    alphr += xnorm * (xnorm / alpha->r);
	    d__1 = alphr / beta;
	    d__2 = -alphi / beta;
	    z__1.r = d__1, z__1.i = d__2;
	    tau->r = z__1.r, tau->i = z__1.i;
	    d__1 = -alphr;
	    z__1.r = d__1, z__1.i = alphi;
	    alpha->r = z__1.r, alpha->i = z__1.i;
	}
	zladiv_(&z__1, &c_b5, alpha);
	alpha->r = z__1.r, alpha->i = z__1.i;
	i__1 = *n - 1;
	zscal_(&i__1, alpha, &x[1], incx);

/*        If BETA is subnormal, it may lose relative accuracy */

	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= safmin;
/* L20: */
	}
	alpha->r = beta, alpha->i = 0.;
    }

    return 0;

/*     End of ZLARFP */

} /* zlarfp_ */
