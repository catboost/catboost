/* dlanv2.f -- translated by f2c (version 20061008).
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

static doublereal c_b4 = 1.;

/* Subroutine */ int dlanv2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *d__, doublereal *rt1r, doublereal *rt1i, doublereal *rt2r, 
	 doublereal *rt2i, doublereal *cs, doublereal *sn)
{
    /* System generated locals */
    doublereal d__1, d__2;

    /* Builtin functions */
    double d_sign(doublereal *, doublereal *), sqrt(doublereal);

    /* Local variables */
    doublereal p, z__, aa, bb, cc, dd, cs1, sn1, sab, sac, eps, tau, temp, 
	    scale, bcmax, bcmis, sigma;
    extern doublereal dlapy2_(doublereal *, doublereal *), dlamch_(char *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric */
/*  matrix in standard form: */

/*       [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ] */
/*       [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ] */

/*  where either */
/*  1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or */
/*  2) AA = DD and BB*CC < 0, so that AA + or - sqrt(BB*CC) are complex */
/*  conjugate eigenvalues. */

/*  Arguments */
/*  ========= */

/*  A       (input/output) DOUBLE PRECISION */
/*  B       (input/output) DOUBLE PRECISION */
/*  C       (input/output) DOUBLE PRECISION */
/*  D       (input/output) DOUBLE PRECISION */
/*          On entry, the elements of the input matrix. */
/*          On exit, they are overwritten by the elements of the */
/*          standardised Schur form. */

/*  RT1R    (output) DOUBLE PRECISION */
/*  RT1I    (output) DOUBLE PRECISION */
/*  RT2R    (output) DOUBLE PRECISION */
/*  RT2I    (output) DOUBLE PRECISION */
/*          The real and imaginary parts of the eigenvalues. If the */
/*          eigenvalues are a complex conjugate pair, RT1I > 0. */

/*  CS      (output) DOUBLE PRECISION */
/*  SN      (output) DOUBLE PRECISION */
/*          Parameters of the rotation matrix. */

/*  Further Details */
/*  =============== */

/*  Modified by V. Sima, Research Institute for Informatics, Bucharest, */
/*  Romania, to reduce the risk of cancellation errors, */
/*  when computing real eigenvalues, and to ensure, if possible, that */
/*  abs(RT1R) >= abs(RT2R). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    eps = dlamch_("P");
    if (*c__ == 0.) {
	*cs = 1.;
	*sn = 0.;
	goto L10;

    } else if (*b == 0.) {

/*        Swap rows and columns */

	*cs = 0.;
	*sn = 1.;
	temp = *d__;
	*d__ = *a;
	*a = temp;
	*b = -(*c__);
	*c__ = 0.;
	goto L10;
    } else if (*a - *d__ == 0. && d_sign(&c_b4, b) != d_sign(&c_b4, c__)) {
	*cs = 1.;
	*sn = 0.;
	goto L10;
    } else {

	temp = *a - *d__;
	p = temp * .5;
/* Computing MAX */
	d__1 = abs(*b), d__2 = abs(*c__);
	bcmax = max(d__1,d__2);
/* Computing MIN */
	d__1 = abs(*b), d__2 = abs(*c__);
	bcmis = min(d__1,d__2) * d_sign(&c_b4, b) * d_sign(&c_b4, c__);
/* Computing MAX */
	d__1 = abs(p);
	scale = max(d__1,bcmax);
	z__ = p / scale * p + bcmax / scale * bcmis;

/*        If Z is of the order of the machine accuracy, postpone the */
/*        decision on the nature of eigenvalues */

	if (z__ >= eps * 4.) {

/*           Real eigenvalues. Compute A and D. */

	    d__1 = sqrt(scale) * sqrt(z__);
	    z__ = p + d_sign(&d__1, &p);
	    *a = *d__ + z__;
	    *d__ -= bcmax / z__ * bcmis;

/*           Compute B and the rotation matrix */

	    tau = dlapy2_(c__, &z__);
	    *cs = z__ / tau;
	    *sn = *c__ / tau;
	    *b -= *c__;
	    *c__ = 0.;
	} else {

/*           Complex eigenvalues, or real (almost) equal eigenvalues. */
/*           Make diagonal elements equal. */

	    sigma = *b + *c__;
	    tau = dlapy2_(&sigma, &temp);
	    *cs = sqrt((abs(sigma) / tau + 1.) * .5);
	    *sn = -(p / (tau * *cs)) * d_sign(&c_b4, &sigma);

/*           Compute [ AA  BB ] = [ A  B ] [ CS -SN ] */
/*                   [ CC  DD ]   [ C  D ] [ SN  CS ] */

	    aa = *a * *cs + *b * *sn;
	    bb = -(*a) * *sn + *b * *cs;
	    cc = *c__ * *cs + *d__ * *sn;
	    dd = -(*c__) * *sn + *d__ * *cs;

/*           Compute [ A  B ] = [ CS  SN ] [ AA  BB ] */
/*                   [ C  D ]   [-SN  CS ] [ CC  DD ] */

	    *a = aa * *cs + cc * *sn;
	    *b = bb * *cs + dd * *sn;
	    *c__ = -aa * *sn + cc * *cs;
	    *d__ = -bb * *sn + dd * *cs;

	    temp = (*a + *d__) * .5;
	    *a = temp;
	    *d__ = temp;

	    if (*c__ != 0.) {
		if (*b != 0.) {
		    if (d_sign(&c_b4, b) == d_sign(&c_b4, c__)) {

/*                    Real eigenvalues: reduce to upper triangular form */

			sab = sqrt((abs(*b)));
			sac = sqrt((abs(*c__)));
			d__1 = sab * sac;
			p = d_sign(&d__1, c__);
			tau = 1. / sqrt((d__1 = *b + *c__, abs(d__1)));
			*a = temp + p;
			*d__ = temp - p;
			*b -= *c__;
			*c__ = 0.;
			cs1 = sab * tau;
			sn1 = sac * tau;
			temp = *cs * cs1 - *sn * sn1;
			*sn = *cs * sn1 + *sn * cs1;
			*cs = temp;
		    }
		} else {
		    *b = -(*c__);
		    *c__ = 0.;
		    temp = *cs;
		    *cs = -(*sn);
		    *sn = temp;
		}
	    }
	}

    }

L10:

/*     Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I). */

    *rt1r = *a;
    *rt2r = *d__;
    if (*c__ == 0.) {
	*rt1i = 0.;
	*rt2i = 0.;
    } else {
	*rt1i = sqrt((abs(*b))) * sqrt((abs(*c__)));
	*rt2i = -(*rt1i);
    }
    return 0;

/*     End of DLANV2 */

} /* dlanv2_ */
