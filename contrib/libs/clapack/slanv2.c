/* slanv2.f -- translated by f2c (version 20061008).
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

static real c_b4 = 1.f;

/* Subroutine */ int slanv2_(real *a, real *b, real *c__, real *d__, real *
	rt1r, real *rt1i, real *rt2r, real *rt2i, real *cs, real *sn)
{
    /* System generated locals */
    real r__1, r__2;

    /* Builtin functions */
    double r_sign(real *, real *), sqrt(doublereal);

    /* Local variables */
    real p, z__, aa, bb, cc, dd, cs1, sn1, sab, sac, eps, tau, temp, scale, 
	    bcmax, bcmis, sigma;
    extern doublereal slapy2_(real *, real *), slamch_(char *);


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric */
/*  matrix in standard form: */

/*       [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ] */
/*       [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ] */

/*  where either */
/*  1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or */
/*  2) AA = DD and BB*CC < 0, so that AA + or - sqrt(BB*CC) are complex */
/*  conjugate eigenvalues. */

/*  Arguments */
/*  ========= */

/*  A       (input/output) REAL */
/*  B       (input/output) REAL */
/*  C       (input/output) REAL */
/*  D       (input/output) REAL */
/*          On entry, the elements of the input matrix. */
/*          On exit, they are overwritten by the elements of the */
/*          standardised Schur form. */

/*  RT1R    (output) REAL */
/*  RT1I    (output) REAL */
/*  RT2R    (output) REAL */
/*  RT2I    (output) REAL */
/*          The real and imaginary parts of the eigenvalues. If the */
/*          eigenvalues are a complex conjugate pair, RT1I > 0. */

/*  CS      (output) REAL */
/*  SN      (output) REAL */
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

    eps = slamch_("P");
    if (*c__ == 0.f) {
	*cs = 1.f;
	*sn = 0.f;
	goto L10;

    } else if (*b == 0.f) {

/*        Swap rows and columns */

	*cs = 0.f;
	*sn = 1.f;
	temp = *d__;
	*d__ = *a;
	*a = temp;
	*b = -(*c__);
	*c__ = 0.f;
	goto L10;
    } else if (*a - *d__ == 0.f && r_sign(&c_b4, b) != r_sign(&c_b4, c__)) {
	*cs = 1.f;
	*sn = 0.f;
	goto L10;
    } else {

	temp = *a - *d__;
	p = temp * .5f;
/* Computing MAX */
	r__1 = dabs(*b), r__2 = dabs(*c__);
	bcmax = dmax(r__1,r__2);
/* Computing MIN */
	r__1 = dabs(*b), r__2 = dabs(*c__);
	bcmis = dmin(r__1,r__2) * r_sign(&c_b4, b) * r_sign(&c_b4, c__);
/* Computing MAX */
	r__1 = dabs(p);
	scale = dmax(r__1,bcmax);
	z__ = p / scale * p + bcmax / scale * bcmis;

/*        If Z is of the order of the machine accuracy, postpone the */
/*        decision on the nature of eigenvalues */

	if (z__ >= eps * 4.f) {

/*           Real eigenvalues. Compute A and D. */

	    r__1 = sqrt(scale) * sqrt(z__);
	    z__ = p + r_sign(&r__1, &p);
	    *a = *d__ + z__;
	    *d__ -= bcmax / z__ * bcmis;

/*           Compute B and the rotation matrix */

	    tau = slapy2_(c__, &z__);
	    *cs = z__ / tau;
	    *sn = *c__ / tau;
	    *b -= *c__;
	    *c__ = 0.f;
	} else {

/*           Complex eigenvalues, or real (almost) equal eigenvalues. */
/*           Make diagonal elements equal. */

	    sigma = *b + *c__;
	    tau = slapy2_(&sigma, &temp);
	    *cs = sqrt((dabs(sigma) / tau + 1.f) * .5f);
	    *sn = -(p / (tau * *cs)) * r_sign(&c_b4, &sigma);

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

	    temp = (*a + *d__) * .5f;
	    *a = temp;
	    *d__ = temp;

	    if (*c__ != 0.f) {
		if (*b != 0.f) {
		    if (r_sign(&c_b4, b) == r_sign(&c_b4, c__)) {

/*                    Real eigenvalues: reduce to upper triangular form */

			sab = sqrt((dabs(*b)));
			sac = sqrt((dabs(*c__)));
			r__1 = sab * sac;
			p = r_sign(&r__1, c__);
			tau = 1.f / sqrt((r__1 = *b + *c__, dabs(r__1)));
			*a = temp + p;
			*d__ = temp - p;
			*b -= *c__;
			*c__ = 0.f;
			cs1 = sab * tau;
			sn1 = sac * tau;
			temp = *cs * cs1 - *sn * sn1;
			*sn = *cs * sn1 + *sn * cs1;
			*cs = temp;
		    }
		} else {
		    *b = -(*c__);
		    *c__ = 0.f;
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
    if (*c__ == 0.f) {
	*rt1i = 0.f;
	*rt2i = 0.f;
    } else {
	*rt1i = sqrt((dabs(*b))) * sqrt((dabs(*c__)));
	*rt2i = -(*rt1i);
    }
    return 0;

/*     End of SLANV2 */

} /* slanv2_ */
