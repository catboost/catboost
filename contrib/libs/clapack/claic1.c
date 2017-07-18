/* claic1.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int claic1_(integer *job, integer *j, complex *x, real *sest, 
	 complex *w, complex *gamma, real *sestpr, complex *s, complex *c__)
{
    /* System generated locals */
    real r__1, r__2;
    complex q__1, q__2, q__3, q__4, q__5, q__6;

    /* Builtin functions */
    double c_abs(complex *);
    void r_cnjg(complex *, complex *), c_sqrt(complex *, complex *);
    double sqrt(doublereal);
    void c_div(complex *, complex *, complex *);

    /* Local variables */
    real b, t, s1, s2, scl, eps, tmp;
    complex sine;
    real test, zeta1, zeta2;
    complex alpha;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    real norma, absgam, absalp;
    extern doublereal slamch_(char *);
    complex cosine;
    real absest;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLAIC1 applies one step of incremental condition estimation in */
/*  its simplest version: */

/*  Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j */
/*  lower triangular matrix L, such that */
/*           twonorm(L*x) = sest */
/*  Then CLAIC1 computes sestpr, s, c such that */
/*  the vector */
/*                  [ s*x ] */
/*           xhat = [  c  ] */
/*  is an approximate singular vector of */
/*                  [ L     0  ] */
/*           Lhat = [ w' gamma ] */
/*  in the sense that */
/*           twonorm(Lhat*xhat) = sestpr. */

/*  Depending on JOB, an estimate for the largest or smallest singular */
/*  value is computed. */

/*  Note that [s c]' and sestpr**2 is an eigenpair of the system */

/*      diag(sest*sest, 0) + [alpha  gamma] * [ conjg(alpha) ] */
/*                                            [ conjg(gamma) ] */

/*  where  alpha =  conjg(x)'*w. */

/*  Arguments */
/*  ========= */

/*  JOB     (input) INTEGER */
/*          = 1: an estimate for the largest singular value is computed. */
/*          = 2: an estimate for the smallest singular value is computed. */

/*  J       (input) INTEGER */
/*          Length of X and W */

/*  X       (input) COMPLEX array, dimension (J) */
/*          The j-vector x. */

/*  SEST    (input) REAL */
/*          Estimated singular value of j by j matrix L */

/*  W       (input) COMPLEX array, dimension (J) */
/*          The j-vector w. */

/*  GAMMA   (input) COMPLEX */
/*          The diagonal element gamma. */

/*  SESTPR  (output) REAL */
/*          Estimated singular value of (j+1) by (j+1) matrix Lhat. */

/*  S       (output) COMPLEX */
/*          Sine needed in forming xhat. */

/*  C       (output) COMPLEX */
/*          Cosine needed in forming xhat. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --w;
    --x;

    /* Function Body */
    eps = slamch_("Epsilon");
    cdotc_(&q__1, j, &x[1], &c__1, &w[1], &c__1);
    alpha.r = q__1.r, alpha.i = q__1.i;

    absalp = c_abs(&alpha);
    absgam = c_abs(gamma);
    absest = dabs(*sest);

    if (*job == 1) {

/*        Estimating largest singular value */

/*        special cases */

	if (*sest == 0.f) {
	    s1 = dmax(absgam,absalp);
	    if (s1 == 0.f) {
		s->r = 0.f, s->i = 0.f;
		c__->r = 1.f, c__->i = 0.f;
		*sestpr = 0.f;
	    } else {
		q__1.r = alpha.r / s1, q__1.i = alpha.i / s1;
		s->r = q__1.r, s->i = q__1.i;
		q__1.r = gamma->r / s1, q__1.i = gamma->i / s1;
		c__->r = q__1.r, c__->i = q__1.i;
		r_cnjg(&q__4, s);
		q__3.r = s->r * q__4.r - s->i * q__4.i, q__3.i = s->r * 
			q__4.i + s->i * q__4.r;
		r_cnjg(&q__6, c__);
		q__5.r = c__->r * q__6.r - c__->i * q__6.i, q__5.i = c__->r * 
			q__6.i + c__->i * q__6.r;
		q__2.r = q__3.r + q__5.r, q__2.i = q__3.i + q__5.i;
		c_sqrt(&q__1, &q__2);
		tmp = q__1.r;
		q__1.r = s->r / tmp, q__1.i = s->i / tmp;
		s->r = q__1.r, s->i = q__1.i;
		q__1.r = c__->r / tmp, q__1.i = c__->i / tmp;
		c__->r = q__1.r, c__->i = q__1.i;
		*sestpr = s1 * tmp;
	    }
	    return 0;
	} else if (absgam <= eps * absest) {
	    s->r = 1.f, s->i = 0.f;
	    c__->r = 0.f, c__->i = 0.f;
	    tmp = dmax(absest,absalp);
	    s1 = absest / tmp;
	    s2 = absalp / tmp;
	    *sestpr = tmp * sqrt(s1 * s1 + s2 * s2);
	    return 0;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		s->r = 1.f, s->i = 0.f;
		c__->r = 0.f, c__->i = 0.f;
		*sestpr = s2;
	    } else {
		s->r = 0.f, s->i = 0.f;
		c__->r = 1.f, c__->i = 0.f;
		*sestpr = s1;
	    }
	    return 0;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		scl = sqrt(tmp * tmp + 1.f);
		*sestpr = s2 * scl;
		q__2.r = alpha.r / s2, q__2.i = alpha.i / s2;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		s->r = q__1.r, s->i = q__1.i;
		q__2.r = gamma->r / s2, q__2.i = gamma->i / s2;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		c__->r = q__1.r, c__->i = q__1.i;
	    } else {
		tmp = s2 / s1;
		scl = sqrt(tmp * tmp + 1.f);
		*sestpr = s1 * scl;
		q__2.r = alpha.r / s1, q__2.i = alpha.i / s1;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		s->r = q__1.r, s->i = q__1.i;
		q__2.r = gamma->r / s1, q__2.i = gamma->i / s1;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		c__->r = q__1.r, c__->i = q__1.i;
	    }
	    return 0;
	} else {

/*           normal case */

	    zeta1 = absalp / absest;
	    zeta2 = absgam / absest;

	    b = (1.f - zeta1 * zeta1 - zeta2 * zeta2) * .5f;
	    r__1 = zeta1 * zeta1;
	    c__->r = r__1, c__->i = 0.f;
	    if (b > 0.f) {
		r__1 = b * b;
		q__4.r = r__1 + c__->r, q__4.i = c__->i;
		c_sqrt(&q__3, &q__4);
		q__2.r = b + q__3.r, q__2.i = q__3.i;
		c_div(&q__1, c__, &q__2);
		t = q__1.r;
	    } else {
		r__1 = b * b;
		q__3.r = r__1 + c__->r, q__3.i = c__->i;
		c_sqrt(&q__2, &q__3);
		q__1.r = q__2.r - b, q__1.i = q__2.i;
		t = q__1.r;
	    }

	    q__3.r = alpha.r / absest, q__3.i = alpha.i / absest;
	    q__2.r = -q__3.r, q__2.i = -q__3.i;
	    q__1.r = q__2.r / t, q__1.i = q__2.i / t;
	    sine.r = q__1.r, sine.i = q__1.i;
	    q__3.r = gamma->r / absest, q__3.i = gamma->i / absest;
	    q__2.r = -q__3.r, q__2.i = -q__3.i;
	    r__1 = t + 1.f;
	    q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
	    cosine.r = q__1.r, cosine.i = q__1.i;
	    r_cnjg(&q__4, &sine);
	    q__3.r = sine.r * q__4.r - sine.i * q__4.i, q__3.i = sine.r * 
		    q__4.i + sine.i * q__4.r;
	    r_cnjg(&q__6, &cosine);
	    q__5.r = cosine.r * q__6.r - cosine.i * q__6.i, q__5.i = cosine.r 
		    * q__6.i + cosine.i * q__6.r;
	    q__2.r = q__3.r + q__5.r, q__2.i = q__3.i + q__5.i;
	    c_sqrt(&q__1, &q__2);
	    tmp = q__1.r;
	    q__1.r = sine.r / tmp, q__1.i = sine.i / tmp;
	    s->r = q__1.r, s->i = q__1.i;
	    q__1.r = cosine.r / tmp, q__1.i = cosine.i / tmp;
	    c__->r = q__1.r, c__->i = q__1.i;
	    *sestpr = sqrt(t + 1.f) * absest;
	    return 0;
	}

    } else if (*job == 2) {

/*        Estimating smallest singular value */

/*        special cases */

	if (*sest == 0.f) {
	    *sestpr = 0.f;
	    if (dmax(absgam,absalp) == 0.f) {
		sine.r = 1.f, sine.i = 0.f;
		cosine.r = 0.f, cosine.i = 0.f;
	    } else {
		r_cnjg(&q__2, gamma);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		sine.r = q__1.r, sine.i = q__1.i;
		r_cnjg(&q__1, &alpha);
		cosine.r = q__1.r, cosine.i = q__1.i;
	    }
/* Computing MAX */
	    r__1 = c_abs(&sine), r__2 = c_abs(&cosine);
	    s1 = dmax(r__1,r__2);
	    q__1.r = sine.r / s1, q__1.i = sine.i / s1;
	    s->r = q__1.r, s->i = q__1.i;
	    q__1.r = cosine.r / s1, q__1.i = cosine.i / s1;
	    c__->r = q__1.r, c__->i = q__1.i;
	    r_cnjg(&q__4, s);
	    q__3.r = s->r * q__4.r - s->i * q__4.i, q__3.i = s->r * q__4.i + 
		    s->i * q__4.r;
	    r_cnjg(&q__6, c__);
	    q__5.r = c__->r * q__6.r - c__->i * q__6.i, q__5.i = c__->r * 
		    q__6.i + c__->i * q__6.r;
	    q__2.r = q__3.r + q__5.r, q__2.i = q__3.i + q__5.i;
	    c_sqrt(&q__1, &q__2);
	    tmp = q__1.r;
	    q__1.r = s->r / tmp, q__1.i = s->i / tmp;
	    s->r = q__1.r, s->i = q__1.i;
	    q__1.r = c__->r / tmp, q__1.i = c__->i / tmp;
	    c__->r = q__1.r, c__->i = q__1.i;
	    return 0;
	} else if (absgam <= eps * absest) {
	    s->r = 0.f, s->i = 0.f;
	    c__->r = 1.f, c__->i = 0.f;
	    *sestpr = absgam;
	    return 0;
	} else if (absalp <= eps * absest) {
	    s1 = absgam;
	    s2 = absest;
	    if (s1 <= s2) {
		s->r = 0.f, s->i = 0.f;
		c__->r = 1.f, c__->i = 0.f;
		*sestpr = s1;
	    } else {
		s->r = 1.f, s->i = 0.f;
		c__->r = 0.f, c__->i = 0.f;
		*sestpr = s2;
	    }
	    return 0;
	} else if (absest <= eps * absalp || absest <= eps * absgam) {
	    s1 = absgam;
	    s2 = absalp;
	    if (s1 <= s2) {
		tmp = s1 / s2;
		scl = sqrt(tmp * tmp + 1.f);
		*sestpr = absest * (tmp / scl);
		r_cnjg(&q__4, gamma);
		q__3.r = q__4.r / s2, q__3.i = q__4.i / s2;
		q__2.r = -q__3.r, q__2.i = -q__3.i;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		s->r = q__1.r, s->i = q__1.i;
		r_cnjg(&q__3, &alpha);
		q__2.r = q__3.r / s2, q__2.i = q__3.i / s2;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		c__->r = q__1.r, c__->i = q__1.i;
	    } else {
		tmp = s2 / s1;
		scl = sqrt(tmp * tmp + 1.f);
		*sestpr = absest / scl;
		r_cnjg(&q__4, gamma);
		q__3.r = q__4.r / s1, q__3.i = q__4.i / s1;
		q__2.r = -q__3.r, q__2.i = -q__3.i;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		s->r = q__1.r, s->i = q__1.i;
		r_cnjg(&q__3, &alpha);
		q__2.r = q__3.r / s1, q__2.i = q__3.i / s1;
		q__1.r = q__2.r / scl, q__1.i = q__2.i / scl;
		c__->r = q__1.r, c__->i = q__1.i;
	    }
	    return 0;
	} else {

/*           normal case */

	    zeta1 = absalp / absest;
	    zeta2 = absgam / absest;

/* Computing MAX */
	    r__1 = zeta1 * zeta1 + 1.f + zeta1 * zeta2, r__2 = zeta1 * zeta2 
		    + zeta2 * zeta2;
	    norma = dmax(r__1,r__2);

/*           See if root is closer to zero or to ONE */

	    test = (zeta1 - zeta2) * 2.f * (zeta1 + zeta2) + 1.f;
	    if (test >= 0.f) {

/*              root is close to zero, compute directly */

		b = (zeta1 * zeta1 + zeta2 * zeta2 + 1.f) * .5f;
		r__1 = zeta2 * zeta2;
		c__->r = r__1, c__->i = 0.f;
		r__2 = b * b;
		q__2.r = r__2 - c__->r, q__2.i = -c__->i;
		r__1 = b + sqrt(c_abs(&q__2));
		q__1.r = c__->r / r__1, q__1.i = c__->i / r__1;
		t = q__1.r;
		q__2.r = alpha.r / absest, q__2.i = alpha.i / absest;
		r__1 = 1.f - t;
		q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
		sine.r = q__1.r, sine.i = q__1.i;
		q__3.r = gamma->r / absest, q__3.i = gamma->i / absest;
		q__2.r = -q__3.r, q__2.i = -q__3.i;
		q__1.r = q__2.r / t, q__1.i = q__2.i / t;
		cosine.r = q__1.r, cosine.i = q__1.i;
		*sestpr = sqrt(t + eps * 4.f * eps * norma) * absest;
	    } else {

/*              root is closer to ONE, shift by that amount */

		b = (zeta2 * zeta2 + zeta1 * zeta1 - 1.f) * .5f;
		r__1 = zeta1 * zeta1;
		c__->r = r__1, c__->i = 0.f;
		if (b >= 0.f) {
		    q__2.r = -c__->r, q__2.i = -c__->i;
		    r__1 = b * b;
		    q__5.r = r__1 + c__->r, q__5.i = c__->i;
		    c_sqrt(&q__4, &q__5);
		    q__3.r = b + q__4.r, q__3.i = q__4.i;
		    c_div(&q__1, &q__2, &q__3);
		    t = q__1.r;
		} else {
		    r__1 = b * b;
		    q__3.r = r__1 + c__->r, q__3.i = c__->i;
		    c_sqrt(&q__2, &q__3);
		    q__1.r = b - q__2.r, q__1.i = -q__2.i;
		    t = q__1.r;
		}
		q__3.r = alpha.r / absest, q__3.i = alpha.i / absest;
		q__2.r = -q__3.r, q__2.i = -q__3.i;
		q__1.r = q__2.r / t, q__1.i = q__2.i / t;
		sine.r = q__1.r, sine.i = q__1.i;
		q__3.r = gamma->r / absest, q__3.i = gamma->i / absest;
		q__2.r = -q__3.r, q__2.i = -q__3.i;
		r__1 = t + 1.f;
		q__1.r = q__2.r / r__1, q__1.i = q__2.i / r__1;
		cosine.r = q__1.r, cosine.i = q__1.i;
		*sestpr = sqrt(t + 1.f + eps * 4.f * eps * norma) * absest;
	    }
	    r_cnjg(&q__4, &sine);
	    q__3.r = sine.r * q__4.r - sine.i * q__4.i, q__3.i = sine.r * 
		    q__4.i + sine.i * q__4.r;
	    r_cnjg(&q__6, &cosine);
	    q__5.r = cosine.r * q__6.r - cosine.i * q__6.i, q__5.i = cosine.r 
		    * q__6.i + cosine.i * q__6.r;
	    q__2.r = q__3.r + q__5.r, q__2.i = q__3.i + q__5.i;
	    c_sqrt(&q__1, &q__2);
	    tmp = q__1.r;
	    q__1.r = sine.r / tmp, q__1.i = sine.i / tmp;
	    s->r = q__1.r, s->i = q__1.i;
	    q__1.r = cosine.r / tmp, q__1.i = cosine.i / tmp;
	    c__->r = q__1.r, c__->i = q__1.i;
	    return 0;

	}
    }
    return 0;

/*     End of CLAIC1 */

} /* claic1_ */
