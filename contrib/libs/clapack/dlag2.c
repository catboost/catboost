/* dlag2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dlag2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *safmin, doublereal *scale1, doublereal *
	scale2, doublereal *wr1, doublereal *wr2, doublereal *wi)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */
    double sqrt(doublereal), d_sign(doublereal *, doublereal *);

    /* Local variables */
    doublereal r__, c1, c2, c3, c4, c5, s1, s2, a11, a12, a21, a22, b11, b12, 
	    b22, pp, qq, ss, as11, as12, as22, sum, abi22, diff, bmin, wbig, 
	    wabs, wdet, binv11, binv22, discr, anorm, bnorm, bsize, shift, 
	    rtmin, rtmax, wsize, ascale, bscale, wscale, safmax, wsmall;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAG2 computes the eigenvalues of a 2 x 2 generalized eigenvalue */
/*  problem  A - w B, with scaling as necessary to avoid over-/underflow. */

/*  The scaling factor "s" results in a modified eigenvalue equation */

/*      s A - w B */

/*  where  s  is a non-negative scaling factor chosen so that  w,  w B, */
/*  and  s A  do not overflow and, if possible, do not underflow, either. */

/*  Arguments */
/*  ========= */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA, 2) */
/*          On entry, the 2 x 2 matrix A.  It is assumed that its 1-norm */
/*          is less than 1/SAFMIN.  Entries less than */
/*          sqrt(SAFMIN)*norm(A) are subject to being treated as zero. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= 2. */

/*  B       (input) DOUBLE PRECISION array, dimension (LDB, 2) */
/*          On entry, the 2 x 2 upper triangular matrix B.  It is */
/*          assumed that the one-norm of B is less than 1/SAFMIN.  The */
/*          diagonals should be at least sqrt(SAFMIN) times the largest */
/*          element of B (in absolute value); if a diagonal is smaller */
/*          than that, then  +/- sqrt(SAFMIN) will be used instead of */
/*          that diagonal. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= 2. */

/*  SAFMIN  (input) DOUBLE PRECISION */
/*          The smallest positive number s.t. 1/SAFMIN does not */
/*          overflow.  (This should always be DLAMCH('S') -- it is an */
/*          argument in order to avoid having to call DLAMCH frequently.) */

/*  SCALE1  (output) DOUBLE PRECISION */
/*          A scaling factor used to avoid over-/underflow in the */
/*          eigenvalue equation which defines the first eigenvalue.  If */
/*          the eigenvalues are complex, then the eigenvalues are */
/*          ( WR1  +/-  WI i ) / SCALE1  (which may lie outside the */
/*          exponent range of the machine), SCALE1=SCALE2, and SCALE1 */
/*          will always be positive.  If the eigenvalues are real, then */
/*          the first (real) eigenvalue is  WR1 / SCALE1 , but this may */
/*          overflow or underflow, and in fact, SCALE1 may be zero or */
/*          less than the underflow threshhold if the exact eigenvalue */
/*          is sufficiently large. */

/*  SCALE2  (output) DOUBLE PRECISION */
/*          A scaling factor used to avoid over-/underflow in the */
/*          eigenvalue equation which defines the second eigenvalue.  If */
/*          the eigenvalues are complex, then SCALE2=SCALE1.  If the */
/*          eigenvalues are real, then the second (real) eigenvalue is */
/*          WR2 / SCALE2 , but this may overflow or underflow, and in */
/*          fact, SCALE2 may be zero or less than the underflow */
/*          threshhold if the exact eigenvalue is sufficiently large. */

/*  WR1     (output) DOUBLE PRECISION */
/*          If the eigenvalue is real, then WR1 is SCALE1 times the */
/*          eigenvalue closest to the (2,2) element of A B**(-1).  If the */
/*          eigenvalue is complex, then WR1=WR2 is SCALE1 times the real */
/*          part of the eigenvalues. */

/*  WR2     (output) DOUBLE PRECISION */
/*          If the eigenvalue is real, then WR2 is SCALE2 times the */
/*          other eigenvalue.  If the eigenvalue is complex, then */
/*          WR1=WR2 is SCALE1 times the real part of the eigenvalues. */

/*  WI      (output) DOUBLE PRECISION */
/*          If the eigenvalue is real, then WI is zero.  If the */
/*          eigenvalue is complex, then WI is SCALE1 times the imaginary */
/*          part of the eigenvalues.  WI will always be non-negative. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    rtmin = sqrt(*safmin);
    rtmax = 1. / rtmin;
    safmax = 1. / *safmin;

/*     Scale A */

/* Computing MAX */
    d__5 = (d__1 = a[a_dim1 + 1], abs(d__1)) + (d__2 = a[a_dim1 + 2], abs(
	    d__2)), d__6 = (d__3 = a[(a_dim1 << 1) + 1], abs(d__3)) + (d__4 = 
	    a[(a_dim1 << 1) + 2], abs(d__4)), d__5 = max(d__5,d__6);
    anorm = max(d__5,*safmin);
    ascale = 1. / anorm;
    a11 = ascale * a[a_dim1 + 1];
    a21 = ascale * a[a_dim1 + 2];
    a12 = ascale * a[(a_dim1 << 1) + 1];
    a22 = ascale * a[(a_dim1 << 1) + 2];

/*     Perturb B if necessary to insure non-singularity */

    b11 = b[b_dim1 + 1];
    b12 = b[(b_dim1 << 1) + 1];
    b22 = b[(b_dim1 << 1) + 2];
/* Computing MAX */
    d__1 = abs(b11), d__2 = abs(b12), d__1 = max(d__1,d__2), d__2 = abs(b22), 
	    d__1 = max(d__1,d__2);
    bmin = rtmin * max(d__1,rtmin);
    if (abs(b11) < bmin) {
	b11 = d_sign(&bmin, &b11);
    }
    if (abs(b22) < bmin) {
	b22 = d_sign(&bmin, &b22);
    }

/*     Scale B */

/* Computing MAX */
    d__1 = abs(b11), d__2 = abs(b12) + abs(b22), d__1 = max(d__1,d__2);
    bnorm = max(d__1,*safmin);
/* Computing MAX */
    d__1 = abs(b11), d__2 = abs(b22);
    bsize = max(d__1,d__2);
    bscale = 1. / bsize;
    b11 *= bscale;
    b12 *= bscale;
    b22 *= bscale;

/*     Compute larger eigenvalue by method described by C. van Loan */

/*     ( AS is A shifted by -SHIFT*B ) */

    binv11 = 1. / b11;
    binv22 = 1. / b22;
    s1 = a11 * binv11;
    s2 = a22 * binv22;
    if (abs(s1) <= abs(s2)) {
	as12 = a12 - s1 * b12;
	as22 = a22 - s1 * b22;
	ss = a21 * (binv11 * binv22);
	abi22 = as22 * binv22 - ss * b12;
	pp = abi22 * .5;
	shift = s1;
    } else {
	as12 = a12 - s2 * b12;
	as11 = a11 - s2 * b11;
	ss = a21 * (binv11 * binv22);
	abi22 = -ss * b12;
	pp = (as11 * binv11 + abi22) * .5;
	shift = s2;
    }
    qq = ss * as12;
    if ((d__1 = pp * rtmin, abs(d__1)) >= 1.) {
/* Computing 2nd power */
	d__1 = rtmin * pp;
	discr = d__1 * d__1 + qq * *safmin;
	r__ = sqrt((abs(discr))) * rtmax;
    } else {
/* Computing 2nd power */
	d__1 = pp;
	if (d__1 * d__1 + abs(qq) <= *safmin) {
/* Computing 2nd power */
	    d__1 = rtmax * pp;
	    discr = d__1 * d__1 + qq * safmax;
	    r__ = sqrt((abs(discr))) * rtmin;
	} else {
/* Computing 2nd power */
	    d__1 = pp;
	    discr = d__1 * d__1 + qq;
	    r__ = sqrt((abs(discr)));
	}
    }

/*     Note: the test of R in the following IF is to cover the case when */
/*           DISCR is small and negative and is flushed to zero during */
/*           the calculation of R.  On machines which have a consistent */
/*           flush-to-zero threshhold and handle numbers above that */
/*           threshhold correctly, it would not be necessary. */

    if (discr >= 0. || r__ == 0.) {
	sum = pp + d_sign(&r__, &pp);
	diff = pp - d_sign(&r__, &pp);
	wbig = shift + sum;

/*        Compute smaller eigenvalue */

	wsmall = shift + diff;
/* Computing MAX */
	d__1 = abs(wsmall);
	if (abs(wbig) * .5 > max(d__1,*safmin)) {
	    wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
	    wsmall = wdet / wbig;
	}

/*        Choose (real) eigenvalue closest to 2,2 element of A*B**(-1) */
/*        for WR1. */

	if (pp > abi22) {
	    *wr1 = min(wbig,wsmall);
	    *wr2 = max(wbig,wsmall);
	} else {
	    *wr1 = max(wbig,wsmall);
	    *wr2 = min(wbig,wsmall);
	}
	*wi = 0.;
    } else {

/*        Complex eigenvalues */

	*wr1 = shift + pp;
	*wr2 = *wr1;
	*wi = r__;
    }

/*     Further scaling to avoid underflow and overflow in computing */
/*     SCALE1 and overflow in computing w*B. */

/*     This scale factor (WSCALE) is bounded from above using C1 and C2, */
/*     and from below using C3 and C4. */
/*        C1 implements the condition  s A  must never overflow. */
/*        C2 implements the condition  w B  must never overflow. */
/*        C3, with C2, */
/*           implement the condition that s A - w B must never overflow. */
/*        C4 implements the condition  s    should not underflow. */
/*        C5 implements the condition  max(s,|w|) should be at least 2. */

    c1 = bsize * (*safmin * max(1.,ascale));
    c2 = *safmin * max(1.,bnorm);
    c3 = bsize * *safmin;
    if (ascale <= 1. && bsize <= 1.) {
/* Computing MIN */
	d__1 = 1., d__2 = ascale / *safmin * bsize;
	c4 = min(d__1,d__2);
    } else {
	c4 = 1.;
    }
    if (ascale <= 1. || bsize <= 1.) {
/* Computing MIN */
	d__1 = 1., d__2 = ascale * bsize;
	c5 = min(d__1,d__2);
    } else {
	c5 = 1.;
    }

/*     Scale first eigenvalue */

    wabs = abs(*wr1) + abs(*wi);
/* Computing MAX */
/* Computing MIN */
    d__3 = c4, d__4 = max(wabs,c5) * .5;
    d__1 = max(*safmin,c1), d__2 = (wabs * c2 + c3) * 1.0000100000000001, 
	    d__1 = max(d__1,d__2), d__2 = min(d__3,d__4);
    wsize = max(d__1,d__2);
    if (wsize != 1.) {
	wscale = 1. / wsize;
	if (wsize > 1.) {
	    *scale1 = max(ascale,bsize) * wscale * min(ascale,bsize);
	} else {
	    *scale1 = min(ascale,bsize) * wscale * max(ascale,bsize);
	}
	*wr1 *= wscale;
	if (*wi != 0.) {
	    *wi *= wscale;
	    *wr2 = *wr1;
	    *scale2 = *scale1;
	}
    } else {
	*scale1 = ascale * bsize;
	*scale2 = *scale1;
    }

/*     Scale second eigenvalue (if real) */

    if (*wi == 0.) {
/* Computing MAX */
/* Computing MIN */
/* Computing MAX */
	d__5 = abs(*wr2);
	d__3 = c4, d__4 = max(d__5,c5) * .5;
	d__1 = max(*safmin,c1), d__2 = (abs(*wr2) * c2 + c3) * 
		1.0000100000000001, d__1 = max(d__1,d__2), d__2 = min(d__3,
		d__4);
	wsize = max(d__1,d__2);
	if (wsize != 1.) {
	    wscale = 1. / wsize;
	    if (wsize > 1.) {
		*scale2 = max(ascale,bsize) * wscale * min(ascale,bsize);
	    } else {
		*scale2 = min(ascale,bsize) * wscale * max(ascale,bsize);
	    }
	    *wr2 *= wscale;
	} else {
	    *scale2 = ascale * bsize;
	}
    }

/*     End of DLAG2 */

    return 0;
} /* dlag2_ */
