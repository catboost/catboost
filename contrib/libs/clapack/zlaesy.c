/* zlaesy.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {1.,0.};
static integer c__2 = 2;

/* Subroutine */ int zlaesy_(doublecomplex *a, doublecomplex *b, 
	doublecomplex *c__, doublecomplex *rt1, doublecomplex *rt2, 
	doublecomplex *evscal, doublecomplex *cs1, doublecomplex *sn1)
{
    /* System generated locals */
    doublereal d__1, d__2;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7;

    /* Builtin functions */
    double z_abs(doublecomplex *);
    void pow_zi(doublecomplex *, doublecomplex *, integer *), z_sqrt(
	    doublecomplex *, doublecomplex *), z_div(doublecomplex *, 
	    doublecomplex *, doublecomplex *);

    /* Local variables */
    doublecomplex s, t;
    doublereal z__;
    doublecomplex tmp;
    doublereal babs, tabs, evnorm;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLAESY computes the eigendecomposition of a 2-by-2 symmetric matrix */
/*     ( ( A, B );( B, C ) ) */
/*  provided the norm of the matrix of eigenvectors is larger than */
/*  some threshold value. */

/*  RT1 is the eigenvalue of larger absolute value, and RT2 of */
/*  smaller absolute value.  If the eigenvectors are computed, then */
/*  on return ( CS1, SN1 ) is the unit eigenvector for RT1, hence */

/*  [  CS1     SN1   ] . [ A  B ] . [ CS1    -SN1   ] = [ RT1  0  ] */
/*  [ -SN1     CS1   ]   [ B  C ]   [ SN1     CS1   ]   [  0  RT2 ] */

/*  Arguments */
/*  ========= */

/*  A       (input) COMPLEX*16 */
/*          The ( 1, 1 ) element of input matrix. */

/*  B       (input) COMPLEX*16 */
/*          The ( 1, 2 ) element of input matrix.  The ( 2, 1 ) element */
/*          is also given by B, since the 2-by-2 matrix is symmetric. */

/*  C       (input) COMPLEX*16 */
/*          The ( 2, 2 ) element of input matrix. */

/*  RT1     (output) COMPLEX*16 */
/*          The eigenvalue of larger modulus. */

/*  RT2     (output) COMPLEX*16 */
/*          The eigenvalue of smaller modulus. */

/*  EVSCAL  (output) COMPLEX*16 */
/*          The complex value by which the eigenvector matrix was scaled */
/*          to make it orthonormal.  If EVSCAL is zero, the eigenvectors */
/*          were not computed.  This means one of two things:  the 2-by-2 */
/*          matrix could not be diagonalized, or the norm of the matrix */
/*          of eigenvectors before scaling was larger than the threshold */
/*          value THRESH (set below). */

/*  CS1     (output) COMPLEX*16 */
/*  SN1     (output) COMPLEX*16 */
/*          If EVSCAL .NE. 0,  ( CS1, SN1 ) is the unit right eigenvector */
/*          for RT1. */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */


/*     Special case:  The matrix is actually diagonal. */
/*     To avoid divide by zero later, we treat this case separately. */

    if (z_abs(b) == 0.) {
	rt1->r = a->r, rt1->i = a->i;
	rt2->r = c__->r, rt2->i = c__->i;
	if (z_abs(rt1) < z_abs(rt2)) {
	    tmp.r = rt1->r, tmp.i = rt1->i;
	    rt1->r = rt2->r, rt1->i = rt2->i;
	    rt2->r = tmp.r, rt2->i = tmp.i;
	    cs1->r = 0., cs1->i = 0.;
	    sn1->r = 1., sn1->i = 0.;
	} else {
	    cs1->r = 1., cs1->i = 0.;
	    sn1->r = 0., sn1->i = 0.;
	}
    } else {

/*        Compute the eigenvalues and eigenvectors. */
/*        The characteristic equation is */
/*           lambda **2 - (A+C) lambda + (A*C - B*B) */
/*        and we solve it using the quadratic formula. */

	z__2.r = a->r + c__->r, z__2.i = a->i + c__->i;
	z__1.r = z__2.r * .5, z__1.i = z__2.i * .5;
	s.r = z__1.r, s.i = z__1.i;
	z__2.r = a->r - c__->r, z__2.i = a->i - c__->i;
	z__1.r = z__2.r * .5, z__1.i = z__2.i * .5;
	t.r = z__1.r, t.i = z__1.i;

/*        Take the square root carefully to avoid over/under flow. */

	babs = z_abs(b);
	tabs = z_abs(&t);
	z__ = max(babs,tabs);
	if (z__ > 0.) {
	    z__5.r = t.r / z__, z__5.i = t.i / z__;
	    pow_zi(&z__4, &z__5, &c__2);
	    z__7.r = b->r / z__, z__7.i = b->i / z__;
	    pow_zi(&z__6, &z__7, &c__2);
	    z__3.r = z__4.r + z__6.r, z__3.i = z__4.i + z__6.i;
	    z_sqrt(&z__2, &z__3);
	    z__1.r = z__ * z__2.r, z__1.i = z__ * z__2.i;
	    t.r = z__1.r, t.i = z__1.i;
	}

/*        Compute the two eigenvalues.  RT1 and RT2 are exchanged */
/*        if necessary so that RT1 will have the greater magnitude. */

	z__1.r = s.r + t.r, z__1.i = s.i + t.i;
	rt1->r = z__1.r, rt1->i = z__1.i;
	z__1.r = s.r - t.r, z__1.i = s.i - t.i;
	rt2->r = z__1.r, rt2->i = z__1.i;
	if (z_abs(rt1) < z_abs(rt2)) {
	    tmp.r = rt1->r, tmp.i = rt1->i;
	    rt1->r = rt2->r, rt1->i = rt2->i;
	    rt2->r = tmp.r, rt2->i = tmp.i;
	}

/*        Choose CS1 = 1 and SN1 to satisfy the first equation, then */
/*        scale the components of this eigenvector so that the matrix */
/*        of eigenvectors X satisfies  X * X' = I .  (No scaling is */
/*        done if the norm of the eigenvalue matrix is less than THRESH.) */

	z__2.r = rt1->r - a->r, z__2.i = rt1->i - a->i;
	z_div(&z__1, &z__2, b);
	sn1->r = z__1.r, sn1->i = z__1.i;
	tabs = z_abs(sn1);
	if (tabs > 1.) {
/* Computing 2nd power */
	    d__2 = 1. / tabs;
	    d__1 = d__2 * d__2;
	    z__5.r = sn1->r / tabs, z__5.i = sn1->i / tabs;
	    pow_zi(&z__4, &z__5, &c__2);
	    z__3.r = d__1 + z__4.r, z__3.i = z__4.i;
	    z_sqrt(&z__2, &z__3);
	    z__1.r = tabs * z__2.r, z__1.i = tabs * z__2.i;
	    t.r = z__1.r, t.i = z__1.i;
	} else {
	    z__3.r = sn1->r * sn1->r - sn1->i * sn1->i, z__3.i = sn1->r * 
		    sn1->i + sn1->i * sn1->r;
	    z__2.r = z__3.r + 1., z__2.i = z__3.i + 0.;
	    z_sqrt(&z__1, &z__2);
	    t.r = z__1.r, t.i = z__1.i;
	}
	evnorm = z_abs(&t);
	if (evnorm >= .1) {
	    z_div(&z__1, &c_b1, &t);
	    evscal->r = z__1.r, evscal->i = z__1.i;
	    cs1->r = evscal->r, cs1->i = evscal->i;
	    z__1.r = sn1->r * evscal->r - sn1->i * evscal->i, z__1.i = sn1->r 
		    * evscal->i + sn1->i * evscal->r;
	    sn1->r = z__1.r, sn1->i = z__1.i;
	} else {
	    evscal->r = 0., evscal->i = 0.;
	}
    }
    return 0;

/*     End of ZLAESY */

} /* zlaesy_ */
