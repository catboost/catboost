/* dlaqr1.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dlaqr1_(integer *n, doublereal *h__, integer *ldh, 
	doublereal *sr1, doublereal *si1, doublereal *sr2, doublereal *si2, 
	doublereal *v)
{
    /* System generated locals */
    integer h_dim1, h_offset;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    doublereal s, h21s, h31s;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*       Given a 2-by-2 or 3-by-3 matrix H, DLAQR1 sets v to a */
/*       scalar multiple of the first column of the product */

/*       (*)  K = (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I) */

/*       scaling to avoid overflows and most underflows. It */
/*       is assumed that either */

/*               1) sr1 = sr2 and si1 = -si2 */
/*           or */
/*               2) si1 = si2 = 0. */

/*       This is useful for starting double implicit shift bulges */
/*       in the QR algorithm. */


/*       N      (input) integer */
/*              Order of the matrix H. N must be either 2 or 3. */

/*       H      (input) DOUBLE PRECISION array of dimension (LDH,N) */
/*              The 2-by-2 or 3-by-3 matrix H in (*). */

/*       LDH    (input) integer */
/*              The leading dimension of H as declared in */
/*              the calling procedure.  LDH.GE.N */

/*       SR1    (input) DOUBLE PRECISION */
/*       SI1    The shifts in (*). */
/*       SR2 */
/*       SI2 */

/*       V      (output) DOUBLE PRECISION array of dimension N */
/*              A scalar multiple of the first column of the */
/*              matrix K in (*). */

/*     ================================================================ */
/*     Based on contributions by */
/*        Karen Braman and Ralph Byers, Department of Mathematics, */
/*        University of Kansas, USA */

/*     ================================================================ */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --v;

    /* Function Body */
    if (*n == 2) {
	s = (d__1 = h__[h_dim1 + 1] - *sr2, abs(d__1)) + abs(*si2) + (d__2 = 
		h__[h_dim1 + 2], abs(d__2));
	if (s == 0.) {
	    v[1] = 0.;
	    v[2] = 0.;
	} else {
	    h21s = h__[h_dim1 + 2] / s;
	    v[1] = h21s * h__[(h_dim1 << 1) + 1] + (h__[h_dim1 + 1] - *sr1) * 
		    ((h__[h_dim1 + 1] - *sr2) / s) - *si1 * (*si2 / s);
	    v[2] = h21s * (h__[h_dim1 + 1] + h__[(h_dim1 << 1) + 2] - *sr1 - *
		    sr2);
	}
    } else {
	s = (d__1 = h__[h_dim1 + 1] - *sr2, abs(d__1)) + abs(*si2) + (d__2 = 
		h__[h_dim1 + 2], abs(d__2)) + (d__3 = h__[h_dim1 + 3], abs(
		d__3));
	if (s == 0.) {
	    v[1] = 0.;
	    v[2] = 0.;
	    v[3] = 0.;
	} else {
	    h21s = h__[h_dim1 + 2] / s;
	    h31s = h__[h_dim1 + 3] / s;
	    v[1] = (h__[h_dim1 + 1] - *sr1) * ((h__[h_dim1 + 1] - *sr2) / s) 
		    - *si1 * (*si2 / s) + h__[(h_dim1 << 1) + 1] * h21s + h__[
		    h_dim1 * 3 + 1] * h31s;
	    v[2] = h21s * (h__[h_dim1 + 1] + h__[(h_dim1 << 1) + 2] - *sr1 - *
		    sr2) + h__[h_dim1 * 3 + 2] * h31s;
	    v[3] = h31s * (h__[h_dim1 + 1] + h__[h_dim1 * 3 + 3] - *sr1 - *
		    sr2) + h21s * h__[(h_dim1 << 1) + 3];
	}
    }
    return 0;
} /* dlaqr1_ */
