/* zlaqr1.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlaqr1_(integer *n, doublecomplex *h__, integer *ldh, 
	doublecomplex *s1, doublecomplex *s2, doublecomplex *v)
{
    /* System generated locals */
    integer h_dim1, h_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7, z__8;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    doublereal s;
    doublecomplex h21s, h31s;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*       Given a 2-by-2 or 3-by-3 matrix H, ZLAQR1 sets v to a */
/*       scalar multiple of the first column of the product */

/*       (*)  K = (H - s1*I)*(H - s2*I) */

/*       scaling to avoid overflows and most underflows. */

/*       This is useful for starting double implicit shift bulges */
/*       in the QR algorithm. */


/*       N      (input) integer */
/*              Order of the matrix H. N must be either 2 or 3. */

/*       H      (input) COMPLEX*16 array of dimension (LDH,N) */
/*              The 2-by-2 or 3-by-3 matrix H in (*). */

/*       LDH    (input) integer */
/*              The leading dimension of H as declared in */
/*              the calling procedure.  LDH.GE.N */

/*       S1     (input) COMPLEX*16 */
/*       S2     S1 and S2 are the shifts defining K in (*) above. */

/*       V      (output) COMPLEX*16 array of dimension N */
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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --v;

    /* Function Body */
    if (*n == 2) {
	i__1 = h_dim1 + 1;
	z__2.r = h__[i__1].r - s2->r, z__2.i = h__[i__1].i - s2->i;
	z__1.r = z__2.r, z__1.i = z__2.i;
	i__2 = h_dim1 + 2;
	s = (d__1 = z__1.r, abs(d__1)) + (d__2 = d_imag(&z__1), abs(d__2)) + (
		(d__3 = h__[i__2].r, abs(d__3)) + (d__4 = d_imag(&h__[h_dim1 
		+ 2]), abs(d__4)));
	if (s == 0.) {
	    v[1].r = 0., v[1].i = 0.;
	    v[2].r = 0., v[2].i = 0.;
	} else {
	    i__1 = h_dim1 + 2;
	    z__1.r = h__[i__1].r / s, z__1.i = h__[i__1].i / s;
	    h21s.r = z__1.r, h21s.i = z__1.i;
	    i__1 = (h_dim1 << 1) + 1;
	    z__2.r = h21s.r * h__[i__1].r - h21s.i * h__[i__1].i, z__2.i = 
		    h21s.r * h__[i__1].i + h21s.i * h__[i__1].r;
	    i__2 = h_dim1 + 1;
	    z__4.r = h__[i__2].r - s1->r, z__4.i = h__[i__2].i - s1->i;
	    i__3 = h_dim1 + 1;
	    z__6.r = h__[i__3].r - s2->r, z__6.i = h__[i__3].i - s2->i;
	    z__5.r = z__6.r / s, z__5.i = z__6.i / s;
	    z__3.r = z__4.r * z__5.r - z__4.i * z__5.i, z__3.i = z__4.r * 
		    z__5.i + z__4.i * z__5.r;
	    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	    v[1].r = z__1.r, v[1].i = z__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    z__4.r = h__[i__1].r + h__[i__2].r, z__4.i = h__[i__1].i + h__[
		    i__2].i;
	    z__3.r = z__4.r - s1->r, z__3.i = z__4.i - s1->i;
	    z__2.r = z__3.r - s2->r, z__2.i = z__3.i - s2->i;
	    z__1.r = h21s.r * z__2.r - h21s.i * z__2.i, z__1.i = h21s.r * 
		    z__2.i + h21s.i * z__2.r;
	    v[2].r = z__1.r, v[2].i = z__1.i;
	}
    } else {
	i__1 = h_dim1 + 1;
	z__2.r = h__[i__1].r - s2->r, z__2.i = h__[i__1].i - s2->i;
	z__1.r = z__2.r, z__1.i = z__2.i;
	i__2 = h_dim1 + 2;
	i__3 = h_dim1 + 3;
	s = (d__1 = z__1.r, abs(d__1)) + (d__2 = d_imag(&z__1), abs(d__2)) + (
		(d__3 = h__[i__2].r, abs(d__3)) + (d__4 = d_imag(&h__[h_dim1 
		+ 2]), abs(d__4))) + ((d__5 = h__[i__3].r, abs(d__5)) + (d__6 
		= d_imag(&h__[h_dim1 + 3]), abs(d__6)));
	if (s == 0.) {
	    v[1].r = 0., v[1].i = 0.;
	    v[2].r = 0., v[2].i = 0.;
	    v[3].r = 0., v[3].i = 0.;
	} else {
	    i__1 = h_dim1 + 2;
	    z__1.r = h__[i__1].r / s, z__1.i = h__[i__1].i / s;
	    h21s.r = z__1.r, h21s.i = z__1.i;
	    i__1 = h_dim1 + 3;
	    z__1.r = h__[i__1].r / s, z__1.i = h__[i__1].i / s;
	    h31s.r = z__1.r, h31s.i = z__1.i;
	    i__1 = h_dim1 + 1;
	    z__4.r = h__[i__1].r - s1->r, z__4.i = h__[i__1].i - s1->i;
	    i__2 = h_dim1 + 1;
	    z__6.r = h__[i__2].r - s2->r, z__6.i = h__[i__2].i - s2->i;
	    z__5.r = z__6.r / s, z__5.i = z__6.i / s;
	    z__3.r = z__4.r * z__5.r - z__4.i * z__5.i, z__3.i = z__4.r * 
		    z__5.i + z__4.i * z__5.r;
	    i__3 = (h_dim1 << 1) + 1;
	    z__7.r = h__[i__3].r * h21s.r - h__[i__3].i * h21s.i, z__7.i = 
		    h__[i__3].r * h21s.i + h__[i__3].i * h21s.r;
	    z__2.r = z__3.r + z__7.r, z__2.i = z__3.i + z__7.i;
	    i__4 = h_dim1 * 3 + 1;
	    z__8.r = h__[i__4].r * h31s.r - h__[i__4].i * h31s.i, z__8.i = 
		    h__[i__4].r * h31s.i + h__[i__4].i * h31s.r;
	    z__1.r = z__2.r + z__8.r, z__1.i = z__2.i + z__8.i;
	    v[1].r = z__1.r, v[1].i = z__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    z__5.r = h__[i__1].r + h__[i__2].r, z__5.i = h__[i__1].i + h__[
		    i__2].i;
	    z__4.r = z__5.r - s1->r, z__4.i = z__5.i - s1->i;
	    z__3.r = z__4.r - s2->r, z__3.i = z__4.i - s2->i;
	    z__2.r = h21s.r * z__3.r - h21s.i * z__3.i, z__2.i = h21s.r * 
		    z__3.i + h21s.i * z__3.r;
	    i__3 = h_dim1 * 3 + 2;
	    z__6.r = h__[i__3].r * h31s.r - h__[i__3].i * h31s.i, z__6.i = 
		    h__[i__3].r * h31s.i + h__[i__3].i * h31s.r;
	    z__1.r = z__2.r + z__6.r, z__1.i = z__2.i + z__6.i;
	    v[2].r = z__1.r, v[2].i = z__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = h_dim1 * 3 + 3;
	    z__5.r = h__[i__1].r + h__[i__2].r, z__5.i = h__[i__1].i + h__[
		    i__2].i;
	    z__4.r = z__5.r - s1->r, z__4.i = z__5.i - s1->i;
	    z__3.r = z__4.r - s2->r, z__3.i = z__4.i - s2->i;
	    z__2.r = h31s.r * z__3.r - h31s.i * z__3.i, z__2.i = h31s.r * 
		    z__3.i + h31s.i * z__3.r;
	    i__3 = (h_dim1 << 1) + 3;
	    z__6.r = h21s.r * h__[i__3].r - h21s.i * h__[i__3].i, z__6.i = 
		    h21s.r * h__[i__3].i + h21s.i * h__[i__3].r;
	    z__1.r = z__2.r + z__6.r, z__1.i = z__2.i + z__6.i;
	    v[3].r = z__1.r, v[3].i = z__1.i;
	}
    }
    return 0;
} /* zlaqr1_ */
