/* claqr1.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int claqr1_(integer *n, complex *h__, integer *ldh, complex *
	s1, complex *s2, complex *v)
{
    /* System generated locals */
    integer h_dim1, h_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    complex q__1, q__2, q__3, q__4, q__5, q__6, q__7, q__8;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    real s;
    complex h21s, h31s;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*       Given a 2-by-2 or 3-by-3 matrix H, CLAQR1 sets v to a */
/*       scalar multiple of the first column of the product */

/*       (*)  K = (H - s1*I)*(H - s2*I) */

/*       scaling to avoid overflows and most underflows. */

/*       This is useful for starting double implicit shift bulges */
/*       in the QR algorithm. */


/*       N      (input) integer */
/*              Order of the matrix H. N must be either 2 or 3. */

/*       H      (input) COMPLEX array of dimension (LDH,N) */
/*              The 2-by-2 or 3-by-3 matrix H in (*). */

/*       LDH    (input) integer */
/*              The leading dimension of H as declared in */
/*              the calling procedure.  LDH.GE.N */

/*       S1     (input) COMPLEX */
/*       S2     S1 and S2 are the shifts defining K in (*) above. */

/*       V      (output) COMPLEX array of dimension N */
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
	q__2.r = h__[i__1].r - s2->r, q__2.i = h__[i__1].i - s2->i;
	q__1.r = q__2.r, q__1.i = q__2.i;
	i__2 = h_dim1 + 2;
	s = (r__1 = q__1.r, dabs(r__1)) + (r__2 = r_imag(&q__1), dabs(r__2)) 
		+ ((r__3 = h__[i__2].r, dabs(r__3)) + (r__4 = r_imag(&h__[
		h_dim1 + 2]), dabs(r__4)));
	if (s == 0.f) {
	    v[1].r = 0.f, v[1].i = 0.f;
	    v[2].r = 0.f, v[2].i = 0.f;
	} else {
	    i__1 = h_dim1 + 2;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h21s.r = q__1.r, h21s.i = q__1.i;
	    i__1 = (h_dim1 << 1) + 1;
	    q__2.r = h21s.r * h__[i__1].r - h21s.i * h__[i__1].i, q__2.i = 
		    h21s.r * h__[i__1].i + h21s.i * h__[i__1].r;
	    i__2 = h_dim1 + 1;
	    q__4.r = h__[i__2].r - s1->r, q__4.i = h__[i__2].i - s1->i;
	    i__3 = h_dim1 + 1;
	    q__6.r = h__[i__3].r - s2->r, q__6.i = h__[i__3].i - s2->i;
	    q__5.r = q__6.r / s, q__5.i = q__6.i / s;
	    q__3.r = q__4.r * q__5.r - q__4.i * q__5.i, q__3.i = q__4.r * 
		    q__5.i + q__4.i * q__5.r;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    v[1].r = q__1.r, v[1].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    q__4.r = h__[i__1].r + h__[i__2].r, q__4.i = h__[i__1].i + h__[
		    i__2].i;
	    q__3.r = q__4.r - s1->r, q__3.i = q__4.i - s1->i;
	    q__2.r = q__3.r - s2->r, q__2.i = q__3.i - s2->i;
	    q__1.r = h21s.r * q__2.r - h21s.i * q__2.i, q__1.i = h21s.r * 
		    q__2.i + h21s.i * q__2.r;
	    v[2].r = q__1.r, v[2].i = q__1.i;
	}
    } else {
	i__1 = h_dim1 + 1;
	q__2.r = h__[i__1].r - s2->r, q__2.i = h__[i__1].i - s2->i;
	q__1.r = q__2.r, q__1.i = q__2.i;
	i__2 = h_dim1 + 2;
	i__3 = h_dim1 + 3;
	s = (r__1 = q__1.r, dabs(r__1)) + (r__2 = r_imag(&q__1), dabs(r__2)) 
		+ ((r__3 = h__[i__2].r, dabs(r__3)) + (r__4 = r_imag(&h__[
		h_dim1 + 2]), dabs(r__4))) + ((r__5 = h__[i__3].r, dabs(r__5))
		 + (r__6 = r_imag(&h__[h_dim1 + 3]), dabs(r__6)));
	if (s == 0.f) {
	    v[1].r = 0.f, v[1].i = 0.f;
	    v[2].r = 0.f, v[2].i = 0.f;
	    v[3].r = 0.f, v[3].i = 0.f;
	} else {
	    i__1 = h_dim1 + 2;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h21s.r = q__1.r, h21s.i = q__1.i;
	    i__1 = h_dim1 + 3;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h31s.r = q__1.r, h31s.i = q__1.i;
	    i__1 = h_dim1 + 1;
	    q__4.r = h__[i__1].r - s1->r, q__4.i = h__[i__1].i - s1->i;
	    i__2 = h_dim1 + 1;
	    q__6.r = h__[i__2].r - s2->r, q__6.i = h__[i__2].i - s2->i;
	    q__5.r = q__6.r / s, q__5.i = q__6.i / s;
	    q__3.r = q__4.r * q__5.r - q__4.i * q__5.i, q__3.i = q__4.r * 
		    q__5.i + q__4.i * q__5.r;
	    i__3 = (h_dim1 << 1) + 1;
	    q__7.r = h__[i__3].r * h21s.r - h__[i__3].i * h21s.i, q__7.i = 
		    h__[i__3].r * h21s.i + h__[i__3].i * h21s.r;
	    q__2.r = q__3.r + q__7.r, q__2.i = q__3.i + q__7.i;
	    i__4 = h_dim1 * 3 + 1;
	    q__8.r = h__[i__4].r * h31s.r - h__[i__4].i * h31s.i, q__8.i = 
		    h__[i__4].r * h31s.i + h__[i__4].i * h31s.r;
	    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
	    v[1].r = q__1.r, v[1].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    q__5.r = h__[i__1].r + h__[i__2].r, q__5.i = h__[i__1].i + h__[
		    i__2].i;
	    q__4.r = q__5.r - s1->r, q__4.i = q__5.i - s1->i;
	    q__3.r = q__4.r - s2->r, q__3.i = q__4.i - s2->i;
	    q__2.r = h21s.r * q__3.r - h21s.i * q__3.i, q__2.i = h21s.r * 
		    q__3.i + h21s.i * q__3.r;
	    i__3 = h_dim1 * 3 + 2;
	    q__6.r = h__[i__3].r * h31s.r - h__[i__3].i * h31s.i, q__6.i = 
		    h__[i__3].r * h31s.i + h__[i__3].i * h31s.r;
	    q__1.r = q__2.r + q__6.r, q__1.i = q__2.i + q__6.i;
	    v[2].r = q__1.r, v[2].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = h_dim1 * 3 + 3;
	    q__5.r = h__[i__1].r + h__[i__2].r, q__5.i = h__[i__1].i + h__[
		    i__2].i;
	    q__4.r = q__5.r - s1->r, q__4.i = q__5.i - s1->i;
	    q__3.r = q__4.r - s2->r, q__3.i = q__4.i - s2->i;
	    q__2.r = h31s.r * q__3.r - h31s.i * q__3.i, q__2.i = h31s.r * 
		    q__3.i + h31s.i * q__3.r;
	    i__3 = (h_dim1 << 1) + 3;
	    q__6.r = h21s.r * h__[i__3].r - h21s.i * h__[i__3].i, q__6.i = 
		    h21s.r * h__[i__3].i + h21s.i * h__[i__3].r;
	    q__1.r = q__2.r + q__6.r, q__1.i = q__2.i + q__6.i;
	    v[3].r = q__1.r, v[3].i = q__1.i;
	}
    }
    return 0;
} /* claqr1_ */
