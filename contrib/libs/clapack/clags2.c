/* clags2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int clags2_(logical *upper, real *a1, complex *a2, real *a3, 
	real *b1, complex *b2, real *b3, real *csu, complex *snu, real *csv, 
	complex *snv, real *csq, complex *snq)
{
    /* System generated locals */
    real r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8;
    complex q__1, q__2, q__3, q__4, q__5;

    /* Builtin functions */
    double c_abs(complex *), r_imag(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    real a;
    complex b, c__;
    real d__;
    complex r__, d1;
    real s1, s2, fb, fc;
    complex ua11, ua12, ua21, ua22, vb11, vb12, vb21, vb22;
    real csl, csr, snl, snr, aua11, aua12, aua21, aua22, avb11, avb12, avb21, 
	    avb22, ua11r, ua22r, vb11r, vb22r;
    extern /* Subroutine */ int slasv2_(real *, real *, real *, real *, real *
, real *, real *, real *, real *), clartg_(complex *, complex *, 
	    real *, complex *, complex *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLAGS2 computes 2-by-2 unitary matrices U, V and Q, such */
/*  that if ( UPPER ) then */

/*            U'*A*Q = U'*( A1 A2 )*Q = ( x  0  ) */
/*                        ( 0  A3 )     ( x  x  ) */
/*  and */
/*            V'*B*Q = V'*( B1 B2 )*Q = ( x  0  ) */
/*                        ( 0  B3 )     ( x  x  ) */

/*  or if ( .NOT.UPPER ) then */

/*            U'*A*Q = U'*( A1 0  )*Q = ( x  x  ) */
/*                        ( A2 A3 )     ( 0  x  ) */
/*  and */
/*            V'*B*Q = V'*( B1 0  )*Q = ( x  x  ) */
/*                        ( B2 B3 )     ( 0  x  ) */
/*  where */

/*    U = (     CSU      SNU ), V = (     CSV     SNV ), */
/*        ( -CONJG(SNU)  CSU )      ( -CONJG(SNV) CSV ) */

/*    Q = (     CSQ      SNQ ) */
/*        ( -CONJG(SNQ)  CSQ ) */

/*  Z' denotes the conjugate transpose of Z. */

/*  The rows of the transformed A and B are parallel. Moreover, if the */
/*  input 2-by-2 matrix A is not zero, then the transformed (1,1) entry */
/*  of A is not zero. If the input matrices A and B are both not zero, */
/*  then the transformed (2,2) element of B is not zero, except when the */
/*  first rows of input A and B are parallel and the second rows are */
/*  zero. */

/*  Arguments */
/*  ========= */

/*  UPPER   (input) LOGICAL */
/*          = .TRUE.: the input matrices A and B are upper triangular. */
/*          = .FALSE.: the input matrices A and B are lower triangular. */

/*  A1      (input) REAL */
/*  A2      (input) COMPLEX */
/*  A3      (input) REAL */
/*          On entry, A1, A2 and A3 are elements of the input 2-by-2 */
/*          upper (lower) triangular matrix A. */

/*  B1      (input) REAL */
/*  B2      (input) COMPLEX */
/*  B3      (input) REAL */
/*          On entry, B1, B2 and B3 are elements of the input 2-by-2 */
/*          upper (lower) triangular matrix B. */

/*  CSU     (output) REAL */
/*  SNU     (output) COMPLEX */
/*          The desired unitary matrix U. */

/*  CSV     (output) REAL */
/*  SNV     (output) COMPLEX */
/*          The desired unitary matrix V. */

/*  CSQ     (output) REAL */
/*  SNQ     (output) COMPLEX */
/*          The desired unitary matrix Q. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    if (*upper) {

/*        Input matrices A and B are upper triangular matrices */

/*        Form matrix C = A*adj(B) = ( a b ) */
/*                                   ( 0 d ) */

	a = *a1 * *b3;
	d__ = *a3 * *b1;
	q__2.r = *b1 * a2->r, q__2.i = *b1 * a2->i;
	q__3.r = *a1 * b2->r, q__3.i = *a1 * b2->i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	b.r = q__1.r, b.i = q__1.i;
	fb = c_abs(&b);

/*        Transform complex 2-by-2 matrix C to real matrix by unitary */
/*        diagonal matrix diag(1,D1). */

	d1.r = 1.f, d1.i = 0.f;
	if (fb != 0.f) {
	    q__1.r = b.r / fb, q__1.i = b.i / fb;
	    d1.r = q__1.r, d1.i = q__1.i;
	}

/*        The SVD of real 2 by 2 triangular C */

/*         ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 ) */
/*         ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T ) */

	slasv2_(&a, &fb, &d__, &s1, &s2, &snr, &csr, &snl, &csl);

	if (dabs(csl) >= dabs(snl) || dabs(csr) >= dabs(snr)) {

/*           Compute the (1,1) and (1,2) elements of U'*A and V'*B, */
/*           and (1,2) element of |U|'*|A| and |V|'*|B|. */

	    ua11r = csl * *a1;
	    q__2.r = csl * a2->r, q__2.i = csl * a2->i;
	    q__4.r = snl * d1.r, q__4.i = snl * d1.i;
	    q__3.r = *a3 * q__4.r, q__3.i = *a3 * q__4.i;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    ua12.r = q__1.r, ua12.i = q__1.i;

	    vb11r = csr * *b1;
	    q__2.r = csr * b2->r, q__2.i = csr * b2->i;
	    q__4.r = snr * d1.r, q__4.i = snr * d1.i;
	    q__3.r = *b3 * q__4.r, q__3.i = *b3 * q__4.i;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    vb12.r = q__1.r, vb12.i = q__1.i;

	    aua12 = dabs(csl) * ((r__1 = a2->r, dabs(r__1)) + (r__2 = r_imag(
		    a2), dabs(r__2))) + dabs(snl) * dabs(*a3);
	    avb12 = dabs(csr) * ((r__1 = b2->r, dabs(r__1)) + (r__2 = r_imag(
		    b2), dabs(r__2))) + dabs(snr) * dabs(*b3);

/*           zero (1,2) elements of U'*A and V'*B */

	    if (dabs(ua11r) + ((r__1 = ua12.r, dabs(r__1)) + (r__2 = r_imag(&
		    ua12), dabs(r__2))) == 0.f) {
		q__2.r = vb11r, q__2.i = 0.f;
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &vb12);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else if (dabs(vb11r) + ((r__1 = vb12.r, dabs(r__1)) + (r__2 = 
		    r_imag(&vb12), dabs(r__2))) == 0.f) {
		q__2.r = ua11r, q__2.i = 0.f;
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &ua12);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else if (aua12 / (dabs(ua11r) + ((r__1 = ua12.r, dabs(r__1)) + (
		    r__2 = r_imag(&ua12), dabs(r__2)))) <= avb12 / (dabs(
		    vb11r) + ((r__3 = vb12.r, dabs(r__3)) + (r__4 = r_imag(&
		    vb12), dabs(r__4))))) {
		q__2.r = ua11r, q__2.i = 0.f;
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &ua12);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else {
		q__2.r = vb11r, q__2.i = 0.f;
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &vb12);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    }

	    *csu = csl;
	    q__2.r = -d1.r, q__2.i = -d1.i;
	    q__1.r = snl * q__2.r, q__1.i = snl * q__2.i;
	    snu->r = q__1.r, snu->i = q__1.i;
	    *csv = csr;
	    q__2.r = -d1.r, q__2.i = -d1.i;
	    q__1.r = snr * q__2.r, q__1.i = snr * q__2.i;
	    snv->r = q__1.r, snv->i = q__1.i;

	} else {

/*           Compute the (2,1) and (2,2) elements of U'*A and V'*B, */
/*           and (2,2) element of |U|'*|A| and |V|'*|B|. */

	    r_cnjg(&q__4, &d1);
	    q__3.r = -q__4.r, q__3.i = -q__4.i;
	    q__2.r = snl * q__3.r, q__2.i = snl * q__3.i;
	    q__1.r = *a1 * q__2.r, q__1.i = *a1 * q__2.i;
	    ua21.r = q__1.r, ua21.i = q__1.i;
	    r_cnjg(&q__5, &d1);
	    q__4.r = -q__5.r, q__4.i = -q__5.i;
	    q__3.r = snl * q__4.r, q__3.i = snl * q__4.i;
	    q__2.r = q__3.r * a2->r - q__3.i * a2->i, q__2.i = q__3.r * a2->i 
		    + q__3.i * a2->r;
	    r__1 = csl * *a3;
	    q__1.r = q__2.r + r__1, q__1.i = q__2.i;
	    ua22.r = q__1.r, ua22.i = q__1.i;

	    r_cnjg(&q__4, &d1);
	    q__3.r = -q__4.r, q__3.i = -q__4.i;
	    q__2.r = snr * q__3.r, q__2.i = snr * q__3.i;
	    q__1.r = *b1 * q__2.r, q__1.i = *b1 * q__2.i;
	    vb21.r = q__1.r, vb21.i = q__1.i;
	    r_cnjg(&q__5, &d1);
	    q__4.r = -q__5.r, q__4.i = -q__5.i;
	    q__3.r = snr * q__4.r, q__3.i = snr * q__4.i;
	    q__2.r = q__3.r * b2->r - q__3.i * b2->i, q__2.i = q__3.r * b2->i 
		    + q__3.i * b2->r;
	    r__1 = csr * *b3;
	    q__1.r = q__2.r + r__1, q__1.i = q__2.i;
	    vb22.r = q__1.r, vb22.i = q__1.i;

	    aua22 = dabs(snl) * ((r__1 = a2->r, dabs(r__1)) + (r__2 = r_imag(
		    a2), dabs(r__2))) + dabs(csl) * dabs(*a3);
	    avb22 = dabs(snr) * ((r__1 = b2->r, dabs(r__1)) + (r__2 = r_imag(
		    b2), dabs(r__2))) + dabs(csr) * dabs(*b3);

/*           zero (2,2) elements of U'*A and V'*B, and then swap. */

	    if ((r__1 = ua21.r, dabs(r__1)) + (r__2 = r_imag(&ua21), dabs(
		    r__2)) + ((r__3 = ua22.r, dabs(r__3)) + (r__4 = r_imag(&
		    ua22), dabs(r__4))) == 0.f) {
		r_cnjg(&q__2, &vb21);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &vb22);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else if ((r__1 = vb21.r, dabs(r__1)) + (r__2 = r_imag(&vb21), 
		    dabs(r__2)) + c_abs(&vb22) == 0.f) {
		r_cnjg(&q__2, &ua21);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &ua22);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else if (aua22 / ((r__1 = ua21.r, dabs(r__1)) + (r__2 = r_imag(&
		    ua21), dabs(r__2)) + ((r__3 = ua22.r, dabs(r__3)) + (r__4 
		    = r_imag(&ua22), dabs(r__4)))) <= avb22 / ((r__5 = vb21.r,
		     dabs(r__5)) + (r__6 = r_imag(&vb21), dabs(r__6)) + ((
		    r__7 = vb22.r, dabs(r__7)) + (r__8 = r_imag(&vb22), dabs(
		    r__8))))) {
		r_cnjg(&q__2, &ua21);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &ua22);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    } else {
		r_cnjg(&q__2, &vb21);
		q__1.r = -q__2.r, q__1.i = -q__2.i;
		r_cnjg(&q__3, &vb22);
		clartg_(&q__1, &q__3, csq, snq, &r__);
	    }

	    *csu = snl;
	    q__1.r = csl * d1.r, q__1.i = csl * d1.i;
	    snu->r = q__1.r, snu->i = q__1.i;
	    *csv = snr;
	    q__1.r = csr * d1.r, q__1.i = csr * d1.i;
	    snv->r = q__1.r, snv->i = q__1.i;

	}

    } else {

/*        Input matrices A and B are lower triangular matrices */

/*        Form matrix C = A*adj(B) = ( a 0 ) */
/*                                   ( c d ) */

	a = *a1 * *b3;
	d__ = *a3 * *b1;
	q__2.r = *b3 * a2->r, q__2.i = *b3 * a2->i;
	q__3.r = *a3 * b2->r, q__3.i = *a3 * b2->i;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	c__.r = q__1.r, c__.i = q__1.i;
	fc = c_abs(&c__);

/*        Transform complex 2-by-2 matrix C to real matrix by unitary */
/*        diagonal matrix diag(d1,1). */

	d1.r = 1.f, d1.i = 0.f;
	if (fc != 0.f) {
	    q__1.r = c__.r / fc, q__1.i = c__.i / fc;
	    d1.r = q__1.r, d1.i = q__1.i;
	}

/*        The SVD of real 2 by 2 triangular C */

/*         ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 ) */
/*         ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T ) */

	slasv2_(&a, &fc, &d__, &s1, &s2, &snr, &csr, &snl, &csl);

	if (dabs(csr) >= dabs(snr) || dabs(csl) >= dabs(snl)) {

/*           Compute the (2,1) and (2,2) elements of U'*A and V'*B, */
/*           and (2,1) element of |U|'*|A| and |V|'*|B|. */

	    q__4.r = -d1.r, q__4.i = -d1.i;
	    q__3.r = snr * q__4.r, q__3.i = snr * q__4.i;
	    q__2.r = *a1 * q__3.r, q__2.i = *a1 * q__3.i;
	    q__5.r = csr * a2->r, q__5.i = csr * a2->i;
	    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
	    ua21.r = q__1.r, ua21.i = q__1.i;
	    ua22r = csr * *a3;

	    q__4.r = -d1.r, q__4.i = -d1.i;
	    q__3.r = snl * q__4.r, q__3.i = snl * q__4.i;
	    q__2.r = *b1 * q__3.r, q__2.i = *b1 * q__3.i;
	    q__5.r = csl * b2->r, q__5.i = csl * b2->i;
	    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
	    vb21.r = q__1.r, vb21.i = q__1.i;
	    vb22r = csl * *b3;

	    aua21 = dabs(snr) * dabs(*a1) + dabs(csr) * ((r__1 = a2->r, dabs(
		    r__1)) + (r__2 = r_imag(a2), dabs(r__2)));
	    avb21 = dabs(snl) * dabs(*b1) + dabs(csl) * ((r__1 = b2->r, dabs(
		    r__1)) + (r__2 = r_imag(b2), dabs(r__2)));

/*           zero (2,1) elements of U'*A and V'*B. */

	    if ((r__1 = ua21.r, dabs(r__1)) + (r__2 = r_imag(&ua21), dabs(
		    r__2)) + dabs(ua22r) == 0.f) {
		q__1.r = vb22r, q__1.i = 0.f;
		clartg_(&q__1, &vb21, csq, snq, &r__);
	    } else if ((r__1 = vb21.r, dabs(r__1)) + (r__2 = r_imag(&vb21), 
		    dabs(r__2)) + dabs(vb22r) == 0.f) {
		q__1.r = ua22r, q__1.i = 0.f;
		clartg_(&q__1, &ua21, csq, snq, &r__);
	    } else if (aua21 / ((r__1 = ua21.r, dabs(r__1)) + (r__2 = r_imag(&
		    ua21), dabs(r__2)) + dabs(ua22r)) <= avb21 / ((r__3 = 
		    vb21.r, dabs(r__3)) + (r__4 = r_imag(&vb21), dabs(r__4)) 
		    + dabs(vb22r))) {
		q__1.r = ua22r, q__1.i = 0.f;
		clartg_(&q__1, &ua21, csq, snq, &r__);
	    } else {
		q__1.r = vb22r, q__1.i = 0.f;
		clartg_(&q__1, &vb21, csq, snq, &r__);
	    }

	    *csu = csr;
	    r_cnjg(&q__3, &d1);
	    q__2.r = -q__3.r, q__2.i = -q__3.i;
	    q__1.r = snr * q__2.r, q__1.i = snr * q__2.i;
	    snu->r = q__1.r, snu->i = q__1.i;
	    *csv = csl;
	    r_cnjg(&q__3, &d1);
	    q__2.r = -q__3.r, q__2.i = -q__3.i;
	    q__1.r = snl * q__2.r, q__1.i = snl * q__2.i;
	    snv->r = q__1.r, snv->i = q__1.i;

	} else {

/*           Compute the (1,1) and (1,2) elements of U'*A and V'*B, */
/*           and (1,1) element of |U|'*|A| and |V|'*|B|. */

	    r__1 = csr * *a1;
	    r_cnjg(&q__4, &d1);
	    q__3.r = snr * q__4.r, q__3.i = snr * q__4.i;
	    q__2.r = q__3.r * a2->r - q__3.i * a2->i, q__2.i = q__3.r * a2->i 
		    + q__3.i * a2->r;
	    q__1.r = r__1 + q__2.r, q__1.i = q__2.i;
	    ua11.r = q__1.r, ua11.i = q__1.i;
	    r_cnjg(&q__3, &d1);
	    q__2.r = snr * q__3.r, q__2.i = snr * q__3.i;
	    q__1.r = *a3 * q__2.r, q__1.i = *a3 * q__2.i;
	    ua12.r = q__1.r, ua12.i = q__1.i;

	    r__1 = csl * *b1;
	    r_cnjg(&q__4, &d1);
	    q__3.r = snl * q__4.r, q__3.i = snl * q__4.i;
	    q__2.r = q__3.r * b2->r - q__3.i * b2->i, q__2.i = q__3.r * b2->i 
		    + q__3.i * b2->r;
	    q__1.r = r__1 + q__2.r, q__1.i = q__2.i;
	    vb11.r = q__1.r, vb11.i = q__1.i;
	    r_cnjg(&q__3, &d1);
	    q__2.r = snl * q__3.r, q__2.i = snl * q__3.i;
	    q__1.r = *b3 * q__2.r, q__1.i = *b3 * q__2.i;
	    vb12.r = q__1.r, vb12.i = q__1.i;

	    aua11 = dabs(csr) * dabs(*a1) + dabs(snr) * ((r__1 = a2->r, dabs(
		    r__1)) + (r__2 = r_imag(a2), dabs(r__2)));
	    avb11 = dabs(csl) * dabs(*b1) + dabs(snl) * ((r__1 = b2->r, dabs(
		    r__1)) + (r__2 = r_imag(b2), dabs(r__2)));

/*           zero (1,1) elements of U'*A and V'*B, and then swap. */

	    if ((r__1 = ua11.r, dabs(r__1)) + (r__2 = r_imag(&ua11), dabs(
		    r__2)) + ((r__3 = ua12.r, dabs(r__3)) + (r__4 = r_imag(&
		    ua12), dabs(r__4))) == 0.f) {
		clartg_(&vb12, &vb11, csq, snq, &r__);
	    } else if ((r__1 = vb11.r, dabs(r__1)) + (r__2 = r_imag(&vb11), 
		    dabs(r__2)) + ((r__3 = vb12.r, dabs(r__3)) + (r__4 = 
		    r_imag(&vb12), dabs(r__4))) == 0.f) {
		clartg_(&ua12, &ua11, csq, snq, &r__);
	    } else if (aua11 / ((r__1 = ua11.r, dabs(r__1)) + (r__2 = r_imag(&
		    ua11), dabs(r__2)) + ((r__3 = ua12.r, dabs(r__3)) + (r__4 
		    = r_imag(&ua12), dabs(r__4)))) <= avb11 / ((r__5 = vb11.r,
		     dabs(r__5)) + (r__6 = r_imag(&vb11), dabs(r__6)) + ((
		    r__7 = vb12.r, dabs(r__7)) + (r__8 = r_imag(&vb12), dabs(
		    r__8))))) {
		clartg_(&ua12, &ua11, csq, snq, &r__);
	    } else {
		clartg_(&vb12, &vb11, csq, snq, &r__);
	    }

	    *csu = snr;
	    r_cnjg(&q__2, &d1);
	    q__1.r = csr * q__2.r, q__1.i = csr * q__2.i;
	    snu->r = q__1.r, snu->i = q__1.i;
	    *csv = snl;
	    r_cnjg(&q__2, &d1);
	    q__1.r = csl * q__2.r, q__1.i = csl * q__2.i;
	    snv->r = q__1.r, snv->i = q__1.i;

	}

    }

    return 0;

/*     End of CLAGS2 */

} /* clags2_ */
