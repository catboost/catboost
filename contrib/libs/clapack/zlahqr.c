/* zlahqr.f -- translated by f2c (version 20061008).
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
static integer c__2 = 2;

/* Subroutine */ int zlahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublecomplex *h__, integer *ldh, 
	doublecomplex *w, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);
    double z_abs(doublecomplex *);
    void z_sqrt(doublecomplex *, doublecomplex *), pow_zi(doublecomplex *, 
	    doublecomplex *, integer *);

    /* Local variables */
    integer i__, j, k, l, m;
    doublereal s;
    doublecomplex t, u, v[2], x, y;
    integer i1, i2;
    doublecomplex t1;
    doublereal t2;
    doublecomplex v2;
    doublereal aa, ab, ba, bb, h10;
    doublecomplex h11;
    doublereal h21;
    doublecomplex h22, sc;
    integer nh, nz;
    doublereal sx;
    integer jhi;
    doublecomplex h11s;
    integer jlo, its;
    doublereal ulp;
    doublecomplex sum;
    doublereal tst;
    doublecomplex temp;
    extern /* Subroutine */ int zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *);
    doublereal rtemp;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *), dlabad_(doublereal *, doublereal *);
    extern doublereal dlamch_(char *);
    doublereal safmin, safmax;
    extern /* Subroutine */ int zlarfg_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *);
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *, 
	     doublecomplex *);
    doublereal smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     Purpose */
/*     ======= */

/*     ZLAHQR is an auxiliary routine called by CHSEQR to update the */
/*     eigenvalues and Schur decomposition already computed by CHSEQR, by */
/*     dealing with the Hessenberg submatrix in rows and columns ILO to */
/*     IHI. */

/*     Arguments */
/*     ========= */

/*     WANTT   (input) LOGICAL */
/*          = .TRUE. : the full Schur form T is required; */
/*          = .FALSE.: only eigenvalues are required. */

/*     WANTZ   (input) LOGICAL */
/*          = .TRUE. : the matrix of Schur vectors Z is required; */
/*          = .FALSE.: Schur vectors are not required. */

/*     N       (input) INTEGER */
/*          The order of the matrix H.  N >= 0. */

/*     ILO     (input) INTEGER */
/*     IHI     (input) INTEGER */
/*          It is assumed that H is already upper triangular in rows and */
/*          columns IHI+1:N, and that H(ILO,ILO-1) = 0 (unless ILO = 1). */
/*          ZLAHQR works primarily with the Hessenberg submatrix in rows */
/*          and columns ILO to IHI, but applies transformations to all of */
/*          H if WANTT is .TRUE.. */
/*          1 <= ILO <= max(1,IHI); IHI <= N. */

/*     H       (input/output) COMPLEX*16 array, dimension (LDH,N) */
/*          On entry, the upper Hessenberg matrix H. */
/*          On exit, if INFO is zero and if WANTT is .TRUE., then H */
/*          is upper triangular in rows and columns ILO:IHI.  If INFO */
/*          is zero and if WANTT is .FALSE., then the contents of H */
/*          are unspecified on exit.  The output state of H in case */
/*          INF is positive is below under the description of INFO. */

/*     LDH     (input) INTEGER */
/*          The leading dimension of the array H. LDH >= max(1,N). */

/*     W       (output) COMPLEX*16 array, dimension (N) */
/*          The computed eigenvalues ILO to IHI are stored in the */
/*          corresponding elements of W. If WANTT is .TRUE., the */
/*          eigenvalues are stored in the same order as on the diagonal */
/*          of the Schur form returned in H, with W(i) = H(i,i). */

/*     ILOZ    (input) INTEGER */
/*     IHIZ    (input) INTEGER */
/*          Specify the rows of Z to which transformations must be */
/*          applied if WANTZ is .TRUE.. */
/*          1 <= ILOZ <= ILO; IHI <= IHIZ <= N. */

/*     Z       (input/output) COMPLEX*16 array, dimension (LDZ,N) */
/*          If WANTZ is .TRUE., on entry Z must contain the current */
/*          matrix Z of transformations accumulated by CHSEQR, and on */
/*          exit Z has been updated; transformations are applied only to */
/*          the submatrix Z(ILOZ:IHIZ,ILO:IHI). */
/*          If WANTZ is .FALSE., Z is not referenced. */

/*     LDZ     (input) INTEGER */
/*          The leading dimension of the array Z. LDZ >= max(1,N). */

/*     INFO    (output) INTEGER */
/*           =   0: successful exit */
/*          .GT. 0: if INFO = i, ZLAHQR failed to compute all the */
/*                  eigenvalues ILO to IHI in a total of 30 iterations */
/*                  per eigenvalue; elements i+1:ihi of W contain */
/*                  those eigenvalues which have been successfully */
/*                  computed. */

/*                  If INFO .GT. 0 and WANTT is .FALSE., then on exit, */
/*                  the remaining unconverged eigenvalues are the */
/*                  eigenvalues of the upper Hessenberg matrix */
/*                  rows and columns ILO thorugh INFO of the final, */
/*                  output value of H. */

/*                  If INFO .GT. 0 and WANTT is .TRUE., then on exit */
/*          (*)       (initial value of H)*U  = U*(final value of H) */
/*                  where U is an orthognal matrix.    The final */
/*                  value of H is upper Hessenberg and triangular in */
/*                  rows and columns INFO+1 through IHI. */

/*                  If INFO .GT. 0 and WANTZ is .TRUE., then on exit */
/*                      (final value of Z)  = (initial value of Z)*U */
/*                  where U is the orthogonal matrix in (*) */
/*                  (regardless of the value of WANTT.) */

/*     Further Details */
/*     =============== */

/*     02-96 Based on modifications by */
/*     David Day, Sandia National Laboratory, USA */

/*     12-04 Further modifications by */
/*     Ralph Byers, University of Kansas, USA */
/*     This is a modified version of ZLAHQR from LAPACK version 3.0. */
/*     It is (1) more robust against overflow and underflow and */
/*     (2) adopts the more conservative Ahues & Tisseur stopping */
/*     criterion (LAWN 122, 1997). */

/*     ========================================================= */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    if (*ilo == *ihi) {
	i__1 = *ilo;
	i__2 = *ilo + *ilo * h_dim1;
	w[i__1].r = h__[i__2].r, w[i__1].i = h__[i__2].i;
	return 0;
    }

/*     ==== clear out the trash ==== */
    i__1 = *ihi - 3;
    for (j = *ilo; j <= i__1; ++j) {
	i__2 = j + 2 + j * h_dim1;
	h__[i__2].r = 0., h__[i__2].i = 0.;
	i__2 = j + 3 + j * h_dim1;
	h__[i__2].r = 0., h__[i__2].i = 0.;
/* L10: */
    }
    if (*ilo <= *ihi - 2) {
	i__1 = *ihi + (*ihi - 2) * h_dim1;
	h__[i__1].r = 0., h__[i__1].i = 0.;
    }
/*     ==== ensure that subdiagonal entries are real ==== */
    if (*wantt) {
	jlo = 1;
	jhi = *n;
    } else {
	jlo = *ilo;
	jhi = *ihi;
    }
    i__1 = *ihi;
    for (i__ = *ilo + 1; i__ <= i__1; ++i__) {
	if (d_imag(&h__[i__ + (i__ - 1) * h_dim1]) != 0.) {
/*           ==== The following redundant normalization */
/*           .    avoids problems with both gradual and */
/*           .    sudden underflow in ABS(H(I,I-1)) ==== */
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    i__3 = i__ + (i__ - 1) * h_dim1;
	    d__3 = (d__1 = h__[i__3].r, abs(d__1)) + (d__2 = d_imag(&h__[i__ 
		    + (i__ - 1) * h_dim1]), abs(d__2));
	    z__1.r = h__[i__2].r / d__3, z__1.i = h__[i__2].i / d__3;
	    sc.r = z__1.r, sc.i = z__1.i;
	    d_cnjg(&z__2, &sc);
	    d__1 = z_abs(&sc);
	    z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	    sc.r = z__1.r, sc.i = z__1.i;
	    i__2 = i__ + (i__ - 1) * h_dim1;
	    d__1 = z_abs(&h__[i__ + (i__ - 1) * h_dim1]);
	    h__[i__2].r = d__1, h__[i__2].i = 0.;
	    i__2 = jhi - i__ + 1;
	    zscal_(&i__2, &sc, &h__[i__ + i__ * h_dim1], ldh);
/* Computing MIN */
	    i__3 = jhi, i__4 = i__ + 1;
	    i__2 = min(i__3,i__4) - jlo + 1;
	    d_cnjg(&z__1, &sc);
	    zscal_(&i__2, &z__1, &h__[jlo + i__ * h_dim1], &c__1);
	    if (*wantz) {
		i__2 = *ihiz - *iloz + 1;
		d_cnjg(&z__1, &sc);
		zscal_(&i__2, &z__1, &z__[*iloz + i__ * z_dim1], &c__1);
	    }
	}
/* L20: */
    }

    nh = *ihi - *ilo + 1;
    nz = *ihiz - *iloz + 1;

/*     Set machine-dependent constants for the stopping criterion. */

    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((doublereal) nh / ulp);

/*     I1 and I2 are the indices of the first row and last column of H */
/*     to which transformations must be applied. If eigenvalues only are */
/*     being computed, I1 and I2 are set inside the main loop. */

    if (*wantt) {
	i1 = 1;
	i2 = *n;
    }

/*     The main loop begins here. I is the loop index and decreases from */
/*     IHI to ILO in steps of 1. Each iteration of the loop works */
/*     with the active submatrix in rows and columns L to I. */
/*     Eigenvalues I+1 to IHI have already converged. Either L = ILO, or */
/*     H(L,L-1) is negligible so that the matrix splits. */

    i__ = *ihi;
L30:
    if (i__ < *ilo) {
	goto L150;
    }

/*     Perform QR iterations on rows and columns ILO to I until a */
/*     submatrix of order 1 splits off at the bottom because a */
/*     subdiagonal element has become negligible. */

    l = *ilo;
    for (its = 0; its <= 30; ++its) {

/*        Look for a single small subdiagonal element. */

	i__1 = l + 1;
	for (k = i__; k >= i__1; --k) {
	    i__2 = k + (k - 1) * h_dim1;
	    if ((d__1 = h__[i__2].r, abs(d__1)) + (d__2 = d_imag(&h__[k + (k 
		    - 1) * h_dim1]), abs(d__2)) <= smlnum) {
		goto L50;
	    }
	    i__2 = k - 1 + (k - 1) * h_dim1;
	    i__3 = k + k * h_dim1;
	    tst = (d__1 = h__[i__2].r, abs(d__1)) + (d__2 = d_imag(&h__[k - 1 
		    + (k - 1) * h_dim1]), abs(d__2)) + ((d__3 = h__[i__3].r, 
		    abs(d__3)) + (d__4 = d_imag(&h__[k + k * h_dim1]), abs(
		    d__4)));
	    if (tst == 0.) {
		if (k - 2 >= *ilo) {
		    i__2 = k - 1 + (k - 2) * h_dim1;
		    tst += (d__1 = h__[i__2].r, abs(d__1));
		}
		if (k + 1 <= *ihi) {
		    i__2 = k + 1 + k * h_dim1;
		    tst += (d__1 = h__[i__2].r, abs(d__1));
		}
	    }
/*           ==== The following is a conservative small subdiagonal */
/*           .    deflation criterion due to Ahues & Tisseur (LAWN 122, */
/*           .    1997). It has better mathematical foundation and */
/*           .    improves accuracy in some examples.  ==== */
	    i__2 = k + (k - 1) * h_dim1;
	    if ((d__1 = h__[i__2].r, abs(d__1)) <= ulp * tst) {
/* Computing MAX */
		i__2 = k + (k - 1) * h_dim1;
		i__3 = k - 1 + k * h_dim1;
		d__5 = (d__1 = h__[i__2].r, abs(d__1)) + (d__2 = d_imag(&h__[
			k + (k - 1) * h_dim1]), abs(d__2)), d__6 = (d__3 = 
			h__[i__3].r, abs(d__3)) + (d__4 = d_imag(&h__[k - 1 + 
			k * h_dim1]), abs(d__4));
		ab = max(d__5,d__6);
/* Computing MIN */
		i__2 = k + (k - 1) * h_dim1;
		i__3 = k - 1 + k * h_dim1;
		d__5 = (d__1 = h__[i__2].r, abs(d__1)) + (d__2 = d_imag(&h__[
			k + (k - 1) * h_dim1]), abs(d__2)), d__6 = (d__3 = 
			h__[i__3].r, abs(d__3)) + (d__4 = d_imag(&h__[k - 1 + 
			k * h_dim1]), abs(d__4));
		ba = min(d__5,d__6);
		i__2 = k - 1 + (k - 1) * h_dim1;
		i__3 = k + k * h_dim1;
		z__2.r = h__[i__2].r - h__[i__3].r, z__2.i = h__[i__2].i - 
			h__[i__3].i;
		z__1.r = z__2.r, z__1.i = z__2.i;
/* Computing MAX */
		i__4 = k + k * h_dim1;
		d__5 = (d__1 = h__[i__4].r, abs(d__1)) + (d__2 = d_imag(&h__[
			k + k * h_dim1]), abs(d__2)), d__6 = (d__3 = z__1.r, 
			abs(d__3)) + (d__4 = d_imag(&z__1), abs(d__4));
		aa = max(d__5,d__6);
		i__2 = k - 1 + (k - 1) * h_dim1;
		i__3 = k + k * h_dim1;
		z__2.r = h__[i__2].r - h__[i__3].r, z__2.i = h__[i__2].i - 
			h__[i__3].i;
		z__1.r = z__2.r, z__1.i = z__2.i;
/* Computing MIN */
		i__4 = k + k * h_dim1;
		d__5 = (d__1 = h__[i__4].r, abs(d__1)) + (d__2 = d_imag(&h__[
			k + k * h_dim1]), abs(d__2)), d__6 = (d__3 = z__1.r, 
			abs(d__3)) + (d__4 = d_imag(&z__1), abs(d__4));
		bb = min(d__5,d__6);
		s = aa + ab;
/* Computing MAX */
		d__1 = smlnum, d__2 = ulp * (bb * (aa / s));
		if (ba * (ab / s) <= max(d__1,d__2)) {
		    goto L50;
		}
	    }
/* L40: */
	}
L50:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible */

	    i__1 = l + (l - 1) * h_dim1;
	    h__[i__1].r = 0., h__[i__1].i = 0.;
	}

/*        Exit from loop if a submatrix of order 1 has split off. */

	if (l >= i__) {
	    goto L140;
	}

/*        Now the active submatrix is in rows and columns L to I. If */
/*        eigenvalues only are being computed, only the active submatrix */
/*        need be transformed. */

	if (! (*wantt)) {
	    i1 = l;
	    i2 = i__;
	}

	if (its == 10) {

/*           Exceptional shift. */

	    i__1 = l + 1 + l * h_dim1;
	    s = (d__1 = h__[i__1].r, abs(d__1)) * .75;
	    i__1 = l + l * h_dim1;
	    z__1.r = s + h__[i__1].r, z__1.i = h__[i__1].i;
	    t.r = z__1.r, t.i = z__1.i;
	} else if (its == 20) {

/*           Exceptional shift. */

	    i__1 = i__ + (i__ - 1) * h_dim1;
	    s = (d__1 = h__[i__1].r, abs(d__1)) * .75;
	    i__1 = i__ + i__ * h_dim1;
	    z__1.r = s + h__[i__1].r, z__1.i = h__[i__1].i;
	    t.r = z__1.r, t.i = z__1.i;
	} else {

/*           Wilkinson's shift. */

	    i__1 = i__ + i__ * h_dim1;
	    t.r = h__[i__1].r, t.i = h__[i__1].i;
	    z_sqrt(&z__2, &h__[i__ - 1 + i__ * h_dim1]);
	    z_sqrt(&z__3, &h__[i__ + (i__ - 1) * h_dim1]);
	    z__1.r = z__2.r * z__3.r - z__2.i * z__3.i, z__1.i = z__2.r * 
		    z__3.i + z__2.i * z__3.r;
	    u.r = z__1.r, u.i = z__1.i;
	    s = (d__1 = u.r, abs(d__1)) + (d__2 = d_imag(&u), abs(d__2));
	    if (s != 0.) {
		i__1 = i__ - 1 + (i__ - 1) * h_dim1;
		z__2.r = h__[i__1].r - t.r, z__2.i = h__[i__1].i - t.i;
		z__1.r = z__2.r * .5, z__1.i = z__2.i * .5;
		x.r = z__1.r, x.i = z__1.i;
		sx = (d__1 = x.r, abs(d__1)) + (d__2 = d_imag(&x), abs(d__2));
/* Computing MAX */
		d__3 = s, d__4 = (d__1 = x.r, abs(d__1)) + (d__2 = d_imag(&x),
			 abs(d__2));
		s = max(d__3,d__4);
		z__5.r = x.r / s, z__5.i = x.i / s;
		pow_zi(&z__4, &z__5, &c__2);
		z__7.r = u.r / s, z__7.i = u.i / s;
		pow_zi(&z__6, &z__7, &c__2);
		z__3.r = z__4.r + z__6.r, z__3.i = z__4.i + z__6.i;
		z_sqrt(&z__2, &z__3);
		z__1.r = s * z__2.r, z__1.i = s * z__2.i;
		y.r = z__1.r, y.i = z__1.i;
		if (sx > 0.) {
		    z__1.r = x.r / sx, z__1.i = x.i / sx;
		    z__2.r = x.r / sx, z__2.i = x.i / sx;
		    if (z__1.r * y.r + d_imag(&z__2) * d_imag(&y) < 0.) {
			z__3.r = -y.r, z__3.i = -y.i;
			y.r = z__3.r, y.i = z__3.i;
		    }
		}
		z__4.r = x.r + y.r, z__4.i = x.i + y.i;
		zladiv_(&z__3, &u, &z__4);
		z__2.r = u.r * z__3.r - u.i * z__3.i, z__2.i = u.r * z__3.i + 
			u.i * z__3.r;
		z__1.r = t.r - z__2.r, z__1.i = t.i - z__2.i;
		t.r = z__1.r, t.i = z__1.i;
	    }
	}

/*        Look for two consecutive small subdiagonal elements. */

	i__1 = l + 1;
	for (m = i__ - 1; m >= i__1; --m) {

/*           Determine the effect of starting the single-shift QR */
/*           iteration at row M, and see if this would make H(M,M-1) */
/*           negligible. */

	    i__2 = m + m * h_dim1;
	    h11.r = h__[i__2].r, h11.i = h__[i__2].i;
	    i__2 = m + 1 + (m + 1) * h_dim1;
	    h22.r = h__[i__2].r, h22.i = h__[i__2].i;
	    z__1.r = h11.r - t.r, z__1.i = h11.i - t.i;
	    h11s.r = z__1.r, h11s.i = z__1.i;
	    i__2 = m + 1 + m * h_dim1;
	    h21 = h__[i__2].r;
	    s = (d__1 = h11s.r, abs(d__1)) + (d__2 = d_imag(&h11s), abs(d__2))
		     + abs(h21);
	    z__1.r = h11s.r / s, z__1.i = h11s.i / s;
	    h11s.r = z__1.r, h11s.i = z__1.i;
	    h21 /= s;
	    v[0].r = h11s.r, v[0].i = h11s.i;
	    v[1].r = h21, v[1].i = 0.;
	    i__2 = m + (m - 1) * h_dim1;
	    h10 = h__[i__2].r;
	    if (abs(h10) * abs(h21) <= ulp * (((d__1 = h11s.r, abs(d__1)) + (
		    d__2 = d_imag(&h11s), abs(d__2))) * ((d__3 = h11.r, abs(
		    d__3)) + (d__4 = d_imag(&h11), abs(d__4)) + ((d__5 = 
		    h22.r, abs(d__5)) + (d__6 = d_imag(&h22), abs(d__6)))))) {
		goto L70;
	    }
/* L60: */
	}
	i__1 = l + l * h_dim1;
	h11.r = h__[i__1].r, h11.i = h__[i__1].i;
	i__1 = l + 1 + (l + 1) * h_dim1;
	h22.r = h__[i__1].r, h22.i = h__[i__1].i;
	z__1.r = h11.r - t.r, z__1.i = h11.i - t.i;
	h11s.r = z__1.r, h11s.i = z__1.i;
	i__1 = l + 1 + l * h_dim1;
	h21 = h__[i__1].r;
	s = (d__1 = h11s.r, abs(d__1)) + (d__2 = d_imag(&h11s), abs(d__2)) + 
		abs(h21);
	z__1.r = h11s.r / s, z__1.i = h11s.i / s;
	h11s.r = z__1.r, h11s.i = z__1.i;
	h21 /= s;
	v[0].r = h11s.r, v[0].i = h11s.i;
	v[1].r = h21, v[1].i = 0.;
L70:

/*        Single-shift QR step */

	i__1 = i__ - 1;
	for (k = m; k <= i__1; ++k) {

/*           The first iteration of this loop determines a reflection G */
/*           from the vector V and applies it from left and right to H, */
/*           thus creating a nonzero bulge below the subdiagonal. */

/*           Each subsequent iteration determines a reflection G to */
/*           restore the Hessenberg form in the (K-1)th column, and thus */
/*           chases the bulge one step toward the bottom of the active */
/*           submatrix. */

/*           V(2) is always real before the call to ZLARFG, and hence */
/*           after the call T2 ( = T1*V(2) ) is also real. */

	    if (k > m) {
		zcopy_(&c__2, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    zlarfg_(&c__2, v, &v[1], &c__1, &t1);
	    if (k > m) {
		i__2 = k + (k - 1) * h_dim1;
		h__[i__2].r = v[0].r, h__[i__2].i = v[0].i;
		i__2 = k + 1 + (k - 1) * h_dim1;
		h__[i__2].r = 0., h__[i__2].i = 0.;
	    }
	    v2.r = v[1].r, v2.i = v[1].i;
	    z__1.r = t1.r * v2.r - t1.i * v2.i, z__1.i = t1.r * v2.i + t1.i * 
		    v2.r;
	    t2 = z__1.r;

/*           Apply G from the left to transform the rows of the matrix */
/*           in columns K to I2. */

	    i__2 = i2;
	    for (j = k; j <= i__2; ++j) {
		d_cnjg(&z__3, &t1);
		i__3 = k + j * h_dim1;
		z__2.r = z__3.r * h__[i__3].r - z__3.i * h__[i__3].i, z__2.i =
			 z__3.r * h__[i__3].i + z__3.i * h__[i__3].r;
		i__4 = k + 1 + j * h_dim1;
		z__4.r = t2 * h__[i__4].r, z__4.i = t2 * h__[i__4].i;
		z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		sum.r = z__1.r, sum.i = z__1.i;
		i__3 = k + j * h_dim1;
		i__4 = k + j * h_dim1;
		z__1.r = h__[i__4].r - sum.r, z__1.i = h__[i__4].i - sum.i;
		h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
		i__3 = k + 1 + j * h_dim1;
		i__4 = k + 1 + j * h_dim1;
		z__2.r = sum.r * v2.r - sum.i * v2.i, z__2.i = sum.r * v2.i + 
			sum.i * v2.r;
		z__1.r = h__[i__4].r - z__2.r, z__1.i = h__[i__4].i - z__2.i;
		h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
/* L80: */
	    }

/*           Apply G from the right to transform the columns of the */
/*           matrix in rows I1 to min(K+2,I). */

/* Computing MIN */
	    i__3 = k + 2;
	    i__2 = min(i__3,i__);
	    for (j = i1; j <= i__2; ++j) {
		i__3 = j + k * h_dim1;
		z__2.r = t1.r * h__[i__3].r - t1.i * h__[i__3].i, z__2.i = 
			t1.r * h__[i__3].i + t1.i * h__[i__3].r;
		i__4 = j + (k + 1) * h_dim1;
		z__3.r = t2 * h__[i__4].r, z__3.i = t2 * h__[i__4].i;
		z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		sum.r = z__1.r, sum.i = z__1.i;
		i__3 = j + k * h_dim1;
		i__4 = j + k * h_dim1;
		z__1.r = h__[i__4].r - sum.r, z__1.i = h__[i__4].i - sum.i;
		h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
		i__3 = j + (k + 1) * h_dim1;
		i__4 = j + (k + 1) * h_dim1;
		d_cnjg(&z__3, &v2);
		z__2.r = sum.r * z__3.r - sum.i * z__3.i, z__2.i = sum.r * 
			z__3.i + sum.i * z__3.r;
		z__1.r = h__[i__4].r - z__2.r, z__1.i = h__[i__4].i - z__2.i;
		h__[i__3].r = z__1.r, h__[i__3].i = z__1.i;
/* L90: */
	    }

	    if (*wantz) {

/*              Accumulate transformations in the matrix Z */

		i__2 = *ihiz;
		for (j = *iloz; j <= i__2; ++j) {
		    i__3 = j + k * z_dim1;
		    z__2.r = t1.r * z__[i__3].r - t1.i * z__[i__3].i, z__2.i =
			     t1.r * z__[i__3].i + t1.i * z__[i__3].r;
		    i__4 = j + (k + 1) * z_dim1;
		    z__3.r = t2 * z__[i__4].r, z__3.i = t2 * z__[i__4].i;
		    z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		    sum.r = z__1.r, sum.i = z__1.i;
		    i__3 = j + k * z_dim1;
		    i__4 = j + k * z_dim1;
		    z__1.r = z__[i__4].r - sum.r, z__1.i = z__[i__4].i - 
			    sum.i;
		    z__[i__3].r = z__1.r, z__[i__3].i = z__1.i;
		    i__3 = j + (k + 1) * z_dim1;
		    i__4 = j + (k + 1) * z_dim1;
		    d_cnjg(&z__3, &v2);
		    z__2.r = sum.r * z__3.r - sum.i * z__3.i, z__2.i = sum.r *
			     z__3.i + sum.i * z__3.r;
		    z__1.r = z__[i__4].r - z__2.r, z__1.i = z__[i__4].i - 
			    z__2.i;
		    z__[i__3].r = z__1.r, z__[i__3].i = z__1.i;
/* L100: */
		}
	    }

	    if (k == m && m > l) {

/*              If the QR step was started at row M > L because two */
/*              consecutive small subdiagonals were found, then extra */
/*              scaling must be performed to ensure that H(M,M-1) remains */
/*              real. */

		z__1.r = 1. - t1.r, z__1.i = 0. - t1.i;
		temp.r = z__1.r, temp.i = z__1.i;
		d__1 = z_abs(&temp);
		z__1.r = temp.r / d__1, z__1.i = temp.i / d__1;
		temp.r = z__1.r, temp.i = z__1.i;
		i__2 = m + 1 + m * h_dim1;
		i__3 = m + 1 + m * h_dim1;
		d_cnjg(&z__2, &temp);
		z__1.r = h__[i__3].r * z__2.r - h__[i__3].i * z__2.i, z__1.i =
			 h__[i__3].r * z__2.i + h__[i__3].i * z__2.r;
		h__[i__2].r = z__1.r, h__[i__2].i = z__1.i;
		if (m + 2 <= i__) {
		    i__2 = m + 2 + (m + 1) * h_dim1;
		    i__3 = m + 2 + (m + 1) * h_dim1;
		    z__1.r = h__[i__3].r * temp.r - h__[i__3].i * temp.i, 
			    z__1.i = h__[i__3].r * temp.i + h__[i__3].i * 
			    temp.r;
		    h__[i__2].r = z__1.r, h__[i__2].i = z__1.i;
		}
		i__2 = i__;
		for (j = m; j <= i__2; ++j) {
		    if (j != m + 1) {
			if (i2 > j) {
			    i__3 = i2 - j;
			    zscal_(&i__3, &temp, &h__[j + (j + 1) * h_dim1], 
				    ldh);
			}
			i__3 = j - i1;
			d_cnjg(&z__1, &temp);
			zscal_(&i__3, &z__1, &h__[i1 + j * h_dim1], &c__1);
			if (*wantz) {
			    d_cnjg(&z__1, &temp);
			    zscal_(&nz, &z__1, &z__[*iloz + j * z_dim1], &
				    c__1);
			}
		    }
/* L110: */
		}
	    }
/* L120: */
	}

/*        Ensure that H(I,I-1) is real. */

	i__1 = i__ + (i__ - 1) * h_dim1;
	temp.r = h__[i__1].r, temp.i = h__[i__1].i;
	if (d_imag(&temp) != 0.) {
	    rtemp = z_abs(&temp);
	    i__1 = i__ + (i__ - 1) * h_dim1;
	    h__[i__1].r = rtemp, h__[i__1].i = 0.;
	    z__1.r = temp.r / rtemp, z__1.i = temp.i / rtemp;
	    temp.r = z__1.r, temp.i = z__1.i;
	    if (i2 > i__) {
		i__1 = i2 - i__;
		d_cnjg(&z__1, &temp);
		zscal_(&i__1, &z__1, &h__[i__ + (i__ + 1) * h_dim1], ldh);
	    }
	    i__1 = i__ - i1;
	    zscal_(&i__1, &temp, &h__[i1 + i__ * h_dim1], &c__1);
	    if (*wantz) {
		zscal_(&nz, &temp, &z__[*iloz + i__ * z_dim1], &c__1);
	    }
	}

/* L130: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L140:

/*     H(I,I-1) is negligible: one eigenvalue has converged. */

    i__1 = i__;
    i__2 = i__ + i__ * h_dim1;
    w[i__1].r = h__[i__2].r, w[i__1].i = h__[i__2].i;

/*     return to start of the main loop with new value of I. */

    i__ = l - 1;
    goto L30;

L150:
    return 0;

/*     End of ZLAHQR */

} /* zlahqr_ */
