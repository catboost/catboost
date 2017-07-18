/* slahqr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *
	info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, k, l, m;
    real s, v[3];
    integer i1, i2;
    real t1, t2, t3, v2, v3, aa, ab, ba, bb, h11, h12, h21, h22, cs;
    integer nh;
    real sn;
    integer nr;
    real tr;
    integer nz;
    real det, h21s;
    integer its;
    real ulp, sum, tst, rt1i, rt2i, rt1r, rt2r;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *, 
	    integer *, real *, real *), scopy_(integer *, real *, integer *, 
	    real *, integer *), slanv2_(real *, real *, real *, real *, real *
, real *, real *, real *, real *, real *), slabad_(real *, real *)
	    ;
    extern doublereal slamch_(char *);
    real safmin;
    extern /* Subroutine */ int slarfg_(integer *, real *, real *, integer *, 
	    real *);
    real safmax, rtdisc, smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     Purpose */
/*     ======= */

/*     SLAHQR is an auxiliary routine called by SHSEQR to update the */
/*     eigenvalues and Schur decomposition already computed by SHSEQR, by */
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
/*          It is assumed that H is already upper quasi-triangular in */
/*          rows and columns IHI+1:N, and that H(ILO,ILO-1) = 0 (unless */
/*          ILO = 1). SLAHQR works primarily with the Hessenberg */
/*          submatrix in rows and columns ILO to IHI, but applies */
/*          transformations to all of H if WANTT is .TRUE.. */
/*          1 <= ILO <= max(1,IHI); IHI <= N. */

/*     H       (input/output) REAL array, dimension (LDH,N) */
/*          On entry, the upper Hessenberg matrix H. */
/*          On exit, if INFO is zero and if WANTT is .TRUE., H is upper */
/*          quasi-triangular in rows and columns ILO:IHI, with any */
/*          2-by-2 diagonal blocks in standard form. If INFO is zero */
/*          and WANTT is .FALSE., the contents of H are unspecified on */
/*          exit.  The output state of H if INFO is nonzero is given */
/*          below under the description of INFO. */

/*     LDH     (input) INTEGER */
/*          The leading dimension of the array H. LDH >= max(1,N). */

/*     WR      (output) REAL array, dimension (N) */
/*     WI      (output) REAL array, dimension (N) */
/*          The real and imaginary parts, respectively, of the computed */
/*          eigenvalues ILO to IHI are stored in the corresponding */
/*          elements of WR and WI. If two eigenvalues are computed as a */
/*          complex conjugate pair, they are stored in consecutive */
/*          elements of WR and WI, say the i-th and (i+1)th, with */
/*          WI(i) > 0 and WI(i+1) < 0. If WANTT is .TRUE., the */
/*          eigenvalues are stored in the same order as on the diagonal */
/*          of the Schur form returned in H, with WR(i) = H(i,i), and, if */
/*          H(i:i+1,i:i+1) is a 2-by-2 diagonal block, */
/*          WI(i) = sqrt(H(i+1,i)*H(i,i+1)) and WI(i+1) = -WI(i). */

/*     ILOZ    (input) INTEGER */
/*     IHIZ    (input) INTEGER */
/*          Specify the rows of Z to which transformations must be */
/*          applied if WANTZ is .TRUE.. */
/*          1 <= ILOZ <= ILO; IHI <= IHIZ <= N. */

/*     Z       (input/output) REAL array, dimension (LDZ,N) */
/*          If WANTZ is .TRUE., on entry Z must contain the current */
/*          matrix Z of transformations accumulated by SHSEQR, and on */
/*          exit Z has been updated; transformations are applied only to */
/*          the submatrix Z(ILOZ:IHIZ,ILO:IHI). */
/*          If WANTZ is .FALSE., Z is not referenced. */

/*     LDZ     (input) INTEGER */
/*          The leading dimension of the array Z. LDZ >= max(1,N). */

/*     INFO    (output) INTEGER */
/*           =   0: successful exit */
/*          .GT. 0: If INFO = i, SLAHQR failed to compute all the */
/*                  eigenvalues ILO to IHI in a total of 30 iterations */
/*                  per eigenvalue; elements i+1:ihi of WR and WI */
/*                  contain those eigenvalues which have been */
/*                  successfully computed. */

/*                  If INFO .GT. 0 and WANTT is .FALSE., then on exit, */
/*                  the remaining unconverged eigenvalues are the */
/*                  eigenvalues of the upper Hessenberg matrix rows */
/*                  and columns ILO thorugh INFO of the final, output */
/*                  value of H. */

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
/*     This is a modified version of SLAHQR from LAPACK version 3.0. */
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
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
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
	wr[*ilo] = h__[*ilo + *ilo * h_dim1];
	wi[*ilo] = 0.f;
	return 0;
    }

/*     ==== clear out the trash ==== */
    i__1 = *ihi - 3;
    for (j = *ilo; j <= i__1; ++j) {
	h__[j + 2 + j * h_dim1] = 0.f;
	h__[j + 3 + j * h_dim1] = 0.f;
/* L10: */
    }
    if (*ilo <= *ihi - 2) {
	h__[*ihi + (*ihi - 2) * h_dim1] = 0.f;
    }

    nh = *ihi - *ilo + 1;
    nz = *ihiz - *iloz + 1;

/*     Set machine-dependent constants for the stopping criterion. */

    safmin = slamch_("SAFE MINIMUM");
    safmax = 1.f / safmin;
    slabad_(&safmin, &safmax);
    ulp = slamch_("PRECISION");
    smlnum = safmin * ((real) nh / ulp);

/*     I1 and I2 are the indices of the first row and last column of H */
/*     to which transformations must be applied. If eigenvalues only are */
/*     being computed, I1 and I2 are set inside the main loop. */

    if (*wantt) {
	i1 = 1;
	i2 = *n;
    }

/*     The main loop begins here. I is the loop index and decreases from */
/*     IHI to ILO in steps of 1 or 2. Each iteration of the loop works */
/*     with the active submatrix in rows and columns L to I. */
/*     Eigenvalues I+1 to IHI have already converged. Either L = ILO or */
/*     H(L,L-1) is negligible so that the matrix splits. */

    i__ = *ihi;
L20:
    l = *ilo;
    if (i__ < *ilo) {
	goto L160;
    }

/*     Perform QR iterations on rows and columns ILO to I until a */
/*     submatrix of order 1 or 2 splits off at the bottom because a */
/*     subdiagonal element has become negligible. */

    for (its = 0; its <= 30; ++its) {

/*        Look for a single small subdiagonal element. */

	i__1 = l + 1;
	for (k = i__; k >= i__1; --k) {
	    if ((r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)) <= smlnum) {
		goto L40;
	    }
	    tst = (r__1 = h__[k - 1 + (k - 1) * h_dim1], dabs(r__1)) + (r__2 =
		     h__[k + k * h_dim1], dabs(r__2));
	    if (tst == 0.f) {
		if (k - 2 >= *ilo) {
		    tst += (r__1 = h__[k - 1 + (k - 2) * h_dim1], dabs(r__1));
		}
		if (k + 1 <= *ihi) {
		    tst += (r__1 = h__[k + 1 + k * h_dim1], dabs(r__1));
		}
	    }
/*           ==== The following is a conservative small subdiagonal */
/*           .    deflation  criterion due to Ahues & Tisseur (LAWN 122, */
/*           .    1997). It has better mathematical foundation and */
/*           .    improves accuracy in some cases.  ==== */
	    if ((r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)) <= ulp * tst) {
/* Computing MAX */
		r__3 = (r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)), r__4 = 
			(r__2 = h__[k - 1 + k * h_dim1], dabs(r__2));
		ab = dmax(r__3,r__4);
/* Computing MIN */
		r__3 = (r__1 = h__[k + (k - 1) * h_dim1], dabs(r__1)), r__4 = 
			(r__2 = h__[k - 1 + k * h_dim1], dabs(r__2));
		ba = dmin(r__3,r__4);
/* Computing MAX */
		r__3 = (r__1 = h__[k + k * h_dim1], dabs(r__1)), r__4 = (r__2 
			= h__[k - 1 + (k - 1) * h_dim1] - h__[k + k * h_dim1],
			 dabs(r__2));
		aa = dmax(r__3,r__4);
/* Computing MIN */
		r__3 = (r__1 = h__[k + k * h_dim1], dabs(r__1)), r__4 = (r__2 
			= h__[k - 1 + (k - 1) * h_dim1] - h__[k + k * h_dim1],
			 dabs(r__2));
		bb = dmin(r__3,r__4);
		s = aa + ab;
/* Computing MAX */
		r__1 = smlnum, r__2 = ulp * (bb * (aa / s));
		if (ba * (ab / s) <= dmax(r__1,r__2)) {
		    goto L40;
		}
	    }
/* L30: */
	}
L40:
	l = k;
	if (l > *ilo) {

/*           H(L,L-1) is negligible */

	    h__[l + (l - 1) * h_dim1] = 0.f;
	}

/*        Exit from loop if a submatrix of order 1 or 2 has split off. */

	if (l >= i__ - 1) {
	    goto L150;
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

	    s = (r__1 = h__[l + 1 + l * h_dim1], dabs(r__1)) + (r__2 = h__[l 
		    + 2 + (l + 1) * h_dim1], dabs(r__2));
	    h11 = s * .75f + h__[l + l * h_dim1];
	    h12 = s * -.4375f;
	    h21 = s;
	    h22 = h11;
	} else if (its == 20) {

/*           Exceptional shift. */

	    s = (r__1 = h__[i__ + (i__ - 1) * h_dim1], dabs(r__1)) + (r__2 = 
		    h__[i__ - 1 + (i__ - 2) * h_dim1], dabs(r__2));
	    h11 = s * .75f + h__[i__ + i__ * h_dim1];
	    h12 = s * -.4375f;
	    h21 = s;
	    h22 = h11;
	} else {

/*           Prepare to use Francis' double shift */
/*           (i.e. 2nd degree generalized Rayleigh quotient) */

	    h11 = h__[i__ - 1 + (i__ - 1) * h_dim1];
	    h21 = h__[i__ + (i__ - 1) * h_dim1];
	    h12 = h__[i__ - 1 + i__ * h_dim1];
	    h22 = h__[i__ + i__ * h_dim1];
	}
	s = dabs(h11) + dabs(h12) + dabs(h21) + dabs(h22);
	if (s == 0.f) {
	    rt1r = 0.f;
	    rt1i = 0.f;
	    rt2r = 0.f;
	    rt2i = 0.f;
	} else {
	    h11 /= s;
	    h21 /= s;
	    h12 /= s;
	    h22 /= s;
	    tr = (h11 + h22) / 2.f;
	    det = (h11 - tr) * (h22 - tr) - h12 * h21;
	    rtdisc = sqrt((dabs(det)));
	    if (det >= 0.f) {

/*              ==== complex conjugate shifts ==== */

		rt1r = tr * s;
		rt2r = rt1r;
		rt1i = rtdisc * s;
		rt2i = -rt1i;
	    } else {

/*              ==== real shifts (use only one of them)  ==== */

		rt1r = tr + rtdisc;
		rt2r = tr - rtdisc;
		if ((r__1 = rt1r - h22, dabs(r__1)) <= (r__2 = rt2r - h22, 
			dabs(r__2))) {
		    rt1r *= s;
		    rt2r = rt1r;
		} else {
		    rt2r *= s;
		    rt1r = rt2r;
		}
		rt1i = 0.f;
		rt2i = 0.f;
	    }
	}

/*        Look for two consecutive small subdiagonal elements. */

	i__1 = l;
	for (m = i__ - 2; m >= i__1; --m) {
/*           Determine the effect of starting the double-shift QR */
/*           iteration at row M, and see if this would make H(M,M-1) */
/*           negligible.  (The following uses scaling to avoid */
/*           overflows and most underflows.) */

	    h21s = h__[m + 1 + m * h_dim1];
	    s = (r__1 = h__[m + m * h_dim1] - rt2r, dabs(r__1)) + dabs(rt2i) 
		    + dabs(h21s);
	    h21s = h__[m + 1 + m * h_dim1] / s;
	    v[0] = h21s * h__[m + (m + 1) * h_dim1] + (h__[m + m * h_dim1] - 
		    rt1r) * ((h__[m + m * h_dim1] - rt2r) / s) - rt1i * (rt2i 
		    / s);
	    v[1] = h21s * (h__[m + m * h_dim1] + h__[m + 1 + (m + 1) * h_dim1]
		     - rt1r - rt2r);
	    v[2] = h21s * h__[m + 2 + (m + 1) * h_dim1];
	    s = dabs(v[0]) + dabs(v[1]) + dabs(v[2]);
	    v[0] /= s;
	    v[1] /= s;
	    v[2] /= s;
	    if (m == l) {
		goto L60;
	    }
	    if ((r__1 = h__[m + (m - 1) * h_dim1], dabs(r__1)) * (dabs(v[1]) 
		    + dabs(v[2])) <= ulp * dabs(v[0]) * ((r__2 = h__[m - 1 + (
		    m - 1) * h_dim1], dabs(r__2)) + (r__3 = h__[m + m * 
		    h_dim1], dabs(r__3)) + (r__4 = h__[m + 1 + (m + 1) * 
		    h_dim1], dabs(r__4)))) {
		goto L60;
	    }
/* L50: */
	}
L60:

/*        Double-shift QR step */

	i__1 = i__ - 1;
	for (k = m; k <= i__1; ++k) {

/*           The first iteration of this loop determines a reflection G */
/*           from the vector V and applies it from left and right to H, */
/*           thus creating a nonzero bulge below the subdiagonal. */

/*           Each subsequent iteration determines a reflection G to */
/*           restore the Hessenberg form in the (K-1)th column, and thus */
/*           chases the bulge one step toward the bottom of the active */
/*           submatrix. NR is the order of G. */

/* Computing MIN */
	    i__2 = 3, i__3 = i__ - k + 1;
	    nr = min(i__2,i__3);
	    if (k > m) {
		scopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    slarfg_(&nr, v, &v[1], &c__1, &t1);
	    if (k > m) {
		h__[k + (k - 1) * h_dim1] = v[0];
		h__[k + 1 + (k - 1) * h_dim1] = 0.f;
		if (k < i__ - 1) {
		    h__[k + 2 + (k - 1) * h_dim1] = 0.f;
		}
	    } else if (m > l) {
/*               ==== Use the following instead of */
/*               .    H( K, K-1 ) = -H( K, K-1 ) to */
/*               .    avoid a bug when v(2) and v(3) */
/*               .    underflow. ==== */
		h__[k + (k - 1) * h_dim1] *= 1.f - t1;
	    }
	    v2 = v[1];
	    t2 = t1 * v2;
	    if (nr == 3) {
		v3 = v[2];
		t3 = t1 * v3;

/*              Apply G from the left to transform the rows of the matrix */
/*              in columns K to I2. */

		i__2 = i2;
		for (j = k; j <= i__2; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1] 
			    + v3 * h__[k + 2 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
		    h__[k + 2 + j * h_dim1] -= sum * t3;
/* L70: */
		}

/*              Apply G from the right to transform the columns of the */
/*              matrix in rows I1 to min(K+3,I). */

/* Computing MIN */
		i__3 = k + 3;
		i__2 = min(i__3,i__);
		for (j = i1; j <= i__2; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			     + v3 * h__[j + (k + 2) * h_dim1];
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
		    h__[j + (k + 2) * h_dim1] -= sum * t3;
/* L80: */
		}

		if (*wantz) {

/*                 Accumulate transformations in the matrix Z */

		    i__2 = *ihiz;
		    for (j = *iloz; j <= i__2; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) * 
				z_dim1] + v3 * z__[j + (k + 2) * z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
			z__[j + (k + 2) * z_dim1] -= sum * t3;
/* L90: */
		    }
		}
	    } else if (nr == 2) {

/*              Apply G from the left to transform the rows of the matrix */
/*              in columns K to I2. */

		i__2 = i2;
		for (j = k; j <= i__2; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
/* L100: */
		}

/*              Apply G from the right to transform the columns of the */
/*              matrix in rows I1 to min(K+3,I). */

		i__2 = i__;
		for (j = i1; j <= i__2; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			    ;
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
/* L110: */
		}

		if (*wantz) {

/*                 Accumulate transformations in the matrix Z */

		    i__2 = *ihiz;
		    for (j = *iloz; j <= i__2; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) * 
				z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
/* L120: */
		    }
		}
	    }
/* L130: */
	}

/* L140: */
    }

/*     Failure to converge in remaining number of iterations */

    *info = i__;
    return 0;

L150:

    if (l == i__) {

/*        H(I,I-1) is negligible: one eigenvalue has converged. */

	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.f;
    } else if (l == i__ - 1) {

/*        H(I-1,I-2) is negligible: a pair of eigenvalues have converged. */

/*        Transform the 2-by-2 submatrix to standard Schur form, */
/*        and compute and store the eigenvalues. */

	slanv2_(&h__[i__ - 1 + (i__ - 1) * h_dim1], &h__[i__ - 1 + i__ * 
		h_dim1], &h__[i__ + (i__ - 1) * h_dim1], &h__[i__ + i__ * 
		h_dim1], &wr[i__ - 1], &wi[i__ - 1], &wr[i__], &wi[i__], &cs, 
		&sn);

	if (*wantt) {

/*           Apply the transformation to the rest of H. */

	    if (i2 > i__) {
		i__1 = i2 - i__;
		srot_(&i__1, &h__[i__ - 1 + (i__ + 1) * h_dim1], ldh, &h__[
			i__ + (i__ + 1) * h_dim1], ldh, &cs, &sn);
	    }
	    i__1 = i__ - i1 - 1;
	    srot_(&i__1, &h__[i1 + (i__ - 1) * h_dim1], &c__1, &h__[i1 + i__ *
		     h_dim1], &c__1, &cs, &sn);
	}
	if (*wantz) {

/*           Apply the transformation to Z. */

	    srot_(&nz, &z__[*iloz + (i__ - 1) * z_dim1], &c__1, &z__[*iloz + 
		    i__ * z_dim1], &c__1, &cs, &sn);
	}
    }

/*     return to start of the main loop with new value of I. */

    i__ = l - 1;
    goto L20;

L160:
    return 0;

/*     End of SLAHQR */

} /* slahqr_ */
