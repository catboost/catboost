/* dlaqr5.f -- translated by f2c (version 20061008).
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

static doublereal c_b7 = 0.;
static doublereal c_b8 = 1.;
static integer c__3 = 3;
static integer c__1 = 1;
static integer c__2 = 2;

/* Subroutine */ int dlaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, doublereal 
	*sr, doublereal *si, doublereal *h__, integer *ldh, integer *iloz, 
	integer *ihiz, doublereal *z__, integer *ldz, doublereal *v, integer *
	ldv, doublereal *u, integer *ldu, integer *nv, doublereal *wv, 
	integer *ldwv, integer *nh, doublereal *wh, integer *ldwh)
{
    /* System generated locals */
    integer h_dim1, h_offset, u_dim1, u_offset, v_dim1, v_offset, wh_dim1, 
	    wh_offset, wv_dim1, wv_offset, z_dim1, z_offset, i__1, i__2, i__3,
	     i__4, i__5, i__6, i__7;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Local variables */
    integer i__, j, k, m, i2, j2, i4, j4, k1;
    doublereal h11, h12, h21, h22;
    integer m22, ns, nu;
    doublereal vt[3], scl;
    integer kdu, kms;
    doublereal ulp;
    integer knz, kzs;
    doublereal tst1, tst2, beta;
    logical blk22, bmp22;
    integer mend, jcol, jlen, jbot, mbot;
    doublereal swap;
    integer jtop, jrow, mtop;
    doublereal alpha;
    logical accum;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    integer ndcol, incol, krcol, nbmps;
    extern /* Subroutine */ int dtrmm_(char *, char *, char *, char *, 
	    integer *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *), dlaqr1_(
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *), dlabad_(doublereal *, 
	    doublereal *);
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlarfg_(integer *, doublereal *, doublereal *, 
	     integer *, doublereal *), dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *);
    doublereal safmin;
    extern /* Subroutine */ int dlaset_(char *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *);
    doublereal safmax, refsum;
    integer mstart;
    doublereal smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     This auxiliary subroutine called by DLAQR0 performs a */
/*     single small-bulge multi-shift QR sweep. */

/*      WANTT  (input) logical scalar */
/*             WANTT = .true. if the quasi-triangular Schur factor */
/*             is being computed.  WANTT is set to .false. otherwise. */

/*      WANTZ  (input) logical scalar */
/*             WANTZ = .true. if the orthogonal Schur factor is being */
/*             computed.  WANTZ is set to .false. otherwise. */

/*      KACC22 (input) integer with value 0, 1, or 2. */
/*             Specifies the computation mode of far-from-diagonal */
/*             orthogonal updates. */
/*        = 0: DLAQR5 does not accumulate reflections and does not */
/*             use matrix-matrix multiply to update far-from-diagonal */
/*             matrix entries. */
/*        = 1: DLAQR5 accumulates reflections and uses matrix-matrix */
/*             multiply to update the far-from-diagonal matrix entries. */
/*        = 2: DLAQR5 accumulates reflections, uses matrix-matrix */
/*             multiply to update the far-from-diagonal matrix entries, */
/*             and takes advantage of 2-by-2 block structure during */
/*             matrix multiplies. */

/*      N      (input) integer scalar */
/*             N is the order of the Hessenberg matrix H upon which this */
/*             subroutine operates. */

/*      KTOP   (input) integer scalar */
/*      KBOT   (input) integer scalar */
/*             These are the first and last rows and columns of an */
/*             isolated diagonal block upon which the QR sweep is to be */
/*             applied. It is assumed without a check that */
/*                       either KTOP = 1  or   H(KTOP,KTOP-1) = 0 */
/*             and */
/*                       either KBOT = N  or   H(KBOT+1,KBOT) = 0. */

/*      NSHFTS (input) integer scalar */
/*             NSHFTS gives the number of simultaneous shifts.  NSHFTS */
/*             must be positive and even. */

/*      SR     (input/output) DOUBLE PRECISION array of size (NSHFTS) */
/*      SI     (input/output) DOUBLE PRECISION array of size (NSHFTS) */
/*             SR contains the real parts and SI contains the imaginary */
/*             parts of the NSHFTS shifts of origin that define the */
/*             multi-shift QR sweep.  On output SR and SI may be */
/*             reordered. */

/*      H      (input/output) DOUBLE PRECISION array of size (LDH,N) */
/*             On input H contains a Hessenberg matrix.  On output a */
/*             multi-shift QR sweep with shifts SR(J)+i*SI(J) is applied */
/*             to the isolated diagonal block in rows and columns KTOP */
/*             through KBOT. */

/*      LDH    (input) integer scalar */
/*             LDH is the leading dimension of H just as declared in the */
/*             calling procedure.  LDH.GE.MAX(1,N). */

/*      ILOZ   (input) INTEGER */
/*      IHIZ   (input) INTEGER */
/*             Specify the rows of Z to which transformations must be */
/*             applied if WANTZ is .TRUE.. 1 .LE. ILOZ .LE. IHIZ .LE. N */

/*      Z      (input/output) DOUBLE PRECISION array of size (LDZ,IHI) */
/*             If WANTZ = .TRUE., then the QR Sweep orthogonal */
/*             similarity transformation is accumulated into */
/*             Z(ILOZ:IHIZ,ILO:IHI) from the right. */
/*             If WANTZ = .FALSE., then Z is unreferenced. */

/*      LDZ    (input) integer scalar */
/*             LDA is the leading dimension of Z just as declared in */
/*             the calling procedure. LDZ.GE.N. */

/*      V      (workspace) DOUBLE PRECISION array of size (LDV,NSHFTS/2) */

/*      LDV    (input) integer scalar */
/*             LDV is the leading dimension of V as declared in the */
/*             calling procedure.  LDV.GE.3. */

/*      U      (workspace) DOUBLE PRECISION array of size */
/*             (LDU,3*NSHFTS-3) */

/*      LDU    (input) integer scalar */
/*             LDU is the leading dimension of U just as declared in the */
/*             in the calling subroutine.  LDU.GE.3*NSHFTS-3. */

/*      NH     (input) integer scalar */
/*             NH is the number of columns in array WH available for */
/*             workspace. NH.GE.1. */

/*      WH     (workspace) DOUBLE PRECISION array of size (LDWH,NH) */

/*      LDWH   (input) integer scalar */
/*             Leading dimension of WH just as declared in the */
/*             calling procedure.  LDWH.GE.3*NSHFTS-3. */

/*      NV     (input) integer scalar */
/*             NV is the number of rows in WV agailable for workspace. */
/*             NV.GE.1. */

/*      WV     (workspace) DOUBLE PRECISION array of size */
/*             (LDWV,3*NSHFTS-3) */

/*      LDWV   (input) integer scalar */
/*             LDWV is the leading dimension of WV as declared in the */
/*             in the calling subroutine.  LDWV.GE.NV. */

/*     ================================================================ */
/*     Based on contributions by */
/*        Karen Braman and Ralph Byers, Department of Mathematics, */
/*        University of Kansas, USA */

/*     ================================================================ */
/*     Reference: */

/*     K. Braman, R. Byers and R. Mathias, The Multi-Shift QR */
/*     Algorithm Part I: Maintaining Well Focused Shifts, and */
/*     Level 3 Performance, SIAM Journal of Matrix Analysis, */
/*     volume 23, pages 929--947, 2002. */

/*     ================================================================ */
/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */

/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     ==== If there are no shifts, then there is nothing to do. ==== */

    /* Parameter adjustments */
    --sr;
    --si;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    wh_dim1 = *ldwh;
    wh_offset = 1 + wh_dim1;
    wh -= wh_offset;

    /* Function Body */
    if (*nshfts < 2) {
	return 0;
    }

/*     ==== If the active block is empty or 1-by-1, then there */
/*     .    is nothing to do. ==== */

    if (*ktop >= *kbot) {
	return 0;
    }

/*     ==== Shuffle shifts into pairs of real shifts and pairs */
/*     .    of complex conjugate shifts assuming complex */
/*     .    conjugate shifts are already adjacent to one */
/*     .    another. ==== */

    i__1 = *nshfts - 2;
    for (i__ = 1; i__ <= i__1; i__ += 2) {
	if (si[i__] != -si[i__ + 1]) {

	    swap = sr[i__];
	    sr[i__] = sr[i__ + 1];
	    sr[i__ + 1] = sr[i__ + 2];
	    sr[i__ + 2] = swap;

	    swap = si[i__];
	    si[i__] = si[i__ + 1];
	    si[i__ + 1] = si[i__ + 2];
	    si[i__ + 2] = swap;
	}
/* L10: */
    }

/*     ==== NSHFTS is supposed to be even, but if it is odd, */
/*     .    then simply reduce it by one.  The shuffle above */
/*     .    ensures that the dropped shift is real and that */
/*     .    the remaining shifts are paired. ==== */

    ns = *nshfts - *nshfts % 2;

/*     ==== Machine constants for deflation ==== */

    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((doublereal) (*n) / ulp);

/*     ==== Use accumulated reflections to update far-from-diagonal */
/*     .    entries ? ==== */

    accum = *kacc22 == 1 || *kacc22 == 2;

/*     ==== If so, exploit the 2-by-2 block structure? ==== */

    blk22 = ns > 2 && *kacc22 == 2;

/*     ==== clear trash ==== */

    if (*ktop + 2 <= *kbot) {
	h__[*ktop + 2 + *ktop * h_dim1] = 0.;
    }

/*     ==== NBMPS = number of 2-shift bulges in the chain ==== */

    nbmps = ns / 2;

/*     ==== KDU = width of slab ==== */

    kdu = nbmps * 6 - 3;

/*     ==== Create and chase chains of NBMPS bulges ==== */

    i__1 = *kbot - 2;
    i__2 = nbmps * 3 - 2;
    for (incol = (1 - nbmps) * 3 + *ktop - 1; i__2 < 0 ? incol >= i__1 : 
	    incol <= i__1; incol += i__2) {
	ndcol = incol + kdu;
	if (accum) {
	    dlaset_("ALL", &kdu, &kdu, &c_b7, &c_b8, &u[u_offset], ldu);
	}

/*        ==== Near-the-diagonal bulge chase.  The following loop */
/*        .    performs the near-the-diagonal part of a small bulge */
/*        .    multi-shift QR sweep.  Each 6*NBMPS-2 column diagonal */
/*        .    chunk extends from column INCOL to column NDCOL */
/*        .    (including both column INCOL and column NDCOL). The */
/*        .    following loop chases a 3*NBMPS column long chain of */
/*        .    NBMPS bulges 3*NBMPS-2 columns to the right.  (INCOL */
/*        .    may be less than KTOP and and NDCOL may be greater than */
/*        .    KBOT indicating phantom columns from which to chase */
/*        .    bulges before they are actually introduced or to which */
/*        .    to chase bulges beyond column KBOT.)  ==== */

/* Computing MIN */
	i__4 = incol + nbmps * 3 - 3, i__5 = *kbot - 2;
	i__3 = min(i__4,i__5);
	for (krcol = incol; krcol <= i__3; ++krcol) {

/*           ==== Bulges number MTOP to MBOT are active double implicit */
/*           .    shift bulges.  There may or may not also be small */
/*           .    2-by-2 bulge, if there is room.  The inactive bulges */
/*           .    (if any) must wait until the active bulges have moved */
/*           .    down the diagonal to make room.  The phantom matrix */
/*           .    paradigm described above helps keep track.  ==== */

/* Computing MAX */
	    i__4 = 1, i__5 = (*ktop - 1 - krcol + 2) / 3 + 1;
	    mtop = max(i__4,i__5);
/* Computing MIN */
	    i__4 = nbmps, i__5 = (*kbot - krcol) / 3;
	    mbot = min(i__4,i__5);
	    m22 = mbot + 1;
	    bmp22 = mbot < nbmps && krcol + (m22 - 1) * 3 == *kbot - 2;

/*           ==== Generate reflections to chase the chain right */
/*           .    one column.  (The minimum value of K is KTOP-1.) ==== */

	    i__4 = mbot;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		if (k == *ktop - 1) {
		    dlaqr1_(&c__3, &h__[*ktop + *ktop * h_dim1], ldh, &sr[(m 
			    << 1) - 1], &si[(m << 1) - 1], &sr[m * 2], &si[m *
			     2], &v[m * v_dim1 + 1]);
		    alpha = v[m * v_dim1 + 1];
		    dlarfg_(&c__3, &alpha, &v[m * v_dim1 + 2], &c__1, &v[m * 
			    v_dim1 + 1]);
		} else {
		    beta = h__[k + 1 + k * h_dim1];
		    v[m * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
		    v[m * v_dim1 + 3] = h__[k + 3 + k * h_dim1];
		    dlarfg_(&c__3, &beta, &v[m * v_dim1 + 2], &c__1, &v[m * 
			    v_dim1 + 1]);

/*                 ==== A Bulge may collapse because of vigilant */
/*                 .    deflation or destructive underflow.  In the */
/*                 .    underflow case, try the two-small-subdiagonals */
/*                 .    trick to try to reinflate the bulge.  ==== */

		    if (h__[k + 3 + k * h_dim1] != 0. || h__[k + 3 + (k + 1) *
			     h_dim1] != 0. || h__[k + 3 + (k + 2) * h_dim1] ==
			     0.) {

/*                    ==== Typical case: not collapsed (yet). ==== */

			h__[k + 1 + k * h_dim1] = beta;
			h__[k + 2 + k * h_dim1] = 0.;
			h__[k + 3 + k * h_dim1] = 0.;
		    } else {

/*                    ==== Atypical case: collapsed.  Attempt to */
/*                    .    reintroduce ignoring H(K+1,K) and H(K+2,K). */
/*                    .    If the fill resulting from the new */
/*                    .    reflector is too large, then abandon it. */
/*                    .    Otherwise, use the new one. ==== */

			dlaqr1_(&c__3, &h__[k + 1 + (k + 1) * h_dim1], ldh, &
				sr[(m << 1) - 1], &si[(m << 1) - 1], &sr[m * 
				2], &si[m * 2], vt);
			alpha = vt[0];
			dlarfg_(&c__3, &alpha, &vt[1], &c__1, vt);
			refsum = vt[0] * (h__[k + 1 + k * h_dim1] + vt[1] * 
				h__[k + 2 + k * h_dim1]);

			if ((d__1 = h__[k + 2 + k * h_dim1] - refsum * vt[1], 
				abs(d__1)) + (d__2 = refsum * vt[2], abs(d__2)
				) > ulp * ((d__3 = h__[k + k * h_dim1], abs(
				d__3)) + (d__4 = h__[k + 1 + (k + 1) * h_dim1]
				, abs(d__4)) + (d__5 = h__[k + 2 + (k + 2) * 
				h_dim1], abs(d__5)))) {

/*                       ==== Starting a new bulge here would */
/*                       .    create non-negligible fill.  Use */
/*                       .    the old one with trepidation. ==== */

			    h__[k + 1 + k * h_dim1] = beta;
			    h__[k + 2 + k * h_dim1] = 0.;
			    h__[k + 3 + k * h_dim1] = 0.;
			} else {

/*                       ==== Stating a new bulge here would */
/*                       .    create only negligible fill. */
/*                       .    Replace the old reflector with */
/*                       .    the new one. ==== */

			    h__[k + 1 + k * h_dim1] -= refsum;
			    h__[k + 2 + k * h_dim1] = 0.;
			    h__[k + 3 + k * h_dim1] = 0.;
			    v[m * v_dim1 + 1] = vt[0];
			    v[m * v_dim1 + 2] = vt[1];
			    v[m * v_dim1 + 3] = vt[2];
			}
		    }
		}
/* L20: */
	    }

/*           ==== Generate a 2-by-2 reflection, if needed. ==== */

	    k = krcol + (m22 - 1) * 3;
	    if (bmp22) {
		if (k == *ktop - 1) {
		    dlaqr1_(&c__2, &h__[k + 1 + (k + 1) * h_dim1], ldh, &sr[(
			    m22 << 1) - 1], &si[(m22 << 1) - 1], &sr[m22 * 2], 
			     &si[m22 * 2], &v[m22 * v_dim1 + 1]);
		    beta = v[m22 * v_dim1 + 1];
		    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 
			    * v_dim1 + 1]);
		} else {
		    beta = h__[k + 1 + k * h_dim1];
		    v[m22 * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
		    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 
			    * v_dim1 + 1]);
		    h__[k + 1 + k * h_dim1] = beta;
		    h__[k + 2 + k * h_dim1] = 0.;
		}
	    }

/*           ==== Multiply H by reflections from the left ==== */

	    if (accum) {
		jbot = min(ndcol,*kbot);
	    } else if (*wantt) {
		jbot = *n;
	    } else {
		jbot = *kbot;
	    }
	    i__4 = jbot;
	    for (j = max(*ktop,krcol); j <= i__4; ++j) {
/* Computing MIN */
		i__5 = mbot, i__6 = (j - krcol + 2) / 3;
		mend = min(i__5,i__6);
		i__5 = mend;
		for (m = mtop; m <= i__5; ++m) {
		    k = krcol + (m - 1) * 3;
		    refsum = v[m * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] + v[
			    m * v_dim1 + 2] * h__[k + 2 + j * h_dim1] + v[m * 
			    v_dim1 + 3] * h__[k + 3 + j * h_dim1]);
		    h__[k + 1 + j * h_dim1] -= refsum;
		    h__[k + 2 + j * h_dim1] -= refsum * v[m * v_dim1 + 2];
		    h__[k + 3 + j * h_dim1] -= refsum * v[m * v_dim1 + 3];
/* L30: */
		}
/* L40: */
	    }
	    if (bmp22) {
		k = krcol + (m22 - 1) * 3;
/* Computing MAX */
		i__4 = k + 1;
		i__5 = jbot;
		for (j = max(i__4,*ktop); j <= i__5; ++j) {
		    refsum = v[m22 * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] + 
			    v[m22 * v_dim1 + 2] * h__[k + 2 + j * h_dim1]);
		    h__[k + 1 + j * h_dim1] -= refsum;
		    h__[k + 2 + j * h_dim1] -= refsum * v[m22 * v_dim1 + 2];
/* L50: */
		}
	    }

/*           ==== Multiply H by reflections from the right. */
/*           .    Delay filling in the last row until the */
/*           .    vigilant deflation check is complete. ==== */

	    if (accum) {
		jtop = max(*ktop,incol);
	    } else if (*wantt) {
		jtop = 1;
	    } else {
		jtop = *ktop;
	    }
	    i__5 = mbot;
	    for (m = mtop; m <= i__5; ++m) {
		if (v[m * v_dim1 + 1] != 0.) {
		    k = krcol + (m - 1) * 3;
/* Computing MIN */
		    i__6 = *kbot, i__7 = k + 3;
		    i__4 = min(i__6,i__7);
		    for (j = jtop; j <= i__4; ++j) {
			refsum = v[m * v_dim1 + 1] * (h__[j + (k + 1) * 
				h_dim1] + v[m * v_dim1 + 2] * h__[j + (k + 2) 
				* h_dim1] + v[m * v_dim1 + 3] * h__[j + (k + 
				3) * h_dim1]);
			h__[j + (k + 1) * h_dim1] -= refsum;
			h__[j + (k + 2) * h_dim1] -= refsum * v[m * v_dim1 + 
				2];
			h__[j + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 + 
				3];
/* L60: */
		    }

		    if (accum) {

/*                    ==== Accumulate U. (If necessary, update Z later */
/*                    .    with with an efficient matrix-matrix */
/*                    .    multiply.) ==== */

			kms = k - incol;
/* Computing MAX */
			i__4 = 1, i__6 = *ktop - incol;
			i__7 = kdu;
			for (j = max(i__4,i__6); j <= i__7; ++j) {
			    refsum = v[m * v_dim1 + 1] * (u[j + (kms + 1) * 
				    u_dim1] + v[m * v_dim1 + 2] * u[j + (kms 
				    + 2) * u_dim1] + v[m * v_dim1 + 3] * u[j 
				    + (kms + 3) * u_dim1]);
			    u[j + (kms + 1) * u_dim1] -= refsum;
			    u[j + (kms + 2) * u_dim1] -= refsum * v[m * 
				    v_dim1 + 2];
			    u[j + (kms + 3) * u_dim1] -= refsum * v[m * 
				    v_dim1 + 3];
/* L70: */
			}
		    } else if (*wantz) {

/*                    ==== U is not accumulated, so update Z */
/*                    .    now by multiplying by reflections */
/*                    .    from the right. ==== */

			i__7 = *ihiz;
			for (j = *iloz; j <= i__7; ++j) {
			    refsum = v[m * v_dim1 + 1] * (z__[j + (k + 1) * 
				    z_dim1] + v[m * v_dim1 + 2] * z__[j + (k 
				    + 2) * z_dim1] + v[m * v_dim1 + 3] * z__[
				    j + (k + 3) * z_dim1]);
			    z__[j + (k + 1) * z_dim1] -= refsum;
			    z__[j + (k + 2) * z_dim1] -= refsum * v[m * 
				    v_dim1 + 2];
			    z__[j + (k + 3) * z_dim1] -= refsum * v[m * 
				    v_dim1 + 3];
/* L80: */
			}
		    }
		}
/* L90: */
	    }

/*           ==== Special case: 2-by-2 reflection (if needed) ==== */

	    k = krcol + (m22 - 1) * 3;
	    if (bmp22 && v[m22 * v_dim1 + 1] != 0.) {
/* Computing MIN */
		i__7 = *kbot, i__4 = k + 3;
		i__5 = min(i__7,i__4);
		for (j = jtop; j <= i__5; ++j) {
		    refsum = v[m22 * v_dim1 + 1] * (h__[j + (k + 1) * h_dim1] 
			    + v[m22 * v_dim1 + 2] * h__[j + (k + 2) * h_dim1])
			    ;
		    h__[j + (k + 1) * h_dim1] -= refsum;
		    h__[j + (k + 2) * h_dim1] -= refsum * v[m22 * v_dim1 + 2];
/* L100: */
		}

		if (accum) {
		    kms = k - incol;
/* Computing MAX */
		    i__5 = 1, i__7 = *ktop - incol;
		    i__4 = kdu;
		    for (j = max(i__5,i__7); j <= i__4; ++j) {
			refsum = v[m22 * v_dim1 + 1] * (u[j + (kms + 1) * 
				u_dim1] + v[m22 * v_dim1 + 2] * u[j + (kms + 
				2) * u_dim1]);
			u[j + (kms + 1) * u_dim1] -= refsum;
			u[j + (kms + 2) * u_dim1] -= refsum * v[m22 * v_dim1 
				+ 2];
/* L110: */
		    }
		} else if (*wantz) {
		    i__4 = *ihiz;
		    for (j = *iloz; j <= i__4; ++j) {
			refsum = v[m22 * v_dim1 + 1] * (z__[j + (k + 1) * 
				z_dim1] + v[m22 * v_dim1 + 2] * z__[j + (k + 
				2) * z_dim1]);
			z__[j + (k + 1) * z_dim1] -= refsum;
			z__[j + (k + 2) * z_dim1] -= refsum * v[m22 * v_dim1 
				+ 2];
/* L120: */
		    }
		}
	    }

/*           ==== Vigilant deflation check ==== */

	    mstart = mtop;
	    if (krcol + (mstart - 1) * 3 < *ktop) {
		++mstart;
	    }
	    mend = mbot;
	    if (bmp22) {
		++mend;
	    }
	    if (krcol == *kbot - 2) {
		++mend;
	    }
	    i__4 = mend;
	    for (m = mstart; m <= i__4; ++m) {
/* Computing MIN */
		i__5 = *kbot - 1, i__7 = krcol + (m - 1) * 3;
		k = min(i__5,i__7);

/*              ==== The following convergence test requires that */
/*              .    the tradition small-compared-to-nearby-diagonals */
/*              .    criterion and the Ahues & Tisseur (LAWN 122, 1997) */
/*              .    criteria both be satisfied.  The latter improves */
/*              .    accuracy in some examples. Falling back on an */
/*              .    alternate convergence criterion when TST1 or TST2 */
/*              .    is zero (as done here) is traditional but probably */
/*              .    unnecessary. ==== */

		if (h__[k + 1 + k * h_dim1] != 0.) {
		    tst1 = (d__1 = h__[k + k * h_dim1], abs(d__1)) + (d__2 = 
			    h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
		    if (tst1 == 0.) {
			if (k >= *ktop + 1) {
			    tst1 += (d__1 = h__[k + (k - 1) * h_dim1], abs(
				    d__1));
			}
			if (k >= *ktop + 2) {
			    tst1 += (d__1 = h__[k + (k - 2) * h_dim1], abs(
				    d__1));
			}
			if (k >= *ktop + 3) {
			    tst1 += (d__1 = h__[k + (k - 3) * h_dim1], abs(
				    d__1));
			}
			if (k <= *kbot - 2) {
			    tst1 += (d__1 = h__[k + 2 + (k + 1) * h_dim1], 
				    abs(d__1));
			}
			if (k <= *kbot - 3) {
			    tst1 += (d__1 = h__[k + 3 + (k + 1) * h_dim1], 
				    abs(d__1));
			}
			if (k <= *kbot - 4) {
			    tst1 += (d__1 = h__[k + 4 + (k + 1) * h_dim1], 
				    abs(d__1));
			}
		    }
/* Computing MAX */
		    d__2 = smlnum, d__3 = ulp * tst1;
		    if ((d__1 = h__[k + 1 + k * h_dim1], abs(d__1)) <= max(
			    d__2,d__3)) {
/* Computing MAX */
			d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)), 
				d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(
				d__2));
			h12 = max(d__3,d__4);
/* Computing MIN */
			d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)), 
				d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(
				d__2));
			h21 = min(d__3,d__4);
/* Computing MAX */
			d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(
				d__1)), d__4 = (d__2 = h__[k + k * h_dim1] - 
				h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
			h11 = max(d__3,d__4);
/* Computing MIN */
			d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(
				d__1)), d__4 = (d__2 = h__[k + k * h_dim1] - 
				h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
			h22 = min(d__3,d__4);
			scl = h11 + h12;
			tst2 = h22 * (h11 / scl);

/* Computing MAX */
			d__1 = smlnum, d__2 = ulp * tst2;
			if (tst2 == 0. || h21 * (h12 / scl) <= max(d__1,d__2))
				 {
			    h__[k + 1 + k * h_dim1] = 0.;
			}
		    }
		}
/* L130: */
	    }

/*           ==== Fill in the last row of each bulge. ==== */

/* Computing MIN */
	    i__4 = nbmps, i__5 = (*kbot - krcol - 1) / 3;
	    mend = min(i__4,i__5);
	    i__4 = mend;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		refsum = v[m * v_dim1 + 1] * v[m * v_dim1 + 3] * h__[k + 4 + (
			k + 3) * h_dim1];
		h__[k + 4 + (k + 1) * h_dim1] = -refsum;
		h__[k + 4 + (k + 2) * h_dim1] = -refsum * v[m * v_dim1 + 2];
		h__[k + 4 + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 + 3];
/* L140: */
	    }

/*           ==== End of near-the-diagonal bulge chase. ==== */

/* L150: */
	}

/*        ==== Use U (if accumulated) to update far-from-diagonal */
/*        .    entries in H.  If required, use U to update Z as */
/*        .    well. ==== */

	if (accum) {
	    if (*wantt) {
		jtop = 1;
		jbot = *n;
	    } else {
		jtop = *ktop;
		jbot = *kbot;
	    }
	    if (! blk22 || incol < *ktop || ndcol > *kbot || ns <= 2) {

/*              ==== Updates not exploiting the 2-by-2 block */
/*              .    structure of U.  K1 and NU keep track of */
/*              .    the location and size of U in the special */
/*              .    cases of introducing bulges and chasing */
/*              .    bulges off the bottom.  In these special */
/*              .    cases and in case the number of shifts */
/*              .    is NS = 2, there is no 2-by-2 block */
/*              .    structure to exploit.  ==== */

/* Computing MAX */
		i__3 = 1, i__4 = *ktop - incol;
		k1 = max(i__3,i__4);
/* Computing MAX */
		i__3 = 0, i__4 = ndcol - *kbot;
		nu = kdu - max(i__3,i__4) - k1 + 1;

/*              ==== Horizontal Multiply ==== */

		i__3 = jbot;
		i__4 = *nh;
		for (jcol = min(ndcol,*kbot) + 1; i__4 < 0 ? jcol >= i__3 : 
			jcol <= i__3; jcol += i__4) {
/* Computing MIN */
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = min(i__5,i__7);
		    dgemm_("C", "N", &nu, &jlen, &nu, &c_b8, &u[k1 + k1 * 
			    u_dim1], ldu, &h__[incol + k1 + jcol * h_dim1], 
			    ldh, &c_b7, &wh[wh_offset], ldwh);
		    dlacpy_("ALL", &nu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + k1 + jcol * h_dim1], ldh);
/* L160: */
		}

/*              ==== Vertical multiply ==== */

		i__4 = max(*ktop,incol) - 1;
		i__3 = *nv;
		for (jrow = jtop; i__3 < 0 ? jrow >= i__4 : jrow <= i__4; 
			jrow += i__3) {
/* Computing MIN */
		    i__5 = *nv, i__7 = max(*ktop,incol) - jrow;
		    jlen = min(i__5,i__7);
		    dgemm_("N", "N", &jlen, &nu, &nu, &c_b8, &h__[jrow + (
			    incol + k1) * h_dim1], ldh, &u[k1 + k1 * u_dim1], 
			    ldu, &c_b7, &wv[wv_offset], ldwv);
		    dlacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + k1) * h_dim1], ldh);
/* L170: */
		}

/*              ==== Z multiply (also vertical) ==== */

		if (*wantz) {
		    i__3 = *ihiz;
		    i__4 = *nv;
		    for (jrow = *iloz; i__4 < 0 ? jrow >= i__3 : jrow <= i__3;
			     jrow += i__4) {
/* Computing MIN */
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = min(i__5,i__7);
			dgemm_("N", "N", &jlen, &nu, &nu, &c_b8, &z__[jrow + (
				incol + k1) * z_dim1], ldz, &u[k1 + k1 * 
				u_dim1], ldu, &c_b7, &wv[wv_offset], ldwv);
			dlacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &z__[
				jrow + (incol + k1) * z_dim1], ldz)
				;
/* L180: */
		    }
		}
	    } else {

/*              ==== Updates exploiting U's 2-by-2 block structure. */
/*              .    (I2, I4, J2, J4 are the last rows and columns */
/*              .    of the blocks.) ==== */

		i2 = (kdu + 1) / 2;
		i4 = kdu;
		j2 = i4 - i2;
		j4 = kdu;

/*              ==== KZS and KNZ deal with the band of zeros */
/*              .    along the diagonal of one of the triangular */
/*              .    blocks. ==== */

		kzs = j4 - j2 - (ns + 1);
		knz = ns + 1;

/*              ==== Horizontal multiply ==== */

		i__4 = jbot;
		i__3 = *nh;
		for (jcol = min(ndcol,*kbot) + 1; i__3 < 0 ? jcol >= i__4 : 
			jcol <= i__4; jcol += i__3) {
/* Computing MIN */
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = min(i__5,i__7);

/*                 ==== Copy bottom of H to top+KZS of scratch ==== */
/*                  (The first KZS rows get multiplied by zero.) ==== */

		    dlacpy_("ALL", &knz, &jlen, &h__[incol + 1 + j2 + jcol * 
			    h_dim1], ldh, &wh[kzs + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U21' ==== */

		    dlaset_("ALL", &kzs, &jlen, &c_b7, &c_b7, &wh[wh_offset], 
			    ldwh);
		    dtrmm_("L", "U", "C", "N", &knz, &jlen, &c_b8, &u[j2 + 1 
			    + (kzs + 1) * u_dim1], ldu, &wh[kzs + 1 + wh_dim1]
, ldwh);

/*                 ==== Multiply top of H by U11' ==== */

		    dgemm_("C", "N", &i2, &jlen, &j2, &c_b8, &u[u_offset], 
			    ldu, &h__[incol + 1 + jcol * h_dim1], ldh, &c_b8, 
			    &wh[wh_offset], ldwh);

/*                 ==== Copy top of H to bottom of WH ==== */

		    dlacpy_("ALL", &j2, &jlen, &h__[incol + 1 + jcol * h_dim1]
, ldh, &wh[i2 + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U21' ==== */

		    dtrmm_("L", "L", "C", "N", &j2, &jlen, &c_b8, &u[(i2 + 1) 
			    * u_dim1 + 1], ldu, &wh[i2 + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U22 ==== */

		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    dgemm_("C", "N", &i__5, &jlen, &i__7, &c_b8, &u[j2 + 1 + (
			    i2 + 1) * u_dim1], ldu, &h__[incol + 1 + j2 + 
			    jcol * h_dim1], ldh, &c_b8, &wh[i2 + 1 + wh_dim1], 
			     ldwh);

/*                 ==== Copy it back ==== */

		    dlacpy_("ALL", &kdu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + 1 + jcol * h_dim1], ldh);
/* L190: */
		}

/*              ==== Vertical multiply ==== */

		i__3 = max(incol,*ktop) - 1;
		i__4 = *nv;
		for (jrow = jtop; i__4 < 0 ? jrow >= i__3 : jrow <= i__3; 
			jrow += i__4) {
/* Computing MIN */
		    i__5 = *nv, i__7 = max(incol,*ktop) - jrow;
		    jlen = min(i__5,i__7);

/*                 ==== Copy right of H to scratch (the first KZS */
/*                 .    columns get multiplied by zero) ==== */

		    dlacpy_("ALL", &jlen, &knz, &h__[jrow + (incol + 1 + j2) *
			     h_dim1], ldh, &wv[(kzs + 1) * wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U21 ==== */

		    dlaset_("ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[wv_offset], 
			    ldwv);
		    dtrmm_("R", "U", "N", "N", &jlen, &knz, &c_b8, &u[j2 + 1 
			    + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) * 
			    wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U11 ==== */

		    dgemm_("N", "N", &jlen, &i2, &j2, &c_b8, &h__[jrow + (
			    incol + 1) * h_dim1], ldh, &u[u_offset], ldu, &
			    c_b8, &wv[wv_offset], ldwv);

/*                 ==== Copy left of H to right of scratch ==== */

		    dlacpy_("ALL", &jlen, &j2, &h__[jrow + (incol + 1) * 
			    h_dim1], ldh, &wv[(i2 + 1) * wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U21 ==== */

		    i__5 = i4 - i2;
		    dtrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b8, &u[(i2 + 
			    1) * u_dim1 + 1], ldu, &wv[(i2 + 1) * wv_dim1 + 1]
, ldwv);

/*                 ==== Multiply by U22 ==== */

		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    dgemm_("N", "N", &jlen, &i__5, &i__7, &c_b8, &h__[jrow + (
			    incol + 1 + j2) * h_dim1], ldh, &u[j2 + 1 + (i2 + 
			    1) * u_dim1], ldu, &c_b8, &wv[(i2 + 1) * wv_dim1 
			    + 1], ldwv);

/*                 ==== Copy it back ==== */

		    dlacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + 1) * h_dim1], ldh);
/* L200: */
		}

/*              ==== Multiply Z (also vertical) ==== */

		if (*wantz) {
		    i__4 = *ihiz;
		    i__3 = *nv;
		    for (jrow = *iloz; i__3 < 0 ? jrow >= i__4 : jrow <= i__4;
			     jrow += i__3) {
/* Computing MIN */
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = min(i__5,i__7);

/*                    ==== Copy right of Z to left of scratch (first */
/*                    .     KZS columns get multiplied by zero) ==== */

			dlacpy_("ALL", &jlen, &knz, &z__[jrow + (incol + 1 + 
				j2) * z_dim1], ldz, &wv[(kzs + 1) * wv_dim1 + 
				1], ldwv);

/*                    ==== Multiply by U12 ==== */

			dlaset_("ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[
				wv_offset], ldwv);
			dtrmm_("R", "U", "N", "N", &jlen, &knz, &c_b8, &u[j2 
				+ 1 + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) 
				* wv_dim1 + 1], ldwv);

/*                    ==== Multiply by U11 ==== */

			dgemm_("N", "N", &jlen, &i2, &j2, &c_b8, &z__[jrow + (
				incol + 1) * z_dim1], ldz, &u[u_offset], ldu, 
				&c_b8, &wv[wv_offset], ldwv);

/*                    ==== Copy left of Z to right of scratch ==== */

			dlacpy_("ALL", &jlen, &j2, &z__[jrow + (incol + 1) * 
				z_dim1], ldz, &wv[(i2 + 1) * wv_dim1 + 1], 
				ldwv);

/*                    ==== Multiply by U21 ==== */

			i__5 = i4 - i2;
			dtrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b8, &u[(
				i2 + 1) * u_dim1 + 1], ldu, &wv[(i2 + 1) * 
				wv_dim1 + 1], ldwv);

/*                    ==== Multiply by U22 ==== */

			i__5 = i4 - i2;
			i__7 = j4 - j2;
			dgemm_("N", "N", &jlen, &i__5, &i__7, &c_b8, &z__[
				jrow + (incol + 1 + j2) * z_dim1], ldz, &u[j2 
				+ 1 + (i2 + 1) * u_dim1], ldu, &c_b8, &wv[(i2 
				+ 1) * wv_dim1 + 1], ldwv);

/*                    ==== Copy the result back to Z ==== */

			dlacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &
				z__[jrow + (incol + 1) * z_dim1], ldz);
/* L210: */
		    }
		}
	    }
	}
/* L220: */
    }

/*     ==== End of DLAQR5 ==== */

    return 0;
} /* dlaqr5_ */
