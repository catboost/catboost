/* slaqr4.f -- translated by f2c (version 20061008).
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

static integer c__13 = 13;
static integer c__15 = 15;
static integer c_n1 = -1;
static integer c__12 = 12;
static integer c__14 = 14;
static integer c__16 = 16;
static logical c_false = FALSE_;
static integer c__1 = 1;
static integer c__3 = 3;

/* Subroutine */ int slaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;

    /* Local variables */
    integer i__, k;
    real aa, bb, cc, dd;
    integer ld;
    real cs;
    integer nh, it, ks, kt;
    real sn;
    integer ku, kv, ls, ns;
    real ss;
    integer nw, inf, kdu, nho, nve, kwh, nsr, nwr, kwv, ndec, ndfl, kbot, 
	    nmin;
    real swap;
    integer ktop;
    real zdum[1]	/* was [1][1] */;
    integer kacc22, itmax, nsmax, nwmax, kwtop;
    extern /* Subroutine */ int slaqr2_(logical *, logical *, integer *, 
	    integer *, integer *, integer *, real *, integer *, integer *, 
	    integer *, real *, integer *, integer *, integer *, real *, real *
, real *, integer *, integer *, real *, integer *, integer *, 
	    real *, integer *, real *, integer *), slanv2_(real *, real *, 
	    real *, real *, real *, real *, real *, real *, real *, real *), 
	    slaqr5_(logical *, logical *, integer *, integer *, integer *, 
	    integer *, integer *, real *, real *, real *, integer *, integer *
, integer *, real *, integer *, real *, integer *, real *, 
	    integer *, integer *, real *, integer *, integer *, real *, 
	    integer *);
    integer nibble;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    char jbcmpz[1];
    extern /* Subroutine */ int slahqr_(logical *, logical *, integer *, 
	    integer *, integer *, real *, integer *, real *, real *, integer *
, integer *, real *, integer *, integer *), slacpy_(char *, 
	    integer *, integer *, real *, integer *, real *, integer *);
    integer nwupbd;
    logical sorted;
    integer lwkopt;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     This subroutine implements one level of recursion for SLAQR0. */
/*     It is a complete implementation of the small bulge multi-shift */
/*     QR algorithm.  It may be called by SLAQR0 and, for large enough */
/*     deflation window size, it may be called by SLAQR3.  This */
/*     subroutine is identical to SLAQR0 except that it calls SLAQR2 */
/*     instead of SLAQR3. */

/*     Purpose */
/*     ======= */

/*     SLAQR4 computes the eigenvalues of a Hessenberg matrix H */
/*     and, optionally, the matrices T and Z from the Schur decomposition */
/*     H = Z T Z**T, where T is an upper quasi-triangular matrix (the */
/*     Schur form), and Z is the orthogonal matrix of Schur vectors. */

/*     Optionally Z may be postmultiplied into an input orthogonal */
/*     matrix Q so that this routine can give the Schur factorization */
/*     of a matrix A which has been reduced to the Hessenberg form H */
/*     by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T. */

/*     Arguments */
/*     ========= */

/*     WANTT   (input) LOGICAL */
/*          = .TRUE. : the full Schur form T is required; */
/*          = .FALSE.: only eigenvalues are required. */

/*     WANTZ   (input) LOGICAL */
/*          = .TRUE. : the matrix of Schur vectors Z is required; */
/*          = .FALSE.: Schur vectors are not required. */

/*     N     (input) INTEGER */
/*           The order of the matrix H.  N .GE. 0. */

/*     ILO   (input) INTEGER */
/*     IHI   (input) INTEGER */
/*           It is assumed that H is already upper triangular in rows */
/*           and columns 1:ILO-1 and IHI+1:N and, if ILO.GT.1, */
/*           H(ILO,ILO-1) is zero. ILO and IHI are normally set by a */
/*           previous call to SGEBAL, and then passed to SGEHRD when the */
/*           matrix output by SGEBAL is reduced to Hessenberg form. */
/*           Otherwise, ILO and IHI should be set to 1 and N, */
/*           respectively.  If N.GT.0, then 1.LE.ILO.LE.IHI.LE.N. */
/*           If N = 0, then ILO = 1 and IHI = 0. */

/*     H     (input/output) REAL array, dimension (LDH,N) */
/*           On entry, the upper Hessenberg matrix H. */
/*           On exit, if INFO = 0 and WANTT is .TRUE., then H contains */
/*           the upper quasi-triangular matrix T from the Schur */
/*           decomposition (the Schur form); 2-by-2 diagonal blocks */
/*           (corresponding to complex conjugate pairs of eigenvalues) */
/*           are returned in standard form, with H(i,i) = H(i+1,i+1) */
/*           and H(i+1,i)*H(i,i+1).LT.0. If INFO = 0 and WANTT is */
/*           .FALSE., then the contents of H are unspecified on exit. */
/*           (The output value of H when INFO.GT.0 is given under the */
/*           description of INFO below.) */

/*           This subroutine may explicitly set H(i,j) = 0 for i.GT.j and */
/*           j = 1, 2, ... ILO-1 or j = IHI+1, IHI+2, ... N. */

/*     LDH   (input) INTEGER */
/*           The leading dimension of the array H. LDH .GE. max(1,N). */

/*     WR    (output) REAL array, dimension (IHI) */
/*     WI    (output) REAL array, dimension (IHI) */
/*           The real and imaginary parts, respectively, of the computed */
/*           eigenvalues of H(ILO:IHI,ILO:IHI) are stored in WR(ILO:IHI) */
/*           and WI(ILO:IHI). If two eigenvalues are computed as a */
/*           complex conjugate pair, they are stored in consecutive */
/*           elements of WR and WI, say the i-th and (i+1)th, with */
/*           WI(i) .GT. 0 and WI(i+1) .LT. 0. If WANTT is .TRUE., then */
/*           the eigenvalues are stored in the same order as on the */
/*           diagonal of the Schur form returned in H, with */
/*           WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2 diagonal */
/*           block, WI(i) = sqrt(-H(i+1,i)*H(i,i+1)) and */
/*           WI(i+1) = -WI(i). */

/*     ILOZ     (input) INTEGER */
/*     IHIZ     (input) INTEGER */
/*           Specify the rows of Z to which transformations must be */
/*           applied if WANTZ is .TRUE.. */
/*           1 .LE. ILOZ .LE. ILO; IHI .LE. IHIZ .LE. N. */

/*     Z     (input/output) REAL array, dimension (LDZ,IHI) */
/*           If WANTZ is .FALSE., then Z is not referenced. */
/*           If WANTZ is .TRUE., then Z(ILO:IHI,ILOZ:IHIZ) is */
/*           replaced by Z(ILO:IHI,ILOZ:IHIZ)*U where U is the */
/*           orthogonal Schur factor of H(ILO:IHI,ILO:IHI). */
/*           (The output value of Z when INFO.GT.0 is given under */
/*           the description of INFO below.) */

/*     LDZ   (input) INTEGER */
/*           The leading dimension of the array Z.  if WANTZ is .TRUE. */
/*           then LDZ.GE.MAX(1,IHIZ).  Otherwize, LDZ.GE.1. */

/*     WORK  (workspace/output) REAL array, dimension LWORK */
/*           On exit, if LWORK = -1, WORK(1) returns an estimate of */
/*           the optimal value for LWORK. */

/*     LWORK (input) INTEGER */
/*           The dimension of the array WORK.  LWORK .GE. max(1,N) */
/*           is sufficient, but LWORK typically as large as 6*N may */
/*           be required for optimal performance.  A workspace query */
/*           to determine the optimal workspace size is recommended. */

/*           If LWORK = -1, then SLAQR4 does a workspace query. */
/*           In this case, SLAQR4 checks the input parameters and */
/*           estimates the optimal workspace size for the given */
/*           values of N, ILO and IHI.  The estimate is returned */
/*           in WORK(1).  No error message related to LWORK is */
/*           issued by XERBLA.  Neither H nor Z are accessed. */


/*     INFO  (output) INTEGER */
/*             =  0:  successful exit */
/*           .GT. 0:  if INFO = i, SLAQR4 failed to compute all of */
/*                the eigenvalues.  Elements 1:ilo-1 and i+1:n of WR */
/*                and WI contain those eigenvalues which have been */
/*                successfully computed.  (Failures are rare.) */

/*                If INFO .GT. 0 and WANT is .FALSE., then on exit, */
/*                the remaining unconverged eigenvalues are the eigen- */
/*                values of the upper Hessenberg matrix rows and */
/*                columns ILO through INFO of the final, output */
/*                value of H. */

/*                If INFO .GT. 0 and WANTT is .TRUE., then on exit */

/*           (*)  (initial value of H)*U  = U*(final value of H) */

/*                where U is an orthogonal matrix.  The final */
/*                value of H is upper Hessenberg and quasi-triangular */
/*                in rows and columns INFO+1 through IHI. */

/*                If INFO .GT. 0 and WANTZ is .TRUE., then on exit */

/*                  (final value of Z(ILO:IHI,ILOZ:IHIZ) */
/*                   =  (initial value of Z(ILO:IHI,ILOZ:IHIZ)*U */

/*                where U is the orthogonal matrix in (*) (regard- */
/*                less of the value of WANTT.) */

/*                If INFO .GT. 0 and WANTZ is .FALSE., then Z is not */
/*                accessed. */

/*     ================================================================ */
/*     Based on contributions by */
/*        Karen Braman and Ralph Byers, Department of Mathematics, */
/*        University of Kansas, USA */

/*     ================================================================ */
/*     References: */
/*       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR */
/*       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3 */
/*       Performance, SIAM Journal of Matrix Analysis, volume 23, pages */
/*       929--947, 2002. */

/*       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR */
/*       Algorithm Part II: Aggressive Early Deflation, SIAM Journal */
/*       of Matrix Analysis, volume 23, pages 948--973, 2002. */

/*     ================================================================ */
/*     .. Parameters .. */

/*     ==== Matrices of order NTINY or smaller must be processed by */
/*     .    SLAHQR because of insufficient subdiagonal scratch space. */
/*     .    (This is a hard limit.) ==== */

/*     ==== Exceptional deflation windows:  try to cure rare */
/*     .    slow convergence by varying the size of the */
/*     .    deflation window after KEXNW iterations. ==== */

/*     ==== Exceptional shifts: try to cure rare slow convergence */
/*     .    with ad-hoc exceptional shifts every KEXSH iterations. */
/*     .    ==== */

/*     ==== The constants WILK1 and WILK2 are used to form the */
/*     .    exceptional shifts. ==== */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Local Arrays .. */
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
    --work;

    /* Function Body */
    *info = 0;

/*     ==== Quick return for N = 0: nothing to do. ==== */

    if (*n == 0) {
	work[1] = 1.f;
	return 0;
    }

    if (*n <= 11) {

/*        ==== Tiny matrices must use SLAHQR. ==== */

	lwkopt = 1;
	if (*lwork != -1) {
	    slahqr_(wantt, wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1], &
		    wi[1], iloz, ihiz, &z__[z_offset], ldz, info);
	}
    } else {

/*        ==== Use small bulge multi-shift QR with aggressive early */
/*        .    deflation on larger-than-tiny matrices. ==== */

/*        ==== Hope for the best. ==== */

	*info = 0;

/*        ==== Set up job flags for ILAENV. ==== */

	if (*wantt) {
	    *(unsigned char *)jbcmpz = 'S';
	} else {
	    *(unsigned char *)jbcmpz = 'E';
	}
	if (*wantz) {
	    *(unsigned char *)&jbcmpz[1] = 'V';
	} else {
	    *(unsigned char *)&jbcmpz[1] = 'N';
	}

/*        ==== NWR = recommended deflation window size.  At this */
/*        .    point,  N .GT. NTINY = 11, so there is enough */
/*        .    subdiagonal workspace for NWR.GE.2 as required. */
/*        .    (In fact, there is enough subdiagonal space for */
/*        .    NWR.GE.3.) ==== */

	nwr = ilaenv_(&c__13, "SLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nwr = max(2,nwr);
/* Computing MIN */
	i__1 = *ihi - *ilo + 1, i__2 = (*n - 1) / 3, i__1 = min(i__1,i__2);
	nwr = min(i__1,nwr);

/*        ==== NSR = recommended number of simultaneous shifts. */
/*        .    At this point N .GT. NTINY = 11, so there is at */
/*        .    enough subdiagonal workspace for NSR to be even */
/*        .    and greater than or equal to two as required. ==== */

	nsr = ilaenv_(&c__15, "SLAQR4", jbcmpz, n, ilo, ihi, lwork);
/* Computing MIN */
	i__1 = nsr, i__2 = (*n + 6) / 9, i__1 = min(i__1,i__2), i__2 = *ihi - 
		*ilo;
	nsr = min(i__1,i__2);
/* Computing MAX */
	i__1 = 2, i__2 = nsr - nsr % 2;
	nsr = max(i__1,i__2);

/*        ==== Estimate optimal workspace ==== */

/*        ==== Workspace query call to SLAQR2 ==== */

	i__1 = nwr + 1;
	slaqr2_(wantt, wantz, n, ilo, ihi, &i__1, &h__[h_offset], ldh, iloz, 
		ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1], &h__[
		h_offset], ldh, n, &h__[h_offset], ldh, n, &h__[h_offset], 
		ldh, &work[1], &c_n1);

/*        ==== Optimal workspace = MAX(SLAQR5, SLAQR2) ==== */

/* Computing MAX */
	i__1 = nsr * 3 / 2, i__2 = (integer) work[1];
	lwkopt = max(i__1,i__2);

/*        ==== Quick return in case of workspace query. ==== */

	if (*lwork == -1) {
	    work[1] = (real) lwkopt;
	    return 0;
	}

/*        ==== SLAHQR/SLAQR0 crossover point ==== */

	nmin = ilaenv_(&c__12, "SLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nmin = max(11,nmin);

/*        ==== Nibble crossover point ==== */

	nibble = ilaenv_(&c__14, "SLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nibble = max(0,nibble);

/*        ==== Accumulate reflections during ttswp?  Use block */
/*        .    2-by-2 structure during matrix-matrix multiply? ==== */

	kacc22 = ilaenv_(&c__16, "SLAQR4", jbcmpz, n, ilo, ihi, lwork);
	kacc22 = max(0,kacc22);
	kacc22 = min(2,kacc22);

/*        ==== NWMAX = the largest possible deflation window for */
/*        .    which there is sufficient workspace. ==== */

/* Computing MIN */
	i__1 = (*n - 1) / 3, i__2 = *lwork / 2;
	nwmax = min(i__1,i__2);
	nw = nwmax;

/*        ==== NSMAX = the Largest number of simultaneous shifts */
/*        .    for which there is sufficient workspace. ==== */

/* Computing MIN */
	i__1 = (*n + 6) / 9, i__2 = (*lwork << 1) / 3;
	nsmax = min(i__1,i__2);
	nsmax -= nsmax % 2;

/*        ==== NDFL: an iteration count restarted at deflation. ==== */

	ndfl = 1;

/*        ==== ITMAX = iteration limit ==== */

/* Computing MAX */
	i__1 = 10, i__2 = *ihi - *ilo + 1;
	itmax = max(i__1,i__2) * 30;

/*        ==== Last row and column in the active block ==== */

	kbot = *ihi;

/*        ==== Main Loop ==== */

	i__1 = itmax;
	for (it = 1; it <= i__1; ++it) {

/*           ==== Done when KBOT falls below ILO ==== */

	    if (kbot < *ilo) {
		goto L90;
	    }

/*           ==== Locate active block ==== */

	    i__2 = *ilo + 1;
	    for (k = kbot; k >= i__2; --k) {
		if (h__[k + (k - 1) * h_dim1] == 0.f) {
		    goto L20;
		}
/* L10: */
	    }
	    k = *ilo;
L20:
	    ktop = k;

/*           ==== Select deflation window size: */
/*           .    Typical Case: */
/*           .      If possible and advisable, nibble the entire */
/*           .      active block.  If not, use size MIN(NWR,NWMAX) */
/*           .      or MIN(NWR+1,NWMAX) depending upon which has */
/*           .      the smaller corresponding subdiagonal entry */
/*           .      (a heuristic). */
/*           . */
/*           .    Exceptional Case: */
/*           .      If there have been no deflations in KEXNW or */
/*           .      more iterations, then vary the deflation window */
/*           .      size.   At first, because, larger windows are, */
/*           .      in general, more powerful than smaller ones, */
/*           .      rapidly increase the window to the maximum possible. */
/*           .      Then, gradually reduce the window size. ==== */

	    nh = kbot - ktop + 1;
	    nwupbd = min(nh,nwmax);
	    if (ndfl < 5) {
		nw = min(nwupbd,nwr);
	    } else {
/* Computing MIN */
		i__2 = nwupbd, i__3 = nw << 1;
		nw = min(i__2,i__3);
	    }
	    if (nw < nwmax) {
		if (nw >= nh - 1) {
		    nw = nh;
		} else {
		    kwtop = kbot - nw + 1;
		    if ((r__1 = h__[kwtop + (kwtop - 1) * h_dim1], dabs(r__1))
			     > (r__2 = h__[kwtop - 1 + (kwtop - 2) * h_dim1], 
			    dabs(r__2))) {
			++nw;
		    }
		}
	    }
	    if (ndfl < 5) {
		ndec = -1;
	    } else if (ndec >= 0 || nw >= nwupbd) {
		++ndec;
		if (nw - ndec < 2) {
		    ndec = 0;
		}
		nw -= ndec;
	    }

/*           ==== Aggressive early deflation: */
/*           .    split workspace under the subdiagonal into */
/*           .      - an nw-by-nw work array V in the lower */
/*           .        left-hand-corner, */
/*           .      - an NW-by-at-least-NW-but-more-is-better */
/*           .        (NW-by-NHO) horizontal work array along */
/*           .        the bottom edge, */
/*           .      - an at-least-NW-but-more-is-better (NHV-by-NW) */
/*           .        vertical work array along the left-hand-edge. */
/*           .        ==== */

	    kv = *n - nw + 1;
	    kt = nw + 1;
	    nho = *n - nw - 1 - kt + 1;
	    kwv = nw + 2;
	    nve = *n - nw - kwv + 1;

/*           ==== Aggressive early deflation ==== */

	    slaqr2_(wantt, wantz, n, &ktop, &kbot, &nw, &h__[h_offset], ldh, 
		    iloz, ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1], 
		     &h__[kv + h_dim1], ldh, &nho, &h__[kv + kt * h_dim1], 
		    ldh, &nve, &h__[kwv + h_dim1], ldh, &work[1], lwork);

/*           ==== Adjust KBOT accounting for new deflations. ==== */

	    kbot -= ld;

/*           ==== KS points to the shifts. ==== */

	    ks = kbot - ls + 1;

/*           ==== Skip an expensive QR sweep if there is a (partly */
/*           .    heuristic) reason to expect that many eigenvalues */
/*           .    will deflate without it.  Here, the QR sweep is */
/*           .    skipped if many eigenvalues have just been deflated */
/*           .    or if the remaining active block is small. */

	    if (ld == 0 || ld * 100 <= nw * nibble && kbot - ktop + 1 > min(
		    nmin,nwmax)) {

/*              ==== NS = nominal number of simultaneous shifts. */
/*              .    This may be lowered (slightly) if SLAQR2 */
/*              .    did not provide that many shifts. ==== */

/* Computing MIN */
/* Computing MAX */
		i__4 = 2, i__5 = kbot - ktop;
		i__2 = min(nsmax,nsr), i__3 = max(i__4,i__5);
		ns = min(i__2,i__3);
		ns -= ns % 2;

/*              ==== If there have been no deflations */
/*              .    in a multiple of KEXSH iterations, */
/*              .    then try exceptional shifts. */
/*              .    Otherwise use shifts provided by */
/*              .    SLAQR2 above or from the eigenvalues */
/*              .    of a trailing principal submatrix. ==== */

		if (ndfl % 6 == 0) {
		    ks = kbot - ns + 1;
/* Computing MAX */
		    i__3 = ks + 1, i__4 = ktop + 2;
		    i__2 = max(i__3,i__4);
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			ss = (r__1 = h__[i__ + (i__ - 1) * h_dim1], dabs(r__1)
				) + (r__2 = h__[i__ - 1 + (i__ - 2) * h_dim1],
				 dabs(r__2));
			aa = ss * .75f + h__[i__ + i__ * h_dim1];
			bb = ss;
			cc = ss * -.4375f;
			dd = aa;
			slanv2_(&aa, &bb, &cc, &dd, &wr[i__ - 1], &wi[i__ - 1]
, &wr[i__], &wi[i__], &cs, &sn);
/* L30: */
		    }
		    if (ks == ktop) {
			wr[ks + 1] = h__[ks + 1 + (ks + 1) * h_dim1];
			wi[ks + 1] = 0.f;
			wr[ks] = wr[ks + 1];
			wi[ks] = wi[ks + 1];
		    }
		} else {

/*                 ==== Got NS/2 or fewer shifts? Use SLAHQR */
/*                 .    on a trailing principal submatrix to */
/*                 .    get more. (Since NS.LE.NSMAX.LE.(N+6)/9, */
/*                 .    there is enough space below the subdiagonal */
/*                 .    to fit an NS-by-NS scratch array.) ==== */

		    if (kbot - ks + 1 <= ns / 2) {
			ks = kbot - ns + 1;
			kt = *n - ns + 1;
			slacpy_("A", &ns, &ns, &h__[ks + ks * h_dim1], ldh, &
				h__[kt + h_dim1], ldh);
			slahqr_(&c_false, &c_false, &ns, &c__1, &ns, &h__[kt 
				+ h_dim1], ldh, &wr[ks], &wi[ks], &c__1, &
				c__1, zdum, &c__1, &inf);
			ks += inf;

/*                    ==== In case of a rare QR failure use */
/*                    .    eigenvalues of the trailing 2-by-2 */
/*                    .    principal submatrix.  ==== */

			if (ks >= kbot) {
			    aa = h__[kbot - 1 + (kbot - 1) * h_dim1];
			    cc = h__[kbot + (kbot - 1) * h_dim1];
			    bb = h__[kbot - 1 + kbot * h_dim1];
			    dd = h__[kbot + kbot * h_dim1];
			    slanv2_(&aa, &bb, &cc, &dd, &wr[kbot - 1], &wi[
				    kbot - 1], &wr[kbot], &wi[kbot], &cs, &sn)
				    ;
			    ks = kbot - 1;
			}
		    }

		    if (kbot - ks + 1 > ns) {

/*                    ==== Sort the shifts (Helps a little) */
/*                    .    Bubble sort keeps complex conjugate */
/*                    .    pairs together. ==== */

			sorted = FALSE_;
			i__2 = ks + 1;
			for (k = kbot; k >= i__2; --k) {
			    if (sorted) {
				goto L60;
			    }
			    sorted = TRUE_;
			    i__3 = k - 1;
			    for (i__ = ks; i__ <= i__3; ++i__) {
				if ((r__1 = wr[i__], dabs(r__1)) + (r__2 = wi[
					i__], dabs(r__2)) < (r__3 = wr[i__ + 
					1], dabs(r__3)) + (r__4 = wi[i__ + 1],
					 dabs(r__4))) {
				    sorted = FALSE_;

				    swap = wr[i__];
				    wr[i__] = wr[i__ + 1];
				    wr[i__ + 1] = swap;

				    swap = wi[i__];
				    wi[i__] = wi[i__ + 1];
				    wi[i__ + 1] = swap;
				}
/* L40: */
			    }
/* L50: */
			}
L60:
			;
		    }

/*                 ==== Shuffle shifts into pairs of real shifts */
/*                 .    and pairs of complex conjugate shifts */
/*                 .    assuming complex conjugate shifts are */
/*                 .    already adjacent to one another. (Yes, */
/*                 .    they are.)  ==== */

		    i__2 = ks + 2;
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			if (wi[i__] != -wi[i__ - 1]) {

			    swap = wr[i__];
			    wr[i__] = wr[i__ - 1];
			    wr[i__ - 1] = wr[i__ - 2];
			    wr[i__ - 2] = swap;

			    swap = wi[i__];
			    wi[i__] = wi[i__ - 1];
			    wi[i__ - 1] = wi[i__ - 2];
			    wi[i__ - 2] = swap;
			}
/* L70: */
		    }
		}

/*              ==== If there are only two shifts and both are */
/*              .    real, then use only one.  ==== */

		if (kbot - ks + 1 == 2) {
		    if (wi[kbot] == 0.f) {
			if ((r__1 = wr[kbot] - h__[kbot + kbot * h_dim1], 
				dabs(r__1)) < (r__2 = wr[kbot - 1] - h__[kbot 
				+ kbot * h_dim1], dabs(r__2))) {
			    wr[kbot - 1] = wr[kbot];
			} else {
			    wr[kbot] = wr[kbot - 1];
			}
		    }
		}

/*              ==== Use up to NS of the the smallest magnatiude */
/*              .    shifts.  If there aren't NS shifts available, */
/*              .    then use them all, possibly dropping one to */
/*              .    make the number of shifts even. ==== */

/* Computing MIN */
		i__2 = ns, i__3 = kbot - ks + 1;
		ns = min(i__2,i__3);
		ns -= ns % 2;
		ks = kbot - ns + 1;

/*              ==== Small-bulge multi-shift QR sweep: */
/*              .    split workspace under the subdiagonal into */
/*              .    - a KDU-by-KDU work array U in the lower */
/*              .      left-hand-corner, */
/*              .    - a KDU-by-at-least-KDU-but-more-is-better */
/*              .      (KDU-by-NHo) horizontal work array WH along */
/*              .      the bottom edge, */
/*              .    - and an at-least-KDU-but-more-is-better-by-KDU */
/*              .      (NVE-by-KDU) vertical work WV arrow along */
/*              .      the left-hand-edge. ==== */

		kdu = ns * 3 - 3;
		ku = *n - kdu + 1;
		kwh = kdu + 1;
		nho = *n - kdu - 3 - (kdu + 1) + 1;
		kwv = kdu + 4;
		nve = *n - kdu - kwv + 1;

/*              ==== Small-bulge multi-shift QR sweep ==== */

		slaqr5_(wantt, wantz, &kacc22, n, &ktop, &kbot, &ns, &wr[ks], 
			&wi[ks], &h__[h_offset], ldh, iloz, ihiz, &z__[
			z_offset], ldz, &work[1], &c__3, &h__[ku + h_dim1], 
			ldh, &nve, &h__[kwv + h_dim1], ldh, &nho, &h__[ku + 
			kwh * h_dim1], ldh);
	    }

/*           ==== Note progress (or the lack of it). ==== */

	    if (ld > 0) {
		ndfl = 1;
	    } else {
		++ndfl;
	    }

/*           ==== End of main loop ==== */
/* L80: */
	}

/*        ==== Iteration limit exceeded.  Set INFO to show where */
/*        .    the problem occurred and exit. ==== */

	*info = kbot;
L90:
	;
    }

/*     ==== Return the optimal value of LWORK. ==== */

    work[1] = (real) lwkopt;

/*     ==== End of SLAQR4 ==== */

    return 0;
} /* slaqr4_ */
