/* zlaqr3.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {0.,0.};
static doublecomplex c_b2 = {1.,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static logical c_true = TRUE_;
static integer c__12 = 12;

/* Subroutine */ int zlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublecomplex *h__, 
	integer *ldh, integer *iloz, integer *ihiz, doublecomplex *z__, 
	integer *ldz, integer *ns, integer *nd, doublecomplex *sh, 
	doublecomplex *v, integer *ldv, integer *nh, doublecomplex *t, 
	integer *ldt, integer *nv, doublecomplex *wv, integer *ldwv, 
	doublecomplex *work, integer *lwork)
{
    /* System generated locals */
    integer h_dim1, h_offset, t_dim1, t_offset, v_dim1, v_offset, wv_dim1, 
	    wv_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;
    doublecomplex z__1, z__2;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j;
    doublecomplex s;
    integer jw;
    doublereal foo;
    integer kln;
    doublecomplex tau;
    integer knt;
    doublereal ulp;
    integer lwk1, lwk2, lwk3;
    doublecomplex beta;
    integer kcol, info, nmin, ifst, ilst, ltop, krow;
    extern /* Subroutine */ int zlarf_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, doublecomplex *);
    integer infqr;
    extern /* Subroutine */ int zgemm_(char *, char *, integer *, integer *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *);
    integer kwtop;
    extern /* Subroutine */ int zcopy_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *), dlabad_(doublereal *, doublereal *), 
	    zlaqr4_(logical *, logical *, integer *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *, 
	     doublecomplex *, integer *, doublecomplex *, integer *, integer *
);
    extern doublereal dlamch_(char *);
    doublereal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    doublereal safmax;
    extern /* Subroutine */ int zgehrd_(integer *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *, integer *), zlarfg_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *), zlahqr_(logical *, 
	    logical *, integer *, integer *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, integer *, integer *, doublecomplex *, 
	     integer *, integer *), zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), 
	    zlaset_(char *, integer *, integer *, doublecomplex *, 
	    doublecomplex *, doublecomplex *, integer *);
    doublereal smlnum;
    extern /* Subroutine */ int ztrexc_(char *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, integer *, integer *, integer *, 
	    integer *);
    integer lwkopt;
    extern /* Subroutine */ int zunmhr_(char *, char *, integer *, integer *, 
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *, 
	     doublecomplex *, integer *, doublecomplex *, integer *, integer *
);


/*  -- LAPACK auxiliary routine (version 3.2.1)                        -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.. */
/*  -- April 2009                                                      -- */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*     ****************************************************************** */
/*     Aggressive early deflation: */

/*     This subroutine accepts as input an upper Hessenberg matrix */
/*     H and performs an unitary similarity transformation */
/*     designed to detect and deflate fully converged eigenvalues from */
/*     a trailing principal submatrix.  On output H has been over- */
/*     written by a new Hessenberg matrix that is a perturbation of */
/*     an unitary similarity transformation of H.  It is to be */
/*     hoped that the final version of H has many zero subdiagonal */
/*     entries. */

/*     ****************************************************************** */
/*     WANTT   (input) LOGICAL */
/*          If .TRUE., then the Hessenberg matrix H is fully updated */
/*          so that the triangular Schur factor may be */
/*          computed (in cooperation with the calling subroutine). */
/*          If .FALSE., then only enough of H is updated to preserve */
/*          the eigenvalues. */

/*     WANTZ   (input) LOGICAL */
/*          If .TRUE., then the unitary matrix Z is updated so */
/*          so that the unitary Schur factor may be computed */
/*          (in cooperation with the calling subroutine). */
/*          If .FALSE., then Z is not referenced. */

/*     N       (input) INTEGER */
/*          The order of the matrix H and (if WANTZ is .TRUE.) the */
/*          order of the unitary matrix Z. */

/*     KTOP    (input) INTEGER */
/*          It is assumed that either KTOP = 1 or H(KTOP,KTOP-1)=0. */
/*          KBOT and KTOP together determine an isolated block */
/*          along the diagonal of the Hessenberg matrix. */

/*     KBOT    (input) INTEGER */
/*          It is assumed without a check that either */
/*          KBOT = N or H(KBOT+1,KBOT)=0.  KBOT and KTOP together */
/*          determine an isolated block along the diagonal of the */
/*          Hessenberg matrix. */

/*     NW      (input) INTEGER */
/*          Deflation window size.  1 .LE. NW .LE. (KBOT-KTOP+1). */

/*     H       (input/output) COMPLEX*16 array, dimension (LDH,N) */
/*          On input the initial N-by-N section of H stores the */
/*          Hessenberg matrix undergoing aggressive early deflation. */
/*          On output H has been transformed by a unitary */
/*          similarity transformation, perturbed, and the returned */
/*          to Hessenberg form that (it is to be hoped) has some */
/*          zero subdiagonal entries. */

/*     LDH     (input) integer */
/*          Leading dimension of H just as declared in the calling */
/*          subroutine.  N .LE. LDH */

/*     ILOZ    (input) INTEGER */
/*     IHIZ    (input) INTEGER */
/*          Specify the rows of Z to which transformations must be */
/*          applied if WANTZ is .TRUE.. 1 .LE. ILOZ .LE. IHIZ .LE. N. */

/*     Z       (input/output) COMPLEX*16 array, dimension (LDZ,N) */
/*          IF WANTZ is .TRUE., then on output, the unitary */
/*          similarity transformation mentioned above has been */
/*          accumulated into Z(ILOZ:IHIZ,ILO:IHI) from the right. */
/*          If WANTZ is .FALSE., then Z is unreferenced. */

/*     LDZ     (input) integer */
/*          The leading dimension of Z just as declared in the */
/*          calling subroutine.  1 .LE. LDZ. */

/*     NS      (output) integer */
/*          The number of unconverged (ie approximate) eigenvalues */
/*          returned in SR and SI that may be used as shifts by the */
/*          calling subroutine. */

/*     ND      (output) integer */
/*          The number of converged eigenvalues uncovered by this */
/*          subroutine. */

/*     SH      (output) COMPLEX*16 array, dimension KBOT */
/*          On output, approximate eigenvalues that may */
/*          be used for shifts are stored in SH(KBOT-ND-NS+1) */
/*          through SR(KBOT-ND).  Converged eigenvalues are */
/*          stored in SH(KBOT-ND+1) through SH(KBOT). */

/*     V       (workspace) COMPLEX*16 array, dimension (LDV,NW) */
/*          An NW-by-NW work array. */

/*     LDV     (input) integer scalar */
/*          The leading dimension of V just as declared in the */
/*          calling subroutine.  NW .LE. LDV */

/*     NH      (input) integer scalar */
/*          The number of columns of T.  NH.GE.NW. */

/*     T       (workspace) COMPLEX*16 array, dimension (LDT,NW) */

/*     LDT     (input) integer */
/*          The leading dimension of T just as declared in the */
/*          calling subroutine.  NW .LE. LDT */

/*     NV      (input) integer */
/*          The number of rows of work array WV available for */
/*          workspace.  NV.GE.NW. */

/*     WV      (workspace) COMPLEX*16 array, dimension (LDWV,NW) */

/*     LDWV    (input) integer */
/*          The leading dimension of W just as declared in the */
/*          calling subroutine.  NW .LE. LDV */

/*     WORK    (workspace) COMPLEX*16 array, dimension LWORK. */
/*          On exit, WORK(1) is set to an estimate of the optimal value */
/*          of LWORK for the given values of N, NW, KTOP and KBOT. */

/*     LWORK   (input) integer */
/*          The dimension of the work array WORK.  LWORK = 2*NW */
/*          suffices, but greater efficiency may result from larger */
/*          values of LWORK. */

/*          If LWORK = -1, then a workspace query is assumed; ZLAQR3 */
/*          only estimates the optimal workspace size for the given */
/*          values of N, NW, KTOP and KBOT.  The estimate is returned */
/*          in WORK(1).  No error message related to LWORK is issued */
/*          by XERBLA.  Neither H nor Z are accessed. */

/*     ================================================================ */
/*     Based on contributions by */
/*        Karen Braman and Ralph Byers, Department of Mathematics, */
/*        University of Kansas, USA */

/*     ================================================================ */
/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
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

/*     ==== Estimate optimal workspace. ==== */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --sh;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    --work;

    /* Function Body */
/* Computing MIN */
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    if (jw <= 2) {
	lwkopt = 1;
    } else {

/*        ==== Workspace query call to ZGEHRD ==== */

	i__1 = jw - 1;
	zgehrd_(&jw, &c__1, &i__1, &t[t_offset], ldt, &work[1], &work[1], &
		c_n1, &info);
	lwk1 = (integer) work[1].r;

/*        ==== Workspace query call to ZUNMHR ==== */

	i__1 = jw - 1;
	zunmhr_("R", "N", &jw, &jw, &c__1, &i__1, &t[t_offset], ldt, &work[1], 
		 &v[v_offset], ldv, &work[1], &c_n1, &info);
	lwk2 = (integer) work[1].r;

/*        ==== Workspace query call to ZLAQR4 ==== */

	zlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[1], 
		&c__1, &jw, &v[v_offset], ldv, &work[1], &c_n1, &infqr);
	lwk3 = (integer) work[1].r;

/*        ==== Optimal workspace ==== */

/* Computing MAX */
	i__1 = jw + max(lwk1,lwk2);
	lwkopt = max(i__1,lwk3);
    }

/*     ==== Quick return in case of workspace query. ==== */

    if (*lwork == -1) {
	d__1 = (doublereal) lwkopt;
	z__1.r = d__1, z__1.i = 0.;
	work[1].r = z__1.r, work[1].i = z__1.i;
	return 0;
    }

/*     ==== Nothing to do ... */
/*     ... for an empty active block ... ==== */
    *ns = 0;
    *nd = 0;
    work[1].r = 1., work[1].i = 0.;
    if (*ktop > *kbot) {
	return 0;
    }
/*     ... nor for an empty deflation window. ==== */
    if (*nw < 1) {
	return 0;
    }

/*     ==== Machine constants ==== */

    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((doublereal) (*n) / ulp);

/*     ==== Setup deflation window ==== */

/* Computing MIN */
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    kwtop = *kbot - jw + 1;
    if (kwtop == *ktop) {
	s.r = 0., s.i = 0.;
    } else {
	i__1 = kwtop + (kwtop - 1) * h_dim1;
	s.r = h__[i__1].r, s.i = h__[i__1].i;
    }

    if (*kbot == kwtop) {

/*        ==== 1-by-1 deflation window: not much to do ==== */

	i__1 = kwtop;
	i__2 = kwtop + kwtop * h_dim1;
	sh[i__1].r = h__[i__2].r, sh[i__1].i = h__[i__2].i;
	*ns = 1;
	*nd = 0;
/* Computing MAX */
	i__1 = kwtop + kwtop * h_dim1;
	d__5 = smlnum, d__6 = ulp * ((d__1 = h__[i__1].r, abs(d__1)) + (d__2 =
		 d_imag(&h__[kwtop + kwtop * h_dim1]), abs(d__2)));
	if ((d__3 = s.r, abs(d__3)) + (d__4 = d_imag(&s), abs(d__4)) <= max(
		d__5,d__6)) {
	    *ns = 0;
	    *nd = 1;
	    if (kwtop > *ktop) {
		i__1 = kwtop + (kwtop - 1) * h_dim1;
		h__[i__1].r = 0., h__[i__1].i = 0.;
	    }
	}
	work[1].r = 1., work[1].i = 0.;
	return 0;
    }

/*     ==== Convert to spike-triangular form.  (In case of a */
/*     .    rare QR failure, this routine continues to do */
/*     .    aggressive early deflation using that part of */
/*     .    the deflation window that converged using INFQR */
/*     .    here and there to keep track.) ==== */

    zlacpy_("U", &jw, &jw, &h__[kwtop + kwtop * h_dim1], ldh, &t[t_offset], 
	    ldt);
    i__1 = jw - 1;
    i__2 = *ldh + 1;
    i__3 = *ldt + 1;
    zcopy_(&i__1, &h__[kwtop + 1 + kwtop * h_dim1], &i__2, &t[t_dim1 + 2], &
	    i__3);

    zlaset_("A", &jw, &jw, &c_b1, &c_b2, &v[v_offset], ldv);
    nmin = ilaenv_(&c__12, "ZLAQR3", "SV", &jw, &c__1, &jw, lwork);
    if (jw > nmin) {
	zlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[
		kwtop], &c__1, &jw, &v[v_offset], ldv, &work[1], lwork, &
		infqr);
    } else {
	zlahqr_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sh[
		kwtop], &c__1, &jw, &v[v_offset], ldv, &infqr);
    }

/*     ==== Deflation detection loop ==== */

    *ns = jw;
    ilst = infqr + 1;
    i__1 = jw;
    for (knt = infqr + 1; knt <= i__1; ++knt) {

/*        ==== Small spike tip deflation test ==== */

	i__2 = *ns + *ns * t_dim1;
	foo = (d__1 = t[i__2].r, abs(d__1)) + (d__2 = d_imag(&t[*ns + *ns * 
		t_dim1]), abs(d__2));
	if (foo == 0.) {
	    foo = (d__1 = s.r, abs(d__1)) + (d__2 = d_imag(&s), abs(d__2));
	}
	i__2 = *ns * v_dim1 + 1;
/* Computing MAX */
	d__5 = smlnum, d__6 = ulp * foo;
	if (((d__1 = s.r, abs(d__1)) + (d__2 = d_imag(&s), abs(d__2))) * ((
		d__3 = v[i__2].r, abs(d__3)) + (d__4 = d_imag(&v[*ns * v_dim1 
		+ 1]), abs(d__4))) <= max(d__5,d__6)) {

/*           ==== One more converged eigenvalue ==== */

	    --(*ns);
	} else {

/*           ==== One undeflatable eigenvalue.  Move it up out of the */
/*           .    way.   (ZTREXC can not fail in this case.) ==== */

	    ifst = *ns;
	    ztrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst, &
		    ilst, &info);
	    ++ilst;
	}
/* L10: */
    }

/*        ==== Return to Hessenberg form ==== */

    if (*ns == 0) {
	s.r = 0., s.i = 0.;
    }

    if (*ns < jw) {

/*        ==== sorting the diagonal of T improves accuracy for */
/*        .    graded matrices.  ==== */

	i__1 = *ns;
	for (i__ = infqr + 1; i__ <= i__1; ++i__) {
	    ifst = i__;
	    i__2 = *ns;
	    for (j = i__ + 1; j <= i__2; ++j) {
		i__3 = j + j * t_dim1;
		i__4 = ifst + ifst * t_dim1;
		if ((d__1 = t[i__3].r, abs(d__1)) + (d__2 = d_imag(&t[j + j * 
			t_dim1]), abs(d__2)) > (d__3 = t[i__4].r, abs(d__3)) 
			+ (d__4 = d_imag(&t[ifst + ifst * t_dim1]), abs(d__4))
			) {
		    ifst = j;
		}
/* L20: */
	    }
	    ilst = i__;
	    if (ifst != ilst) {
		ztrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst, 
			 &ilst, &info);
	    }
/* L30: */
	}
    }

/*     ==== Restore shift/eigenvalue array from T ==== */

    i__1 = jw;
    for (i__ = infqr + 1; i__ <= i__1; ++i__) {
	i__2 = kwtop + i__ - 1;
	i__3 = i__ + i__ * t_dim1;
	sh[i__2].r = t[i__3].r, sh[i__2].i = t[i__3].i;
/* L40: */
    }


    if (*ns < jw || s.r == 0. && s.i == 0.) {
	if (*ns > 1 && (s.r != 0. || s.i != 0.)) {

/*           ==== Reflect spike back into lower triangle ==== */

	    zcopy_(ns, &v[v_offset], ldv, &work[1], &c__1);
	    i__1 = *ns;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		d_cnjg(&z__1, &work[i__]);
		work[i__2].r = z__1.r, work[i__2].i = z__1.i;
/* L50: */
	    }
	    beta.r = work[1].r, beta.i = work[1].i;
	    zlarfg_(ns, &beta, &work[2], &c__1, &tau);
	    work[1].r = 1., work[1].i = 0.;

	    i__1 = jw - 2;
	    i__2 = jw - 2;
	    zlaset_("L", &i__1, &i__2, &c_b1, &c_b1, &t[t_dim1 + 3], ldt);

	    d_cnjg(&z__1, &tau);
	    zlarf_("L", ns, &jw, &work[1], &c__1, &z__1, &t[t_offset], ldt, &
		    work[jw + 1]);
	    zlarf_("R", ns, ns, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    zlarf_("R", &jw, ns, &work[1], &c__1, &tau, &v[v_offset], ldv, &
		    work[jw + 1]);

	    i__1 = *lwork - jw;
	    zgehrd_(&jw, &c__1, ns, &t[t_offset], ldt, &work[1], &work[jw + 1]
, &i__1, &info);
	}

/*        ==== Copy updated reduced window into place ==== */

	if (kwtop > 1) {
	    i__1 = kwtop + (kwtop - 1) * h_dim1;
	    d_cnjg(&z__2, &v[v_dim1 + 1]);
	    z__1.r = s.r * z__2.r - s.i * z__2.i, z__1.i = s.r * z__2.i + s.i 
		    * z__2.r;
	    h__[i__1].r = z__1.r, h__[i__1].i = z__1.i;
	}
	zlacpy_("U", &jw, &jw, &t[t_offset], ldt, &h__[kwtop + kwtop * h_dim1]
, ldh);
	i__1 = jw - 1;
	i__2 = *ldt + 1;
	i__3 = *ldh + 1;
	zcopy_(&i__1, &t[t_dim1 + 2], &i__2, &h__[kwtop + 1 + kwtop * h_dim1], 
		 &i__3);

/*        ==== Accumulate orthogonal matrix in order update */
/*        .    H and Z, if requested.  ==== */

	if (*ns > 1 && (s.r != 0. || s.i != 0.)) {
	    i__1 = *lwork - jw;
	    zunmhr_("R", "N", &jw, ns, &c__1, ns, &t[t_offset], ldt, &work[1], 
		     &v[v_offset], ldv, &work[jw + 1], &i__1, &info);
	}

/*        ==== Update vertical slab in H ==== */

	if (*wantt) {
	    ltop = 1;
	} else {
	    ltop = *ktop;
	}
	i__1 = kwtop - 1;
	i__2 = *nv;
	for (krow = ltop; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow += 
		i__2) {
/* Computing MIN */
	    i__3 = *nv, i__4 = kwtop - krow;
	    kln = min(i__3,i__4);
	    zgemm_("N", "N", &kln, &jw, &jw, &c_b2, &h__[krow + kwtop * 
		    h_dim1], ldh, &v[v_offset], ldv, &c_b1, &wv[wv_offset], 
		    ldwv);
	    zlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &h__[krow + kwtop * 
		    h_dim1], ldh);
/* L60: */
	}

/*        ==== Update horizontal slab in H ==== */

	if (*wantt) {
	    i__2 = *n;
	    i__1 = *nh;
	    for (kcol = *kbot + 1; i__1 < 0 ? kcol >= i__2 : kcol <= i__2; 
		    kcol += i__1) {
/* Computing MIN */
		i__3 = *nh, i__4 = *n - kcol + 1;
		kln = min(i__3,i__4);
		zgemm_("C", "N", &jw, &kln, &jw, &c_b2, &v[v_offset], ldv, &
			h__[kwtop + kcol * h_dim1], ldh, &c_b1, &t[t_offset], 
			ldt);
		zlacpy_("A", &jw, &kln, &t[t_offset], ldt, &h__[kwtop + kcol *
			 h_dim1], ldh);
/* L70: */
	    }
	}

/*        ==== Update vertical slab in Z ==== */

	if (*wantz) {
	    i__1 = *ihiz;
	    i__2 = *nv;
	    for (krow = *iloz; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		     i__2) {
/* Computing MIN */
		i__3 = *nv, i__4 = *ihiz - krow + 1;
		kln = min(i__3,i__4);
		zgemm_("N", "N", &kln, &jw, &jw, &c_b2, &z__[krow + kwtop * 
			z_dim1], ldz, &v[v_offset], ldv, &c_b1, &wv[wv_offset]
, ldwv);
		zlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &z__[krow + 
			kwtop * z_dim1], ldz);
/* L80: */
	    }
	}
    }

/*     ==== Return the number of deflations ... ==== */

    *nd = jw - *ns;

/*     ==== ... and the number of shifts. (Subtracting */
/*     .    INFQR from the spike length takes care */
/*     .    of the case of a rare QR failure while */
/*     .    calculating eigenvalues of the deflation */
/*     .    window.)  ==== */

    *ns -= infqr;

/*      ==== Return optimal workspace. ==== */

    d__1 = (doublereal) lwkopt;
    z__1.r = d__1, z__1.i = 0.;
    work[1].r = z__1.r, work[1].i = z__1.i;

/*     ==== End of ZLAQR3 ==== */

    return 0;
} /* zlaqr3_ */
