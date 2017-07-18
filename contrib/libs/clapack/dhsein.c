/* dhsein.f -- translated by f2c (version 20061008).
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

static logical c_false = FALSE_;
static logical c_true = TRUE_;

/* Subroutine */ int dhsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *mm, integer *m, doublereal *work, integer *
	ifaill, integer *ifailr, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1, 
	    i__2;
    doublereal d__1, d__2;

    /* Local variables */
    integer i__, k, kl, kr, kln, ksi;
    doublereal wki;
    integer ksr;
    doublereal ulp, wkr, eps3;
    logical pair;
    doublereal unfl;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical leftv, bothv;
    doublereal hnorm;
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlaein_(logical *, logical *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, integer *, doublereal *, doublereal *
, doublereal *, doublereal *, integer *);
    extern doublereal dlanhs_(char *, integer *, doublereal *, integer *, 
	    doublereal *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;
    logical noinit;
    integer ldwork;
    logical rightv, fromqr;
    doublereal smlnum;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DHSEIN uses inverse iteration to find specified right and/or left */
/*  eigenvectors of a real upper Hessenberg matrix H. */

/*  The right eigenvector x and the left eigenvector y of the matrix H */
/*  corresponding to an eigenvalue w are defined by: */

/*               H * x = w * x,     y**h * H = w * y**h */

/*  where y**h denotes the conjugate transpose of the vector y. */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'R': compute right eigenvectors only; */
/*          = 'L': compute left eigenvectors only; */
/*          = 'B': compute both right and left eigenvectors. */

/*  EIGSRC  (input) CHARACTER*1 */
/*          Specifies the source of eigenvalues supplied in (WR,WI): */
/*          = 'Q': the eigenvalues were found using DHSEQR; thus, if */
/*                 H has zero subdiagonal elements, and so is */
/*                 block-triangular, then the j-th eigenvalue can be */
/*                 assumed to be an eigenvalue of the block containing */
/*                 the j-th row/column.  This property allows DHSEIN to */
/*                 perform inverse iteration on just one diagonal block. */
/*          = 'N': no assumptions are made on the correspondence */
/*                 between eigenvalues and diagonal blocks.  In this */
/*                 case, DHSEIN must always perform inverse iteration */
/*                 using the whole matrix H. */

/*  INITV   (input) CHARACTER*1 */
/*          = 'N': no initial vectors are supplied; */
/*          = 'U': user-supplied initial vectors are stored in the arrays */
/*                 VL and/or VR. */

/*  SELECT  (input/output) LOGICAL array, dimension (N) */
/*          Specifies the eigenvectors to be computed. To select the */
/*          real eigenvector corresponding to a real eigenvalue WR(j), */
/*          SELECT(j) must be set to .TRUE.. To select the complex */
/*          eigenvector corresponding to a complex eigenvalue */
/*          (WR(j),WI(j)), with complex conjugate (WR(j+1),WI(j+1)), */
/*          either SELECT(j) or SELECT(j+1) or both must be set to */
/*          .TRUE.; then on exit SELECT(j) is .TRUE. and SELECT(j+1) is */
/*          .FALSE.. */

/*  N       (input) INTEGER */
/*          The order of the matrix H.  N >= 0. */

/*  H       (input) DOUBLE PRECISION array, dimension (LDH,N) */
/*          The upper Hessenberg matrix H. */

/*  LDH     (input) INTEGER */
/*          The leading dimension of the array H.  LDH >= max(1,N). */

/*  WR      (input/output) DOUBLE PRECISION array, dimension (N) */
/*  WI      (input) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the real and imaginary parts of the eigenvalues of */
/*          H; a complex conjugate pair of eigenvalues must be stored in */
/*          consecutive elements of WR and WI. */
/*          On exit, WR may have been altered since close eigenvalues */
/*          are perturbed slightly in searching for independent */
/*          eigenvectors. */

/*  VL      (input/output) DOUBLE PRECISION array, dimension (LDVL,MM) */
/*          On entry, if INITV = 'U' and SIDE = 'L' or 'B', VL must */
/*          contain starting vectors for the inverse iteration for the */
/*          left eigenvectors; the starting vector for each eigenvector */
/*          must be in the same column(s) in which the eigenvector will */
/*          be stored. */
/*          On exit, if SIDE = 'L' or 'B', the left eigenvectors */
/*          specified by SELECT will be stored consecutively in the */
/*          columns of VL, in the same order as their eigenvalues. A */
/*          complex eigenvector corresponding to a complex eigenvalue is */
/*          stored in two consecutive columns, the first holding the real */
/*          part and the second the imaginary part. */
/*          If SIDE = 'R', VL is not referenced. */

/*  LDVL    (input) INTEGER */
/*          The leading dimension of the array VL. */
/*          LDVL >= max(1,N) if SIDE = 'L' or 'B'; LDVL >= 1 otherwise. */

/*  VR      (input/output) DOUBLE PRECISION array, dimension (LDVR,MM) */
/*          On entry, if INITV = 'U' and SIDE = 'R' or 'B', VR must */
/*          contain starting vectors for the inverse iteration for the */
/*          right eigenvectors; the starting vector for each eigenvector */
/*          must be in the same column(s) in which the eigenvector will */
/*          be stored. */
/*          On exit, if SIDE = 'R' or 'B', the right eigenvectors */
/*          specified by SELECT will be stored consecutively in the */
/*          columns of VR, in the same order as their eigenvalues. A */
/*          complex eigenvector corresponding to a complex eigenvalue is */
/*          stored in two consecutive columns, the first holding the real */
/*          part and the second the imaginary part. */
/*          If SIDE = 'L', VR is not referenced. */

/*  LDVR    (input) INTEGER */
/*          The leading dimension of the array VR. */
/*          LDVR >= max(1,N) if SIDE = 'R' or 'B'; LDVR >= 1 otherwise. */

/*  MM      (input) INTEGER */
/*          The number of columns in the arrays VL and/or VR. MM >= M. */

/*  M       (output) INTEGER */
/*          The number of columns in the arrays VL and/or VR required to */
/*          store the eigenvectors; each selected real eigenvector */
/*          occupies one column and each selected complex eigenvector */
/*          occupies two columns. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension ((N+2)*N) */

/*  IFAILL  (output) INTEGER array, dimension (MM) */
/*          If SIDE = 'L' or 'B', IFAILL(i) = j > 0 if the left */
/*          eigenvector in the i-th column of VL (corresponding to the */
/*          eigenvalue w(j)) failed to converge; IFAILL(i) = 0 if the */
/*          eigenvector converged satisfactorily. If the i-th and (i+1)th */
/*          columns of VL hold a complex eigenvector, then IFAILL(i) and */
/*          IFAILL(i+1) are set to the same value. */
/*          If SIDE = 'R', IFAILL is not referenced. */

/*  IFAILR  (output) INTEGER array, dimension (MM) */
/*          If SIDE = 'R' or 'B', IFAILR(i) = j > 0 if the right */
/*          eigenvector in the i-th column of VR (corresponding to the */
/*          eigenvalue w(j)) failed to converge; IFAILR(i) = 0 if the */
/*          eigenvector converged satisfactorily. If the i-th and (i+1)th */
/*          columns of VR hold a complex eigenvector, then IFAILR(i) and */
/*          IFAILR(i+1) are set to the same value. */
/*          If SIDE = 'L', IFAILR is not referenced. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, i is the number of eigenvectors which */
/*                failed to converge; see IFAILL and IFAILR for further */
/*                details. */

/*  Further Details */
/*  =============== */

/*  Each eigenvector is normalized so that the element of largest */
/*  magnitude has magnitude 1; here the magnitude of a complex number */
/*  (x,y) is taken to be |x|+|y|. */

/*  ===================================================================== */

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
/*     .. Executable Statements .. */

/*     Decode and test the input parameters. */

    /* Parameter adjustments */
    --select;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;
    --ifaill;
    --ifailr;

    /* Function Body */
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;

    fromqr = lsame_(eigsrc, "Q");

    noinit = lsame_(initv, "N");

/*     Set M to the number of columns required to store the selected */
/*     eigenvectors, and standardize the array SELECT. */

    *m = 0;
    pair = FALSE_;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (pair) {
	    pair = FALSE_;
	    select[k] = FALSE_;
	} else {
	    if (wi[k] == 0.) {
		if (select[k]) {
		    ++(*m);
		}
	    } else {
		pair = TRUE_;
		if (select[k] || select[k + 1]) {
		    select[k] = TRUE_;
		    *m += 2;
		}
	    }
	}
/* L10: */
    }

    *info = 0;
    if (! rightv && ! leftv) {
	*info = -1;
    } else if (! fromqr && ! lsame_(eigsrc, "N")) {
	*info = -2;
    } else if (! noinit && ! lsame_(initv, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -5;
    } else if (*ldh < max(1,*n)) {
	*info = -7;
    } else if (*ldvl < 1 || leftv && *ldvl < *n) {
	*info = -11;
    } else if (*ldvr < 1 || rightv && *ldvr < *n) {
	*info = -13;
    } else if (*mm < *m) {
	*info = -14;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DHSEIN", &i__1);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

/*     Set machine-dependent constants. */

    unfl = dlamch_("Safe minimum");
    ulp = dlamch_("Precision");
    smlnum = unfl * (*n / ulp);
    bignum = (1. - ulp) / smlnum;

    ldwork = *n + 1;

    kl = 1;
    kln = 0;
    if (fromqr) {
	kr = 0;
    } else {
	kr = *n;
    }
    ksr = 1;

    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	if (select[k]) {

/*           Compute eigenvector(s) corresponding to W(K). */

	    if (fromqr) {

/*              If affiliation of eigenvalues is known, check whether */
/*              the matrix splits. */

/*              Determine KL and KR such that 1 <= KL <= K <= KR <= N */
/*              and H(KL,KL-1) and H(KR+1,KR) are zero (or KL = 1 or */
/*              KR = N). */

/*              Then inverse iteration can be performed with the */
/*              submatrix H(KL:N,KL:N) for a left eigenvector, and with */
/*              the submatrix H(1:KR,1:KR) for a right eigenvector. */

		i__2 = kl + 1;
		for (i__ = k; i__ >= i__2; --i__) {
		    if (h__[i__ + (i__ - 1) * h_dim1] == 0.) {
			goto L30;
		    }
/* L20: */
		}
L30:
		kl = i__;
		if (k > kr) {
		    i__2 = *n - 1;
		    for (i__ = k; i__ <= i__2; ++i__) {
			if (h__[i__ + 1 + i__ * h_dim1] == 0.) {
			    goto L50;
			}
/* L40: */
		    }
L50:
		    kr = i__;
		}
	    }

	    if (kl != kln) {
		kln = kl;

/*              Compute infinity-norm of submatrix H(KL:KR,KL:KR) if it */
/*              has not ben computed before. */

		i__2 = kr - kl + 1;
		hnorm = dlanhs_("I", &i__2, &h__[kl + kl * h_dim1], ldh, &
			work[1]);
		if (hnorm > 0.) {
		    eps3 = hnorm * ulp;
		} else {
		    eps3 = smlnum;
		}
	    }

/*           Perturb eigenvalue if it is close to any previous */
/*           selected eigenvalues affiliated to the submatrix */
/*           H(KL:KR,KL:KR). Close roots are modified by EPS3. */

	    wkr = wr[k];
	    wki = wi[k];
L60:
	    i__2 = kl;
	    for (i__ = k - 1; i__ >= i__2; --i__) {
		if (select[i__] && (d__1 = wr[i__] - wkr, abs(d__1)) + (d__2 =
			 wi[i__] - wki, abs(d__2)) < eps3) {
		    wkr += eps3;
		    goto L60;
		}
/* L70: */
	    }
	    wr[k] = wkr;

	    pair = wki != 0.;
	    if (pair) {
		ksi = ksr + 1;
	    } else {
		ksi = ksr;
	    }
	    if (leftv) {

/*              Compute left eigenvector. */

		i__2 = *n - kl + 1;
		dlaein_(&c_false, &noinit, &i__2, &h__[kl + kl * h_dim1], ldh, 
			 &wkr, &wki, &vl[kl + ksr * vl_dim1], &vl[kl + ksi * 
			vl_dim1], &work[1], &ldwork, &work[*n * *n + *n + 1], 
			&eps3, &smlnum, &bignum, &iinfo);
		if (iinfo > 0) {
		    if (pair) {
			*info += 2;
		    } else {
			++(*info);
		    }
		    ifaill[ksr] = k;
		    ifaill[ksi] = k;
		} else {
		    ifaill[ksr] = 0;
		    ifaill[ksi] = 0;
		}
		i__2 = kl - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    vl[i__ + ksr * vl_dim1] = 0.;
/* L80: */
		}
		if (pair) {
		    i__2 = kl - 1;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			vl[i__ + ksi * vl_dim1] = 0.;
/* L90: */
		    }
		}
	    }
	    if (rightv) {

/*              Compute right eigenvector. */

		dlaein_(&c_true, &noinit, &kr, &h__[h_offset], ldh, &wkr, &
			wki, &vr[ksr * vr_dim1 + 1], &vr[ksi * vr_dim1 + 1], &
			work[1], &ldwork, &work[*n * *n + *n + 1], &eps3, &
			smlnum, &bignum, &iinfo);
		if (iinfo > 0) {
		    if (pair) {
			*info += 2;
		    } else {
			++(*info);
		    }
		    ifailr[ksr] = k;
		    ifailr[ksi] = k;
		} else {
		    ifailr[ksr] = 0;
		    ifailr[ksi] = 0;
		}
		i__2 = *n;
		for (i__ = kr + 1; i__ <= i__2; ++i__) {
		    vr[i__ + ksr * vr_dim1] = 0.;
/* L100: */
		}
		if (pair) {
		    i__2 = *n;
		    for (i__ = kr + 1; i__ <= i__2; ++i__) {
			vr[i__ + ksi * vr_dim1] = 0.;
/* L110: */
		    }
		}
	    }

	    if (pair) {
		ksr += 2;
	    } else {
		++ksr;
	    }
	}
/* L120: */
    }

    return 0;

/*     End of DHSEIN */

} /* dhsein_ */
