/* ztrsna.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ztrsna_(char *job, char *howmny, logical *select, 
	integer *n, doublecomplex *t, integer *ldt, doublecomplex *vl, 
	integer *ldvl, doublecomplex *vr, integer *ldvr, doublereal *s, 
	doublereal *sep, integer *mm, integer *m, doublecomplex *work, 
	integer *ldwork, doublereal *rwork, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, 
	    work_dim1, work_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Builtin functions */
    double z_abs(doublecomplex *), d_imag(doublecomplex *);

    /* Local variables */
    integer i__, j, k, ks, ix;
    doublereal eps, est;
    integer kase, ierr;
    doublecomplex prod;
    doublereal lnrm, rnrm, scale;
    extern logical lsame_(char *, char *);
    integer isave[3];
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    doublecomplex dummy[1];
    logical wants;
    doublereal xnorm;
    extern /* Subroutine */ int zlacn2_(integer *, doublecomplex *, 
	    doublecomplex *, doublereal *, integer *, integer *), dlabad_(
	    doublereal *, doublereal *);
    extern doublereal dznrm2_(integer *, doublecomplex *, integer *), dlamch_(
	    char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;
    logical wantbh;
    extern integer izamax_(integer *, doublecomplex *, integer *);
    logical somcon;
    extern /* Subroutine */ int zdrscl_(integer *, doublereal *, 
	    doublecomplex *, integer *);
    char normin[1];
    extern /* Subroutine */ int zlacpy_(char *, integer *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    doublereal smlnum;
    logical wantsp;
    extern /* Subroutine */ int zlatrs_(char *, char *, char *, char *, 
	    integer *, doublecomplex *, integer *, doublecomplex *, 
	    doublereal *, doublereal *, integer *), ztrexc_(char *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, integer *, integer *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     Modified to call ZLACN2 in place of ZLACON, 10 Feb 03, SJH. */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTRSNA estimates reciprocal condition numbers for specified */
/*  eigenvalues and/or right eigenvectors of a complex upper triangular */
/*  matrix T (or of any matrix Q*T*Q**H with Q unitary). */

/*  Arguments */
/*  ========= */

/*  JOB     (input) CHARACTER*1 */
/*          Specifies whether condition numbers are required for */
/*          eigenvalues (S) or eigenvectors (SEP): */
/*          = 'E': for eigenvalues only (S); */
/*          = 'V': for eigenvectors only (SEP); */
/*          = 'B': for both eigenvalues and eigenvectors (S and SEP). */

/*  HOWMNY  (input) CHARACTER*1 */
/*          = 'A': compute condition numbers for all eigenpairs; */
/*          = 'S': compute condition numbers for selected eigenpairs */
/*                 specified by the array SELECT. */

/*  SELECT  (input) LOGICAL array, dimension (N) */
/*          If HOWMNY = 'S', SELECT specifies the eigenpairs for which */
/*          condition numbers are required. To select condition numbers */
/*          for the j-th eigenpair, SELECT(j) must be set to .TRUE.. */
/*          If HOWMNY = 'A', SELECT is not referenced. */

/*  N       (input) INTEGER */
/*          The order of the matrix T. N >= 0. */

/*  T       (input) COMPLEX*16 array, dimension (LDT,N) */
/*          The upper triangular matrix T. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= max(1,N). */

/*  VL      (input) COMPLEX*16 array, dimension (LDVL,M) */
/*          If JOB = 'E' or 'B', VL must contain left eigenvectors of T */
/*          (or of any Q*T*Q**H with Q unitary), corresponding to the */
/*          eigenpairs specified by HOWMNY and SELECT. The eigenvectors */
/*          must be stored in consecutive columns of VL, as returned by */
/*          ZHSEIN or ZTREVC. */
/*          If JOB = 'V', VL is not referenced. */

/*  LDVL    (input) INTEGER */
/*          The leading dimension of the array VL. */
/*          LDVL >= 1; and if JOB = 'E' or 'B', LDVL >= N. */

/*  VR      (input) COMPLEX*16 array, dimension (LDVR,M) */
/*          If JOB = 'E' or 'B', VR must contain right eigenvectors of T */
/*          (or of any Q*T*Q**H with Q unitary), corresponding to the */
/*          eigenpairs specified by HOWMNY and SELECT. The eigenvectors */
/*          must be stored in consecutive columns of VR, as returned by */
/*          ZHSEIN or ZTREVC. */
/*          If JOB = 'V', VR is not referenced. */

/*  LDVR    (input) INTEGER */
/*          The leading dimension of the array VR. */
/*          LDVR >= 1; and if JOB = 'E' or 'B', LDVR >= N. */

/*  S       (output) DOUBLE PRECISION array, dimension (MM) */
/*          If JOB = 'E' or 'B', the reciprocal condition numbers of the */
/*          selected eigenvalues, stored in consecutive elements of the */
/*          array. Thus S(j), SEP(j), and the j-th columns of VL and VR */
/*          all correspond to the same eigenpair (but not in general the */
/*          j-th eigenpair, unless all eigenpairs are selected). */
/*          If JOB = 'V', S is not referenced. */

/*  SEP     (output) DOUBLE PRECISION array, dimension (MM) */
/*          If JOB = 'V' or 'B', the estimated reciprocal condition */
/*          numbers of the selected eigenvectors, stored in consecutive */
/*          elements of the array. */
/*          If JOB = 'E', SEP is not referenced. */

/*  MM      (input) INTEGER */
/*          The number of elements in the arrays S (if JOB = 'E' or 'B') */
/*           and/or SEP (if JOB = 'V' or 'B'). MM >= M. */

/*  M       (output) INTEGER */
/*          The number of elements of the arrays S and/or SEP actually */
/*          used to store the estimated condition numbers. */
/*          If HOWMNY = 'A', M is set to N. */

/*  WORK    (workspace) COMPLEX*16 array, dimension (LDWORK,N+6) */
/*          If JOB = 'E', WORK is not referenced. */

/*  LDWORK  (input) INTEGER */
/*          The leading dimension of the array WORK. */
/*          LDWORK >= 1; and if JOB = 'V' or 'B', LDWORK >= N. */

/*  RWORK   (workspace) DOUBLE PRECISION array, dimension (N) */
/*          If JOB = 'E', RWORK is not referenced. */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The reciprocal of the condition number of an eigenvalue lambda is */
/*  defined as */

/*          S(lambda) = |v'*u| / (norm(u)*norm(v)) */

/*  where u and v are the right and left eigenvectors of T corresponding */
/*  to lambda; v' denotes the conjugate transpose of v, and norm(u) */
/*  denotes the Euclidean norm. These reciprocal condition numbers always */
/*  lie between zero (very badly conditioned) and one (very well */
/*  conditioned). If n = 1, S(lambda) is defined to be 1. */

/*  An approximate error bound for a computed eigenvalue W(i) is given by */

/*                      EPS * norm(T) / S(i) */

/*  where EPS is the machine precision. */

/*  The reciprocal of the condition number of the right eigenvector u */
/*  corresponding to lambda is defined as follows. Suppose */

/*              T = ( lambda  c  ) */
/*                  (   0    T22 ) */

/*  Then the reciprocal condition number is */

/*          SEP( lambda, T22 ) = sigma-min( T22 - lambda*I ) */

/*  where sigma-min denotes the smallest singular value. We approximate */
/*  the smallest singular value by the reciprocal of an estimate of the */
/*  one-norm of the inverse of T22 - lambda*I. If n = 1, SEP(1) is */
/*  defined to be abs(T(1,1)). */

/*  An approximate error bound for a computed right eigenvector VR(i) */
/*  is given by */

/*                      EPS * norm(T) / SEP(i) */

/*  ===================================================================== */

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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Decode and test the input parameters */

    /* Parameter adjustments */
    --select;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --s;
    --sep;
    work_dim1 = *ldwork;
    work_offset = 1 + work_dim1;
    work -= work_offset;
    --rwork;

    /* Function Body */
    wantbh = lsame_(job, "B");
    wants = lsame_(job, "E") || wantbh;
    wantsp = lsame_(job, "V") || wantbh;

    somcon = lsame_(howmny, "S");

/*     Set M to the number of eigenpairs for which condition numbers are */
/*     to be computed. */

    if (somcon) {
	*m = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (select[j]) {
		++(*m);
	    }
/* L10: */
	}
    } else {
	*m = *n;
    }

    *info = 0;
    if (! wants && ! wantsp) {
	*info = -1;
    } else if (! lsame_(howmny, "A") && ! somcon) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ldt < max(1,*n)) {
	*info = -6;
    } else if (*ldvl < 1 || wants && *ldvl < *n) {
	*info = -8;
    } else if (*ldvr < 1 || wants && *ldvr < *n) {
	*info = -10;
    } else if (*mm < *m) {
	*info = -13;
    } else if (*ldwork < 1 || wantsp && *ldwork < *n) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTRSNA", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (somcon) {
	    if (! select[1]) {
		return 0;
	    }
	}
	if (wants) {
	    s[1] = 1.;
	}
	if (wantsp) {
	    sep[1] = z_abs(&t[t_dim1 + 1]);
	}
	return 0;
    }

/*     Get machine constants */

    eps = dlamch_("P");
    smlnum = dlamch_("S") / eps;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);

    ks = 1;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {

	if (somcon) {
	    if (! select[k]) {
		goto L50;
	    }
	}

	if (wants) {

/*           Compute the reciprocal condition number of the k-th */
/*           eigenvalue. */

	    zdotc_(&z__1, n, &vr[ks * vr_dim1 + 1], &c__1, &vl[ks * vl_dim1 + 
		    1], &c__1);
	    prod.r = z__1.r, prod.i = z__1.i;
	    rnrm = dznrm2_(n, &vr[ks * vr_dim1 + 1], &c__1);
	    lnrm = dznrm2_(n, &vl[ks * vl_dim1 + 1], &c__1);
	    s[ks] = z_abs(&prod) / (rnrm * lnrm);

	}

	if (wantsp) {

/*           Estimate the reciprocal condition number of the k-th */
/*           eigenvector. */

/*           Copy the matrix T to the array WORK and swap the k-th */
/*           diagonal element to the (1,1) position. */

	    zlacpy_("Full", n, n, &t[t_offset], ldt, &work[work_offset], 
		    ldwork);
	    ztrexc_("No Q", n, &work[work_offset], ldwork, dummy, &c__1, &k, &
		    c__1, &ierr);

/*           Form  C = T22 - lambda*I in WORK(2:N,2:N). */

	    i__2 = *n;
	    for (i__ = 2; i__ <= i__2; ++i__) {
		i__3 = i__ + i__ * work_dim1;
		i__4 = i__ + i__ * work_dim1;
		i__5 = work_dim1 + 1;
		z__1.r = work[i__4].r - work[i__5].r, z__1.i = work[i__4].i - 
			work[i__5].i;
		work[i__3].r = z__1.r, work[i__3].i = z__1.i;
/* L20: */
	    }

/*           Estimate a lower bound for the 1-norm of inv(C'). The 1st */
/*           and (N+1)th columns of WORK are used to store work vectors. */

	    sep[ks] = 0.;
	    est = 0.;
	    kase = 0;
	    *(unsigned char *)normin = 'N';
L30:
	    i__2 = *n - 1;
	    zlacn2_(&i__2, &work[(*n + 1) * work_dim1 + 1], &work[work_offset]
, &est, &kase, isave);

	    if (kase != 0) {
		if (kase == 1) {

/*                 Solve C'*x = scale*b */

		    i__2 = *n - 1;
		    zlatrs_("Upper", "Conjugate transpose", "Nonunit", normin, 
			     &i__2, &work[(work_dim1 << 1) + 2], ldwork, &
			    work[work_offset], &scale, &rwork[1], &ierr);
		} else {

/*                 Solve C*x = scale*b */

		    i__2 = *n - 1;
		    zlatrs_("Upper", "No transpose", "Nonunit", normin, &i__2, 
			     &work[(work_dim1 << 1) + 2], ldwork, &work[
			    work_offset], &scale, &rwork[1], &ierr);
		}
		*(unsigned char *)normin = 'Y';
		if (scale != 1.) {

/*                 Multiply by 1/SCALE if doing so will not cause */
/*                 overflow. */

		    i__2 = *n - 1;
		    ix = izamax_(&i__2, &work[work_offset], &c__1);
		    i__2 = ix + work_dim1;
		    xnorm = (d__1 = work[i__2].r, abs(d__1)) + (d__2 = d_imag(
			    &work[ix + work_dim1]), abs(d__2));
		    if (scale < xnorm * smlnum || scale == 0.) {
			goto L40;
		    }
		    zdrscl_(n, &scale, &work[work_offset], &c__1);
		}
		goto L30;
	    }

	    sep[ks] = 1. / max(est,smlnum);
	}

L40:
	++ks;
L50:
	;
    }
    return 0;

/*     End of ZTRSNA */

} /* ztrsna_ */
