/* dtrsna.f -- translated by f2c (version 20061008).
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
static logical c_true = TRUE_;
static logical c_false = FALSE_;

/* Subroutine */ int dtrsna_(char *job, char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, doublereal *s, doublereal *sep, 
	integer *mm, integer *m, doublereal *work, integer *ldwork, integer *
	iwork, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, 
	    work_dim1, work_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, k, n2;
    doublereal cs;
    integer nn, ks;
    doublereal sn, mu, eps, est;
    integer kase;
    doublereal cond;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    logical pair;
    integer ierr;
    doublereal dumm, prod;
    integer ifst;
    doublereal lnrm;
    integer ilst;
    doublereal rnrm;
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    doublereal prod1, prod2, scale, delta;
    extern logical lsame_(char *, char *);
    integer isave[3];
    logical wants;
    doublereal dummy[1];
    extern /* Subroutine */ int dlacn2_(integer *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, integer *);
    extern doublereal dlapy2_(doublereal *, doublereal *);
    extern /* Subroutine */ int dlabad_(doublereal *, doublereal *);
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *), 
	    xerbla_(char *, integer *);
    doublereal bignum;
    logical wantbh;
    extern /* Subroutine */ int dlaqtr_(logical *, logical *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, integer *), dtrexc_(char *, integer *
, doublereal *, integer *, doublereal *, integer *, integer *, 
	    integer *, doublereal *, integer *);
    logical somcon;
    doublereal smlnum;
    logical wantsp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     Modified to call DLACN2 in place of DLACON, 5 Feb 03, SJH. */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTRSNA estimates reciprocal condition numbers for specified */
/*  eigenvalues and/or right eigenvectors of a real upper */
/*  quasi-triangular matrix T (or of any matrix Q*T*Q**T with Q */
/*  orthogonal). */

/*  T must be in Schur canonical form (as returned by DHSEQR), that is, */
/*  block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each */
/*  2-by-2 diagonal block has its diagonal elements equal and its */
/*  off-diagonal elements of opposite sign. */

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
/*          for the eigenpair corresponding to a real eigenvalue w(j), */
/*          SELECT(j) must be set to .TRUE.. To select condition numbers */
/*          corresponding to a complex conjugate pair of eigenvalues w(j) */
/*          and w(j+1), either SELECT(j) or SELECT(j+1) or both, must be */
/*          set to .TRUE.. */
/*          If HOWMNY = 'A', SELECT is not referenced. */

/*  N       (input) INTEGER */
/*          The order of the matrix T. N >= 0. */

/*  T       (input) DOUBLE PRECISION array, dimension (LDT,N) */
/*          The upper quasi-triangular matrix T, in Schur canonical form. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= max(1,N). */

/*  VL      (input) DOUBLE PRECISION array, dimension (LDVL,M) */
/*          If JOB = 'E' or 'B', VL must contain left eigenvectors of T */
/*          (or of any Q*T*Q**T with Q orthogonal), corresponding to the */
/*          eigenpairs specified by HOWMNY and SELECT. The eigenvectors */
/*          must be stored in consecutive columns of VL, as returned by */
/*          DHSEIN or DTREVC. */
/*          If JOB = 'V', VL is not referenced. */

/*  LDVL    (input) INTEGER */
/*          The leading dimension of the array VL. */
/*          LDVL >= 1; and if JOB = 'E' or 'B', LDVL >= N. */

/*  VR      (input) DOUBLE PRECISION array, dimension (LDVR,M) */
/*          If JOB = 'E' or 'B', VR must contain right eigenvectors of T */
/*          (or of any Q*T*Q**T with Q orthogonal), corresponding to the */
/*          eigenpairs specified by HOWMNY and SELECT. The eigenvectors */
/*          must be stored in consecutive columns of VR, as returned by */
/*          DHSEIN or DTREVC. */
/*          If JOB = 'V', VR is not referenced. */

/*  LDVR    (input) INTEGER */
/*          The leading dimension of the array VR. */
/*          LDVR >= 1; and if JOB = 'E' or 'B', LDVR >= N. */

/*  S       (output) DOUBLE PRECISION array, dimension (MM) */
/*          If JOB = 'E' or 'B', the reciprocal condition numbers of the */
/*          selected eigenvalues, stored in consecutive elements of the */
/*          array. For a complex conjugate pair of eigenvalues two */
/*          consecutive elements of S are set to the same value. Thus */
/*          S(j), SEP(j), and the j-th columns of VL and VR all */
/*          correspond to the same eigenpair (but not in general the */
/*          j-th eigenpair, unless all eigenpairs are selected). */
/*          If JOB = 'V', S is not referenced. */

/*  SEP     (output) DOUBLE PRECISION array, dimension (MM) */
/*          If JOB = 'V' or 'B', the estimated reciprocal condition */
/*          numbers of the selected eigenvectors, stored in consecutive */
/*          elements of the array. For a complex eigenvector two */
/*          consecutive elements of SEP are set to the same value. If */
/*          the eigenvalues cannot be reordered to compute SEP(j), SEP(j) */
/*          is set to 0; this can only occur when the true value would be */
/*          very small anyway. */
/*          If JOB = 'E', SEP is not referenced. */

/*  MM      (input) INTEGER */
/*          The number of elements in the arrays S (if JOB = 'E' or 'B') */
/*           and/or SEP (if JOB = 'V' or 'B'). MM >= M. */

/*  M       (output) INTEGER */
/*          The number of elements of the arrays S and/or SEP actually */
/*          used to store the estimated condition numbers. */
/*          If HOWMNY = 'A', M is set to N. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDWORK,N+6) */
/*          If JOB = 'E', WORK is not referenced. */

/*  LDWORK  (input) INTEGER */
/*          The leading dimension of the array WORK. */
/*          LDWORK >= 1; and if JOB = 'V' or 'B', LDWORK >= N. */

/*  IWORK   (workspace) INTEGER array, dimension (2*(N-1)) */
/*          If JOB = 'E', IWORK is not referenced. */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The reciprocal of the condition number of an eigenvalue lambda is */
/*  defined as */

/*          S(lambda) = |v'*u| / (norm(u)*norm(v)) */

/*  where u and v are the right and left eigenvectors of T corresponding */
/*  to lambda; v' denotes the conjugate-transpose of v, and norm(u) */
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
    --iwork;

    /* Function Body */
    wantbh = lsame_(job, "B");
    wants = lsame_(job, "E") || wantbh;
    wantsp = lsame_(job, "V") || wantbh;

    somcon = lsame_(howmny, "S");

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
    } else {

/*        Set M to the number of eigenpairs for which condition numbers */
/*        are required, and test MM. */

	if (somcon) {
	    *m = 0;
	    pair = FALSE_;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (pair) {
		    pair = FALSE_;
		} else {
		    if (k < *n) {
			if (t[k + 1 + k * t_dim1] == 0.) {
			    if (select[k]) {
				++(*m);
			    }
			} else {
			    pair = TRUE_;
			    if (select[k] || select[k + 1]) {
				*m += 2;
			    }
			}
		    } else {
			if (select[*n]) {
			    ++(*m);
			}
		    }
		}
/* L10: */
	    }
	} else {
	    *m = *n;
	}

	if (*mm < *m) {
	    *info = -13;
	} else if (*ldwork < 1 || wantsp && *ldwork < *n) {
	    *info = -16;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTRSNA", &i__1);
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
	    sep[1] = (d__1 = t[t_dim1 + 1], abs(d__1));
	}
	return 0;
    }

/*     Get machine constants */

    eps = dlamch_("P");
    smlnum = dlamch_("S") / eps;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);

    ks = 0;
    pair = FALSE_;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {

/*        Determine whether T(k,k) begins a 1-by-1 or 2-by-2 block. */

	if (pair) {
	    pair = FALSE_;
	    goto L60;
	} else {
	    if (k < *n) {
		pair = t[k + 1 + k * t_dim1] != 0.;
	    }
	}

/*        Determine whether condition numbers are required for the k-th */
/*        eigenpair. */

	if (somcon) {
	    if (pair) {
		if (! select[k] && ! select[k + 1]) {
		    goto L60;
		}
	    } else {
		if (! select[k]) {
		    goto L60;
		}
	    }
	}

	++ks;

	if (wants) {

/*           Compute the reciprocal condition number of the k-th */
/*           eigenvalue. */

	    if (! pair) {

/*              Real eigenvalue. */

		prod = ddot_(n, &vr[ks * vr_dim1 + 1], &c__1, &vl[ks * 
			vl_dim1 + 1], &c__1);
		rnrm = dnrm2_(n, &vr[ks * vr_dim1 + 1], &c__1);
		lnrm = dnrm2_(n, &vl[ks * vl_dim1 + 1], &c__1);
		s[ks] = abs(prod) / (rnrm * lnrm);
	    } else {

/*              Complex eigenvalue. */

		prod1 = ddot_(n, &vr[ks * vr_dim1 + 1], &c__1, &vl[ks * 
			vl_dim1 + 1], &c__1);
		prod1 += ddot_(n, &vr[(ks + 1) * vr_dim1 + 1], &c__1, &vl[(ks 
			+ 1) * vl_dim1 + 1], &c__1);
		prod2 = ddot_(n, &vl[ks * vl_dim1 + 1], &c__1, &vr[(ks + 1) * 
			vr_dim1 + 1], &c__1);
		prod2 -= ddot_(n, &vl[(ks + 1) * vl_dim1 + 1], &c__1, &vr[ks *
			 vr_dim1 + 1], &c__1);
		d__1 = dnrm2_(n, &vr[ks * vr_dim1 + 1], &c__1);
		d__2 = dnrm2_(n, &vr[(ks + 1) * vr_dim1 + 1], &c__1);
		rnrm = dlapy2_(&d__1, &d__2);
		d__1 = dnrm2_(n, &vl[ks * vl_dim1 + 1], &c__1);
		d__2 = dnrm2_(n, &vl[(ks + 1) * vl_dim1 + 1], &c__1);
		lnrm = dlapy2_(&d__1, &d__2);
		cond = dlapy2_(&prod1, &prod2) / (rnrm * lnrm);
		s[ks] = cond;
		s[ks + 1] = cond;
	    }
	}

	if (wantsp) {

/*           Estimate the reciprocal condition number of the k-th */
/*           eigenvector. */

/*           Copy the matrix T to the array WORK and swap the diagonal */
/*           block beginning at T(k,k) to the (1,1) position. */

	    dlacpy_("Full", n, n, &t[t_offset], ldt, &work[work_offset], 
		    ldwork);
	    ifst = k;
	    ilst = 1;
	    dtrexc_("No Q", n, &work[work_offset], ldwork, dummy, &c__1, &
		    ifst, &ilst, &work[(*n + 1) * work_dim1 + 1], &ierr);

	    if (ierr == 1 || ierr == 2) {

/*              Could not swap because blocks not well separated */

		scale = 1.;
		est = bignum;
	    } else {

/*              Reordering successful */

		if (work[work_dim1 + 2] == 0.) {

/*                 Form C = T22 - lambda*I in WORK(2:N,2:N). */

		    i__2 = *n;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			work[i__ + i__ * work_dim1] -= work[work_dim1 + 1];
/* L20: */
		    }
		    n2 = 1;
		    nn = *n - 1;
		} else {

/*                 Triangularize the 2 by 2 block by unitary */
/*                 transformation U = [  cs   i*ss ] */
/*                                    [ i*ss   cs  ]. */
/*                 such that the (1,1) position of WORK is complex */
/*                 eigenvalue lambda with positive imaginary part. (2,2) */
/*                 position of WORK is the complex eigenvalue lambda */
/*                 with negative imaginary  part. */

		    mu = sqrt((d__1 = work[(work_dim1 << 1) + 1], abs(d__1))) 
			    * sqrt((d__2 = work[work_dim1 + 2], abs(d__2)));
		    delta = dlapy2_(&mu, &work[work_dim1 + 2]);
		    cs = mu / delta;
		    sn = -work[work_dim1 + 2] / delta;

/*                 Form */

/*                 C' = WORK(2:N,2:N) + i*[rwork(1) ..... rwork(n-1) ] */
/*                                        [   mu                     ] */
/*                                        [         ..               ] */
/*                                        [             ..           ] */
/*                                        [                  mu      ] */
/*                 where C' is conjugate transpose of complex matrix C, */
/*                 and RWORK is stored starting in the N+1-st column of */
/*                 WORK. */

		    i__2 = *n;
		    for (j = 3; j <= i__2; ++j) {
			work[j * work_dim1 + 2] = cs * work[j * work_dim1 + 2]
				;
			work[j + j * work_dim1] -= work[work_dim1 + 1];
/* L30: */
		    }
		    work[(work_dim1 << 1) + 2] = 0.;

		    work[(*n + 1) * work_dim1 + 1] = mu * 2.;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			work[i__ + (*n + 1) * work_dim1] = sn * work[(i__ + 1)
				 * work_dim1 + 1];
/* L40: */
		    }
		    n2 = 2;
		    nn = *n - 1 << 1;
		}

/*              Estimate norm(inv(C')) */

		est = 0.;
		kase = 0;
L50:
		dlacn2_(&nn, &work[(*n + 2) * work_dim1 + 1], &work[(*n + 4) *
			 work_dim1 + 1], &iwork[1], &est, &kase, isave);
		if (kase != 0) {
		    if (kase == 1) {
			if (n2 == 1) {

/*                       Real eigenvalue: solve C'*x = scale*c. */

			    i__2 = *n - 1;
			    dlaqtr_(&c_true, &c_true, &i__2, &work[(work_dim1 
				    << 1) + 2], ldwork, dummy, &dumm, &scale, 
				    &work[(*n + 4) * work_dim1 + 1], &work[(*
				    n + 6) * work_dim1 + 1], &ierr);
			} else {

/*                       Complex eigenvalue: solve */
/*                       C'*(p+iq) = scale*(c+id) in real arithmetic. */

			    i__2 = *n - 1;
			    dlaqtr_(&c_true, &c_false, &i__2, &work[(
				    work_dim1 << 1) + 2], ldwork, &work[(*n + 
				    1) * work_dim1 + 1], &mu, &scale, &work[(*
				    n + 4) * work_dim1 + 1], &work[(*n + 6) * 
				    work_dim1 + 1], &ierr);
			}
		    } else {
			if (n2 == 1) {

/*                       Real eigenvalue: solve C*x = scale*c. */

			    i__2 = *n - 1;
			    dlaqtr_(&c_false, &c_true, &i__2, &work[(
				    work_dim1 << 1) + 2], ldwork, dummy, &
				    dumm, &scale, &work[(*n + 4) * work_dim1 
				    + 1], &work[(*n + 6) * work_dim1 + 1], &
				    ierr);
			} else {

/*                       Complex eigenvalue: solve */
/*                       C*(p+iq) = scale*(c+id) in real arithmetic. */

			    i__2 = *n - 1;
			    dlaqtr_(&c_false, &c_false, &i__2, &work[(
				    work_dim1 << 1) + 2], ldwork, &work[(*n + 
				    1) * work_dim1 + 1], &mu, &scale, &work[(*
				    n + 4) * work_dim1 + 1], &work[(*n + 6) * 
				    work_dim1 + 1], &ierr);

			}
		    }

		    goto L50;
		}
	    }

	    sep[ks] = scale / max(est,smlnum);
	    if (pair) {
		sep[ks + 1] = sep[ks];
	    }
	}

	if (pair) {
	    ++ks;
	}

L60:
	;
    }
    return 0;

/*     End of DTRSNA */

} /* dtrsna_ */
