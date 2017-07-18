/* chgeqz.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {0.f,0.f};
static complex c_b2 = {1.f,0.f};
static integer c__1 = 1;
static integer c__2 = 2;

/* Subroutine */ int chgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, complex *h__, integer *ldh, complex *t, 
	integer *ldt, complex *alpha, complex *beta, complex *q, integer *ldq, 
	 complex *z__, integer *ldz, complex *work, integer *lwork, real *
	rwork, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, q_dim1, q_offset, t_dim1, t_offset, z_dim1, 
	    z_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    complex q__1, q__2, q__3, q__4, q__5, q__6;

    /* Builtin functions */
    double c_abs(complex *);
    void r_cnjg(complex *, complex *);
    double r_imag(complex *);
    void c_div(complex *, complex *, complex *), pow_ci(complex *, complex *, 
	    integer *), c_sqrt(complex *, complex *);

    /* Local variables */
    real c__;
    integer j;
    complex s, t1;
    integer jc, in;
    complex u12;
    integer jr;
    complex ad11, ad12, ad21, ad22;
    integer jch;
    logical ilq, ilz;
    real ulp;
    complex abi22;
    real absb, atol, btol, temp;
    extern /* Subroutine */ int crot_(integer *, complex *, integer *, 
	    complex *, integer *, real *, complex *);
    real temp2;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *, 
	    integer *);
    extern logical lsame_(char *, char *);
    complex ctemp;
    integer iiter, ilast, jiter;
    real anorm, bnorm;
    integer maxit;
    complex shift;
    real tempr;
    complex ctemp2, ctemp3;
    logical ilazr2;
    real ascale, bscale;
    complex signbc;
    extern doublereal slamch_(char *), clanhs_(char *, integer *, 
	    complex *, integer *, real *);
    extern /* Subroutine */ int claset_(char *, integer *, integer *, complex 
	    *, complex *, complex *, integer *), clartg_(complex *, 
	    complex *, real *, complex *, complex *);
    real safmin;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    complex eshift;
    logical ilschr;
    integer icompq, ilastm;
    complex rtdisc;
    integer ischur;
    logical ilazro;
    integer icompz, ifirst, ifrstm, istart;
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHGEQZ computes the eigenvalues of a complex matrix pair (H,T), */
/*  where H is an upper Hessenberg matrix and T is upper triangular, */
/*  using the single-shift QZ method. */
/*  Matrix pairs of this type are produced by the reduction to */
/*  generalized upper Hessenberg form of a complex matrix pair (A,B): */

/*     A = Q1*H*Z1**H,  B = Q1*T*Z1**H, */

/*  as computed by CGGHRD. */

/*  If JOB='S', then the Hessenberg-triangular pair (H,T) is */
/*  also reduced to generalized Schur form, */

/*     H = Q*S*Z**H,  T = Q*P*Z**H, */

/*  where Q and Z are unitary matrices and S and P are upper triangular. */

/*  Optionally, the unitary matrix Q from the generalized Schur */
/*  factorization may be postmultiplied into an input matrix Q1, and the */
/*  unitary matrix Z may be postmultiplied into an input matrix Z1. */
/*  If Q1 and Z1 are the unitary matrices from CGGHRD that reduced */
/*  the matrix pair (A,B) to generalized Hessenberg form, then the output */
/*  matrices Q1*Q and Z1*Z are the unitary factors from the generalized */
/*  Schur factorization of (A,B): */

/*     A = (Q1*Q)*S*(Z1*Z)**H,  B = (Q1*Q)*P*(Z1*Z)**H. */

/*  To avoid overflow, eigenvalues of the matrix pair (H,T) */
/*  (equivalently, of (A,B)) are computed as a pair of complex values */
/*  (alpha,beta).  If beta is nonzero, lambda = alpha / beta is an */
/*  eigenvalue of the generalized nonsymmetric eigenvalue problem (GNEP) */
/*     A*x = lambda*B*x */
/*  and if alpha is nonzero, mu = beta / alpha is an eigenvalue of the */
/*  alternate form of the GNEP */
/*     mu*A*y = B*y. */
/*  The values of alpha and beta for the i-th eigenvalue can be read */
/*  directly from the generalized Schur form:  alpha = S(i,i), */
/*  beta = P(i,i). */

/*  Ref: C.B. Moler & G.W. Stewart, "An Algorithm for Generalized Matrix */
/*       Eigenvalue Problems", SIAM J. Numer. Anal., 10(1973), */
/*       pp. 241--256. */

/*  Arguments */
/*  ========= */

/*  JOB     (input) CHARACTER*1 */
/*          = 'E': Compute eigenvalues only; */
/*          = 'S': Computer eigenvalues and the Schur form. */

/*  COMPQ   (input) CHARACTER*1 */
/*          = 'N': Left Schur vectors (Q) are not computed; */
/*          = 'I': Q is initialized to the unit matrix and the matrix Q */
/*                 of left Schur vectors of (H,T) is returned; */
/*          = 'V': Q must contain a unitary matrix Q1 on entry and */
/*                 the product Q1*Q is returned. */

/*  COMPZ   (input) CHARACTER*1 */
/*          = 'N': Right Schur vectors (Z) are not computed; */
/*          = 'I': Q is initialized to the unit matrix and the matrix Z */
/*                 of right Schur vectors of (H,T) is returned; */
/*          = 'V': Z must contain a unitary matrix Z1 on entry and */
/*                 the product Z1*Z is returned. */

/*  N       (input) INTEGER */
/*          The order of the matrices H, T, Q, and Z.  N >= 0. */

/*  ILO     (input) INTEGER */
/*  IHI     (input) INTEGER */
/*          ILO and IHI mark the rows and columns of H which are in */
/*          Hessenberg form.  It is assumed that A is already upper */
/*          triangular in rows and columns 1:ILO-1 and IHI+1:N. */
/*          If N > 0, 1 <= ILO <= IHI <= N; if N = 0, ILO=1 and IHI=0. */

/*  H       (input/output) COMPLEX array, dimension (LDH, N) */
/*          On entry, the N-by-N upper Hessenberg matrix H. */
/*          On exit, if JOB = 'S', H contains the upper triangular */
/*          matrix S from the generalized Schur factorization. */
/*          If JOB = 'E', the diagonal of H matches that of S, but */
/*          the rest of H is unspecified. */

/*  LDH     (input) INTEGER */
/*          The leading dimension of the array H.  LDH >= max( 1, N ). */

/*  T       (input/output) COMPLEX array, dimension (LDT, N) */
/*          On entry, the N-by-N upper triangular matrix T. */
/*          On exit, if JOB = 'S', T contains the upper triangular */
/*          matrix P from the generalized Schur factorization. */
/*          If JOB = 'E', the diagonal of T matches that of P, but */
/*          the rest of T is unspecified. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T.  LDT >= max( 1, N ). */

/*  ALPHA   (output) COMPLEX array, dimension (N) */
/*          The complex scalars alpha that define the eigenvalues of */
/*          GNEP.  ALPHA(i) = S(i,i) in the generalized Schur */
/*          factorization. */

/*  BETA    (output) COMPLEX array, dimension (N) */
/*          The real non-negative scalars beta that define the */
/*          eigenvalues of GNEP.  BETA(i) = P(i,i) in the generalized */
/*          Schur factorization. */

/*          Together, the quantities alpha = ALPHA(j) and beta = BETA(j) */
/*          represent the j-th eigenvalue of the matrix pair (A,B), in */
/*          one of the forms lambda = alpha/beta or mu = beta/alpha. */
/*          Since either lambda or mu may overflow, they should not, */
/*          in general, be computed. */

/*  Q       (input/output) COMPLEX array, dimension (LDQ, N) */
/*          On entry, if COMPZ = 'V', the unitary matrix Q1 used in the */
/*          reduction of (A,B) to generalized Hessenberg form. */
/*          On exit, if COMPZ = 'I', the unitary matrix of left Schur */
/*          vectors of (H,T), and if COMPZ = 'V', the unitary matrix of */
/*          left Schur vectors of (A,B). */
/*          Not referenced if COMPZ = 'N'. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q.  LDQ >= 1. */
/*          If COMPQ='V' or 'I', then LDQ >= N. */

/*  Z       (input/output) COMPLEX array, dimension (LDZ, N) */
/*          On entry, if COMPZ = 'V', the unitary matrix Z1 used in the */
/*          reduction of (A,B) to generalized Hessenberg form. */
/*          On exit, if COMPZ = 'I', the unitary matrix of right Schur */
/*          vectors of (H,T), and if COMPZ = 'V', the unitary matrix of */
/*          right Schur vectors of (A,B). */
/*          Not referenced if COMPZ = 'N'. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1. */
/*          If COMPZ='V' or 'I', then LDZ >= N. */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO >= 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK.  LWORK >= max(1,N). */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  RWORK   (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */
/*          = 1,...,N: the QZ iteration did not converge.  (H,T) is not */
/*                     in Schur form, but ALPHA(i) and BETA(i), */
/*                     i=INFO+1,...,N should be correct. */
/*          = N+1,...,2*N: the shift calculation failed.  (H,T) is not */
/*                     in Schur form, but ALPHA(i) and BETA(i), */
/*                     i=INFO-N+1,...,N should be correct. */

/*  Further Details */
/*  =============== */

/*  We assume that complex ABS works as long as its value is less than */
/*  overflow. */

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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Decode JOB, COMPQ, COMPZ */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    --alpha;
    --beta;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --rwork;

    /* Function Body */
    if (lsame_(job, "E")) {
	ilschr = FALSE_;
	ischur = 1;
    } else if (lsame_(job, "S")) {
	ilschr = TRUE_;
	ischur = 2;
    } else {
	ischur = 0;
    }

    if (lsame_(compq, "N")) {
	ilq = FALSE_;
	icompq = 1;
    } else if (lsame_(compq, "V")) {
	ilq = TRUE_;
	icompq = 2;
    } else if (lsame_(compq, "I")) {
	ilq = TRUE_;
	icompq = 3;
    } else {
	icompq = 0;
    }

    if (lsame_(compz, "N")) {
	ilz = FALSE_;
	icompz = 1;
    } else if (lsame_(compz, "V")) {
	ilz = TRUE_;
	icompz = 2;
    } else if (lsame_(compz, "I")) {
	ilz = TRUE_;
	icompz = 3;
    } else {
	icompz = 0;
    }

/*     Check Argument Values */

    *info = 0;
    i__1 = max(1,*n);
    work[1].r = (real) i__1, work[1].i = 0.f;
    lquery = *lwork == -1;
    if (ischur == 0) {
	*info = -1;
    } else if (icompq == 0) {
	*info = -2;
    } else if (icompz == 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ilo < 1) {
	*info = -5;
    } else if (*ihi > *n || *ihi < *ilo - 1) {
	*info = -6;
    } else if (*ldh < *n) {
	*info = -8;
    } else if (*ldt < *n) {
	*info = -10;
    } else if (*ldq < 1 || ilq && *ldq < *n) {
	*info = -14;
    } else if (*ldz < 1 || ilz && *ldz < *n) {
	*info = -16;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -18;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHGEQZ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

/*     WORK( 1 ) = CMPLX( 1 ) */
    if (*n <= 0) {
	work[1].r = 1.f, work[1].i = 0.f;
	return 0;
    }

/*     Initialize Q and Z */

    if (icompq == 3) {
	claset_("Full", n, n, &c_b1, &c_b2, &q[q_offset], ldq);
    }
    if (icompz == 3) {
	claset_("Full", n, n, &c_b1, &c_b2, &z__[z_offset], ldz);
    }

/*     Machine Constants */

    in = *ihi + 1 - *ilo;
    safmin = slamch_("S");
    ulp = slamch_("E") * slamch_("B");
    anorm = clanhs_("F", &in, &h__[*ilo + *ilo * h_dim1], ldh, &rwork[1]);
    bnorm = clanhs_("F", &in, &t[*ilo + *ilo * t_dim1], ldt, &rwork[1]);
/* Computing MAX */
    r__1 = safmin, r__2 = ulp * anorm;
    atol = dmax(r__1,r__2);
/* Computing MAX */
    r__1 = safmin, r__2 = ulp * bnorm;
    btol = dmax(r__1,r__2);
    ascale = 1.f / dmax(safmin,anorm);
    bscale = 1.f / dmax(safmin,bnorm);


/*     Set Eigenvalues IHI+1:N */

    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	absb = c_abs(&t[j + j * t_dim1]);
	if (absb > safmin) {
	    i__2 = j + j * t_dim1;
	    q__2.r = t[i__2].r / absb, q__2.i = t[i__2].i / absb;
	    r_cnjg(&q__1, &q__2);
	    signbc.r = q__1.r, signbc.i = q__1.i;
	    i__2 = j + j * t_dim1;
	    t[i__2].r = absb, t[i__2].i = 0.f;
	    if (ilschr) {
		i__2 = j - 1;
		cscal_(&i__2, &signbc, &t[j * t_dim1 + 1], &c__1);
		cscal_(&j, &signbc, &h__[j * h_dim1 + 1], &c__1);
	    } else {
		i__2 = j + j * h_dim1;
		i__3 = j + j * h_dim1;
		q__1.r = h__[i__3].r * signbc.r - h__[i__3].i * signbc.i, 
			q__1.i = h__[i__3].r * signbc.i + h__[i__3].i * 
			signbc.r;
		h__[i__2].r = q__1.r, h__[i__2].i = q__1.i;
	    }
	    if (ilz) {
		cscal_(n, &signbc, &z__[j * z_dim1 + 1], &c__1);
	    }
	} else {
	    i__2 = j + j * t_dim1;
	    t[i__2].r = 0.f, t[i__2].i = 0.f;
	}
	i__2 = j;
	i__3 = j + j * h_dim1;
	alpha[i__2].r = h__[i__3].r, alpha[i__2].i = h__[i__3].i;
	i__2 = j;
	i__3 = j + j * t_dim1;
	beta[i__2].r = t[i__3].r, beta[i__2].i = t[i__3].i;
/* L10: */
    }

/*     If IHI < ILO, skip QZ steps */

    if (*ihi < *ilo) {
	goto L190;
    }

/*     MAIN QZ ITERATION LOOP */

/*     Initialize dynamic indices */

/*     Eigenvalues ILAST+1:N have been found. */
/*        Column operations modify rows IFRSTM:whatever */
/*        Row operations modify columns whatever:ILASTM */

/*     If only eigenvalues are being computed, then */
/*        IFRSTM is the row of the last splitting row above row ILAST; */
/*        this is always at least ILO. */
/*     IITER counts iterations since the last eigenvalue was found, */
/*        to tell when to use an extraordinary shift. */
/*     MAXIT is the maximum number of QZ sweeps allowed. */

    ilast = *ihi;
    if (ilschr) {
	ifrstm = 1;
	ilastm = *n;
    } else {
	ifrstm = *ilo;
	ilastm = *ihi;
    }
    iiter = 0;
    eshift.r = 0.f, eshift.i = 0.f;
    maxit = (*ihi - *ilo + 1) * 30;

    i__1 = maxit;
    for (jiter = 1; jiter <= i__1; ++jiter) {

/*        Check for too many iterations. */

	if (jiter > maxit) {
	    goto L180;
	}

/*        Split the matrix if possible. */

/*        Two tests: */
/*           1: H(j,j-1)=0  or  j=ILO */
/*           2: T(j,j)=0 */

/*        Special case: j=ILAST */

	if (ilast == *ilo) {
	    goto L60;
	} else {
	    i__2 = ilast + (ilast - 1) * h_dim1;
	    if ((r__1 = h__[i__2].r, dabs(r__1)) + (r__2 = r_imag(&h__[ilast 
		    + (ilast - 1) * h_dim1]), dabs(r__2)) <= atol) {
		i__2 = ilast + (ilast - 1) * h_dim1;
		h__[i__2].r = 0.f, h__[i__2].i = 0.f;
		goto L60;
	    }
	}

	if (c_abs(&t[ilast + ilast * t_dim1]) <= btol) {
	    i__2 = ilast + ilast * t_dim1;
	    t[i__2].r = 0.f, t[i__2].i = 0.f;
	    goto L50;
	}

/*        General case: j<ILAST */

	i__2 = *ilo;
	for (j = ilast - 1; j >= i__2; --j) {

/*           Test 1: for H(j,j-1)=0 or j=ILO */

	    if (j == *ilo) {
		ilazro = TRUE_;
	    } else {
		i__3 = j + (j - 1) * h_dim1;
		if ((r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = r_imag(&h__[j 
			+ (j - 1) * h_dim1]), dabs(r__2)) <= atol) {
		    i__3 = j + (j - 1) * h_dim1;
		    h__[i__3].r = 0.f, h__[i__3].i = 0.f;
		    ilazro = TRUE_;
		} else {
		    ilazro = FALSE_;
		}
	    }

/*           Test 2: for T(j,j)=0 */

	    if (c_abs(&t[j + j * t_dim1]) < btol) {
		i__3 = j + j * t_dim1;
		t[i__3].r = 0.f, t[i__3].i = 0.f;

/*              Test 1a: Check for 2 consecutive small subdiagonals in A */

		ilazr2 = FALSE_;
		if (! ilazro) {
		    i__3 = j + (j - 1) * h_dim1;
		    i__4 = j + 1 + j * h_dim1;
		    i__5 = j + j * h_dim1;
		    if (((r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = r_imag(&
			    h__[j + (j - 1) * h_dim1]), dabs(r__2))) * (
			    ascale * ((r__3 = h__[i__4].r, dabs(r__3)) + (
			    r__4 = r_imag(&h__[j + 1 + j * h_dim1]), dabs(
			    r__4)))) <= ((r__5 = h__[i__5].r, dabs(r__5)) + (
			    r__6 = r_imag(&h__[j + j * h_dim1]), dabs(r__6))) 
			    * (ascale * atol)) {
			ilazr2 = TRUE_;
		    }
		}

/*              If both tests pass (1 & 2), i.e., the leading diagonal */
/*              element of B in the block is zero, split a 1x1 block off */
/*              at the top. (I.e., at the J-th row/column) The leading */
/*              diagonal element of the remainder can also be zero, so */
/*              this may have to be done repeatedly. */

		if (ilazro || ilazr2) {
		    i__3 = ilast - 1;
		    for (jch = j; jch <= i__3; ++jch) {
			i__4 = jch + jch * h_dim1;
			ctemp.r = h__[i__4].r, ctemp.i = h__[i__4].i;
			clartg_(&ctemp, &h__[jch + 1 + jch * h_dim1], &c__, &
				s, &h__[jch + jch * h_dim1]);
			i__4 = jch + 1 + jch * h_dim1;
			h__[i__4].r = 0.f, h__[i__4].i = 0.f;
			i__4 = ilastm - jch;
			crot_(&i__4, &h__[jch + (jch + 1) * h_dim1], ldh, &
				h__[jch + 1 + (jch + 1) * h_dim1], ldh, &c__, 
				&s);
			i__4 = ilastm - jch;
			crot_(&i__4, &t[jch + (jch + 1) * t_dim1], ldt, &t[
				jch + 1 + (jch + 1) * t_dim1], ldt, &c__, &s);
			if (ilq) {
			    r_cnjg(&q__1, &s);
			    crot_(n, &q[jch * q_dim1 + 1], &c__1, &q[(jch + 1)
				     * q_dim1 + 1], &c__1, &c__, &q__1);
			}
			if (ilazr2) {
			    i__4 = jch + (jch - 1) * h_dim1;
			    i__5 = jch + (jch - 1) * h_dim1;
			    q__1.r = c__ * h__[i__5].r, q__1.i = c__ * h__[
				    i__5].i;
			    h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
			}
			ilazr2 = FALSE_;
			i__4 = jch + 1 + (jch + 1) * t_dim1;
			if ((r__1 = t[i__4].r, dabs(r__1)) + (r__2 = r_imag(&
				t[jch + 1 + (jch + 1) * t_dim1]), dabs(r__2)) 
				>= btol) {
			    if (jch + 1 >= ilast) {
				goto L60;
			    } else {
				ifirst = jch + 1;
				goto L70;
			    }
			}
			i__4 = jch + 1 + (jch + 1) * t_dim1;
			t[i__4].r = 0.f, t[i__4].i = 0.f;
/* L20: */
		    }
		    goto L50;
		} else {

/*                 Only test 2 passed -- chase the zero to T(ILAST,ILAST) */
/*                 Then process as in the case T(ILAST,ILAST)=0 */

		    i__3 = ilast - 1;
		    for (jch = j; jch <= i__3; ++jch) {
			i__4 = jch + (jch + 1) * t_dim1;
			ctemp.r = t[i__4].r, ctemp.i = t[i__4].i;
			clartg_(&ctemp, &t[jch + 1 + (jch + 1) * t_dim1], &
				c__, &s, &t[jch + (jch + 1) * t_dim1]);
			i__4 = jch + 1 + (jch + 1) * t_dim1;
			t[i__4].r = 0.f, t[i__4].i = 0.f;
			if (jch < ilastm - 1) {
			    i__4 = ilastm - jch - 1;
			    crot_(&i__4, &t[jch + (jch + 2) * t_dim1], ldt, &
				    t[jch + 1 + (jch + 2) * t_dim1], ldt, &
				    c__, &s);
			}
			i__4 = ilastm - jch + 2;
			crot_(&i__4, &h__[jch + (jch - 1) * h_dim1], ldh, &
				h__[jch + 1 + (jch - 1) * h_dim1], ldh, &c__, 
				&s);
			if (ilq) {
			    r_cnjg(&q__1, &s);
			    crot_(n, &q[jch * q_dim1 + 1], &c__1, &q[(jch + 1)
				     * q_dim1 + 1], &c__1, &c__, &q__1);
			}
			i__4 = jch + 1 + jch * h_dim1;
			ctemp.r = h__[i__4].r, ctemp.i = h__[i__4].i;
			clartg_(&ctemp, &h__[jch + 1 + (jch - 1) * h_dim1], &
				c__, &s, &h__[jch + 1 + jch * h_dim1]);
			i__4 = jch + 1 + (jch - 1) * h_dim1;
			h__[i__4].r = 0.f, h__[i__4].i = 0.f;
			i__4 = jch + 1 - ifrstm;
			crot_(&i__4, &h__[ifrstm + jch * h_dim1], &c__1, &h__[
				ifrstm + (jch - 1) * h_dim1], &c__1, &c__, &s)
				;
			i__4 = jch - ifrstm;
			crot_(&i__4, &t[ifrstm + jch * t_dim1], &c__1, &t[
				ifrstm + (jch - 1) * t_dim1], &c__1, &c__, &s)
				;
			if (ilz) {
			    crot_(n, &z__[jch * z_dim1 + 1], &c__1, &z__[(jch 
				    - 1) * z_dim1 + 1], &c__1, &c__, &s);
			}
/* L30: */
		    }
		    goto L50;
		}
	    } else if (ilazro) {

/*              Only test 1 passed -- work on J:ILAST */

		ifirst = j;
		goto L70;
	    }

/*           Neither test passed -- try next J */

/* L40: */
	}

/*        (Drop-through is "impossible") */

	*info = (*n << 1) + 1;
	goto L210;

/*        T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a */
/*        1x1 block. */

L50:
	i__2 = ilast + ilast * h_dim1;
	ctemp.r = h__[i__2].r, ctemp.i = h__[i__2].i;
	clartg_(&ctemp, &h__[ilast + (ilast - 1) * h_dim1], &c__, &s, &h__[
		ilast + ilast * h_dim1]);
	i__2 = ilast + (ilast - 1) * h_dim1;
	h__[i__2].r = 0.f, h__[i__2].i = 0.f;
	i__2 = ilast - ifrstm;
	crot_(&i__2, &h__[ifrstm + ilast * h_dim1], &c__1, &h__[ifrstm + (
		ilast - 1) * h_dim1], &c__1, &c__, &s);
	i__2 = ilast - ifrstm;
	crot_(&i__2, &t[ifrstm + ilast * t_dim1], &c__1, &t[ifrstm + (ilast - 
		1) * t_dim1], &c__1, &c__, &s);
	if (ilz) {
	    crot_(n, &z__[ilast * z_dim1 + 1], &c__1, &z__[(ilast - 1) * 
		    z_dim1 + 1], &c__1, &c__, &s);
	}

/*        H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHA and BETA */

L60:
	absb = c_abs(&t[ilast + ilast * t_dim1]);
	if (absb > safmin) {
	    i__2 = ilast + ilast * t_dim1;
	    q__2.r = t[i__2].r / absb, q__2.i = t[i__2].i / absb;
	    r_cnjg(&q__1, &q__2);
	    signbc.r = q__1.r, signbc.i = q__1.i;
	    i__2 = ilast + ilast * t_dim1;
	    t[i__2].r = absb, t[i__2].i = 0.f;
	    if (ilschr) {
		i__2 = ilast - ifrstm;
		cscal_(&i__2, &signbc, &t[ifrstm + ilast * t_dim1], &c__1);
		i__2 = ilast + 1 - ifrstm;
		cscal_(&i__2, &signbc, &h__[ifrstm + ilast * h_dim1], &c__1);
	    } else {
		i__2 = ilast + ilast * h_dim1;
		i__3 = ilast + ilast * h_dim1;
		q__1.r = h__[i__3].r * signbc.r - h__[i__3].i * signbc.i, 
			q__1.i = h__[i__3].r * signbc.i + h__[i__3].i * 
			signbc.r;
		h__[i__2].r = q__1.r, h__[i__2].i = q__1.i;
	    }
	    if (ilz) {
		cscal_(n, &signbc, &z__[ilast * z_dim1 + 1], &c__1);
	    }
	} else {
	    i__2 = ilast + ilast * t_dim1;
	    t[i__2].r = 0.f, t[i__2].i = 0.f;
	}
	i__2 = ilast;
	i__3 = ilast + ilast * h_dim1;
	alpha[i__2].r = h__[i__3].r, alpha[i__2].i = h__[i__3].i;
	i__2 = ilast;
	i__3 = ilast + ilast * t_dim1;
	beta[i__2].r = t[i__3].r, beta[i__2].i = t[i__3].i;

/*        Go to next block -- exit if finished. */

	--ilast;
	if (ilast < *ilo) {
	    goto L190;
	}

/*        Reset counters */

	iiter = 0;
	eshift.r = 0.f, eshift.i = 0.f;
	if (! ilschr) {
	    ilastm = ilast;
	    if (ifrstm > ilast) {
		ifrstm = *ilo;
	    }
	}
	goto L160;

/*        QZ step */

/*        This iteration only involves rows/columns IFIRST:ILAST.  We */
/*        assume IFIRST < ILAST, and that the diagonal of B is non-zero. */

L70:
	++iiter;
	if (! ilschr) {
	    ifrstm = ifirst;
	}

/*        Compute the Shift. */

/*        At this point, IFIRST < ILAST, and the diagonal elements of */
/*        T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in */
/*        magnitude) */

	if (iiter / 10 * 10 != iiter) {

/*           The Wilkinson shift (AEP p.512), i.e., the eigenvalue of */
/*           the bottom-right 2x2 block of A inv(B) which is nearest to */
/*           the bottom-right element. */

/*           We factor B as U*D, where U has unit diagonals, and */
/*           compute (A*inv(D))*inv(U). */

	    i__2 = ilast - 1 + ilast * t_dim1;
	    q__2.r = bscale * t[i__2].r, q__2.i = bscale * t[i__2].i;
	    i__3 = ilast + ilast * t_dim1;
	    q__3.r = bscale * t[i__3].r, q__3.i = bscale * t[i__3].i;
	    c_div(&q__1, &q__2, &q__3);
	    u12.r = q__1.r, u12.i = q__1.i;
	    i__2 = ilast - 1 + (ilast - 1) * h_dim1;
	    q__2.r = ascale * h__[i__2].r, q__2.i = ascale * h__[i__2].i;
	    i__3 = ilast - 1 + (ilast - 1) * t_dim1;
	    q__3.r = bscale * t[i__3].r, q__3.i = bscale * t[i__3].i;
	    c_div(&q__1, &q__2, &q__3);
	    ad11.r = q__1.r, ad11.i = q__1.i;
	    i__2 = ilast + (ilast - 1) * h_dim1;
	    q__2.r = ascale * h__[i__2].r, q__2.i = ascale * h__[i__2].i;
	    i__3 = ilast - 1 + (ilast - 1) * t_dim1;
	    q__3.r = bscale * t[i__3].r, q__3.i = bscale * t[i__3].i;
	    c_div(&q__1, &q__2, &q__3);
	    ad21.r = q__1.r, ad21.i = q__1.i;
	    i__2 = ilast - 1 + ilast * h_dim1;
	    q__2.r = ascale * h__[i__2].r, q__2.i = ascale * h__[i__2].i;
	    i__3 = ilast + ilast * t_dim1;
	    q__3.r = bscale * t[i__3].r, q__3.i = bscale * t[i__3].i;
	    c_div(&q__1, &q__2, &q__3);
	    ad12.r = q__1.r, ad12.i = q__1.i;
	    i__2 = ilast + ilast * h_dim1;
	    q__2.r = ascale * h__[i__2].r, q__2.i = ascale * h__[i__2].i;
	    i__3 = ilast + ilast * t_dim1;
	    q__3.r = bscale * t[i__3].r, q__3.i = bscale * t[i__3].i;
	    c_div(&q__1, &q__2, &q__3);
	    ad22.r = q__1.r, ad22.i = q__1.i;
	    q__2.r = u12.r * ad21.r - u12.i * ad21.i, q__2.i = u12.r * ad21.i 
		    + u12.i * ad21.r;
	    q__1.r = ad22.r - q__2.r, q__1.i = ad22.i - q__2.i;
	    abi22.r = q__1.r, abi22.i = q__1.i;

	    q__2.r = ad11.r + abi22.r, q__2.i = ad11.i + abi22.i;
	    q__1.r = q__2.r * .5f, q__1.i = q__2.i * .5f;
	    t1.r = q__1.r, t1.i = q__1.i;
	    pow_ci(&q__4, &t1, &c__2);
	    q__5.r = ad12.r * ad21.r - ad12.i * ad21.i, q__5.i = ad12.r * 
		    ad21.i + ad12.i * ad21.r;
	    q__3.r = q__4.r + q__5.r, q__3.i = q__4.i + q__5.i;
	    q__6.r = ad11.r * ad22.r - ad11.i * ad22.i, q__6.i = ad11.r * 
		    ad22.i + ad11.i * ad22.r;
	    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
	    c_sqrt(&q__1, &q__2);
	    rtdisc.r = q__1.r, rtdisc.i = q__1.i;
	    q__1.r = t1.r - abi22.r, q__1.i = t1.i - abi22.i;
	    q__2.r = t1.r - abi22.r, q__2.i = t1.i - abi22.i;
	    temp = q__1.r * rtdisc.r + r_imag(&q__2) * r_imag(&rtdisc);
	    if (temp <= 0.f) {
		q__1.r = t1.r + rtdisc.r, q__1.i = t1.i + rtdisc.i;
		shift.r = q__1.r, shift.i = q__1.i;
	    } else {
		q__1.r = t1.r - rtdisc.r, q__1.i = t1.i - rtdisc.i;
		shift.r = q__1.r, shift.i = q__1.i;
	    }
	} else {

/*           Exceptional shift.  Chosen for no particularly good reason. */

	    i__2 = ilast - 1 + ilast * h_dim1;
	    q__4.r = ascale * h__[i__2].r, q__4.i = ascale * h__[i__2].i;
	    i__3 = ilast - 1 + (ilast - 1) * t_dim1;
	    q__5.r = bscale * t[i__3].r, q__5.i = bscale * t[i__3].i;
	    c_div(&q__3, &q__4, &q__5);
	    r_cnjg(&q__2, &q__3);
	    q__1.r = eshift.r + q__2.r, q__1.i = eshift.i + q__2.i;
	    eshift.r = q__1.r, eshift.i = q__1.i;
	    shift.r = eshift.r, shift.i = eshift.i;
	}

/*        Now check for two consecutive small subdiagonals. */

	i__2 = ifirst + 1;
	for (j = ilast - 1; j >= i__2; --j) {
	    istart = j;
	    i__3 = j + j * h_dim1;
	    q__2.r = ascale * h__[i__3].r, q__2.i = ascale * h__[i__3].i;
	    i__4 = j + j * t_dim1;
	    q__4.r = bscale * t[i__4].r, q__4.i = bscale * t[i__4].i;
	    q__3.r = shift.r * q__4.r - shift.i * q__4.i, q__3.i = shift.r * 
		    q__4.i + shift.i * q__4.r;
	    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	    ctemp.r = q__1.r, ctemp.i = q__1.i;
	    temp = (r__1 = ctemp.r, dabs(r__1)) + (r__2 = r_imag(&ctemp), 
		    dabs(r__2));
	    i__3 = j + 1 + j * h_dim1;
	    temp2 = ascale * ((r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = 
		    r_imag(&h__[j + 1 + j * h_dim1]), dabs(r__2)));
	    tempr = dmax(temp,temp2);
	    if (tempr < 1.f && tempr != 0.f) {
		temp /= tempr;
		temp2 /= tempr;
	    }
	    i__3 = j + (j - 1) * h_dim1;
	    if (((r__1 = h__[i__3].r, dabs(r__1)) + (r__2 = r_imag(&h__[j + (
		    j - 1) * h_dim1]), dabs(r__2))) * temp2 <= temp * atol) {
		goto L90;
	    }
/* L80: */
	}

	istart = ifirst;
	i__2 = ifirst + ifirst * h_dim1;
	q__2.r = ascale * h__[i__2].r, q__2.i = ascale * h__[i__2].i;
	i__3 = ifirst + ifirst * t_dim1;
	q__4.r = bscale * t[i__3].r, q__4.i = bscale * t[i__3].i;
	q__3.r = shift.r * q__4.r - shift.i * q__4.i, q__3.i = shift.r * 
		q__4.i + shift.i * q__4.r;
	q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
	ctemp.r = q__1.r, ctemp.i = q__1.i;
L90:

/*        Do an implicit-shift QZ sweep. */

/*        Initial Q */

	i__2 = istart + 1 + istart * h_dim1;
	q__1.r = ascale * h__[i__2].r, q__1.i = ascale * h__[i__2].i;
	ctemp2.r = q__1.r, ctemp2.i = q__1.i;
	clartg_(&ctemp, &ctemp2, &c__, &s, &ctemp3);

/*        Sweep */

	i__2 = ilast - 1;
	for (j = istart; j <= i__2; ++j) {
	    if (j > istart) {
		i__3 = j + (j - 1) * h_dim1;
		ctemp.r = h__[i__3].r, ctemp.i = h__[i__3].i;
		clartg_(&ctemp, &h__[j + 1 + (j - 1) * h_dim1], &c__, &s, &
			h__[j + (j - 1) * h_dim1]);
		i__3 = j + 1 + (j - 1) * h_dim1;
		h__[i__3].r = 0.f, h__[i__3].i = 0.f;
	    }

	    i__3 = ilastm;
	    for (jc = j; jc <= i__3; ++jc) {
		i__4 = j + jc * h_dim1;
		q__2.r = c__ * h__[i__4].r, q__2.i = c__ * h__[i__4].i;
		i__5 = j + 1 + jc * h_dim1;
		q__3.r = s.r * h__[i__5].r - s.i * h__[i__5].i, q__3.i = s.r *
			 h__[i__5].i + s.i * h__[i__5].r;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		ctemp.r = q__1.r, ctemp.i = q__1.i;
		i__4 = j + 1 + jc * h_dim1;
		r_cnjg(&q__4, &s);
		q__3.r = -q__4.r, q__3.i = -q__4.i;
		i__5 = j + jc * h_dim1;
		q__2.r = q__3.r * h__[i__5].r - q__3.i * h__[i__5].i, q__2.i =
			 q__3.r * h__[i__5].i + q__3.i * h__[i__5].r;
		i__6 = j + 1 + jc * h_dim1;
		q__5.r = c__ * h__[i__6].r, q__5.i = c__ * h__[i__6].i;
		q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
		i__4 = j + jc * h_dim1;
		h__[i__4].r = ctemp.r, h__[i__4].i = ctemp.i;
		i__4 = j + jc * t_dim1;
		q__2.r = c__ * t[i__4].r, q__2.i = c__ * t[i__4].i;
		i__5 = j + 1 + jc * t_dim1;
		q__3.r = s.r * t[i__5].r - s.i * t[i__5].i, q__3.i = s.r * t[
			i__5].i + s.i * t[i__5].r;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		ctemp2.r = q__1.r, ctemp2.i = q__1.i;
		i__4 = j + 1 + jc * t_dim1;
		r_cnjg(&q__4, &s);
		q__3.r = -q__4.r, q__3.i = -q__4.i;
		i__5 = j + jc * t_dim1;
		q__2.r = q__3.r * t[i__5].r - q__3.i * t[i__5].i, q__2.i = 
			q__3.r * t[i__5].i + q__3.i * t[i__5].r;
		i__6 = j + 1 + jc * t_dim1;
		q__5.r = c__ * t[i__6].r, q__5.i = c__ * t[i__6].i;
		q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
		t[i__4].r = q__1.r, t[i__4].i = q__1.i;
		i__4 = j + jc * t_dim1;
		t[i__4].r = ctemp2.r, t[i__4].i = ctemp2.i;
/* L100: */
	    }
	    if (ilq) {
		i__3 = *n;
		for (jr = 1; jr <= i__3; ++jr) {
		    i__4 = jr + j * q_dim1;
		    q__2.r = c__ * q[i__4].r, q__2.i = c__ * q[i__4].i;
		    r_cnjg(&q__4, &s);
		    i__5 = jr + (j + 1) * q_dim1;
		    q__3.r = q__4.r * q[i__5].r - q__4.i * q[i__5].i, q__3.i =
			     q__4.r * q[i__5].i + q__4.i * q[i__5].r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ctemp.r = q__1.r, ctemp.i = q__1.i;
		    i__4 = jr + (j + 1) * q_dim1;
		    q__3.r = -s.r, q__3.i = -s.i;
		    i__5 = jr + j * q_dim1;
		    q__2.r = q__3.r * q[i__5].r - q__3.i * q[i__5].i, q__2.i =
			     q__3.r * q[i__5].i + q__3.i * q[i__5].r;
		    i__6 = jr + (j + 1) * q_dim1;
		    q__4.r = c__ * q[i__6].r, q__4.i = c__ * q[i__6].i;
		    q__1.r = q__2.r + q__4.r, q__1.i = q__2.i + q__4.i;
		    q[i__4].r = q__1.r, q[i__4].i = q__1.i;
		    i__4 = jr + j * q_dim1;
		    q[i__4].r = ctemp.r, q[i__4].i = ctemp.i;
/* L110: */
		}
	    }

	    i__3 = j + 1 + (j + 1) * t_dim1;
	    ctemp.r = t[i__3].r, ctemp.i = t[i__3].i;
	    clartg_(&ctemp, &t[j + 1 + j * t_dim1], &c__, &s, &t[j + 1 + (j + 
		    1) * t_dim1]);
	    i__3 = j + 1 + j * t_dim1;
	    t[i__3].r = 0.f, t[i__3].i = 0.f;

/* Computing MIN */
	    i__4 = j + 2;
	    i__3 = min(i__4,ilast);
	    for (jr = ifrstm; jr <= i__3; ++jr) {
		i__4 = jr + (j + 1) * h_dim1;
		q__2.r = c__ * h__[i__4].r, q__2.i = c__ * h__[i__4].i;
		i__5 = jr + j * h_dim1;
		q__3.r = s.r * h__[i__5].r - s.i * h__[i__5].i, q__3.i = s.r *
			 h__[i__5].i + s.i * h__[i__5].r;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		ctemp.r = q__1.r, ctemp.i = q__1.i;
		i__4 = jr + j * h_dim1;
		r_cnjg(&q__4, &s);
		q__3.r = -q__4.r, q__3.i = -q__4.i;
		i__5 = jr + (j + 1) * h_dim1;
		q__2.r = q__3.r * h__[i__5].r - q__3.i * h__[i__5].i, q__2.i =
			 q__3.r * h__[i__5].i + q__3.i * h__[i__5].r;
		i__6 = jr + j * h_dim1;
		q__5.r = c__ * h__[i__6].r, q__5.i = c__ * h__[i__6].i;
		q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
		h__[i__4].r = q__1.r, h__[i__4].i = q__1.i;
		i__4 = jr + (j + 1) * h_dim1;
		h__[i__4].r = ctemp.r, h__[i__4].i = ctemp.i;
/* L120: */
	    }
	    i__3 = j;
	    for (jr = ifrstm; jr <= i__3; ++jr) {
		i__4 = jr + (j + 1) * t_dim1;
		q__2.r = c__ * t[i__4].r, q__2.i = c__ * t[i__4].i;
		i__5 = jr + j * t_dim1;
		q__3.r = s.r * t[i__5].r - s.i * t[i__5].i, q__3.i = s.r * t[
			i__5].i + s.i * t[i__5].r;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		ctemp.r = q__1.r, ctemp.i = q__1.i;
		i__4 = jr + j * t_dim1;
		r_cnjg(&q__4, &s);
		q__3.r = -q__4.r, q__3.i = -q__4.i;
		i__5 = jr + (j + 1) * t_dim1;
		q__2.r = q__3.r * t[i__5].r - q__3.i * t[i__5].i, q__2.i = 
			q__3.r * t[i__5].i + q__3.i * t[i__5].r;
		i__6 = jr + j * t_dim1;
		q__5.r = c__ * t[i__6].r, q__5.i = c__ * t[i__6].i;
		q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
		t[i__4].r = q__1.r, t[i__4].i = q__1.i;
		i__4 = jr + (j + 1) * t_dim1;
		t[i__4].r = ctemp.r, t[i__4].i = ctemp.i;
/* L130: */
	    }
	    if (ilz) {
		i__3 = *n;
		for (jr = 1; jr <= i__3; ++jr) {
		    i__4 = jr + (j + 1) * z_dim1;
		    q__2.r = c__ * z__[i__4].r, q__2.i = c__ * z__[i__4].i;
		    i__5 = jr + j * z_dim1;
		    q__3.r = s.r * z__[i__5].r - s.i * z__[i__5].i, q__3.i = 
			    s.r * z__[i__5].i + s.i * z__[i__5].r;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ctemp.r = q__1.r, ctemp.i = q__1.i;
		    i__4 = jr + j * z_dim1;
		    r_cnjg(&q__4, &s);
		    q__3.r = -q__4.r, q__3.i = -q__4.i;
		    i__5 = jr + (j + 1) * z_dim1;
		    q__2.r = q__3.r * z__[i__5].r - q__3.i * z__[i__5].i, 
			    q__2.i = q__3.r * z__[i__5].i + q__3.i * z__[i__5]
			    .r;
		    i__6 = jr + j * z_dim1;
		    q__5.r = c__ * z__[i__6].r, q__5.i = c__ * z__[i__6].i;
		    q__1.r = q__2.r + q__5.r, q__1.i = q__2.i + q__5.i;
		    z__[i__4].r = q__1.r, z__[i__4].i = q__1.i;
		    i__4 = jr + (j + 1) * z_dim1;
		    z__[i__4].r = ctemp.r, z__[i__4].i = ctemp.i;
/* L140: */
		}
	    }
/* L150: */
	}

L160:

/* L170: */
	;
    }

/*     Drop-through = non-convergence */

L180:
    *info = ilast;
    goto L210;

/*     Successful completion of all QZ steps */

L190:

/*     Set Eigenvalues 1:ILO-1 */

    i__1 = *ilo - 1;
    for (j = 1; j <= i__1; ++j) {
	absb = c_abs(&t[j + j * t_dim1]);
	if (absb > safmin) {
	    i__2 = j + j * t_dim1;
	    q__2.r = t[i__2].r / absb, q__2.i = t[i__2].i / absb;
	    r_cnjg(&q__1, &q__2);
	    signbc.r = q__1.r, signbc.i = q__1.i;
	    i__2 = j + j * t_dim1;
	    t[i__2].r = absb, t[i__2].i = 0.f;
	    if (ilschr) {
		i__2 = j - 1;
		cscal_(&i__2, &signbc, &t[j * t_dim1 + 1], &c__1);
		cscal_(&j, &signbc, &h__[j * h_dim1 + 1], &c__1);
	    } else {
		i__2 = j + j * h_dim1;
		i__3 = j + j * h_dim1;
		q__1.r = h__[i__3].r * signbc.r - h__[i__3].i * signbc.i, 
			q__1.i = h__[i__3].r * signbc.i + h__[i__3].i * 
			signbc.r;
		h__[i__2].r = q__1.r, h__[i__2].i = q__1.i;
	    }
	    if (ilz) {
		cscal_(n, &signbc, &z__[j * z_dim1 + 1], &c__1);
	    }
	} else {
	    i__2 = j + j * t_dim1;
	    t[i__2].r = 0.f, t[i__2].i = 0.f;
	}
	i__2 = j;
	i__3 = j + j * h_dim1;
	alpha[i__2].r = h__[i__3].r, alpha[i__2].i = h__[i__3].i;
	i__2 = j;
	i__3 = j + j * t_dim1;
	beta[i__2].r = t[i__3].r, beta[i__2].i = t[i__3].i;
/* L200: */
    }

/*     Normal Termination */

    *info = 0;

/*     Exit (other than argument error) -- return optimal workspace size */

L210:
    q__1.r = (real) (*n), q__1.i = 0.f;
    work[1].r = q__1.r, work[1].i = q__1.i;
    return 0;

/*     End of CHGEQZ */

} /* chgeqz_ */
