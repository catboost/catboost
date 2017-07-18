/* dtgsja.f -- translated by f2c (version 20061008).
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

static doublereal c_b13 = 0.;
static doublereal c_b14 = 1.;
static integer c__1 = 1;
static doublereal c_b43 = -1.;

/* Subroutine */ int dtgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, doublereal *a, 
	integer *lda, doublereal *b, integer *ldb, doublereal *tola, 
	doublereal *tolb, doublereal *alpha, doublereal *beta, doublereal *u, 
	integer *ldu, doublereal *v, integer *ldv, doublereal *q, integer *
	ldq, doublereal *work, integer *ncycle, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, u_dim1, 
	    u_offset, v_dim1, v_offset, i__1, i__2, i__3, i__4;
    doublereal d__1;

    /* Local variables */
    integer i__, j;
    doublereal a1, a2, a3, b1, b2, b3, csq, csu, csv, snq, rwk, snu, snv;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    doublereal gamma;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    logical initq, initu, initv, wantq, upper;
    doublereal error, ssmin;
    logical wantu, wantv;
    extern /* Subroutine */ int dlags2_(logical *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), dlapll_(integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *);
    integer kcycle;
    extern /* Subroutine */ int dlartg_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *), dlaset_(char *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    integer *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTGSJA computes the generalized singular value decomposition (GSVD) */
/*  of two real upper triangular (or trapezoidal) matrices A and B. */

/*  On entry, it is assumed that matrices A and B have the following */
/*  forms, which may be obtained by the preprocessing subroutine DGGSVP */
/*  from a general M-by-N matrix A and P-by-N matrix B: */

/*               N-K-L  K    L */
/*     A =    K ( 0    A12  A13 ) if M-K-L >= 0; */
/*            L ( 0     0   A23 ) */
/*        M-K-L ( 0     0    0  ) */

/*             N-K-L  K    L */
/*     A =  K ( 0    A12  A13 ) if M-K-L < 0; */
/*        M-K ( 0     0   A23 ) */

/*             N-K-L  K    L */
/*     B =  L ( 0     0   B13 ) */
/*        P-L ( 0     0    0  ) */

/*  where the K-by-K matrix A12 and L-by-L matrix B13 are nonsingular */
/*  upper triangular; A23 is L-by-L upper triangular if M-K-L >= 0, */
/*  otherwise A23 is (M-K)-by-L upper trapezoidal. */

/*  On exit, */

/*              U'*A*Q = D1*( 0 R ),    V'*B*Q = D2*( 0 R ), */

/*  where U, V and Q are orthogonal matrices, Z' denotes the transpose */
/*  of Z, R is a nonsingular upper triangular matrix, and D1 and D2 are */
/*  ``diagonal'' matrices, which are of the following structures: */

/*  If M-K-L >= 0, */

/*                      K  L */
/*         D1 =     K ( I  0 ) */
/*                  L ( 0  C ) */
/*              M-K-L ( 0  0 ) */

/*                    K  L */
/*         D2 = L   ( 0  S ) */
/*              P-L ( 0  0 ) */

/*                 N-K-L  K    L */
/*    ( 0 R ) = K (  0   R11  R12 ) K */
/*              L (  0    0   R22 ) L */

/*  where */

/*    C = diag( ALPHA(K+1), ... , ALPHA(K+L) ), */
/*    S = diag( BETA(K+1),  ... , BETA(K+L) ), */
/*    C**2 + S**2 = I. */

/*    R is stored in A(1:K+L,N-K-L+1:N) on exit. */

/*  If M-K-L < 0, */

/*                 K M-K K+L-M */
/*      D1 =   K ( I  0    0   ) */
/*           M-K ( 0  C    0   ) */

/*                   K M-K K+L-M */
/*      D2 =   M-K ( 0  S    0   ) */
/*           K+L-M ( 0  0    I   ) */
/*             P-L ( 0  0    0   ) */

/*                 N-K-L  K   M-K  K+L-M */
/* ( 0 R ) =    K ( 0    R11  R12  R13  ) */
/*            M-K ( 0     0   R22  R23  ) */
/*          K+L-M ( 0     0    0   R33  ) */

/*  where */
/*  C = diag( ALPHA(K+1), ... , ALPHA(M) ), */
/*  S = diag( BETA(K+1),  ... , BETA(M) ), */
/*  C**2 + S**2 = I. */

/*  R = ( R11 R12 R13 ) is stored in A(1:M, N-K-L+1:N) and R33 is stored */
/*      (  0  R22 R23 ) */
/*  in B(M-K+1:L,N+M-K-L+1:N) on exit. */

/*  The computation of the orthogonal transformation matrices U, V or Q */
/*  is optional.  These matrices may either be formed explicitly, or they */
/*  may be postmultiplied into input matrices U1, V1, or Q1. */

/*  Arguments */
/*  ========= */

/*  JOBU    (input) CHARACTER*1 */
/*          = 'U':  U must contain an orthogonal matrix U1 on entry, and */
/*                  the product U1*U is returned; */
/*          = 'I':  U is initialized to the unit matrix, and the */
/*                  orthogonal matrix U is returned; */
/*          = 'N':  U is not computed. */

/*  JOBV    (input) CHARACTER*1 */
/*          = 'V':  V must contain an orthogonal matrix V1 on entry, and */
/*                  the product V1*V is returned; */
/*          = 'I':  V is initialized to the unit matrix, and the */
/*                  orthogonal matrix V is returned; */
/*          = 'N':  V is not computed. */

/*  JOBQ    (input) CHARACTER*1 */
/*          = 'Q':  Q must contain an orthogonal matrix Q1 on entry, and */
/*                  the product Q1*Q is returned; */
/*          = 'I':  Q is initialized to the unit matrix, and the */
/*                  orthogonal matrix Q is returned; */
/*          = 'N':  Q is not computed. */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  P       (input) INTEGER */
/*          The number of rows of the matrix B.  P >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrices A and B.  N >= 0. */

/*  K       (input) INTEGER */
/*  L       (input) INTEGER */
/*          K and L specify the subblocks in the input matrices A and B: */
/*          A23 = A(K+1:MIN(K+L,M),N-L+1:N) and B13 = B(1:L,N-L+1:N) */
/*          of A and B, whose GSVD is going to be computed by DTGSJA. */
/*          See Further details. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, A(N-K+1:N,1:MIN(K+L,M) ) contains the triangular */
/*          matrix R or part of R.  See Purpose for details. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,N) */
/*          On entry, the P-by-N matrix B. */
/*          On exit, if necessary, B(M-K+1:L,N+M-K-L+1:N) contains */
/*          a part of R.  See Purpose for details. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,P). */

/*  TOLA    (input) DOUBLE PRECISION */
/*  TOLB    (input) DOUBLE PRECISION */
/*          TOLA and TOLB are the convergence criteria for the Jacobi- */
/*          Kogbetliantz iteration procedure. Generally, they are the */
/*          same as used in the preprocessing step, say */
/*              TOLA = max(M,N)*norm(A)*MAZHEPS, */
/*              TOLB = max(P,N)*norm(B)*MAZHEPS. */

/*  ALPHA   (output) DOUBLE PRECISION array, dimension (N) */
/*  BETA    (output) DOUBLE PRECISION array, dimension (N) */
/*          On exit, ALPHA and BETA contain the generalized singular */
/*          value pairs of A and B; */
/*            ALPHA(1:K) = 1, */
/*            BETA(1:K)  = 0, */
/*          and if M-K-L >= 0, */
/*            ALPHA(K+1:K+L) = diag(C), */
/*            BETA(K+1:K+L)  = diag(S), */
/*          or if M-K-L < 0, */
/*            ALPHA(K+1:M)= C, ALPHA(M+1:K+L)= 0 */
/*            BETA(K+1:M) = S, BETA(M+1:K+L) = 1. */
/*          Furthermore, if K+L < N, */
/*            ALPHA(K+L+1:N) = 0 and */
/*            BETA(K+L+1:N)  = 0. */

/*  U       (input/output) DOUBLE PRECISION array, dimension (LDU,M) */
/*          On entry, if JOBU = 'U', U must contain a matrix U1 (usually */
/*          the orthogonal matrix returned by DGGSVP). */
/*          On exit, */
/*          if JOBU = 'I', U contains the orthogonal matrix U; */
/*          if JOBU = 'U', U contains the product U1*U. */
/*          If JOBU = 'N', U is not referenced. */

/*  LDU     (input) INTEGER */
/*          The leading dimension of the array U. LDU >= max(1,M) if */
/*          JOBU = 'U'; LDU >= 1 otherwise. */

/*  V       (input/output) DOUBLE PRECISION array, dimension (LDV,P) */
/*          On entry, if JOBV = 'V', V must contain a matrix V1 (usually */
/*          the orthogonal matrix returned by DGGSVP). */
/*          On exit, */
/*          if JOBV = 'I', V contains the orthogonal matrix V; */
/*          if JOBV = 'V', V contains the product V1*V. */
/*          If JOBV = 'N', V is not referenced. */

/*  LDV     (input) INTEGER */
/*          The leading dimension of the array V. LDV >= max(1,P) if */
/*          JOBV = 'V'; LDV >= 1 otherwise. */

/*  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N) */
/*          On entry, if JOBQ = 'Q', Q must contain a matrix Q1 (usually */
/*          the orthogonal matrix returned by DGGSVP). */
/*          On exit, */
/*          if JOBQ = 'I', Q contains the orthogonal matrix Q; */
/*          if JOBQ = 'Q', Q contains the product Q1*Q. */
/*          If JOBQ = 'N', Q is not referenced. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. LDQ >= max(1,N) if */
/*          JOBQ = 'Q'; LDQ >= 1 otherwise. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N) */

/*  NCYCLE  (output) INTEGER */
/*          The number of cycles required for convergence. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          = 1:  the procedure does not converge after MAXIT cycles. */

/*  Internal Parameters */
/*  =================== */

/*  MAXIT   INTEGER */
/*          MAXIT specifies the total loops that the iterative procedure */
/*          may take. If after MAXIT cycles, the routine fails to */
/*          converge, we return INFO = 1. */

/*  Further Details */
/*  =============== */

/*  DTGSJA essentially uses a variant of Kogbetliantz algorithm to reduce */
/*  min(L,M-K)-by-L triangular (or trapezoidal) matrix A23 and L-by-L */
/*  matrix B13 to the form: */

/*           U1'*A13*Q1 = C1*R1; V1'*B13*Q1 = S1*R1, */

/*  where U1, V1 and Q1 are orthogonal matrix, and Z' is the transpose */
/*  of Z.  C1 and S1 are diagonal matrices satisfying */

/*                C1**2 + S1**2 = I, */

/*  and R1 is an L-by-L nonsingular upper triangular matrix. */

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

/*     Decode and test the input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --alpha;
    --beta;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --work;

    /* Function Body */
    initu = lsame_(jobu, "I");
    wantu = initu || lsame_(jobu, "U");

    initv = lsame_(jobv, "I");
    wantv = initv || lsame_(jobv, "V");

    initq = lsame_(jobq, "I");
    wantq = initq || lsame_(jobq, "Q");

    *info = 0;
    if (! (initu || wantu || lsame_(jobu, "N"))) {
	*info = -1;
    } else if (! (initv || wantv || lsame_(jobv, "N"))) 
	    {
	*info = -2;
    } else if (! (initq || wantq || lsame_(jobq, "N"))) 
	    {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*p < 0) {
	*info = -5;
    } else if (*n < 0) {
	*info = -6;
    } else if (*lda < max(1,*m)) {
	*info = -10;
    } else if (*ldb < max(1,*p)) {
	*info = -12;
    } else if (*ldu < 1 || wantu && *ldu < *m) {
	*info = -18;
    } else if (*ldv < 1 || wantv && *ldv < *p) {
	*info = -20;
    } else if (*ldq < 1 || wantq && *ldq < *n) {
	*info = -22;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTGSJA", &i__1);
	return 0;
    }

/*     Initialize U, V and Q, if necessary */

    if (initu) {
	dlaset_("Full", m, m, &c_b13, &c_b14, &u[u_offset], ldu);
    }
    if (initv) {
	dlaset_("Full", p, p, &c_b13, &c_b14, &v[v_offset], ldv);
    }
    if (initq) {
	dlaset_("Full", n, n, &c_b13, &c_b14, &q[q_offset], ldq);
    }

/*     Loop until convergence */

    upper = FALSE_;
    for (kcycle = 1; kcycle <= 40; ++kcycle) {

	upper = ! upper;

	i__1 = *l - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = *l;
	    for (j = i__ + 1; j <= i__2; ++j) {

		a1 = 0.;
		a2 = 0.;
		a3 = 0.;
		if (*k + i__ <= *m) {
		    a1 = a[*k + i__ + (*n - *l + i__) * a_dim1];
		}
		if (*k + j <= *m) {
		    a3 = a[*k + j + (*n - *l + j) * a_dim1];
		}

		b1 = b[i__ + (*n - *l + i__) * b_dim1];
		b3 = b[j + (*n - *l + j) * b_dim1];

		if (upper) {
		    if (*k + i__ <= *m) {
			a2 = a[*k + i__ + (*n - *l + j) * a_dim1];
		    }
		    b2 = b[i__ + (*n - *l + j) * b_dim1];
		} else {
		    if (*k + j <= *m) {
			a2 = a[*k + j + (*n - *l + i__) * a_dim1];
		    }
		    b2 = b[j + (*n - *l + i__) * b_dim1];
		}

		dlags2_(&upper, &a1, &a2, &a3, &b1, &b2, &b3, &csu, &snu, &
			csv, &snv, &csq, &snq);

/*              Update (K+I)-th and (K+J)-th rows of matrix A: U'*A */

		if (*k + j <= *m) {
		    drot_(l, &a[*k + j + (*n - *l + 1) * a_dim1], lda, &a[*k 
			    + i__ + (*n - *l + 1) * a_dim1], lda, &csu, &snu);
		}

/*              Update I-th and J-th rows of matrix B: V'*B */

		drot_(l, &b[j + (*n - *l + 1) * b_dim1], ldb, &b[i__ + (*n - *
			l + 1) * b_dim1], ldb, &csv, &snv);

/*              Update (N-L+I)-th and (N-L+J)-th columns of matrices */
/*              A and B: A*Q and B*Q */

/* Computing MIN */
		i__4 = *k + *l;
		i__3 = min(i__4,*m);
		drot_(&i__3, &a[(*n - *l + j) * a_dim1 + 1], &c__1, &a[(*n - *
			l + i__) * a_dim1 + 1], &c__1, &csq, &snq);

		drot_(l, &b[(*n - *l + j) * b_dim1 + 1], &c__1, &b[(*n - *l + 
			i__) * b_dim1 + 1], &c__1, &csq, &snq);

		if (upper) {
		    if (*k + i__ <= *m) {
			a[*k + i__ + (*n - *l + j) * a_dim1] = 0.;
		    }
		    b[i__ + (*n - *l + j) * b_dim1] = 0.;
		} else {
		    if (*k + j <= *m) {
			a[*k + j + (*n - *l + i__) * a_dim1] = 0.;
		    }
		    b[j + (*n - *l + i__) * b_dim1] = 0.;
		}

/*              Update orthogonal matrices U, V, Q, if desired. */

		if (wantu && *k + j <= *m) {
		    drot_(m, &u[(*k + j) * u_dim1 + 1], &c__1, &u[(*k + i__) *
			     u_dim1 + 1], &c__1, &csu, &snu);
		}

		if (wantv) {
		    drot_(p, &v[j * v_dim1 + 1], &c__1, &v[i__ * v_dim1 + 1], 
			    &c__1, &csv, &snv);
		}

		if (wantq) {
		    drot_(n, &q[(*n - *l + j) * q_dim1 + 1], &c__1, &q[(*n - *
			    l + i__) * q_dim1 + 1], &c__1, &csq, &snq);
		}

/* L10: */
	    }
/* L20: */
	}

	if (! upper) {

/*           The matrices A13 and B13 were lower triangular at the start */
/*           of the cycle, and are now upper triangular. */

/*           Convergence test: test the parallelism of the corresponding */
/*           rows of A and B. */

	    error = 0.;
/* Computing MIN */
	    i__2 = *l, i__3 = *m - *k;
	    i__1 = min(i__2,i__3);
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *l - i__ + 1;
		dcopy_(&i__2, &a[*k + i__ + (*n - *l + i__) * a_dim1], lda, &
			work[1], &c__1);
		i__2 = *l - i__ + 1;
		dcopy_(&i__2, &b[i__ + (*n - *l + i__) * b_dim1], ldb, &work[*
			l + 1], &c__1);
		i__2 = *l - i__ + 1;
		dlapll_(&i__2, &work[1], &c__1, &work[*l + 1], &c__1, &ssmin);
		error = max(error,ssmin);
/* L30: */
	    }

	    if (abs(error) <= min(*tola,*tolb)) {
		goto L50;
	    }
	}

/*        End of cycle loop */

/* L40: */
    }

/*     The algorithm has not converged after MAXIT cycles. */

    *info = 1;
    goto L100;

L50:

/*     If ERROR <= MIN(TOLA,TOLB), then the algorithm has converged. */
/*     Compute the generalized singular value pairs (ALPHA, BETA), and */
/*     set the triangular matrix R to array A. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	alpha[i__] = 1.;
	beta[i__] = 0.;
/* L60: */
    }

/* Computing MIN */
    i__2 = *l, i__3 = *m - *k;
    i__1 = min(i__2,i__3);
    for (i__ = 1; i__ <= i__1; ++i__) {

	a1 = a[*k + i__ + (*n - *l + i__) * a_dim1];
	b1 = b[i__ + (*n - *l + i__) * b_dim1];

	if (a1 != 0.) {
	    gamma = b1 / a1;

/*           change sign if necessary */

	    if (gamma < 0.) {
		i__2 = *l - i__ + 1;
		dscal_(&i__2, &c_b43, &b[i__ + (*n - *l + i__) * b_dim1], ldb)
			;
		if (wantv) {
		    dscal_(p, &c_b43, &v[i__ * v_dim1 + 1], &c__1);
		}
	    }

	    d__1 = abs(gamma);
	    dlartg_(&d__1, &c_b14, &beta[*k + i__], &alpha[*k + i__], &rwk);

	    if (alpha[*k + i__] >= beta[*k + i__]) {
		i__2 = *l - i__ + 1;
		d__1 = 1. / alpha[*k + i__];
		dscal_(&i__2, &d__1, &a[*k + i__ + (*n - *l + i__) * a_dim1], 
			lda);
	    } else {
		i__2 = *l - i__ + 1;
		d__1 = 1. / beta[*k + i__];
		dscal_(&i__2, &d__1, &b[i__ + (*n - *l + i__) * b_dim1], ldb);
		i__2 = *l - i__ + 1;
		dcopy_(&i__2, &b[i__ + (*n - *l + i__) * b_dim1], ldb, &a[*k 
			+ i__ + (*n - *l + i__) * a_dim1], lda);
	    }

	} else {

	    alpha[*k + i__] = 0.;
	    beta[*k + i__] = 1.;
	    i__2 = *l - i__ + 1;
	    dcopy_(&i__2, &b[i__ + (*n - *l + i__) * b_dim1], ldb, &a[*k + 
		    i__ + (*n - *l + i__) * a_dim1], lda);

	}

/* L70: */
    }

/*     Post-assignment */

    i__1 = *k + *l;
    for (i__ = *m + 1; i__ <= i__1; ++i__) {
	alpha[i__] = 0.;
	beta[i__] = 1.;
/* L80: */
    }

    if (*k + *l < *n) {
	i__1 = *n;
	for (i__ = *k + *l + 1; i__ <= i__1; ++i__) {
	    alpha[i__] = 0.;
	    beta[i__] = 0.;
/* L90: */
	}
    }

L100:
    *ncycle = kcycle;
    return 0;

/*     End of DTGSJA */

} /* dtgsja_ */
