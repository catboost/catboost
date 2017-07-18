/* cunmhr.f -- translated by f2c (version 20061008).
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
static integer c_n1 = -1;
static integer c__2 = 2;

/* Subroutine */ int cunmhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, complex *a, integer *lda, complex *tau, 
	complex *c__, integer *ldc, complex *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    address a__1[2];
    integer a_dim1, a_offset, c_dim1, c_offset, i__1[2], i__2;
    char ch__1[2];

    /* Builtin functions */
    /* Subroutine */ int s_cat(char *, char **, integer *, integer *, ftnlen);

    /* Local variables */
    integer i1, i2, nb, mi, nh, ni, nq, nw;
    logical left;
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int cunmqr_(char *, char *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, complex *, integer *, 
	    complex *, integer *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CUNMHR overwrites the general complex M-by-N matrix C with */

/*                  SIDE = 'L'     SIDE = 'R' */
/*  TRANS = 'N':      Q * C          C * Q */
/*  TRANS = 'C':      Q**H * C       C * Q**H */

/*  where Q is a complex unitary matrix of order nq, with nq = m if */
/*  SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of */
/*  IHI-ILO elementary reflectors, as returned by CGEHRD: */

/*  Q = H(ilo) H(ilo+1) . . . H(ihi-1). */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'L': apply Q or Q**H from the Left; */
/*          = 'R': apply Q or Q**H from the Right. */

/*  TRANS   (input) CHARACTER*1 */
/*          = 'N': apply Q  (No transpose) */
/*          = 'C': apply Q**H (Conjugate transpose) */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix C. M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix C. N >= 0. */

/*  ILO     (input) INTEGER */
/*  IHI     (input) INTEGER */
/*          ILO and IHI must have the same values as in the previous call */
/*          of CGEHRD. Q is equal to the unit matrix except in the */
/*          submatrix Q(ilo+1:ihi,ilo+1:ihi). */
/*          If SIDE = 'L', then 1 <= ILO <= IHI <= M, if M > 0, and */
/*          ILO = 1 and IHI = 0, if M = 0; */
/*          if SIDE = 'R', then 1 <= ILO <= IHI <= N, if N > 0, and */
/*          ILO = 1 and IHI = 0, if N = 0. */

/*  A       (input) COMPLEX array, dimension */
/*                               (LDA,M) if SIDE = 'L' */
/*                               (LDA,N) if SIDE = 'R' */
/*          The vectors which define the elementary reflectors, as */
/*          returned by CGEHRD. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. */
/*          LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'. */

/*  TAU     (input) COMPLEX array, dimension */
/*                               (M-1) if SIDE = 'L' */
/*                               (N-1) if SIDE = 'R' */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i), as returned by CGEHRD. */

/*  C       (input/output) COMPLEX array, dimension (LDC,N) */
/*          On entry, the M-by-N matrix C. */
/*          On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the array C. LDC >= max(1,M). */

/*  WORK    (workspace/output) COMPLEX array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. */
/*          If SIDE = 'L', LWORK >= max(1,N); */
/*          if SIDE = 'R', LWORK >= max(1,M). */
/*          For optimum performance LWORK >= N*NB if SIDE = 'L', and */
/*          LWORK >= M*NB if SIDE = 'R', where NB is the optimal */
/*          blocksize. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    *info = 0;
    nh = *ihi - *ilo;
    left = lsame_(side, "L");
    lquery = *lwork == -1;

/*     NQ is the order of Q and NW is the minimum dimension of WORK */

    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, 
	    "C")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ilo < 1 || *ilo > max(1,nq)) {
	*info = -5;
    } else if (*ihi < min(*ilo,nq) || *ihi > nq) {
	*info = -6;
    } else if (*lda < max(1,nq)) {
	*info = -8;
    } else if (*ldc < max(1,*m)) {
	*info = -11;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -13;
    }

    if (*info == 0) {
	if (left) {
/* Writing concatenation */
	    i__1[0] = 1, a__1[0] = side;
	    i__1[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
	    nb = ilaenv_(&c__1, "CUNMQR", ch__1, &nh, n, &nh, &c_n1);
	} else {
/* Writing concatenation */
	    i__1[0] = 1, a__1[0] = side;
	    i__1[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__1, &c__2, (ftnlen)2);
	    nb = ilaenv_(&c__1, "CUNMQR", ch__1, m, &nh, &nh, &c_n1);
	}
	lwkopt = max(1,nw) * nb;
	work[1].r = (real) lwkopt, work[1].i = 0.f;
    }

    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("CUNMHR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0 || nh == 0) {
	work[1].r = 1.f, work[1].i = 0.f;
	return 0;
    }

    if (left) {
	mi = nh;
	ni = *n;
	i1 = *ilo + 1;
	i2 = 1;
    } else {
	mi = *m;
	ni = nh;
	i1 = 1;
	i2 = *ilo + 1;
    }

    cunmqr_(side, trans, &mi, &ni, &nh, &a[*ilo + 1 + *ilo * a_dim1], lda, &
	    tau[*ilo], &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);

    work[1].r = (real) lwkopt, work[1].i = 0.f;
    return 0;

/*     End of CUNMHR */

} /* cunmhr_ */
