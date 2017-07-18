/* chbgst.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int chbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, complex *ab, integer *ldab, complex *bb, integer *ldbb, 
	complex *x, integer *ldx, complex *work, real *rwork, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, bb_dim1, bb_offset, x_dim1, x_offset, i__1, 
	    i__2, i__3, i__4, i__5, i__6, i__7, i__8;
    real r__1;
    complex q__1, q__2, q__3, q__4, q__5, q__6, q__7, q__8, q__9, q__10;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j, k, l, m;
    complex t;
    integer i0, i1, i2, j1, j2;
    complex ra;
    integer nr, nx, ka1, kb1;
    complex ra1;
    integer j1t, j2t;
    real bii;
    integer kbt, nrt, inca;
    extern /* Subroutine */ int crot_(integer *, complex *, integer *, 
	    complex *, integer *, real *, complex *), cgerc_(integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgeru_(integer *, integer *, complex *, 
	    complex *, integer *, complex *, integer *, complex *, integer *);
    logical upper, wantx;
    extern /* Subroutine */ int clar2v_(integer *, complex *, complex *, 
	    complex *, integer *, real *, complex *, integer *), clacgv_(
	    integer *, complex *, integer *), csscal_(integer *, real *, 
	    complex *, integer *), claset_(char *, integer *, integer *, 
	    complex *, complex *, complex *, integer *), clartg_(
	    complex *, complex *, real *, complex *, complex *), xerbla_(char 
	    *, integer *), clargv_(integer *, complex *, integer *, 
	    complex *, integer *, real *, integer *);
    logical update;
    extern /* Subroutine */ int clartv_(integer *, complex *, integer *, 
	    complex *, integer *, real *, complex *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CHBGST reduces a complex Hermitian-definite banded generalized */
/*  eigenproblem  A*x = lambda*B*x  to standard form  C*y = lambda*y, */
/*  such that C has the same bandwidth as A. */

/*  B must have been previously factorized as S**H*S by CPBSTF, using a */
/*  split Cholesky factorization. A is overwritten by C = X**H*A*X, where */
/*  X = S**(-1)*Q and Q is a unitary matrix chosen to preserve the */
/*  bandwidth of A. */

/*  Arguments */
/*  ========= */

/*  VECT    (input) CHARACTER*1 */
/*          = 'N':  do not form the transformation matrix X; */
/*          = 'V':  form X. */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B.  N >= 0. */

/*  KA      (input) INTEGER */
/*          The number of superdiagonals of the matrix A if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'.  KA >= 0. */

/*  KB      (input) INTEGER */
/*          The number of superdiagonals of the matrix B if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'.  KA >= KB >= 0. */

/*  AB      (input/output) COMPLEX array, dimension (LDAB,N) */
/*          On entry, the upper or lower triangle of the Hermitian band */
/*          matrix A, stored in the first ka+1 rows of the array.  The */
/*          j-th column of A is stored in the j-th column of the array AB */
/*          as follows: */
/*          if UPLO = 'U', AB(ka+1+i-j,j) = A(i,j) for max(1,j-ka)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+ka). */

/*          On exit, the transformed matrix X**H*A*X, stored in the same */
/*          format as A. */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KA+1. */

/*  BB      (input) COMPLEX array, dimension (LDBB,N) */
/*          The banded factor S from the split Cholesky factorization of */
/*          B, as returned by CPBSTF, stored in the first kb+1 rows of */
/*          the array. */

/*  LDBB    (input) INTEGER */
/*          The leading dimension of the array BB.  LDBB >= KB+1. */

/*  X       (output) COMPLEX array, dimension (LDX,N) */
/*          If VECT = 'V', the n-by-n matrix X. */
/*          If VECT = 'N', the array X is not referenced. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X. */
/*          LDX >= max(1,N) if VECT = 'V'; LDX >= 1 otherwise. */

/*  WORK    (workspace) COMPLEX array, dimension (N) */

/*  RWORK   (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

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

/*     Test the input parameters */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    bb_dim1 = *ldbb;
    bb_offset = 1 + bb_dim1;
    bb -= bb_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --work;
    --rwork;

    /* Function Body */
    wantx = lsame_(vect, "V");
    upper = lsame_(uplo, "U");
    ka1 = *ka + 1;
    kb1 = *kb + 1;
    *info = 0;
    if (! wantx && ! lsame_(vect, "N")) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ka < 0) {
	*info = -4;
    } else if (*kb < 0 || *kb > *ka) {
	*info = -5;
    } else if (*ldab < *ka + 1) {
	*info = -7;
    } else if (*ldbb < *kb + 1) {
	*info = -9;
    } else if (*ldx < 1 || wantx && *ldx < max(1,*n)) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CHBGST", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    inca = *ldab * ka1;

/*     Initialize X to the unit matrix, if needed */

    if (wantx) {
	claset_("Full", n, n, &c_b1, &c_b2, &x[x_offset], ldx);
    }

/*     Set M to the splitting point m. It must be the same value as is */
/*     used in CPBSTF. The chosen value allows the arrays WORK and RWORK */
/*     to be of dimension (N). */

    m = (*n + *kb) / 2;

/*     The routine works in two phases, corresponding to the two halves */
/*     of the split Cholesky factorization of B as S**H*S where */

/*     S = ( U    ) */
/*         ( M  L ) */

/*     with U upper triangular of order m, and L lower triangular of */
/*     order n-m. S has the same bandwidth as B. */

/*     S is treated as a product of elementary matrices: */

/*     S = S(m)*S(m-1)*...*S(2)*S(1)*S(m+1)*S(m+2)*...*S(n-1)*S(n) */

/*     where S(i) is determined by the i-th row of S. */

/*     In phase 1, the index i takes the values n, n-1, ... , m+1; */
/*     in phase 2, it takes the values 1, 2, ... , m. */

/*     For each value of i, the current matrix A is updated by forming */
/*     inv(S(i))**H*A*inv(S(i)). This creates a triangular bulge outside */
/*     the band of A. The bulge is then pushed down toward the bottom of */
/*     A in phase 1, and up toward the top of A in phase 2, by applying */
/*     plane rotations. */

/*     There are kb*(kb+1)/2 elements in the bulge, but at most 2*kb-1 */
/*     of them are linearly independent, so annihilating a bulge requires */
/*     only 2*kb-1 plane rotations. The rotations are divided into a 1st */
/*     set of kb-1 rotations, and a 2nd set of kb rotations. */

/*     Wherever possible, rotations are generated and applied in vector */
/*     operations of length NR between the indices J1 and J2 (sometimes */
/*     replaced by modified values NRT, J1T or J2T). */

/*     The real cosines and complex sines of the rotations are stored in */
/*     the arrays RWORK and WORK, those of the 1st set in elements */
/*     2:m-kb-1, and those of the 2nd set in elements m-kb+1:n. */

/*     The bulges are not formed explicitly; nonzero elements outside the */
/*     band are created only when they are required for generating new */
/*     rotations; they are stored in the array WORK, in positions where */
/*     they are later overwritten by the sines of the rotations which */
/*     annihilate them. */

/*     **************************** Phase 1 ***************************** */

/*     The logical structure of this phase is: */

/*     UPDATE = .TRUE. */
/*     DO I = N, M + 1, -1 */
/*        use S(i) to update A and create a new bulge */
/*        apply rotations to push all bulges KA positions downward */
/*     END DO */
/*     UPDATE = .FALSE. */
/*     DO I = M + KA + 1, N - 1 */
/*        apply rotations to push all bulges KA positions downward */
/*     END DO */

/*     To avoid duplicating code, the two loops are merged. */

    update = TRUE_;
    i__ = *n + 1;
L10:
    if (update) {
	--i__;
/* Computing MIN */
	i__1 = *kb, i__2 = i__ - 1;
	kbt = min(i__1,i__2);
	i0 = i__ - 1;
/* Computing MIN */
	i__1 = *n, i__2 = i__ + *ka;
	i1 = min(i__1,i__2);
	i2 = i__ - kbt + ka1;
	if (i__ < m + 1) {
	    update = FALSE_;
	    ++i__;
	    i0 = m;
	    if (*ka == 0) {
		goto L480;
	    }
	    goto L10;
	}
    } else {
	i__ += *ka;
	if (i__ > *n - 1) {
	    goto L480;
	}
    }

    if (upper) {

/*        Transform A, working with the upper triangle */

	if (update) {

/*           Form  inv(S(i))**H * A * inv(S(i)) */

	    i__1 = kb1 + i__ * bb_dim1;
	    bii = bb[i__1].r;
	    i__1 = ka1 + i__ * ab_dim1;
	    i__2 = ka1 + i__ * ab_dim1;
	    r__1 = ab[i__2].r / bii / bii;
	    ab[i__1].r = r__1, ab[i__1].i = 0.f;
	    i__1 = i1;
	    for (j = i__ + 1; j <= i__1; ++j) {
		i__2 = i__ - j + ka1 + j * ab_dim1;
		i__3 = i__ - j + ka1 + j * ab_dim1;
		q__1.r = ab[i__3].r / bii, q__1.i = ab[i__3].i / bii;
		ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L20: */
	    }
/* Computing MAX */
	    i__1 = 1, i__2 = i__ - *ka;
	    i__3 = i__ - 1;
	    for (j = max(i__1,i__2); j <= i__3; ++j) {
		i__1 = j - i__ + ka1 + i__ * ab_dim1;
		i__2 = j - i__ + ka1 + i__ * ab_dim1;
		q__1.r = ab[i__2].r / bii, q__1.i = ab[i__2].i / bii;
		ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L30: */
	    }
	    i__3 = i__ - 1;
	    for (k = i__ - kbt; k <= i__3; ++k) {
		i__1 = k;
		for (j = i__ - kbt; j <= i__1; ++j) {
		    i__2 = j - k + ka1 + k * ab_dim1;
		    i__4 = j - k + ka1 + k * ab_dim1;
		    i__5 = j - i__ + kb1 + i__ * bb_dim1;
		    r_cnjg(&q__5, &ab[k - i__ + ka1 + i__ * ab_dim1]);
		    q__4.r = bb[i__5].r * q__5.r - bb[i__5].i * q__5.i, 
			    q__4.i = bb[i__5].r * q__5.i + bb[i__5].i * 
			    q__5.r;
		    q__3.r = ab[i__4].r - q__4.r, q__3.i = ab[i__4].i - 
			    q__4.i;
		    r_cnjg(&q__7, &bb[k - i__ + kb1 + i__ * bb_dim1]);
		    i__6 = j - i__ + ka1 + i__ * ab_dim1;
		    q__6.r = q__7.r * ab[i__6].r - q__7.i * ab[i__6].i, 
			    q__6.i = q__7.r * ab[i__6].i + q__7.i * ab[i__6]
			    .r;
		    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		    i__7 = ka1 + i__ * ab_dim1;
		    r__1 = ab[i__7].r;
		    i__8 = j - i__ + kb1 + i__ * bb_dim1;
		    q__9.r = r__1 * bb[i__8].r, q__9.i = r__1 * bb[i__8].i;
		    r_cnjg(&q__10, &bb[k - i__ + kb1 + i__ * bb_dim1]);
		    q__8.r = q__9.r * q__10.r - q__9.i * q__10.i, q__8.i = 
			    q__9.r * q__10.i + q__9.i * q__10.r;
		    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
		    ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L40: */
		}
/* Computing MAX */
		i__1 = 1, i__2 = i__ - *ka;
		i__4 = i__ - kbt - 1;
		for (j = max(i__1,i__2); j <= i__4; ++j) {
		    i__1 = j - k + ka1 + k * ab_dim1;
		    i__2 = j - k + ka1 + k * ab_dim1;
		    r_cnjg(&q__3, &bb[k - i__ + kb1 + i__ * bb_dim1]);
		    i__5 = j - i__ + ka1 + i__ * ab_dim1;
		    q__2.r = q__3.r * ab[i__5].r - q__3.i * ab[i__5].i, 
			    q__2.i = q__3.r * ab[i__5].i + q__3.i * ab[i__5]
			    .r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L50: */
		}
/* L60: */
	    }
	    i__3 = i1;
	    for (j = i__; j <= i__3; ++j) {
/* Computing MAX */
		i__4 = j - *ka, i__1 = i__ - kbt;
		i__2 = i__ - 1;
		for (k = max(i__4,i__1); k <= i__2; ++k) {
		    i__4 = k - j + ka1 + j * ab_dim1;
		    i__1 = k - j + ka1 + j * ab_dim1;
		    i__5 = k - i__ + kb1 + i__ * bb_dim1;
		    i__6 = i__ - j + ka1 + j * ab_dim1;
		    q__2.r = bb[i__5].r * ab[i__6].r - bb[i__5].i * ab[i__6]
			    .i, q__2.i = bb[i__5].r * ab[i__6].i + bb[i__5].i 
			    * ab[i__6].r;
		    q__1.r = ab[i__1].r - q__2.r, q__1.i = ab[i__1].i - 
			    q__2.i;
		    ab[i__4].r = q__1.r, ab[i__4].i = q__1.i;
/* L70: */
		}
/* L80: */
	    }

	    if (wantx) {

/*              post-multiply X by inv(S(i)) */

		i__3 = *n - m;
		r__1 = 1.f / bii;
		csscal_(&i__3, &r__1, &x[m + 1 + i__ * x_dim1], &c__1);
		if (kbt > 0) {
		    i__3 = *n - m;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgerc_(&i__3, &kbt, &q__1, &x[m + 1 + i__ * x_dim1], &
			    c__1, &bb[kb1 - kbt + i__ * bb_dim1], &c__1, &x[m 
			    + 1 + (i__ - kbt) * x_dim1], ldx);
		}
	    }

/*           store a(i,i1) in RA1 for use in next loop over K */

	    i__3 = i__ - i1 + ka1 + i1 * ab_dim1;
	    ra1.r = ab[i__3].r, ra1.i = ab[i__3].i;
	}

/*        Generate and apply vectors of rotations to chase all the */
/*        existing bulges KA positions down toward the bottom of the */
/*        band */

	i__3 = *kb - 1;
	for (k = 1; k <= i__3; ++k) {
	    if (update) {

/*              Determine the rotations which would annihilate the bulge */
/*              which has in theory just been created */

		if (i__ - k + *ka < *n && i__ - k > 1) {

/*                 generate rotation to annihilate a(i,i-k+ka+1) */

		    clartg_(&ab[k + 1 + (i__ - k + *ka) * ab_dim1], &ra1, &
			    rwork[i__ - k + *ka - m], &work[i__ - k + *ka - m]
, &ra);

/*                 create nonzero element a(i-k,i-k+ka+1) outside the */
/*                 band and store it in WORK(i-k) */

		    i__2 = kb1 - k + i__ * bb_dim1;
		    q__2.r = -bb[i__2].r, q__2.i = -bb[i__2].i;
		    q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r 
			    * ra1.i + q__2.i * ra1.r;
		    t.r = q__1.r, t.i = q__1.i;
		    i__2 = i__ - k;
		    i__4 = i__ - k + *ka - m;
		    q__2.r = rwork[i__4] * t.r, q__2.i = rwork[i__4] * t.i;
		    r_cnjg(&q__4, &work[i__ - k + *ka - m]);
		    i__1 = (i__ - k + *ka) * ab_dim1 + 1;
		    q__3.r = q__4.r * ab[i__1].r - q__4.i * ab[i__1].i, 
			    q__3.i = q__4.r * ab[i__1].i + q__4.i * ab[i__1]
			    .r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		    i__2 = (i__ - k + *ka) * ab_dim1 + 1;
		    i__4 = i__ - k + *ka - m;
		    q__2.r = work[i__4].r * t.r - work[i__4].i * t.i, q__2.i =
			     work[i__4].r * t.i + work[i__4].i * t.r;
		    i__1 = i__ - k + *ka - m;
		    i__5 = (i__ - k + *ka) * ab_dim1 + 1;
		    q__3.r = rwork[i__1] * ab[i__5].r, q__3.i = rwork[i__1] * 
			    ab[i__5].i;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
		    ra1.r = ra.r, ra1.i = ra.i;
		}
	    }
/* Computing MAX */
	    i__2 = 1, i__4 = k - i0 + 2;
	    j2 = i__ - k - 1 + max(i__2,i__4) * ka1;
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    if (update) {
/* Computing MAX */
		i__2 = j2, i__4 = i__ + (*ka << 1) - k + 1;
		j2t = max(i__2,i__4);
	    } else {
		j2t = j2;
	    }
	    nrt = (*n - j2t + *ka) / ka1;
	    i__2 = j1;
	    i__4 = ka1;
	    for (j = j2t; i__4 < 0 ? j >= i__2 : j <= i__2; j += i__4) {

/*              create nonzero element a(j-ka,j+1) outside the band */
/*              and store it in WORK(j-m) */

		i__1 = j - m;
		i__5 = j - m;
		i__6 = (j + 1) * ab_dim1 + 1;
		q__1.r = work[i__5].r * ab[i__6].r - work[i__5].i * ab[i__6]
			.i, q__1.i = work[i__5].r * ab[i__6].i + work[i__5].i 
			* ab[i__6].r;
		work[i__1].r = q__1.r, work[i__1].i = q__1.i;
		i__1 = (j + 1) * ab_dim1 + 1;
		i__5 = j - m;
		i__6 = (j + 1) * ab_dim1 + 1;
		q__1.r = rwork[i__5] * ab[i__6].r, q__1.i = rwork[i__5] * ab[
			i__6].i;
		ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L90: */
	    }

/*           generate rotations in 1st set to annihilate elements which */
/*           have been created outside the band */

	    if (nrt > 0) {
		clargv_(&nrt, &ab[j2t * ab_dim1 + 1], &inca, &work[j2t - m], &
			ka1, &rwork[j2t - m], &ka1);
	    }
	    if (nr > 0) {

/*              apply rotations in 1st set from the right */

		i__4 = *ka - 1;
		for (l = 1; l <= i__4; ++l) {
		    clartv_(&nr, &ab[ka1 - l + j2 * ab_dim1], &inca, &ab[*ka 
			    - l + (j2 + 1) * ab_dim1], &inca, &rwork[j2 - m], 
			    &work[j2 - m], &ka1);
/* L100: */
		}

/*              apply rotations in 1st set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[ka1 + j2 * ab_dim1], &ab[ka1 + (j2 + 1) * 
			ab_dim1], &ab[*ka + (j2 + 1) * ab_dim1], &inca, &
			rwork[j2 - m], &work[j2 - m], &ka1);

		clacgv_(&nr, &work[j2 - m], &ka1);
	    }

/*           start applying rotations in 1st set from the left */

	    i__4 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__4; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + (j2 + ka1 - l) * ab_dim1], &inca, &
			    ab[l + 1 + (j2 + ka1 - l) * ab_dim1], &inca, &
			    rwork[j2 - m], &work[j2 - m], &ka1);
		}
/* L110: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 1st set */

		i__4 = j1;
		i__2 = ka1;
		for (j = j2; i__2 < 0 ? j >= i__4 : j <= i__4; j += i__2) {
		    i__1 = *n - m;
		    r_cnjg(&q__1, &work[j - m]);
		    crot_(&i__1, &x[m + 1 + j * x_dim1], &c__1, &x[m + 1 + (j 
			    + 1) * x_dim1], &c__1, &rwork[j - m], &q__1);
/* L120: */
		}
	    }
/* L130: */
	}

	if (update) {
	    if (i2 <= *n && kbt > 0) {

/*              create nonzero element a(i-kbt,i-kbt+ka+1) outside the */
/*              band and store it in WORK(i-kbt) */

		i__3 = i__ - kbt;
		i__2 = kb1 - kbt + i__ * bb_dim1;
		q__2.r = -bb[i__2].r, q__2.i = -bb[i__2].i;
		q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r * 
			ra1.i + q__2.i * ra1.r;
		work[i__3].r = q__1.r, work[i__3].i = q__1.i;
	    }
	}

	for (k = *kb; k >= 1; --k) {
	    if (update) {
/* Computing MAX */
		i__3 = 2, i__2 = k - i0 + 1;
		j2 = i__ - k - 1 + max(i__3,i__2) * ka1;
	    } else {
/* Computing MAX */
		i__3 = 1, i__2 = k - i0 + 1;
		j2 = i__ - k - 1 + max(i__3,i__2) * ka1;
	    }

/*           finish applying rotations in 2nd set from the left */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (*n - j2 + *ka + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + (j2 - l + 1) * ab_dim1], &inca, &ab[
			    l + 1 + (j2 - l + 1) * ab_dim1], &inca, &rwork[j2 
			    - *ka], &work[j2 - *ka], &ka1);
		}
/* L140: */
	    }
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    i__3 = j2;
	    i__2 = -ka1;
	    for (j = j1; i__2 < 0 ? j >= i__3 : j <= i__3; j += i__2) {
		i__4 = j;
		i__1 = j - *ka;
		work[i__4].r = work[i__1].r, work[i__4].i = work[i__1].i;
		rwork[j] = rwork[j - *ka];
/* L150: */
	    }
	    i__2 = j1;
	    i__3 = ka1;
	    for (j = j2; i__3 < 0 ? j >= i__2 : j <= i__2; j += i__3) {

/*              create nonzero element a(j-ka,j+1) outside the band */
/*              and store it in WORK(j) */

		i__4 = j;
		i__1 = j;
		i__5 = (j + 1) * ab_dim1 + 1;
		q__1.r = work[i__1].r * ab[i__5].r - work[i__1].i * ab[i__5]
			.i, q__1.i = work[i__1].r * ab[i__5].i + work[i__1].i 
			* ab[i__5].r;
		work[i__4].r = q__1.r, work[i__4].i = q__1.i;
		i__4 = (j + 1) * ab_dim1 + 1;
		i__1 = j;
		i__5 = (j + 1) * ab_dim1 + 1;
		q__1.r = rwork[i__1] * ab[i__5].r, q__1.i = rwork[i__1] * ab[
			i__5].i;
		ab[i__4].r = q__1.r, ab[i__4].i = q__1.i;
/* L160: */
	    }
	    if (update) {
		if (i__ - k < *n - *ka && k <= kbt) {
		    i__3 = i__ - k + *ka;
		    i__2 = i__ - k;
		    work[i__3].r = work[i__2].r, work[i__3].i = work[i__2].i;
		}
	    }
/* L170: */
	}

	for (k = *kb; k >= 1; --k) {
/* Computing MAX */
	    i__3 = 1, i__2 = k - i0 + 1;
	    j2 = i__ - k - 1 + max(i__3,i__2) * ka1;
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    if (nr > 0) {

/*              generate rotations in 2nd set to annihilate elements */
/*              which have been created outside the band */

		clargv_(&nr, &ab[j2 * ab_dim1 + 1], &inca, &work[j2], &ka1, &
			rwork[j2], &ka1);

/*              apply rotations in 2nd set from the right */

		i__3 = *ka - 1;
		for (l = 1; l <= i__3; ++l) {
		    clartv_(&nr, &ab[ka1 - l + j2 * ab_dim1], &inca, &ab[*ka 
			    - l + (j2 + 1) * ab_dim1], &inca, &rwork[j2], &
			    work[j2], &ka1);
/* L180: */
		}

/*              apply rotations in 2nd set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[ka1 + j2 * ab_dim1], &ab[ka1 + (j2 + 1) * 
			ab_dim1], &ab[*ka + (j2 + 1) * ab_dim1], &inca, &
			rwork[j2], &work[j2], &ka1);

		clacgv_(&nr, &work[j2], &ka1);
	    }

/*           start applying rotations in 2nd set from the left */

	    i__3 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__3; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + (j2 + ka1 - l) * ab_dim1], &inca, &
			    ab[l + 1 + (j2 + ka1 - l) * ab_dim1], &inca, &
			    rwork[j2], &work[j2], &ka1);
		}
/* L190: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 2nd set */

		i__3 = j1;
		i__2 = ka1;
		for (j = j2; i__2 < 0 ? j >= i__3 : j <= i__3; j += i__2) {
		    i__4 = *n - m;
		    r_cnjg(&q__1, &work[j]);
		    crot_(&i__4, &x[m + 1 + j * x_dim1], &c__1, &x[m + 1 + (j 
			    + 1) * x_dim1], &c__1, &rwork[j], &q__1);
/* L200: */
		}
	    }
/* L210: */
	}

	i__2 = *kb - 1;
	for (k = 1; k <= i__2; ++k) {
/* Computing MAX */
	    i__3 = 1, i__4 = k - i0 + 2;
	    j2 = i__ - k - 1 + max(i__3,i__4) * ka1;

/*           finish applying rotations in 1st set from the left */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + (j2 + ka1 - l) * ab_dim1], &inca, &
			    ab[l + 1 + (j2 + ka1 - l) * ab_dim1], &inca, &
			    rwork[j2 - m], &work[j2 - m], &ka1);
		}
/* L220: */
	    }
/* L230: */
	}

	if (*kb > 1) {
	    i__2 = i2 + *ka;
	    for (j = *n - 1; j >= i__2; --j) {
		rwork[j - m] = rwork[j - *ka - m];
		i__3 = j - m;
		i__4 = j - *ka - m;
		work[i__3].r = work[i__4].r, work[i__3].i = work[i__4].i;
/* L240: */
	    }
	}

    } else {

/*        Transform A, working with the lower triangle */

	if (update) {

/*           Form  inv(S(i))**H * A * inv(S(i)) */

	    i__2 = i__ * bb_dim1 + 1;
	    bii = bb[i__2].r;
	    i__2 = i__ * ab_dim1 + 1;
	    i__3 = i__ * ab_dim1 + 1;
	    r__1 = ab[i__3].r / bii / bii;
	    ab[i__2].r = r__1, ab[i__2].i = 0.f;
	    i__2 = i1;
	    for (j = i__ + 1; j <= i__2; ++j) {
		i__3 = j - i__ + 1 + i__ * ab_dim1;
		i__4 = j - i__ + 1 + i__ * ab_dim1;
		q__1.r = ab[i__4].r / bii, q__1.i = ab[i__4].i / bii;
		ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
/* L250: */
	    }
/* Computing MAX */
	    i__2 = 1, i__3 = i__ - *ka;
	    i__4 = i__ - 1;
	    for (j = max(i__2,i__3); j <= i__4; ++j) {
		i__2 = i__ - j + 1 + j * ab_dim1;
		i__3 = i__ - j + 1 + j * ab_dim1;
		q__1.r = ab[i__3].r / bii, q__1.i = ab[i__3].i / bii;
		ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L260: */
	    }
	    i__4 = i__ - 1;
	    for (k = i__ - kbt; k <= i__4; ++k) {
		i__2 = k;
		for (j = i__ - kbt; j <= i__2; ++j) {
		    i__3 = k - j + 1 + j * ab_dim1;
		    i__1 = k - j + 1 + j * ab_dim1;
		    i__5 = i__ - j + 1 + j * bb_dim1;
		    r_cnjg(&q__5, &ab[i__ - k + 1 + k * ab_dim1]);
		    q__4.r = bb[i__5].r * q__5.r - bb[i__5].i * q__5.i, 
			    q__4.i = bb[i__5].r * q__5.i + bb[i__5].i * 
			    q__5.r;
		    q__3.r = ab[i__1].r - q__4.r, q__3.i = ab[i__1].i - 
			    q__4.i;
		    r_cnjg(&q__7, &bb[i__ - k + 1 + k * bb_dim1]);
		    i__6 = i__ - j + 1 + j * ab_dim1;
		    q__6.r = q__7.r * ab[i__6].r - q__7.i * ab[i__6].i, 
			    q__6.i = q__7.r * ab[i__6].i + q__7.i * ab[i__6]
			    .r;
		    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		    i__7 = i__ * ab_dim1 + 1;
		    r__1 = ab[i__7].r;
		    i__8 = i__ - j + 1 + j * bb_dim1;
		    q__9.r = r__1 * bb[i__8].r, q__9.i = r__1 * bb[i__8].i;
		    r_cnjg(&q__10, &bb[i__ - k + 1 + k * bb_dim1]);
		    q__8.r = q__9.r * q__10.r - q__9.i * q__10.i, q__8.i = 
			    q__9.r * q__10.i + q__9.i * q__10.r;
		    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
		    ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
/* L270: */
		}
/* Computing MAX */
		i__2 = 1, i__3 = i__ - *ka;
		i__1 = i__ - kbt - 1;
		for (j = max(i__2,i__3); j <= i__1; ++j) {
		    i__2 = k - j + 1 + j * ab_dim1;
		    i__3 = k - j + 1 + j * ab_dim1;
		    r_cnjg(&q__3, &bb[i__ - k + 1 + k * bb_dim1]);
		    i__5 = i__ - j + 1 + j * ab_dim1;
		    q__2.r = q__3.r * ab[i__5].r - q__3.i * ab[i__5].i, 
			    q__2.i = q__3.r * ab[i__5].i + q__3.i * ab[i__5]
			    .r;
		    q__1.r = ab[i__3].r - q__2.r, q__1.i = ab[i__3].i - 
			    q__2.i;
		    ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L280: */
		}
/* L290: */
	    }
	    i__4 = i1;
	    for (j = i__; j <= i__4; ++j) {
/* Computing MAX */
		i__1 = j - *ka, i__2 = i__ - kbt;
		i__3 = i__ - 1;
		for (k = max(i__1,i__2); k <= i__3; ++k) {
		    i__1 = j - k + 1 + k * ab_dim1;
		    i__2 = j - k + 1 + k * ab_dim1;
		    i__5 = i__ - k + 1 + k * bb_dim1;
		    i__6 = j - i__ + 1 + i__ * ab_dim1;
		    q__2.r = bb[i__5].r * ab[i__6].r - bb[i__5].i * ab[i__6]
			    .i, q__2.i = bb[i__5].r * ab[i__6].i + bb[i__5].i 
			    * ab[i__6].r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L300: */
		}
/* L310: */
	    }

	    if (wantx) {

/*              post-multiply X by inv(S(i)) */

		i__4 = *n - m;
		r__1 = 1.f / bii;
		csscal_(&i__4, &r__1, &x[m + 1 + i__ * x_dim1], &c__1);
		if (kbt > 0) {
		    i__4 = *n - m;
		    q__1.r = -1.f, q__1.i = -0.f;
		    i__3 = *ldbb - 1;
		    cgeru_(&i__4, &kbt, &q__1, &x[m + 1 + i__ * x_dim1], &
			    c__1, &bb[kbt + 1 + (i__ - kbt) * bb_dim1], &i__3, 
			     &x[m + 1 + (i__ - kbt) * x_dim1], ldx);
		}
	    }

/*           store a(i1,i) in RA1 for use in next loop over K */

	    i__4 = i1 - i__ + 1 + i__ * ab_dim1;
	    ra1.r = ab[i__4].r, ra1.i = ab[i__4].i;
	}

/*        Generate and apply vectors of rotations to chase all the */
/*        existing bulges KA positions down toward the bottom of the */
/*        band */

	i__4 = *kb - 1;
	for (k = 1; k <= i__4; ++k) {
	    if (update) {

/*              Determine the rotations which would annihilate the bulge */
/*              which has in theory just been created */

		if (i__ - k + *ka < *n && i__ - k > 1) {

/*                 generate rotation to annihilate a(i-k+ka+1,i) */

		    clartg_(&ab[ka1 - k + i__ * ab_dim1], &ra1, &rwork[i__ - 
			    k + *ka - m], &work[i__ - k + *ka - m], &ra);

/*                 create nonzero element a(i-k+ka+1,i-k) outside the */
/*                 band and store it in WORK(i-k) */

		    i__3 = k + 1 + (i__ - k) * bb_dim1;
		    q__2.r = -bb[i__3].r, q__2.i = -bb[i__3].i;
		    q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r 
			    * ra1.i + q__2.i * ra1.r;
		    t.r = q__1.r, t.i = q__1.i;
		    i__3 = i__ - k;
		    i__1 = i__ - k + *ka - m;
		    q__2.r = rwork[i__1] * t.r, q__2.i = rwork[i__1] * t.i;
		    r_cnjg(&q__4, &work[i__ - k + *ka - m]);
		    i__2 = ka1 + (i__ - k) * ab_dim1;
		    q__3.r = q__4.r * ab[i__2].r - q__4.i * ab[i__2].i, 
			    q__3.i = q__4.r * ab[i__2].i + q__4.i * ab[i__2]
			    .r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
		    i__3 = ka1 + (i__ - k) * ab_dim1;
		    i__1 = i__ - k + *ka - m;
		    q__2.r = work[i__1].r * t.r - work[i__1].i * t.i, q__2.i =
			     work[i__1].r * t.i + work[i__1].i * t.r;
		    i__2 = i__ - k + *ka - m;
		    i__5 = ka1 + (i__ - k) * ab_dim1;
		    q__3.r = rwork[i__2] * ab[i__5].r, q__3.i = rwork[i__2] * 
			    ab[i__5].i;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
		    ra1.r = ra.r, ra1.i = ra.i;
		}
	    }
/* Computing MAX */
	    i__3 = 1, i__1 = k - i0 + 2;
	    j2 = i__ - k - 1 + max(i__3,i__1) * ka1;
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    if (update) {
/* Computing MAX */
		i__3 = j2, i__1 = i__ + (*ka << 1) - k + 1;
		j2t = max(i__3,i__1);
	    } else {
		j2t = j2;
	    }
	    nrt = (*n - j2t + *ka) / ka1;
	    i__3 = j1;
	    i__1 = ka1;
	    for (j = j2t; i__1 < 0 ? j >= i__3 : j <= i__3; j += i__1) {

/*              create nonzero element a(j+1,j-ka) outside the band */
/*              and store it in WORK(j-m) */

		i__2 = j - m;
		i__5 = j - m;
		i__6 = ka1 + (j - *ka + 1) * ab_dim1;
		q__1.r = work[i__5].r * ab[i__6].r - work[i__5].i * ab[i__6]
			.i, q__1.i = work[i__5].r * ab[i__6].i + work[i__5].i 
			* ab[i__6].r;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		i__2 = ka1 + (j - *ka + 1) * ab_dim1;
		i__5 = j - m;
		i__6 = ka1 + (j - *ka + 1) * ab_dim1;
		q__1.r = rwork[i__5] * ab[i__6].r, q__1.i = rwork[i__5] * ab[
			i__6].i;
		ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L320: */
	    }

/*           generate rotations in 1st set to annihilate elements which */
/*           have been created outside the band */

	    if (nrt > 0) {
		clargv_(&nrt, &ab[ka1 + (j2t - *ka) * ab_dim1], &inca, &work[
			j2t - m], &ka1, &rwork[j2t - m], &ka1);
	    }
	    if (nr > 0) {

/*              apply rotations in 1st set from the left */

		i__1 = *ka - 1;
		for (l = 1; l <= i__1; ++l) {
		    clartv_(&nr, &ab[l + 1 + (j2 - l) * ab_dim1], &inca, &ab[
			    l + 2 + (j2 - l) * ab_dim1], &inca, &rwork[j2 - m]
, &work[j2 - m], &ka1);
/* L330: */
		}

/*              apply rotations in 1st set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[j2 * ab_dim1 + 1], &ab[(j2 + 1) * ab_dim1 + 
			1], &ab[j2 * ab_dim1 + 2], &inca, &rwork[j2 - m], &
			work[j2 - m], &ka1);

		clacgv_(&nr, &work[j2 - m], &ka1);
	    }

/*           start applying rotations in 1st set from the right */

	    i__1 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__1; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + j2 * ab_dim1], &inca, &ab[
			    ka1 - l + (j2 + 1) * ab_dim1], &inca, &rwork[j2 - 
			    m], &work[j2 - m], &ka1);
		}
/* L340: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 1st set */

		i__1 = j1;
		i__3 = ka1;
		for (j = j2; i__3 < 0 ? j >= i__1 : j <= i__1; j += i__3) {
		    i__2 = *n - m;
		    crot_(&i__2, &x[m + 1 + j * x_dim1], &c__1, &x[m + 1 + (j 
			    + 1) * x_dim1], &c__1, &rwork[j - m], &work[j - m]
);
/* L350: */
		}
	    }
/* L360: */
	}

	if (update) {
	    if (i2 <= *n && kbt > 0) {

/*              create nonzero element a(i-kbt+ka+1,i-kbt) outside the */
/*              band and store it in WORK(i-kbt) */

		i__4 = i__ - kbt;
		i__3 = kbt + 1 + (i__ - kbt) * bb_dim1;
		q__2.r = -bb[i__3].r, q__2.i = -bb[i__3].i;
		q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r * 
			ra1.i + q__2.i * ra1.r;
		work[i__4].r = q__1.r, work[i__4].i = q__1.i;
	    }
	}

	for (k = *kb; k >= 1; --k) {
	    if (update) {
/* Computing MAX */
		i__4 = 2, i__3 = k - i0 + 1;
		j2 = i__ - k - 1 + max(i__4,i__3) * ka1;
	    } else {
/* Computing MAX */
		i__4 = 1, i__3 = k - i0 + 1;
		j2 = i__ - k - 1 + max(i__4,i__3) * ka1;
	    }

/*           finish applying rotations in 2nd set from the right */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (*n - j2 + *ka + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + (j2 - *ka) * ab_dim1], &
			    inca, &ab[ka1 - l + (j2 - *ka + 1) * ab_dim1], &
			    inca, &rwork[j2 - *ka], &work[j2 - *ka], &ka1);
		}
/* L370: */
	    }
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    i__4 = j2;
	    i__3 = -ka1;
	    for (j = j1; i__3 < 0 ? j >= i__4 : j <= i__4; j += i__3) {
		i__1 = j;
		i__2 = j - *ka;
		work[i__1].r = work[i__2].r, work[i__1].i = work[i__2].i;
		rwork[j] = rwork[j - *ka];
/* L380: */
	    }
	    i__3 = j1;
	    i__4 = ka1;
	    for (j = j2; i__4 < 0 ? j >= i__3 : j <= i__3; j += i__4) {

/*              create nonzero element a(j+1,j-ka) outside the band */
/*              and store it in WORK(j) */

		i__1 = j;
		i__2 = j;
		i__5 = ka1 + (j - *ka + 1) * ab_dim1;
		q__1.r = work[i__2].r * ab[i__5].r - work[i__2].i * ab[i__5]
			.i, q__1.i = work[i__2].r * ab[i__5].i + work[i__2].i 
			* ab[i__5].r;
		work[i__1].r = q__1.r, work[i__1].i = q__1.i;
		i__1 = ka1 + (j - *ka + 1) * ab_dim1;
		i__2 = j;
		i__5 = ka1 + (j - *ka + 1) * ab_dim1;
		q__1.r = rwork[i__2] * ab[i__5].r, q__1.i = rwork[i__2] * ab[
			i__5].i;
		ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L390: */
	    }
	    if (update) {
		if (i__ - k < *n - *ka && k <= kbt) {
		    i__4 = i__ - k + *ka;
		    i__3 = i__ - k;
		    work[i__4].r = work[i__3].r, work[i__4].i = work[i__3].i;
		}
	    }
/* L400: */
	}

	for (k = *kb; k >= 1; --k) {
/* Computing MAX */
	    i__4 = 1, i__3 = k - i0 + 1;
	    j2 = i__ - k - 1 + max(i__4,i__3) * ka1;
	    nr = (*n - j2 + *ka) / ka1;
	    j1 = j2 + (nr - 1) * ka1;
	    if (nr > 0) {

/*              generate rotations in 2nd set to annihilate elements */
/*              which have been created outside the band */

		clargv_(&nr, &ab[ka1 + (j2 - *ka) * ab_dim1], &inca, &work[j2]
, &ka1, &rwork[j2], &ka1);

/*              apply rotations in 2nd set from the left */

		i__4 = *ka - 1;
		for (l = 1; l <= i__4; ++l) {
		    clartv_(&nr, &ab[l + 1 + (j2 - l) * ab_dim1], &inca, &ab[
			    l + 2 + (j2 - l) * ab_dim1], &inca, &rwork[j2], &
			    work[j2], &ka1);
/* L410: */
		}

/*              apply rotations in 2nd set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[j2 * ab_dim1 + 1], &ab[(j2 + 1) * ab_dim1 + 
			1], &ab[j2 * ab_dim1 + 2], &inca, &rwork[j2], &work[
			j2], &ka1);

		clacgv_(&nr, &work[j2], &ka1);
	    }

/*           start applying rotations in 2nd set from the right */

	    i__4 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__4; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + j2 * ab_dim1], &inca, &ab[
			    ka1 - l + (j2 + 1) * ab_dim1], &inca, &rwork[j2], 
			    &work[j2], &ka1);
		}
/* L420: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 2nd set */

		i__4 = j1;
		i__3 = ka1;
		for (j = j2; i__3 < 0 ? j >= i__4 : j <= i__4; j += i__3) {
		    i__1 = *n - m;
		    crot_(&i__1, &x[m + 1 + j * x_dim1], &c__1, &x[m + 1 + (j 
			    + 1) * x_dim1], &c__1, &rwork[j], &work[j]);
/* L430: */
		}
	    }
/* L440: */
	}

	i__3 = *kb - 1;
	for (k = 1; k <= i__3; ++k) {
/* Computing MAX */
	    i__4 = 1, i__1 = k - i0 + 2;
	    j2 = i__ - k - 1 + max(i__4,i__1) * ka1;

/*           finish applying rotations in 1st set from the right */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (*n - j2 + l) / ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + j2 * ab_dim1], &inca, &ab[
			    ka1 - l + (j2 + 1) * ab_dim1], &inca, &rwork[j2 - 
			    m], &work[j2 - m], &ka1);
		}
/* L450: */
	    }
/* L460: */
	}

	if (*kb > 1) {
	    i__3 = i2 + *ka;
	    for (j = *n - 1; j >= i__3; --j) {
		rwork[j - m] = rwork[j - *ka - m];
		i__4 = j - m;
		i__1 = j - *ka - m;
		work[i__4].r = work[i__1].r, work[i__4].i = work[i__1].i;
/* L470: */
	    }
	}

    }

    goto L10;

L480:

/*     **************************** Phase 2 ***************************** */

/*     The logical structure of this phase is: */

/*     UPDATE = .TRUE. */
/*     DO I = 1, M */
/*        use S(i) to update A and create a new bulge */
/*        apply rotations to push all bulges KA positions upward */
/*     END DO */
/*     UPDATE = .FALSE. */
/*     DO I = M - KA - 1, 2, -1 */
/*        apply rotations to push all bulges KA positions upward */
/*     END DO */

/*     To avoid duplicating code, the two loops are merged. */

    update = TRUE_;
    i__ = 0;
L490:
    if (update) {
	++i__;
/* Computing MIN */
	i__3 = *kb, i__4 = m - i__;
	kbt = min(i__3,i__4);
	i0 = i__ + 1;
/* Computing MAX */
	i__3 = 1, i__4 = i__ - *ka;
	i1 = max(i__3,i__4);
	i2 = i__ + kbt - ka1;
	if (i__ > m) {
	    update = FALSE_;
	    --i__;
	    i0 = m + 1;
	    if (*ka == 0) {
		return 0;
	    }
	    goto L490;
	}
    } else {
	i__ -= *ka;
	if (i__ < 2) {
	    return 0;
	}
    }

    if (i__ < m - kbt) {
	nx = m;
    } else {
	nx = *n;
    }

    if (upper) {

/*        Transform A, working with the upper triangle */

	if (update) {

/*           Form  inv(S(i))**H * A * inv(S(i)) */

	    i__3 = kb1 + i__ * bb_dim1;
	    bii = bb[i__3].r;
	    i__3 = ka1 + i__ * ab_dim1;
	    i__4 = ka1 + i__ * ab_dim1;
	    r__1 = ab[i__4].r / bii / bii;
	    ab[i__3].r = r__1, ab[i__3].i = 0.f;
	    i__3 = i__ - 1;
	    for (j = i1; j <= i__3; ++j) {
		i__4 = j - i__ + ka1 + i__ * ab_dim1;
		i__1 = j - i__ + ka1 + i__ * ab_dim1;
		q__1.r = ab[i__1].r / bii, q__1.i = ab[i__1].i / bii;
		ab[i__4].r = q__1.r, ab[i__4].i = q__1.i;
/* L500: */
	    }
/* Computing MIN */
	    i__4 = *n, i__1 = i__ + *ka;
	    i__3 = min(i__4,i__1);
	    for (j = i__ + 1; j <= i__3; ++j) {
		i__4 = i__ - j + ka1 + j * ab_dim1;
		i__1 = i__ - j + ka1 + j * ab_dim1;
		q__1.r = ab[i__1].r / bii, q__1.i = ab[i__1].i / bii;
		ab[i__4].r = q__1.r, ab[i__4].i = q__1.i;
/* L510: */
	    }
	    i__3 = i__ + kbt;
	    for (k = i__ + 1; k <= i__3; ++k) {
		i__4 = i__ + kbt;
		for (j = k; j <= i__4; ++j) {
		    i__1 = k - j + ka1 + j * ab_dim1;
		    i__2 = k - j + ka1 + j * ab_dim1;
		    i__5 = i__ - j + kb1 + j * bb_dim1;
		    r_cnjg(&q__5, &ab[i__ - k + ka1 + k * ab_dim1]);
		    q__4.r = bb[i__5].r * q__5.r - bb[i__5].i * q__5.i, 
			    q__4.i = bb[i__5].r * q__5.i + bb[i__5].i * 
			    q__5.r;
		    q__3.r = ab[i__2].r - q__4.r, q__3.i = ab[i__2].i - 
			    q__4.i;
		    r_cnjg(&q__7, &bb[i__ - k + kb1 + k * bb_dim1]);
		    i__6 = i__ - j + ka1 + j * ab_dim1;
		    q__6.r = q__7.r * ab[i__6].r - q__7.i * ab[i__6].i, 
			    q__6.i = q__7.r * ab[i__6].i + q__7.i * ab[i__6]
			    .r;
		    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		    i__7 = ka1 + i__ * ab_dim1;
		    r__1 = ab[i__7].r;
		    i__8 = i__ - j + kb1 + j * bb_dim1;
		    q__9.r = r__1 * bb[i__8].r, q__9.i = r__1 * bb[i__8].i;
		    r_cnjg(&q__10, &bb[i__ - k + kb1 + k * bb_dim1]);
		    q__8.r = q__9.r * q__10.r - q__9.i * q__10.i, q__8.i = 
			    q__9.r * q__10.i + q__9.i * q__10.r;
		    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L520: */
		}
/* Computing MIN */
		i__1 = *n, i__2 = i__ + *ka;
		i__4 = min(i__1,i__2);
		for (j = i__ + kbt + 1; j <= i__4; ++j) {
		    i__1 = k - j + ka1 + j * ab_dim1;
		    i__2 = k - j + ka1 + j * ab_dim1;
		    r_cnjg(&q__3, &bb[i__ - k + kb1 + k * bb_dim1]);
		    i__5 = i__ - j + ka1 + j * ab_dim1;
		    q__2.r = q__3.r * ab[i__5].r - q__3.i * ab[i__5].i, 
			    q__2.i = q__3.r * ab[i__5].i + q__3.i * ab[i__5]
			    .r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L530: */
		}
/* L540: */
	    }
	    i__3 = i__;
	    for (j = i1; j <= i__3; ++j) {
/* Computing MIN */
		i__1 = j + *ka, i__2 = i__ + kbt;
		i__4 = min(i__1,i__2);
		for (k = i__ + 1; k <= i__4; ++k) {
		    i__1 = j - k + ka1 + k * ab_dim1;
		    i__2 = j - k + ka1 + k * ab_dim1;
		    i__5 = i__ - k + kb1 + k * bb_dim1;
		    i__6 = j - i__ + ka1 + i__ * ab_dim1;
		    q__2.r = bb[i__5].r * ab[i__6].r - bb[i__5].i * ab[i__6]
			    .i, q__2.i = bb[i__5].r * ab[i__6].i + bb[i__5].i 
			    * ab[i__6].r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L550: */
		}
/* L560: */
	    }

	    if (wantx) {

/*              post-multiply X by inv(S(i)) */

		r__1 = 1.f / bii;
		csscal_(&nx, &r__1, &x[i__ * x_dim1 + 1], &c__1);
		if (kbt > 0) {
		    q__1.r = -1.f, q__1.i = -0.f;
		    i__3 = *ldbb - 1;
		    cgeru_(&nx, &kbt, &q__1, &x[i__ * x_dim1 + 1], &c__1, &bb[
			    *kb + (i__ + 1) * bb_dim1], &i__3, &x[(i__ + 1) * 
			    x_dim1 + 1], ldx);
		}
	    }

/*           store a(i1,i) in RA1 for use in next loop over K */

	    i__3 = i1 - i__ + ka1 + i__ * ab_dim1;
	    ra1.r = ab[i__3].r, ra1.i = ab[i__3].i;
	}

/*        Generate and apply vectors of rotations to chase all the */
/*        existing bulges KA positions up toward the top of the band */

	i__3 = *kb - 1;
	for (k = 1; k <= i__3; ++k) {
	    if (update) {

/*              Determine the rotations which would annihilate the bulge */
/*              which has in theory just been created */

		if (i__ + k - ka1 > 0 && i__ + k < m) {

/*                 generate rotation to annihilate a(i+k-ka-1,i) */

		    clartg_(&ab[k + 1 + i__ * ab_dim1], &ra1, &rwork[i__ + k 
			    - *ka], &work[i__ + k - *ka], &ra);

/*                 create nonzero element a(i+k-ka-1,i+k) outside the */
/*                 band and store it in WORK(m-kb+i+k) */

		    i__4 = kb1 - k + (i__ + k) * bb_dim1;
		    q__2.r = -bb[i__4].r, q__2.i = -bb[i__4].i;
		    q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r 
			    * ra1.i + q__2.i * ra1.r;
		    t.r = q__1.r, t.i = q__1.i;
		    i__4 = m - *kb + i__ + k;
		    i__1 = i__ + k - *ka;
		    q__2.r = rwork[i__1] * t.r, q__2.i = rwork[i__1] * t.i;
		    r_cnjg(&q__4, &work[i__ + k - *ka]);
		    i__2 = (i__ + k) * ab_dim1 + 1;
		    q__3.r = q__4.r * ab[i__2].r - q__4.i * ab[i__2].i, 
			    q__3.i = q__4.r * ab[i__2].i + q__4.i * ab[i__2]
			    .r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    work[i__4].r = q__1.r, work[i__4].i = q__1.i;
		    i__4 = (i__ + k) * ab_dim1 + 1;
		    i__1 = i__ + k - *ka;
		    q__2.r = work[i__1].r * t.r - work[i__1].i * t.i, q__2.i =
			     work[i__1].r * t.i + work[i__1].i * t.r;
		    i__2 = i__ + k - *ka;
		    i__5 = (i__ + k) * ab_dim1 + 1;
		    q__3.r = rwork[i__2] * ab[i__5].r, q__3.i = rwork[i__2] * 
			    ab[i__5].i;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ab[i__4].r = q__1.r, ab[i__4].i = q__1.i;
		    ra1.r = ra.r, ra1.i = ra.i;
		}
	    }
/* Computing MAX */
	    i__4 = 1, i__1 = k + i0 - m + 1;
	    j2 = i__ + k + 1 - max(i__4,i__1) * ka1;
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    if (update) {
/* Computing MIN */
		i__4 = j2, i__1 = i__ - (*ka << 1) + k - 1;
		j2t = min(i__4,i__1);
	    } else {
		j2t = j2;
	    }
	    nrt = (j2t + *ka - 1) / ka1;
	    i__4 = j2t;
	    i__1 = ka1;
	    for (j = j1; i__1 < 0 ? j >= i__4 : j <= i__4; j += i__1) {

/*              create nonzero element a(j-1,j+ka) outside the band */
/*              and store it in WORK(j) */

		i__2 = j;
		i__5 = j;
		i__6 = (j + *ka - 1) * ab_dim1 + 1;
		q__1.r = work[i__5].r * ab[i__6].r - work[i__5].i * ab[i__6]
			.i, q__1.i = work[i__5].r * ab[i__6].i + work[i__5].i 
			* ab[i__6].r;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		i__2 = (j + *ka - 1) * ab_dim1 + 1;
		i__5 = j;
		i__6 = (j + *ka - 1) * ab_dim1 + 1;
		q__1.r = rwork[i__5] * ab[i__6].r, q__1.i = rwork[i__5] * ab[
			i__6].i;
		ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L570: */
	    }

/*           generate rotations in 1st set to annihilate elements which */
/*           have been created outside the band */

	    if (nrt > 0) {
		clargv_(&nrt, &ab[(j1 + *ka) * ab_dim1 + 1], &inca, &work[j1], 
			 &ka1, &rwork[j1], &ka1);
	    }
	    if (nr > 0) {

/*              apply rotations in 1st set from the left */

		i__1 = *ka - 1;
		for (l = 1; l <= i__1; ++l) {
		    clartv_(&nr, &ab[ka1 - l + (j1 + l) * ab_dim1], &inca, &
			    ab[*ka - l + (j1 + l) * ab_dim1], &inca, &rwork[
			    j1], &work[j1], &ka1);
/* L580: */
		}

/*              apply rotations in 1st set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[ka1 + j1 * ab_dim1], &ab[ka1 + (j1 - 1) * 
			ab_dim1], &ab[*ka + j1 * ab_dim1], &inca, &rwork[j1], 
			&work[j1], &ka1);

		clacgv_(&nr, &work[j1], &ka1);
	    }

/*           start applying rotations in 1st set from the right */

	    i__1 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__1; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + j1t * ab_dim1], &inca, &ab[l + 1 + (
			    j1t - 1) * ab_dim1], &inca, &rwork[j1t], &work[
			    j1t], &ka1);
		}
/* L590: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 1st set */

		i__1 = j2;
		i__4 = ka1;
		for (j = j1; i__4 < 0 ? j >= i__1 : j <= i__1; j += i__4) {
		    crot_(&nx, &x[j * x_dim1 + 1], &c__1, &x[(j - 1) * x_dim1 
			    + 1], &c__1, &rwork[j], &work[j]);
/* L600: */
		}
	    }
/* L610: */
	}

	if (update) {
	    if (i2 > 0 && kbt > 0) {

/*              create nonzero element a(i+kbt-ka-1,i+kbt) outside the */
/*              band and store it in WORK(m-kb+i+kbt) */

		i__3 = m - *kb + i__ + kbt;
		i__4 = kb1 - kbt + (i__ + kbt) * bb_dim1;
		q__2.r = -bb[i__4].r, q__2.i = -bb[i__4].i;
		q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r * 
			ra1.i + q__2.i * ra1.r;
		work[i__3].r = q__1.r, work[i__3].i = q__1.i;
	    }
	}

	for (k = *kb; k >= 1; --k) {
	    if (update) {
/* Computing MAX */
		i__3 = 2, i__4 = k + i0 - m;
		j2 = i__ + k + 1 - max(i__3,i__4) * ka1;
	    } else {
/* Computing MAX */
		i__3 = 1, i__4 = k + i0 - m;
		j2 = i__ + k + 1 - max(i__3,i__4) * ka1;
	    }

/*           finish applying rotations in 2nd set from the right */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (j2 + *ka + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + (j1t + *ka) * ab_dim1], &inca, &ab[
			    l + 1 + (j1t + *ka - 1) * ab_dim1], &inca, &rwork[
			    m - *kb + j1t + *ka], &work[m - *kb + j1t + *ka], 
			    &ka1);
		}
/* L620: */
	    }
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    i__3 = j2;
	    i__4 = ka1;
	    for (j = j1; i__4 < 0 ? j >= i__3 : j <= i__3; j += i__4) {
		i__1 = m - *kb + j;
		i__2 = m - *kb + j + *ka;
		work[i__1].r = work[i__2].r, work[i__1].i = work[i__2].i;
		rwork[m - *kb + j] = rwork[m - *kb + j + *ka];
/* L630: */
	    }
	    i__4 = j2;
	    i__3 = ka1;
	    for (j = j1; i__3 < 0 ? j >= i__4 : j <= i__4; j += i__3) {

/*              create nonzero element a(j-1,j+ka) outside the band */
/*              and store it in WORK(m-kb+j) */

		i__1 = m - *kb + j;
		i__2 = m - *kb + j;
		i__5 = (j + *ka - 1) * ab_dim1 + 1;
		q__1.r = work[i__2].r * ab[i__5].r - work[i__2].i * ab[i__5]
			.i, q__1.i = work[i__2].r * ab[i__5].i + work[i__2].i 
			* ab[i__5].r;
		work[i__1].r = q__1.r, work[i__1].i = q__1.i;
		i__1 = (j + *ka - 1) * ab_dim1 + 1;
		i__2 = m - *kb + j;
		i__5 = (j + *ka - 1) * ab_dim1 + 1;
		q__1.r = rwork[i__2] * ab[i__5].r, q__1.i = rwork[i__2] * ab[
			i__5].i;
		ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L640: */
	    }
	    if (update) {
		if (i__ + k > ka1 && k <= kbt) {
		    i__3 = m - *kb + i__ + k - *ka;
		    i__4 = m - *kb + i__ + k;
		    work[i__3].r = work[i__4].r, work[i__3].i = work[i__4].i;
		}
	    }
/* L650: */
	}

	for (k = *kb; k >= 1; --k) {
/* Computing MAX */
	    i__3 = 1, i__4 = k + i0 - m;
	    j2 = i__ + k + 1 - max(i__3,i__4) * ka1;
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    if (nr > 0) {

/*              generate rotations in 2nd set to annihilate elements */
/*              which have been created outside the band */

		clargv_(&nr, &ab[(j1 + *ka) * ab_dim1 + 1], &inca, &work[m - *
			kb + j1], &ka1, &rwork[m - *kb + j1], &ka1);

/*              apply rotations in 2nd set from the left */

		i__3 = *ka - 1;
		for (l = 1; l <= i__3; ++l) {
		    clartv_(&nr, &ab[ka1 - l + (j1 + l) * ab_dim1], &inca, &
			    ab[*ka - l + (j1 + l) * ab_dim1], &inca, &rwork[m 
			    - *kb + j1], &work[m - *kb + j1], &ka1);
/* L660: */
		}

/*              apply rotations in 2nd set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[ka1 + j1 * ab_dim1], &ab[ka1 + (j1 - 1) * 
			ab_dim1], &ab[*ka + j1 * ab_dim1], &inca, &rwork[m - *
			kb + j1], &work[m - *kb + j1], &ka1);

		clacgv_(&nr, &work[m - *kb + j1], &ka1);
	    }

/*           start applying rotations in 2nd set from the right */

	    i__3 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__3; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + j1t * ab_dim1], &inca, &ab[l + 1 + (
			    j1t - 1) * ab_dim1], &inca, &rwork[m - *kb + j1t], 
			     &work[m - *kb + j1t], &ka1);
		}
/* L670: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 2nd set */

		i__3 = j2;
		i__4 = ka1;
		for (j = j1; i__4 < 0 ? j >= i__3 : j <= i__3; j += i__4) {
		    crot_(&nx, &x[j * x_dim1 + 1], &c__1, &x[(j - 1) * x_dim1 
			    + 1], &c__1, &rwork[m - *kb + j], &work[m - *kb + 
			    j]);
/* L680: */
		}
	    }
/* L690: */
	}

	i__4 = *kb - 1;
	for (k = 1; k <= i__4; ++k) {
/* Computing MAX */
	    i__3 = 1, i__1 = k + i0 - m + 1;
	    j2 = i__ + k + 1 - max(i__3,i__1) * ka1;

/*           finish applying rotations in 1st set from the right */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[l + j1t * ab_dim1], &inca, &ab[l + 1 + (
			    j1t - 1) * ab_dim1], &inca, &rwork[j1t], &work[
			    j1t], &ka1);
		}
/* L700: */
	    }
/* L710: */
	}

	if (*kb > 1) {
	    i__4 = i2 - *ka;
	    for (j = 2; j <= i__4; ++j) {
		rwork[j] = rwork[j + *ka];
		i__3 = j;
		i__1 = j + *ka;
		work[i__3].r = work[i__1].r, work[i__3].i = work[i__1].i;
/* L720: */
	    }
	}

    } else {

/*        Transform A, working with the lower triangle */

	if (update) {

/*           Form  inv(S(i))**H * A * inv(S(i)) */

	    i__4 = i__ * bb_dim1 + 1;
	    bii = bb[i__4].r;
	    i__4 = i__ * ab_dim1 + 1;
	    i__3 = i__ * ab_dim1 + 1;
	    r__1 = ab[i__3].r / bii / bii;
	    ab[i__4].r = r__1, ab[i__4].i = 0.f;
	    i__4 = i__ - 1;
	    for (j = i1; j <= i__4; ++j) {
		i__3 = i__ - j + 1 + j * ab_dim1;
		i__1 = i__ - j + 1 + j * ab_dim1;
		q__1.r = ab[i__1].r / bii, q__1.i = ab[i__1].i / bii;
		ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
/* L730: */
	    }
/* Computing MIN */
	    i__3 = *n, i__1 = i__ + *ka;
	    i__4 = min(i__3,i__1);
	    for (j = i__ + 1; j <= i__4; ++j) {
		i__3 = j - i__ + 1 + i__ * ab_dim1;
		i__1 = j - i__ + 1 + i__ * ab_dim1;
		q__1.r = ab[i__1].r / bii, q__1.i = ab[i__1].i / bii;
		ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
/* L740: */
	    }
	    i__4 = i__ + kbt;
	    for (k = i__ + 1; k <= i__4; ++k) {
		i__3 = i__ + kbt;
		for (j = k; j <= i__3; ++j) {
		    i__1 = j - k + 1 + k * ab_dim1;
		    i__2 = j - k + 1 + k * ab_dim1;
		    i__5 = j - i__ + 1 + i__ * bb_dim1;
		    r_cnjg(&q__5, &ab[k - i__ + 1 + i__ * ab_dim1]);
		    q__4.r = bb[i__5].r * q__5.r - bb[i__5].i * q__5.i, 
			    q__4.i = bb[i__5].r * q__5.i + bb[i__5].i * 
			    q__5.r;
		    q__3.r = ab[i__2].r - q__4.r, q__3.i = ab[i__2].i - 
			    q__4.i;
		    r_cnjg(&q__7, &bb[k - i__ + 1 + i__ * bb_dim1]);
		    i__6 = j - i__ + 1 + i__ * ab_dim1;
		    q__6.r = q__7.r * ab[i__6].r - q__7.i * ab[i__6].i, 
			    q__6.i = q__7.r * ab[i__6].i + q__7.i * ab[i__6]
			    .r;
		    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		    i__7 = i__ * ab_dim1 + 1;
		    r__1 = ab[i__7].r;
		    i__8 = j - i__ + 1 + i__ * bb_dim1;
		    q__9.r = r__1 * bb[i__8].r, q__9.i = r__1 * bb[i__8].i;
		    r_cnjg(&q__10, &bb[k - i__ + 1 + i__ * bb_dim1]);
		    q__8.r = q__9.r * q__10.r - q__9.i * q__10.i, q__8.i = 
			    q__9.r * q__10.i + q__9.i * q__10.r;
		    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L750: */
		}
/* Computing MIN */
		i__1 = *n, i__2 = i__ + *ka;
		i__3 = min(i__1,i__2);
		for (j = i__ + kbt + 1; j <= i__3; ++j) {
		    i__1 = j - k + 1 + k * ab_dim1;
		    i__2 = j - k + 1 + k * ab_dim1;
		    r_cnjg(&q__3, &bb[k - i__ + 1 + i__ * bb_dim1]);
		    i__5 = j - i__ + 1 + i__ * ab_dim1;
		    q__2.r = q__3.r * ab[i__5].r - q__3.i * ab[i__5].i, 
			    q__2.i = q__3.r * ab[i__5].i + q__3.i * ab[i__5]
			    .r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L760: */
		}
/* L770: */
	    }
	    i__4 = i__;
	    for (j = i1; j <= i__4; ++j) {
/* Computing MIN */
		i__1 = j + *ka, i__2 = i__ + kbt;
		i__3 = min(i__1,i__2);
		for (k = i__ + 1; k <= i__3; ++k) {
		    i__1 = k - j + 1 + j * ab_dim1;
		    i__2 = k - j + 1 + j * ab_dim1;
		    i__5 = k - i__ + 1 + i__ * bb_dim1;
		    i__6 = i__ - j + 1 + j * ab_dim1;
		    q__2.r = bb[i__5].r * ab[i__6].r - bb[i__5].i * ab[i__6]
			    .i, q__2.i = bb[i__5].r * ab[i__6].i + bb[i__5].i 
			    * ab[i__6].r;
		    q__1.r = ab[i__2].r - q__2.r, q__1.i = ab[i__2].i - 
			    q__2.i;
		    ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L780: */
		}
/* L790: */
	    }

	    if (wantx) {

/*              post-multiply X by inv(S(i)) */

		r__1 = 1.f / bii;
		csscal_(&nx, &r__1, &x[i__ * x_dim1 + 1], &c__1);
		if (kbt > 0) {
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgerc_(&nx, &kbt, &q__1, &x[i__ * x_dim1 + 1], &c__1, &bb[
			    i__ * bb_dim1 + 2], &c__1, &x[(i__ + 1) * x_dim1 
			    + 1], ldx);
		}
	    }

/*           store a(i,i1) in RA1 for use in next loop over K */

	    i__4 = i__ - i1 + 1 + i1 * ab_dim1;
	    ra1.r = ab[i__4].r, ra1.i = ab[i__4].i;
	}

/*        Generate and apply vectors of rotations to chase all the */
/*        existing bulges KA positions up toward the top of the band */

	i__4 = *kb - 1;
	for (k = 1; k <= i__4; ++k) {
	    if (update) {

/*              Determine the rotations which would annihilate the bulge */
/*              which has in theory just been created */

		if (i__ + k - ka1 > 0 && i__ + k < m) {

/*                 generate rotation to annihilate a(i,i+k-ka-1) */

		    clartg_(&ab[ka1 - k + (i__ + k - *ka) * ab_dim1], &ra1, &
			    rwork[i__ + k - *ka], &work[i__ + k - *ka], &ra);

/*                 create nonzero element a(i+k,i+k-ka-1) outside the */
/*                 band and store it in WORK(m-kb+i+k) */

		    i__3 = k + 1 + i__ * bb_dim1;
		    q__2.r = -bb[i__3].r, q__2.i = -bb[i__3].i;
		    q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r 
			    * ra1.i + q__2.i * ra1.r;
		    t.r = q__1.r, t.i = q__1.i;
		    i__3 = m - *kb + i__ + k;
		    i__1 = i__ + k - *ka;
		    q__2.r = rwork[i__1] * t.r, q__2.i = rwork[i__1] * t.i;
		    r_cnjg(&q__4, &work[i__ + k - *ka]);
		    i__2 = ka1 + (i__ + k - *ka) * ab_dim1;
		    q__3.r = q__4.r * ab[i__2].r - q__4.i * ab[i__2].i, 
			    q__3.i = q__4.r * ab[i__2].i + q__4.i * ab[i__2]
			    .r;
		    q__1.r = q__2.r - q__3.r, q__1.i = q__2.i - q__3.i;
		    work[i__3].r = q__1.r, work[i__3].i = q__1.i;
		    i__3 = ka1 + (i__ + k - *ka) * ab_dim1;
		    i__1 = i__ + k - *ka;
		    q__2.r = work[i__1].r * t.r - work[i__1].i * t.i, q__2.i =
			     work[i__1].r * t.i + work[i__1].i * t.r;
		    i__2 = i__ + k - *ka;
		    i__5 = ka1 + (i__ + k - *ka) * ab_dim1;
		    q__3.r = rwork[i__2] * ab[i__5].r, q__3.i = rwork[i__2] * 
			    ab[i__5].i;
		    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		    ab[i__3].r = q__1.r, ab[i__3].i = q__1.i;
		    ra1.r = ra.r, ra1.i = ra.i;
		}
	    }
/* Computing MAX */
	    i__3 = 1, i__1 = k + i0 - m + 1;
	    j2 = i__ + k + 1 - max(i__3,i__1) * ka1;
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    if (update) {
/* Computing MIN */
		i__3 = j2, i__1 = i__ - (*ka << 1) + k - 1;
		j2t = min(i__3,i__1);
	    } else {
		j2t = j2;
	    }
	    nrt = (j2t + *ka - 1) / ka1;
	    i__3 = j2t;
	    i__1 = ka1;
	    for (j = j1; i__1 < 0 ? j >= i__3 : j <= i__3; j += i__1) {

/*              create nonzero element a(j+ka,j-1) outside the band */
/*              and store it in WORK(j) */

		i__2 = j;
		i__5 = j;
		i__6 = ka1 + (j - 1) * ab_dim1;
		q__1.r = work[i__5].r * ab[i__6].r - work[i__5].i * ab[i__6]
			.i, q__1.i = work[i__5].r * ab[i__6].i + work[i__5].i 
			* ab[i__6].r;
		work[i__2].r = q__1.r, work[i__2].i = q__1.i;
		i__2 = ka1 + (j - 1) * ab_dim1;
		i__5 = j;
		i__6 = ka1 + (j - 1) * ab_dim1;
		q__1.r = rwork[i__5] * ab[i__6].r, q__1.i = rwork[i__5] * ab[
			i__6].i;
		ab[i__2].r = q__1.r, ab[i__2].i = q__1.i;
/* L800: */
	    }

/*           generate rotations in 1st set to annihilate elements which */
/*           have been created outside the band */

	    if (nrt > 0) {
		clargv_(&nrt, &ab[ka1 + j1 * ab_dim1], &inca, &work[j1], &ka1, 
			 &rwork[j1], &ka1);
	    }
	    if (nr > 0) {

/*              apply rotations in 1st set from the right */

		i__1 = *ka - 1;
		for (l = 1; l <= i__1; ++l) {
		    clartv_(&nr, &ab[l + 1 + j1 * ab_dim1], &inca, &ab[l + 2 
			    + (j1 - 1) * ab_dim1], &inca, &rwork[j1], &work[
			    j1], &ka1);
/* L810: */
		}

/*              apply rotations in 1st set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[j1 * ab_dim1 + 1], &ab[(j1 - 1) * ab_dim1 + 
			1], &ab[(j1 - 1) * ab_dim1 + 2], &inca, &rwork[j1], &
			work[j1], &ka1);

		clacgv_(&nr, &work[j1], &ka1);
	    }

/*           start applying rotations in 1st set from the left */

	    i__1 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__1; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + (j1t - ka1 + l) * ab_dim1]
, &inca, &ab[ka1 - l + (j1t - ka1 + l) * ab_dim1], 
			     &inca, &rwork[j1t], &work[j1t], &ka1);
		}
/* L820: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 1st set */

		i__1 = j2;
		i__3 = ka1;
		for (j = j1; i__3 < 0 ? j >= i__1 : j <= i__1; j += i__3) {
		    r_cnjg(&q__1, &work[j]);
		    crot_(&nx, &x[j * x_dim1 + 1], &c__1, &x[(j - 1) * x_dim1 
			    + 1], &c__1, &rwork[j], &q__1);
/* L830: */
		}
	    }
/* L840: */
	}

	if (update) {
	    if (i2 > 0 && kbt > 0) {

/*              create nonzero element a(i+kbt,i+kbt-ka-1) outside the */
/*              band and store it in WORK(m-kb+i+kbt) */

		i__4 = m - *kb + i__ + kbt;
		i__3 = kbt + 1 + i__ * bb_dim1;
		q__2.r = -bb[i__3].r, q__2.i = -bb[i__3].i;
		q__1.r = q__2.r * ra1.r - q__2.i * ra1.i, q__1.i = q__2.r * 
			ra1.i + q__2.i * ra1.r;
		work[i__4].r = q__1.r, work[i__4].i = q__1.i;
	    }
	}

	for (k = *kb; k >= 1; --k) {
	    if (update) {
/* Computing MAX */
		i__4 = 2, i__3 = k + i0 - m;
		j2 = i__ + k + 1 - max(i__4,i__3) * ka1;
	    } else {
/* Computing MAX */
		i__4 = 1, i__3 = k + i0 - m;
		j2 = i__ + k + 1 - max(i__4,i__3) * ka1;
	    }

/*           finish applying rotations in 2nd set from the left */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (j2 + *ka + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + (j1t + l - 1) * ab_dim1], 
			    &inca, &ab[ka1 - l + (j1t + l - 1) * ab_dim1], &
			    inca, &rwork[m - *kb + j1t + *ka], &work[m - *kb 
			    + j1t + *ka], &ka1);
		}
/* L850: */
	    }
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    i__4 = j2;
	    i__3 = ka1;
	    for (j = j1; i__3 < 0 ? j >= i__4 : j <= i__4; j += i__3) {
		i__1 = m - *kb + j;
		i__2 = m - *kb + j + *ka;
		work[i__1].r = work[i__2].r, work[i__1].i = work[i__2].i;
		rwork[m - *kb + j] = rwork[m - *kb + j + *ka];
/* L860: */
	    }
	    i__3 = j2;
	    i__4 = ka1;
	    for (j = j1; i__4 < 0 ? j >= i__3 : j <= i__3; j += i__4) {

/*              create nonzero element a(j+ka,j-1) outside the band */
/*              and store it in WORK(m-kb+j) */

		i__1 = m - *kb + j;
		i__2 = m - *kb + j;
		i__5 = ka1 + (j - 1) * ab_dim1;
		q__1.r = work[i__2].r * ab[i__5].r - work[i__2].i * ab[i__5]
			.i, q__1.i = work[i__2].r * ab[i__5].i + work[i__2].i 
			* ab[i__5].r;
		work[i__1].r = q__1.r, work[i__1].i = q__1.i;
		i__1 = ka1 + (j - 1) * ab_dim1;
		i__2 = m - *kb + j;
		i__5 = ka1 + (j - 1) * ab_dim1;
		q__1.r = rwork[i__2] * ab[i__5].r, q__1.i = rwork[i__2] * ab[
			i__5].i;
		ab[i__1].r = q__1.r, ab[i__1].i = q__1.i;
/* L870: */
	    }
	    if (update) {
		if (i__ + k > ka1 && k <= kbt) {
		    i__4 = m - *kb + i__ + k - *ka;
		    i__3 = m - *kb + i__ + k;
		    work[i__4].r = work[i__3].r, work[i__4].i = work[i__3].i;
		}
	    }
/* L880: */
	}

	for (k = *kb; k >= 1; --k) {
/* Computing MAX */
	    i__4 = 1, i__3 = k + i0 - m;
	    j2 = i__ + k + 1 - max(i__4,i__3) * ka1;
	    nr = (j2 + *ka - 1) / ka1;
	    j1 = j2 - (nr - 1) * ka1;
	    if (nr > 0) {

/*              generate rotations in 2nd set to annihilate elements */
/*              which have been created outside the band */

		clargv_(&nr, &ab[ka1 + j1 * ab_dim1], &inca, &work[m - *kb + 
			j1], &ka1, &rwork[m - *kb + j1], &ka1);

/*              apply rotations in 2nd set from the right */

		i__4 = *ka - 1;
		for (l = 1; l <= i__4; ++l) {
		    clartv_(&nr, &ab[l + 1 + j1 * ab_dim1], &inca, &ab[l + 2 
			    + (j1 - 1) * ab_dim1], &inca, &rwork[m - *kb + j1]
, &work[m - *kb + j1], &ka1);
/* L890: */
		}

/*              apply rotations in 2nd set from both sides to diagonal */
/*              blocks */

		clar2v_(&nr, &ab[j1 * ab_dim1 + 1], &ab[(j1 - 1) * ab_dim1 + 
			1], &ab[(j1 - 1) * ab_dim1 + 2], &inca, &rwork[m - *
			kb + j1], &work[m - *kb + j1], &ka1);

		clacgv_(&nr, &work[m - *kb + j1], &ka1);
	    }

/*           start applying rotations in 2nd set from the left */

	    i__4 = *kb - k + 1;
	    for (l = *ka - 1; l >= i__4; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + (j1t - ka1 + l) * ab_dim1]
, &inca, &ab[ka1 - l + (j1t - ka1 + l) * ab_dim1], 
			     &inca, &rwork[m - *kb + j1t], &work[m - *kb + 
			    j1t], &ka1);
		}
/* L900: */
	    }

	    if (wantx) {

/*              post-multiply X by product of rotations in 2nd set */

		i__4 = j2;
		i__3 = ka1;
		for (j = j1; i__3 < 0 ? j >= i__4 : j <= i__4; j += i__3) {
		    r_cnjg(&q__1, &work[m - *kb + j]);
		    crot_(&nx, &x[j * x_dim1 + 1], &c__1, &x[(j - 1) * x_dim1 
			    + 1], &c__1, &rwork[m - *kb + j], &q__1);
/* L910: */
		}
	    }
/* L920: */
	}

	i__3 = *kb - 1;
	for (k = 1; k <= i__3; ++k) {
/* Computing MAX */
	    i__4 = 1, i__1 = k + i0 - m + 1;
	    j2 = i__ + k + 1 - max(i__4,i__1) * ka1;

/*           finish applying rotations in 1st set from the left */

	    for (l = *kb - k; l >= 1; --l) {
		nrt = (j2 + l - 1) / ka1;
		j1t = j2 - (nrt - 1) * ka1;
		if (nrt > 0) {
		    clartv_(&nrt, &ab[ka1 - l + 1 + (j1t - ka1 + l) * ab_dim1]
, &inca, &ab[ka1 - l + (j1t - ka1 + l) * ab_dim1], 
			     &inca, &rwork[j1t], &work[j1t], &ka1);
		}
/* L930: */
	    }
/* L940: */
	}

	if (*kb > 1) {
	    i__3 = i2 - *ka;
	    for (j = 2; j <= i__3; ++j) {
		rwork[j] = rwork[j + *ka];
		i__4 = j;
		i__1 = j + *ka;
		work[i__4].r = work[i__1].r, work[i__4].i = work[i__1].i;
/* L950: */
	    }
	}

    }

    goto L490;

/*     End of CHBGST */

} /* chbgst_ */
