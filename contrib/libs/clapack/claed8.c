/* claed8.f -- translated by f2c (version 20061008).
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

static real c_b3 = -1.f;
static integer c__1 = 1;

/* Subroutine */ int claed8_(integer *k, integer *n, integer *qsiz, complex *
	q, integer *ldq, real *d__, real *rho, integer *cutpnt, real *z__, 
	real *dlamda, complex *q2, integer *ldq2, real *w, integer *indxp, 
	integer *indx, integer *indxq, integer *perm, integer *givptr, 
	integer *givcol, real *givnum, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, q2_dim1, q2_offset, i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real c__;
    integer i__, j;
    real s, t;
    integer k2, n1, n2, jp, n1p1;
    real eps, tau, tol;
    integer jlam, imax, jmax;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *), 
	    ccopy_(integer *, complex *, integer *, complex *, integer *), 
	    csrot_(integer *, complex *, integer *, complex *, integer *, 
	    real *, real *), scopy_(integer *, real *, integer *, real *, 
	    integer *);
    extern doublereal slapy2_(real *, real *), slamch_(char *);
    extern /* Subroutine */ int clacpy_(char *, integer *, integer *, complex 
	    *, integer *, complex *, integer *), xerbla_(char *, 
	    integer *);
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slamrg_(integer *, integer *, real *, integer 
	    *, integer *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLAED8 merges the two sets of eigenvalues together into a single */
/*  sorted set.  Then it tries to deflate the size of the problem. */
/*  There are two ways in which deflation can occur:  when two or more */
/*  eigenvalues are close together or if there is a tiny element in the */
/*  Z vector.  For each such occurrence the order of the related secular */
/*  equation problem is reduced by one. */

/*  Arguments */
/*  ========= */

/*  K      (output) INTEGER */
/*         Contains the number of non-deflated eigenvalues. */
/*         This is the order of the related secular equation. */

/*  N      (input) INTEGER */
/*         The dimension of the symmetric tridiagonal matrix.  N >= 0. */

/*  QSIZ   (input) INTEGER */
/*         The dimension of the unitary matrix used to reduce */
/*         the dense or band matrix to tridiagonal form. */
/*         QSIZ >= N if ICOMPQ = 1. */

/*  Q      (input/output) COMPLEX array, dimension (LDQ,N) */
/*         On entry, Q contains the eigenvectors of the partially solved */
/*         system which has been previously updated in matrix */
/*         multiplies with other partially solved eigensystems. */
/*         On exit, Q contains the trailing (N-K) updated eigenvectors */
/*         (those which were deflated) in its last N-K columns. */

/*  LDQ    (input) INTEGER */
/*         The leading dimension of the array Q.  LDQ >= max( 1, N ). */

/*  D      (input/output) REAL array, dimension (N) */
/*         On entry, D contains the eigenvalues of the two submatrices to */
/*         be combined.  On exit, D contains the trailing (N-K) updated */
/*         eigenvalues (those which were deflated) sorted into increasing */
/*         order. */

/*  RHO    (input/output) REAL */
/*         Contains the off diagonal element associated with the rank-1 */
/*         cut which originally split the two submatrices which are now */
/*         being recombined. RHO is modified during the computation to */
/*         the value required by SLAED3. */

/*  CUTPNT (input) INTEGER */
/*         Contains the location of the last eigenvalue in the leading */
/*         sub-matrix.  MIN(1,N) <= CUTPNT <= N. */

/*  Z      (input) REAL array, dimension (N) */
/*         On input this vector contains the updating vector (the last */
/*         row of the first sub-eigenvector matrix and the first row of */
/*         the second sub-eigenvector matrix).  The contents of Z are */
/*         destroyed during the updating process. */

/*  DLAMDA (output) REAL array, dimension (N) */
/*         Contains a copy of the first K eigenvalues which will be used */
/*         by SLAED3 to form the secular equation. */

/*  Q2     (output) COMPLEX array, dimension (LDQ2,N) */
/*         If ICOMPQ = 0, Q2 is not referenced.  Otherwise, */
/*         Contains a copy of the first K eigenvectors which will be used */
/*         by SLAED7 in a matrix multiply (SGEMM) to update the new */
/*         eigenvectors. */

/*  LDQ2   (input) INTEGER */
/*         The leading dimension of the array Q2.  LDQ2 >= max( 1, N ). */

/*  W      (output) REAL array, dimension (N) */
/*         This will hold the first k values of the final */
/*         deflation-altered z-vector and will be passed to SLAED3. */

/*  INDXP  (workspace) INTEGER array, dimension (N) */
/*         This will contain the permutation used to place deflated */
/*         values of D at the end of the array. On output INDXP(1:K) */
/*         points to the nondeflated D-values and INDXP(K+1:N) */
/*         points to the deflated eigenvalues. */

/*  INDX   (workspace) INTEGER array, dimension (N) */
/*         This will contain the permutation used to sort the contents of */
/*         D into ascending order. */

/*  INDXQ  (input) INTEGER array, dimension (N) */
/*         This contains the permutation which separately sorts the two */
/*         sub-problems in D into ascending order.  Note that elements in */
/*         the second half of this permutation must first have CUTPNT */
/*         added to their values in order to be accurate. */

/*  PERM   (output) INTEGER array, dimension (N) */
/*         Contains the permutations (from deflation and sorting) to be */
/*         applied to each eigenblock. */

/*  GIVPTR (output) INTEGER */
/*         Contains the number of Givens rotations which took place in */
/*         this subproblem. */

/*  GIVCOL (output) INTEGER array, dimension (2, N) */
/*         Each pair of numbers indicates a pair of columns to take place */
/*         in a Givens rotation. */

/*  GIVNUM (output) REAL array, dimension (2, N) */
/*         Each number indicates the S value to be used in the */
/*         corresponding Givens rotation. */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
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

/*     Test the input parameters. */

    /* Parameter adjustments */
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --d__;
    --z__;
    --dlamda;
    q2_dim1 = *ldq2;
    q2_offset = 1 + q2_dim1;
    q2 -= q2_offset;
    --w;
    --indxp;
    --indx;
    --indxq;
    --perm;
    givcol -= 3;
    givnum -= 3;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -2;
    } else if (*qsiz < *n) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -5;
    } else if (*cutpnt < min(1,*n) || *cutpnt > *n) {
	*info = -8;
    } else if (*ldq2 < max(1,*n)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CLAED8", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    n1 = *cutpnt;
    n2 = *n - n1;
    n1p1 = n1 + 1;

    if (*rho < 0.f) {
	sscal_(&n2, &c_b3, &z__[n1p1], &c__1);
    }

/*     Normalize z so that norm(z) = 1 */

    t = 1.f / sqrt(2.f);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	indx[j] = j;
/* L10: */
    }
    sscal_(n, &t, &z__[1], &c__1);
    *rho = (r__1 = *rho * 2.f, dabs(r__1));

/*     Sort the eigenvalues into increasing order */

    i__1 = *n;
    for (i__ = *cutpnt + 1; i__ <= i__1; ++i__) {
	indxq[i__] += *cutpnt;
/* L20: */
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = d__[indxq[i__]];
	w[i__] = z__[indxq[i__]];
/* L30: */
    }
    i__ = 1;
    j = *cutpnt + 1;
    slamrg_(&n1, &n2, &dlamda[1], &c__1, &c__1, &indx[1]);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = dlamda[indx[i__]];
	z__[i__] = w[indx[i__]];
/* L40: */
    }

/*     Calculate the allowable deflation tolerance */

    imax = isamax_(n, &z__[1], &c__1);
    jmax = isamax_(n, &d__[1], &c__1);
    eps = slamch_("Epsilon");
    tol = eps * 8.f * (r__1 = d__[jmax], dabs(r__1));

/*     If the rank-1 modifier is small enough, no more needs to be done */
/*     -- except to reorganize Q so that its columns correspond with the */
/*     elements in D. */

    if (*rho * (r__1 = z__[imax], dabs(r__1)) <= tol) {
	*k = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    perm[j] = indxq[indx[j]];
	    ccopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1]
, &c__1);
/* L50: */
	}
	clacpy_("A", qsiz, n, &q2[q2_dim1 + 1], ldq2, &q[q_dim1 + 1], ldq);
	return 0;
    }

/*     If there are multiple eigenvalues then the problem deflates.  Here */
/*     the number of equal eigenvalues are found.  As each equal */
/*     eigenvalue is found, an elementary reflector is computed to rotate */
/*     the corresponding eigensubspace so that the corresponding */
/*     components of Z are zero in this new basis. */

    *k = 0;
    *givptr = 0;
    k2 = *n + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (*rho * (r__1 = z__[j], dabs(r__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    indxp[k2] = j;
	    if (j == *n) {
		goto L100;
	    }
	} else {
	    jlam = j;
	    goto L70;
	}
/* L60: */
    }
L70:
    ++j;
    if (j > *n) {
	goto L90;
    }
    if (*rho * (r__1 = z__[j], dabs(r__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	indxp[k2] = j;
    } else {

/*        Check if eigenvalues are close enough to allow deflation. */

	s = z__[jlam];
	c__ = z__[j];

/*        Find sqrt(a**2+b**2) without overflow or */
/*        destructive underflow. */

	tau = slapy2_(&c__, &s);
	t = d__[j] - d__[jlam];
	c__ /= tau;
	s = -s / tau;
	if ((r__1 = t * c__ * s, dabs(r__1)) <= tol) {

/*           Deflation is possible. */

	    z__[j] = tau;
	    z__[jlam] = 0.f;

/*           Record the appropriate Givens rotation */

	    ++(*givptr);
	    givcol[(*givptr << 1) + 1] = indxq[indx[jlam]];
	    givcol[(*givptr << 1) + 2] = indxq[indx[j]];
	    givnum[(*givptr << 1) + 1] = c__;
	    givnum[(*givptr << 1) + 2] = s;
	    csrot_(qsiz, &q[indxq[indx[jlam]] * q_dim1 + 1], &c__1, &q[indxq[
		    indx[j]] * q_dim1 + 1], &c__1, &c__, &s);
	    t = d__[jlam] * c__ * c__ + d__[j] * s * s;
	    d__[j] = d__[jlam] * s * s + d__[j] * c__ * c__;
	    d__[jlam] = t;
	    --k2;
	    i__ = 1;
L80:
	    if (k2 + i__ <= *n) {
		if (d__[jlam] < d__[indxp[k2 + i__]]) {
		    indxp[k2 + i__ - 1] = indxp[k2 + i__];
		    indxp[k2 + i__] = jlam;
		    ++i__;
		    goto L80;
		} else {
		    indxp[k2 + i__ - 1] = jlam;
		}
	    } else {
		indxp[k2 + i__ - 1] = jlam;
	    }
	    jlam = j;
	} else {
	    ++(*k);
	    w[*k] = z__[jlam];
	    dlamda[*k] = d__[jlam];
	    indxp[*k] = jlam;
	    jlam = j;
	}
    }
    goto L70;
L90:

/*     Record the last eigenvalue. */

    ++(*k);
    w[*k] = z__[jlam];
    dlamda[*k] = d__[jlam];
    indxp[*k] = jlam;

L100:

/*     Sort the eigenvalues and corresponding eigenvectors into DLAMDA */
/*     and Q2 respectively.  The eigenvalues/vectors which were not */
/*     deflated go into the first K slots of DLAMDA and Q2 respectively, */
/*     while those which were deflated go into the last N - K slots. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jp = indxp[j];
	dlamda[j] = d__[jp];
	perm[j] = indxq[indx[jp]];
	ccopy_(qsiz, &q[perm[j] * q_dim1 + 1], &c__1, &q2[j * q2_dim1 + 1], &
		c__1);
/* L110: */
    }

/*     The deflated eigenvalues and their corresponding vectors go back */
/*     into the last N - K slots of D and Q respectively. */

    if (*k < *n) {
	i__1 = *n - *k;
	scopy_(&i__1, &dlamda[*k + 1], &c__1, &d__[*k + 1], &c__1);
	i__1 = *n - *k;
	clacpy_("A", qsiz, &i__1, &q2[(*k + 1) * q2_dim1 + 1], ldq2, &q[(*k + 
		1) * q_dim1 + 1], ldq);
    }

    return 0;

/*     End of CLAED8 */

} /* claed8_ */
