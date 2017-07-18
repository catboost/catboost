/* sgetc2.f -- translated by f2c (version 20061008).
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
static real c_b10 = -1.f;

/* Subroutine */ int sgetc2_(integer *n, real *a, integer *lda, integer *ipiv, 
	 integer *jpiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    real r__1;

    /* Local variables */
    integer i__, j, ip, jp;
    real eps;
    integer ipv, jpv;
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *, 
	    integer *, real *, integer *, real *, integer *);
    real smin, xmax;
    extern /* Subroutine */ int sswap_(integer *, real *, integer *, real *, 
	    integer *), slabad_(real *, real *);
    extern doublereal slamch_(char *);
    real bignum, smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SGETC2 computes an LU factorization with complete pivoting of the */
/*  n-by-n matrix A. The factorization has the form A = P * L * U * Q, */
/*  where P and Q are permutation matrices, L is lower triangular with */
/*  unit diagonal elements and U is upper triangular. */

/*  This is the Level 2 BLAS algorithm. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A. N >= 0. */

/*  A       (input/output) REAL array, dimension (LDA, N) */
/*          On entry, the n-by-n matrix A to be factored. */
/*          On exit, the factors L and U from the factorization */
/*          A = P*L*U*Q; the unit diagonal elements of L are not stored. */
/*          If U(k, k) appears to be less than SMIN, U(k, k) is given the */
/*          value of SMIN, i.e., giving a nonsingular perturbed system. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  IPIV    (output) INTEGER array, dimension(N). */
/*          The pivot indices; for 1 <= i <= N, row i of the */
/*          matrix has been interchanged with row IPIV(i). */

/*  JPIV    (output) INTEGER array, dimension(N). */
/*          The pivot indices; for 1 <= j <= N, column j of the */
/*          matrix has been interchanged with column JPIV(j). */

/*  INFO    (output) INTEGER */
/*           = 0: successful exit */
/*           > 0: if INFO = k, U(k, k) is likely to produce owerflow if */
/*                we try to solve for x in Ax = b. So U is perturbed to */
/*                avoid the overflow. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Set constants to control overflow */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    --jpiv;

    /* Function Body */
    *info = 0;
    eps = slamch_("P");
    smlnum = slamch_("S") / eps;
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);

/*     Factorize A using complete pivoting. */
/*     Set pivots less than SMIN to SMIN. */

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Find max element in matrix A */

	xmax = 0.f;
	i__2 = *n;
	for (ip = i__; ip <= i__2; ++ip) {
	    i__3 = *n;
	    for (jp = i__; jp <= i__3; ++jp) {
		if ((r__1 = a[ip + jp * a_dim1], dabs(r__1)) >= xmax) {
		    xmax = (r__1 = a[ip + jp * a_dim1], dabs(r__1));
		    ipv = ip;
		    jpv = jp;
		}
/* L10: */
	    }
/* L20: */
	}
	if (i__ == 1) {
/* Computing MAX */
	    r__1 = eps * xmax;
	    smin = dmax(r__1,smlnum);
	}

/*        Swap rows */

	if (ipv != i__) {
	    sswap_(n, &a[ipv + a_dim1], lda, &a[i__ + a_dim1], lda);
	}
	ipiv[i__] = ipv;

/*        Swap columns */

	if (jpv != i__) {
	    sswap_(n, &a[jpv * a_dim1 + 1], &c__1, &a[i__ * a_dim1 + 1], &
		    c__1);
	}
	jpiv[i__] = jpv;

/*        Check for singularity */

	if ((r__1 = a[i__ + i__ * a_dim1], dabs(r__1)) < smin) {
	    *info = i__;
	    a[i__ + i__ * a_dim1] = smin;
	}
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    a[j + i__ * a_dim1] /= a[i__ + i__ * a_dim1];
/* L30: */
	}
	i__2 = *n - i__;
	i__3 = *n - i__;
	sger_(&i__2, &i__3, &c_b10, &a[i__ + 1 + i__ * a_dim1], &c__1, &a[i__ 
		+ (i__ + 1) * a_dim1], lda, &a[i__ + 1 + (i__ + 1) * a_dim1], 
		lda);
/* L40: */
    }

    if ((r__1 = a[*n + *n * a_dim1], dabs(r__1)) < smin) {
	*info = *n;
	a[*n + *n * a_dim1] = smin;
    }

    return 0;

/*     End of SGETC2 */

} /* sgetc2_ */
