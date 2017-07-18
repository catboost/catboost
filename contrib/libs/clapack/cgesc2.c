/* cgesc2.f -- translated by f2c (version 20061008).
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
static complex c_b13 = {1.f,0.f};
static integer c_n1 = -1;

/* Subroutine */ int cgesc2_(integer *n, complex *a, integer *lda, complex *
	rhs, integer *ipiv, integer *jpiv, real *scale)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    real r__1;
    complex q__1, q__2, q__3;

    /* Builtin functions */
    double c_abs(complex *);
    void c_div(complex *, complex *, complex *);

    /* Local variables */
    integer i__, j;
    real eps;
    complex temp;
    extern /* Subroutine */ int cscal_(integer *, complex *, complex *, 
	    integer *), slabad_(real *, real *);
    extern integer icamax_(integer *, complex *, integer *);
    extern doublereal slamch_(char *);
    real bignum;
    extern /* Subroutine */ int claswp_(integer *, complex *, integer *, 
	    integer *, integer *, integer *, integer *);
    real smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGESC2 solves a system of linear equations */

/*            A * X = scale* RHS */

/*  with a general N-by-N matrix A using the LU factorization with */
/*  complete pivoting computed by CGETC2. */


/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A. */

/*  A       (input) COMPLEX array, dimension (LDA, N) */
/*          On entry, the  LU part of the factorization of the n-by-n */
/*          matrix A computed by CGETC2:  A = P * L * U * Q */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1, N). */

/*  RHS     (input/output) COMPLEX array, dimension N. */
/*          On entry, the right hand side vector b. */
/*          On exit, the solution vector X. */

/*  IPIV    (input) INTEGER array, dimension (N). */
/*          The pivot indices; for 1 <= i <= N, row i of the */
/*          matrix has been interchanged with row IPIV(i). */

/*  JPIV    (input) INTEGER array, dimension (N). */
/*          The pivot indices; for 1 <= j <= N, column j of the */
/*          matrix has been interchanged with column JPIV(j). */

/*  SCALE    (output) REAL */
/*           On exit, SCALE contains the scale factor. SCALE is chosen */
/*           0 <= SCALE <= 1 to prevent owerflow in the solution. */

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

/*     Set constant to control overflow */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --rhs;
    --ipiv;
    --jpiv;

    /* Function Body */
    eps = slamch_("P");
    smlnum = slamch_("S") / eps;
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);

/*     Apply permutations IPIV to RHS */

    i__1 = *n - 1;
    claswp_(&c__1, &rhs[1], lda, &c__1, &i__1, &ipiv[1], &c__1);

/*     Solve for L part */

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    i__3 = j;
	    i__4 = j;
	    i__5 = j + i__ * a_dim1;
	    i__6 = i__;
	    q__2.r = a[i__5].r * rhs[i__6].r - a[i__5].i * rhs[i__6].i, 
		    q__2.i = a[i__5].r * rhs[i__6].i + a[i__5].i * rhs[i__6]
		    .r;
	    q__1.r = rhs[i__4].r - q__2.r, q__1.i = rhs[i__4].i - q__2.i;
	    rhs[i__3].r = q__1.r, rhs[i__3].i = q__1.i;
/* L10: */
	}
/* L20: */
    }

/*     Solve for U part */

    *scale = 1.f;

/*     Check for scaling */

    i__ = icamax_(n, &rhs[1], &c__1);
    if (smlnum * 2.f * c_abs(&rhs[i__]) > c_abs(&a[*n + *n * a_dim1])) {
	r__1 = c_abs(&rhs[i__]);
	q__1.r = .5f / r__1, q__1.i = 0.f / r__1;
	temp.r = q__1.r, temp.i = q__1.i;
	cscal_(n, &temp, &rhs[1], &c__1);
	*scale *= temp.r;
    }
    for (i__ = *n; i__ >= 1; --i__) {
	c_div(&q__1, &c_b13, &a[i__ + i__ * a_dim1]);
	temp.r = q__1.r, temp.i = q__1.i;
	i__1 = i__;
	i__2 = i__;
	q__1.r = rhs[i__2].r * temp.r - rhs[i__2].i * temp.i, q__1.i = rhs[
		i__2].r * temp.i + rhs[i__2].i * temp.r;
	rhs[i__1].r = q__1.r, rhs[i__1].i = q__1.i;
	i__1 = *n;
	for (j = i__ + 1; j <= i__1; ++j) {
	    i__2 = i__;
	    i__3 = i__;
	    i__4 = j;
	    i__5 = i__ + j * a_dim1;
	    q__3.r = a[i__5].r * temp.r - a[i__5].i * temp.i, q__3.i = a[i__5]
		    .r * temp.i + a[i__5].i * temp.r;
	    q__2.r = rhs[i__4].r * q__3.r - rhs[i__4].i * q__3.i, q__2.i = 
		    rhs[i__4].r * q__3.i + rhs[i__4].i * q__3.r;
	    q__1.r = rhs[i__3].r - q__2.r, q__1.i = rhs[i__3].i - q__2.i;
	    rhs[i__2].r = q__1.r, rhs[i__2].i = q__1.i;
/* L30: */
	}
/* L40: */
    }

/*     Apply permutations JPIV to the solution (RHS) */

    i__1 = *n - 1;
    claswp_(&c__1, &rhs[1], lda, &c__1, &i__1, &jpiv[1], &c_n1);
    return 0;

/*     End of CGESC2 */

} /* cgesc2_ */
