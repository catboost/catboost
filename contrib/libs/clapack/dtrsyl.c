/* dtrsyl.f -- translated by f2c (version 20061008).
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
static logical c_false = FALSE_;
static integer c__2 = 2;
static doublereal c_b26 = 1.;
static doublereal c_b30 = 0.;
static logical c_true = TRUE_;

/* Subroutine */ int dtrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, doublereal *c__, integer *ldc, doublereal *scale, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
	    i__3, i__4;
    doublereal d__1, d__2;

    /* Local variables */
    integer j, k, l;
    doublereal x[4]	/* was [2][2] */;
    integer k1, k2, l1, l2;
    doublereal a11, db, da11, vec[4]	/* was [2][2] */, dum[1], eps, sgn;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    integer ierr;
    doublereal smin, suml, sumr;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    integer knext, lnext;
    doublereal xnorm;
    extern /* Subroutine */ int dlaln2_(logical *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *, 
	     doublereal *, doublereal *, integer *, doublereal *, doublereal *
, doublereal *, integer *, doublereal *, doublereal *, integer *),
	     dlasy2_(logical *, logical *, integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *), dlabad_(doublereal *, doublereal *);
    extern doublereal dlamch_(char *), dlange_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *);
    doublereal scaloc;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;
    logical notrna, notrnb;
    doublereal smlnum;


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

/*  DTRSYL solves the real Sylvester matrix equation: */

/*     op(A)*X + X*op(B) = scale*C or */
/*     op(A)*X - X*op(B) = scale*C, */

/*  where op(A) = A or A**T, and  A and B are both upper quasi- */
/*  triangular. A is M-by-M and B is N-by-N; the right hand side C and */
/*  the solution X are M-by-N; and scale is an output scale factor, set */
/*  <= 1 to avoid overflow in X. */

/*  A and B must be in Schur canonical form (as returned by DHSEQR), that */
/*  is, block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; */
/*  each 2-by-2 diagonal block has its diagonal elements equal and its */
/*  off-diagonal elements of opposite sign. */

/*  Arguments */
/*  ========= */

/*  TRANA   (input) CHARACTER*1 */
/*          Specifies the option op(A): */
/*          = 'N': op(A) = A    (No transpose) */
/*          = 'T': op(A) = A**T (Transpose) */
/*          = 'C': op(A) = A**H (Conjugate transpose = Transpose) */

/*  TRANB   (input) CHARACTER*1 */
/*          Specifies the option op(B): */
/*          = 'N': op(B) = B    (No transpose) */
/*          = 'T': op(B) = B**T (Transpose) */
/*          = 'C': op(B) = B**H (Conjugate transpose = Transpose) */

/*  ISGN    (input) INTEGER */
/*          Specifies the sign in the equation: */
/*          = +1: solve op(A)*X + X*op(B) = scale*C */
/*          = -1: solve op(A)*X - X*op(B) = scale*C */

/*  M       (input) INTEGER */
/*          The order of the matrix A, and the number of rows in the */
/*          matrices X and C. M >= 0. */

/*  N       (input) INTEGER */
/*          The order of the matrix B, and the number of columns in the */
/*          matrices X and C. N >= 0. */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA,M) */
/*          The upper quasi-triangular matrix A, in Schur canonical form. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  B       (input) DOUBLE PRECISION array, dimension (LDB,N) */
/*          The upper quasi-triangular matrix B, in Schur canonical form. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  C       (input/output) DOUBLE PRECISION array, dimension (LDC,N) */
/*          On entry, the M-by-N right hand side matrix C. */
/*          On exit, C is overwritten by the solution matrix X. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the array C. LDC >= max(1,M) */

/*  SCALE   (output) DOUBLE PRECISION */
/*          The scale factor, scale, set <= 1 to avoid overflow in X. */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */
/*          = 1: A and B have common or very close eigenvalues; perturbed */
/*               values were used to solve the equation (but the matrices */
/*               A and B are unchanged). */

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

/*     Decode and Test input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    notrna = lsame_(trana, "N");
    notrnb = lsame_(tranb, "N");

    *info = 0;
    if (! notrna && ! lsame_(trana, "T") && ! lsame_(
	    trana, "C")) {
	*info = -1;
    } else if (! notrnb && ! lsame_(tranb, "T") && ! 
	    lsame_(tranb, "C")) {
	*info = -2;
    } else if (*isgn != 1 && *isgn != -1) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,*m)) {
	*info = -7;
    } else if (*ldb < max(1,*n)) {
	*info = -9;
    } else if (*ldc < max(1,*m)) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTRSYL", &i__1);
	return 0;
    }

/*     Quick return if possible */

    *scale = 1.;
    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Set constants to control overflow */

    eps = dlamch_("P");
    smlnum = dlamch_("S");
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    smlnum = smlnum * (doublereal) (*m * *n) / eps;
    bignum = 1. / smlnum;

/* Computing MAX */
    d__1 = smlnum, d__2 = eps * dlange_("M", m, m, &a[a_offset], lda, dum), d__1 = max(d__1,d__2), d__2 = eps * dlange_("M", n, n, 
	    &b[b_offset], ldb, dum);
    smin = max(d__1,d__2);

    sgn = (doublereal) (*isgn);

    if (notrna && notrnb) {

/*        Solve    A*X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-left corner column by column by */

/*         A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                  M                         L-1 */
/*        R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)]. */
/*                I=K+1                       J=1 */

/*        Start column loop (index = L) */
/*        L1 (L2) : column index of the first (first) row of X(K,L). */

	lnext = 1;
	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    if (l < lnext) {
		goto L60;
	    }
	    if (l == *n) {
		l1 = l;
		l2 = l;
	    } else {
		if (b[l + 1 + l * b_dim1] != 0.) {
		    l1 = l;
		    l2 = l + 1;
		    lnext = l + 2;
		} else {
		    l1 = l;
		    l2 = l;
		    lnext = l + 1;
		}
	    }

/*           Start row loop (index = K) */
/*           K1 (K2): row index of the first (last) row of X(K,L). */

	    knext = *m;
	    for (k = *m; k >= 1; --k) {
		if (k > knext) {
		    goto L50;
		}
		if (k == 1) {
		    k1 = k;
		    k2 = k;
		} else {
		    if (a[k + (k - 1) * a_dim1] != 0.) {
			k1 = k - 1;
			k2 = k;
			knext = k - 2;
		    } else {
			k1 = k;
			k2 = k;
			knext = k - 1;
		    }
		}

		if (l1 == l2 && k1 == k2) {
		    i__2 = *m - k1;
/* Computing MIN */
		    i__3 = k1 + 1;
/* Computing MIN */
		    i__4 = k1 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);
		    scaloc = 1.;

		    a11 = a[k1 + k1 * a_dim1] + sgn * b[l1 + l1 * b_dim1];
		    da11 = abs(a11);
		    if (da11 <= smin) {
			a11 = smin;
			da11 = smin;
			*info = 1;
		    }
		    db = abs(vec[0]);
		    if (da11 < 1. && db > 1.) {
			if (db > bignum * da11) {
			    scaloc = 1. / db;
			}
		    }
		    x[0] = vec[0] * scaloc / a11;

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L10: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];

		} else if (l1 == l2 && k1 != k2) {

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k2 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k2 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    d__1 = -sgn * b[l1 + l1 * b_dim1];
		    dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b26, &a[k1 + k1 
			    * a_dim1], lda, &c_b26, &c_b26, vec, &c__2, &d__1, 
			     &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L20: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k2 + l1 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 == k2) {

		    i__2 = *m - k1;
/* Computing MIN */
		    i__3 = k1 + 1;
/* Computing MIN */
		    i__4 = k1 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = sgn * (c__[k1 + l1 * c_dim1] - (suml + sgn * 
			    sumr));

		    i__2 = *m - k1;
/* Computing MIN */
		    i__3 = k1 + 1;
/* Computing MIN */
		    i__4 = k1 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l2 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = sgn * (c__[k1 + l2 * c_dim1] - (suml + sgn * 
			    sumr));

		    d__1 = -sgn * a[k1 + k1 * a_dim1];
		    dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b26, &b[l1 + l1 *
			     b_dim1], ldb, &c_b26, &c_b26, vec, &c__2, &d__1, 
			    &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L30: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 != k2) {

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k1 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l2 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k1 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[2] = c__[k1 + l2 * c_dim1] - (suml + sgn * sumr);

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k2 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l1 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k2 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = *m - k2;
/* Computing MIN */
		    i__3 = k2 + 1;
/* Computing MIN */
		    i__4 = k2 + 1;
		    suml = ddot_(&i__2, &a[k2 + min(i__3, *m)* a_dim1], lda, &
			    c__[min(i__4, *m)+ l2 * c_dim1], &c__1);
		    i__2 = l1 - 1;
		    sumr = ddot_(&i__2, &c__[k2 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[3] = c__[k2 + l2 * c_dim1] - (suml + sgn * sumr);

		    dlasy2_(&c_false, &c_false, isgn, &c__2, &c__2, &a[k1 + 
			    k1 * a_dim1], lda, &b[l1 + l1 * b_dim1], ldb, vec, 
			     &c__2, &scaloc, x, &c__2, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L40: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[2];
		    c__[k2 + l1 * c_dim1] = x[1];
		    c__[k2 + l2 * c_dim1] = x[3];
		}

L50:
		;
	    }

L60:
	    ;
	}

    } else if (! notrna && notrnb) {

/*        Solve    A' *X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        upper-left corner column by column by */

/*          A(K,K)'*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                   K-1                        L-1 */
/*          R(K,L) = SUM [A(I,K)'*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)] */
/*                   I=1                        J=1 */

/*        Start column loop (index = L) */
/*        L1 (L2): column index of the first (last) row of X(K,L) */

	lnext = 1;
	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    if (l < lnext) {
		goto L120;
	    }
	    if (l == *n) {
		l1 = l;
		l2 = l;
	    } else {
		if (b[l + 1 + l * b_dim1] != 0.) {
		    l1 = l;
		    l2 = l + 1;
		    lnext = l + 2;
		} else {
		    l1 = l;
		    l2 = l;
		    lnext = l + 1;
		}
	    }

/*           Start row loop (index = K) */
/*           K1 (K2): row index of the first (last) row of X(K,L) */

	    knext = 1;
	    i__2 = *m;
	    for (k = 1; k <= i__2; ++k) {
		if (k < knext) {
		    goto L110;
		}
		if (k == *m) {
		    k1 = k;
		    k2 = k;
		} else {
		    if (a[k + 1 + k * a_dim1] != 0.) {
			k1 = k;
			k2 = k + 1;
			knext = k + 2;
		    } else {
			k1 = k;
			k2 = k;
			knext = k + 1;
		    }
		}

		if (l1 == l2 && k1 == k2) {
		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);
		    scaloc = 1.;

		    a11 = a[k1 + k1 * a_dim1] + sgn * b[l1 + l1 * b_dim1];
		    da11 = abs(a11);
		    if (da11 <= smin) {
			a11 = smin;
			da11 = smin;
			*info = 1;
		    }
		    db = abs(vec[0]);
		    if (da11 < 1. && db > 1.) {
			if (db > bignum * da11) {
			    scaloc = 1. / db;
			}
		    }
		    x[0] = vec[0] * scaloc / a11;

		    if (scaloc != 1.) {
			i__3 = *n;
			for (j = 1; j <= i__3; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L70: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];

		} else if (l1 == l2 && k1 != k2) {

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k2 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k2 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    d__1 = -sgn * b[l1 + l1 * b_dim1];
		    dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b26, &a[k1 + k1 *
			     a_dim1], lda, &c_b26, &c_b26, vec, &c__2, &d__1, 
			    &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__3 = *n;
			for (j = 1; j <= i__3; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L80: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k2 + l1 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 == k2) {

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = sgn * (c__[k1 + l1 * c_dim1] - (suml + sgn * 
			    sumr));

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = sgn * (c__[k1 + l2 * c_dim1] - (suml + sgn * 
			    sumr));

		    d__1 = -sgn * a[k1 + k1 * a_dim1];
		    dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b26, &b[l1 + l1 *
			     b_dim1], ldb, &c_b26, &c_b26, vec, &c__2, &d__1, 
			    &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__3 = *n;
			for (j = 1; j <= i__3; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L90: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 != k2) {

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k1 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k1 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[2] = c__[k1 + l2 * c_dim1] - (suml + sgn * sumr);

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k2 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k2 + c_dim1], ldc, &b[l1 * 
			    b_dim1 + 1], &c__1);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__3 = k1 - 1;
		    suml = ddot_(&i__3, &a[k2 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__3 = l1 - 1;
		    sumr = ddot_(&i__3, &c__[k2 + c_dim1], ldc, &b[l2 * 
			    b_dim1 + 1], &c__1);
		    vec[3] = c__[k2 + l2 * c_dim1] - (suml + sgn * sumr);

		    dlasy2_(&c_true, &c_false, isgn, &c__2, &c__2, &a[k1 + k1 
			    * a_dim1], lda, &b[l1 + l1 * b_dim1], ldb, vec, &
			    c__2, &scaloc, x, &c__2, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__3 = *n;
			for (j = 1; j <= i__3; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L100: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[2];
		    c__[k2 + l1 * c_dim1] = x[1];
		    c__[k2 + l2 * c_dim1] = x[3];
		}

L110:
		;
	    }
L120:
	    ;
	}

    } else if (! notrna && ! notrnb) {

/*        Solve    A'*X + ISGN*X*B' = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        top-right corner column by column by */

/*           A(K,K)'*X(K,L) + ISGN*X(K,L)*B(L,L)' = C(K,L) - R(K,L) */

/*        Where */
/*                     K-1                          N */
/*            R(K,L) = SUM [A(I,K)'*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)']. */
/*                     I=1                        J=L+1 */

/*        Start column loop (index = L) */
/*        L1 (L2): column index of the first (last) row of X(K,L) */

	lnext = *n;
	for (l = *n; l >= 1; --l) {
	    if (l > lnext) {
		goto L180;
	    }
	    if (l == 1) {
		l1 = l;
		l2 = l;
	    } else {
		if (b[l + (l - 1) * b_dim1] != 0.) {
		    l1 = l - 1;
		    l2 = l;
		    lnext = l - 2;
		} else {
		    l1 = l;
		    l2 = l;
		    lnext = l - 1;
		}
	    }

/*           Start row loop (index = K) */
/*           K1 (K2): row index of the first (last) row of X(K,L) */

	    knext = 1;
	    i__1 = *m;
	    for (k = 1; k <= i__1; ++k) {
		if (k < knext) {
		    goto L170;
		}
		if (k == *m) {
		    k1 = k;
		    k2 = k;
		} else {
		    if (a[k + 1 + k * a_dim1] != 0.) {
			k1 = k;
			k2 = k + 1;
			knext = k + 2;
		    } else {
			k1 = k;
			k2 = k;
			knext = k + 1;
		    }
		}

		if (l1 == l2 && k1 == k2) {
		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l1;
/* Computing MIN */
		    i__3 = l1 + 1;
/* Computing MIN */
		    i__4 = l1 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);
		    scaloc = 1.;

		    a11 = a[k1 + k1 * a_dim1] + sgn * b[l1 + l1 * b_dim1];
		    da11 = abs(a11);
		    if (da11 <= smin) {
			a11 = smin;
			da11 = smin;
			*info = 1;
		    }
		    db = abs(vec[0]);
		    if (da11 < 1. && db > 1.) {
			if (db > bignum * da11) {
			    scaloc = 1. / db;
			}
		    }
		    x[0] = vec[0] * scaloc / a11;

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L130: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];

		} else if (l1 == l2 && k1 != k2) {

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k2 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k2 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    d__1 = -sgn * b[l1 + l1 * b_dim1];
		    dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b26, &a[k1 + k1 *
			     a_dim1], lda, &c_b26, &c_b26, vec, &c__2, &d__1, 
			    &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L140: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k2 + l1 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 == k2) {

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[0] = sgn * (c__[k1 + l1 * c_dim1] - (suml + sgn * 
			    sumr));

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__4, *n)* b_dim1], ldb);
		    vec[1] = sgn * (c__[k1 + l2 * c_dim1] - (suml + sgn * 
			    sumr));

		    d__1 = -sgn * a[k1 + k1 * a_dim1];
		    dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b26, &b[l1 + l1 
			    * b_dim1], ldb, &c_b26, &c_b26, vec, &c__2, &d__1, 
			     &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L150: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 != k2) {

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k1 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k1 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__4, *n)* b_dim1], ldb);
		    vec[2] = c__[k1 + l2 * c_dim1] - (suml + sgn * sumr);

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k2 * a_dim1 + 1], &c__1, &c__[l1 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k2 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__4, *n)* b_dim1], ldb);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__2 = k1 - 1;
		    suml = ddot_(&i__2, &a[k2 * a_dim1 + 1], &c__1, &c__[l2 * 
			    c_dim1 + 1], &c__1);
		    i__2 = *n - l2;
/* Computing MIN */
		    i__3 = l2 + 1;
/* Computing MIN */
		    i__4 = l2 + 1;
		    sumr = ddot_(&i__2, &c__[k2 + min(i__3, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__4, *n)* b_dim1], ldb);
		    vec[3] = c__[k2 + l2 * c_dim1] - (suml + sgn * sumr);

		    dlasy2_(&c_true, &c_true, isgn, &c__2, &c__2, &a[k1 + k1 *
			     a_dim1], lda, &b[l1 + l1 * b_dim1], ldb, vec, &
			    c__2, &scaloc, x, &c__2, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L160: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[2];
		    c__[k2 + l1 * c_dim1] = x[1];
		    c__[k2 + l2 * c_dim1] = x[3];
		}

L170:
		;
	    }
L180:
	    ;
	}

    } else if (notrna && ! notrnb) {

/*        Solve    A*X + ISGN*X*B' = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-right corner column by column by */

/*            A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)' = C(K,L) - R(K,L) */

/*        Where */
/*                      M                          N */
/*            R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)']. */
/*                    I=K+1                      J=L+1 */

/*        Start column loop (index = L) */
/*        L1 (L2): column index of the first (last) row of X(K,L) */

	lnext = *n;
	for (l = *n; l >= 1; --l) {
	    if (l > lnext) {
		goto L240;
	    }
	    if (l == 1) {
		l1 = l;
		l2 = l;
	    } else {
		if (b[l + (l - 1) * b_dim1] != 0.) {
		    l1 = l - 1;
		    l2 = l;
		    lnext = l - 2;
		} else {
		    l1 = l;
		    l2 = l;
		    lnext = l - 1;
		}
	    }

/*           Start row loop (index = K) */
/*           K1 (K2): row index of the first (last) row of X(K,L) */

	    knext = *m;
	    for (k = *m; k >= 1; --k) {
		if (k > knext) {
		    goto L230;
		}
		if (k == 1) {
		    k1 = k;
		    k2 = k;
		} else {
		    if (a[k + (k - 1) * a_dim1] != 0.) {
			k1 = k - 1;
			k2 = k;
			knext = k - 2;
		    } else {
			k1 = k;
			k2 = k;
			knext = k - 1;
		    }
		}

		if (l1 == l2 && k1 == k2) {
		    i__1 = *m - k1;
/* Computing MIN */
		    i__2 = k1 + 1;
/* Computing MIN */
		    i__3 = k1 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l1;
/* Computing MIN */
		    i__2 = l1 + 1;
/* Computing MIN */
		    i__3 = l1 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);
		    scaloc = 1.;

		    a11 = a[k1 + k1 * a_dim1] + sgn * b[l1 + l1 * b_dim1];
		    da11 = abs(a11);
		    if (da11 <= smin) {
			a11 = smin;
			da11 = smin;
			*info = 1;
		    }
		    db = abs(vec[0]);
		    if (da11 < 1. && db > 1.) {
			if (db > bignum * da11) {
			    scaloc = 1. / db;
			}
		    }
		    x[0] = vec[0] * scaloc / a11;

		    if (scaloc != 1.) {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L190: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];

		} else if (l1 == l2 && k1 != k2) {

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k2 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k2 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    d__1 = -sgn * b[l1 + l1 * b_dim1];
		    dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b26, &a[k1 + k1 
			    * a_dim1], lda, &c_b26, &c_b26, vec, &c__2, &d__1, 
			     &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L200: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k2 + l1 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 == k2) {

		    i__1 = *m - k1;
/* Computing MIN */
		    i__2 = k1 + 1;
/* Computing MIN */
		    i__3 = k1 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[0] = sgn * (c__[k1 + l1 * c_dim1] - (suml + sgn * 
			    sumr));

		    i__1 = *m - k1;
/* Computing MIN */
		    i__2 = k1 + 1;
/* Computing MIN */
		    i__3 = k1 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l2 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__3, *n)* b_dim1], ldb);
		    vec[1] = sgn * (c__[k1 + l2 * c_dim1] - (suml + sgn * 
			    sumr));

		    d__1 = -sgn * a[k1 + k1 * a_dim1];
		    dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b26, &b[l1 + l1 
			    * b_dim1], ldb, &c_b26, &c_b26, vec, &c__2, &d__1, 
			     &c_b30, x, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L210: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[1];

		} else if (l1 != l2 && k1 != k2) {

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[0] = c__[k1 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k1 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l2 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k1 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__3, *n)* b_dim1], ldb);
		    vec[2] = c__[k1 + l2 * c_dim1] - (suml + sgn * sumr);

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k2 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l1 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k2 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l1 + min(i__3, *n)* b_dim1], ldb);
		    vec[1] = c__[k2 + l1 * c_dim1] - (suml + sgn * sumr);

		    i__1 = *m - k2;
/* Computing MIN */
		    i__2 = k2 + 1;
/* Computing MIN */
		    i__3 = k2 + 1;
		    suml = ddot_(&i__1, &a[k2 + min(i__2, *m)* a_dim1], lda, &
			    c__[min(i__3, *m)+ l2 * c_dim1], &c__1);
		    i__1 = *n - l2;
/* Computing MIN */
		    i__2 = l2 + 1;
/* Computing MIN */
		    i__3 = l2 + 1;
		    sumr = ddot_(&i__1, &c__[k2 + min(i__2, *n)* c_dim1], ldc, 
			     &b[l2 + min(i__3, *n)* b_dim1], ldb);
		    vec[3] = c__[k2 + l2 * c_dim1] - (suml + sgn * sumr);

		    dlasy2_(&c_false, &c_true, isgn, &c__2, &c__2, &a[k1 + k1 
			    * a_dim1], lda, &b[l1 + l1 * b_dim1], ldb, vec, &
			    c__2, &scaloc, x, &c__2, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 1;
		    }

		    if (scaloc != 1.) {
			i__1 = *n;
			for (j = 1; j <= i__1; ++j) {
			    dscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L220: */
			}
			*scale *= scaloc;
		    }
		    c__[k1 + l1 * c_dim1] = x[0];
		    c__[k1 + l2 * c_dim1] = x[2];
		    c__[k2 + l1 * c_dim1] = x[1];
		    c__[k2 + l2 * c_dim1] = x[3];
		}

L230:
		;
	    }
L240:
	    ;
	}

    }

    return 0;

/*     End of DTRSYL */

} /* dtrsyl_ */
