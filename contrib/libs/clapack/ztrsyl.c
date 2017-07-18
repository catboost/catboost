/* ztrsyl.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ztrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, doublecomplex *a, integer *lda, doublecomplex *b, 
	integer *ldb, doublecomplex *c__, integer *ldc, doublereal *scale, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
	    i__3, i__4;
    doublereal d__1, d__2;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer j, k, l;
    doublecomplex a11;
    doublereal db;
    doublecomplex x11;
    doublereal da11;
    doublecomplex vec;
    doublereal dum[1], eps, sgn, smin;
    doublecomplex suml, sumr;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), zdotu_(
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *);
    extern /* Subroutine */ int dlabad_(doublereal *, doublereal *);
    extern doublereal dlamch_(char *);
    doublereal scaloc;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern doublereal zlange_(char *, integer *, integer *, doublecomplex *, 
	    integer *, doublereal *);
    doublereal bignum;
    extern /* Subroutine */ int zdscal_(integer *, doublereal *, 
	    doublecomplex *, integer *);
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *, 
	     doublecomplex *);
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

/*  ZTRSYL solves the complex Sylvester matrix equation: */

/*     op(A)*X + X*op(B) = scale*C or */
/*     op(A)*X - X*op(B) = scale*C, */

/*  where op(A) = A or A**H, and A and B are both upper triangular. A is */
/*  M-by-M and B is N-by-N; the right hand side C and the solution X are */
/*  M-by-N; and scale is an output scale factor, set <= 1 to avoid */
/*  overflow in X. */

/*  Arguments */
/*  ========= */

/*  TRANA   (input) CHARACTER*1 */
/*          Specifies the option op(A): */
/*          = 'N': op(A) = A    (No transpose) */
/*          = 'C': op(A) = A**H (Conjugate transpose) */

/*  TRANB   (input) CHARACTER*1 */
/*          Specifies the option op(B): */
/*          = 'N': op(B) = B    (No transpose) */
/*          = 'C': op(B) = B**H (Conjugate transpose) */

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

/*  A       (input) COMPLEX*16 array, dimension (LDA,M) */
/*          The upper triangular matrix A. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  B       (input) COMPLEX*16 array, dimension (LDB,N) */
/*          The upper triangular matrix B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  C       (input/output) COMPLEX*16 array, dimension (LDC,N) */
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
    if (! notrna && ! lsame_(trana, "C")) {
	*info = -1;
    } else if (! notrnb && ! lsame_(tranb, "C")) {
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
	xerbla_("ZTRSYL", &i__1);
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
    d__1 = smlnum, d__2 = eps * zlange_("M", m, m, &a[a_offset], lda, dum), d__1 = max(d__1,d__2), d__2 = eps * zlange_("M", n, n, 
	    &b[b_offset], ldb, dum);
    smin = max(d__1,d__2);
    sgn = (doublereal) (*isgn);

    if (notrna && notrnb) {

/*        Solve    A*X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-left corner column by column by */

/*            A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    M                        L-1 */
/*          R(K,L) = SUM [A(K,I)*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]. */
/*                  I=K+1                      J=1 */

	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    for (k = *m; k >= 1; --k) {

		i__2 = *m - k;
/* Computing MIN */
		i__3 = k + 1;
/* Computing MIN */
		i__4 = k + 1;
		zdotu_(&z__1, &i__2, &a[k + min(i__3, *m)* a_dim1], lda, &c__[
			min(i__4, *m)+ l * c_dim1], &c__1);
		suml.r = z__1.r, suml.i = z__1.i;
		i__2 = l - 1;
		zdotu_(&z__1, &i__2, &c__[k + c_dim1], ldc, &b[l * b_dim1 + 1]
, &c__1);
		sumr.r = z__1.r, sumr.i = z__1.i;
		i__2 = k + l * c_dim1;
		z__3.r = sgn * sumr.r, z__3.i = sgn * sumr.i;
		z__2.r = suml.r + z__3.r, z__2.i = suml.i + z__3.i;
		z__1.r = c__[i__2].r - z__2.r, z__1.i = c__[i__2].i - z__2.i;
		vec.r = z__1.r, vec.i = z__1.i;

		scaloc = 1.;
		i__2 = k + k * a_dim1;
		i__3 = l + l * b_dim1;
		z__2.r = sgn * b[i__3].r, z__2.i = sgn * b[i__3].i;
		z__1.r = a[i__2].r + z__2.r, z__1.i = a[i__2].i + z__2.i;
		a11.r = z__1.r, a11.i = z__1.i;
		da11 = (d__1 = a11.r, abs(d__1)) + (d__2 = d_imag(&a11), abs(
			d__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.;
		    da11 = smin;
		    *info = 1;
		}
		db = (d__1 = vec.r, abs(d__1)) + (d__2 = d_imag(&vec), abs(
			d__2));
		if (da11 < 1. && db > 1.) {
		    if (db > bignum * da11) {
			scaloc = 1. / db;
		    }
		}
		z__3.r = scaloc, z__3.i = 0.;
		z__2.r = vec.r * z__3.r - vec.i * z__3.i, z__2.i = vec.r * 
			z__3.i + vec.i * z__3.r;
		zladiv_(&z__1, &z__2, &a11);
		x11.r = z__1.r, x11.i = z__1.i;

		if (scaloc != 1.) {
		    i__2 = *n;
		    for (j = 1; j <= i__2; ++j) {
			zdscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L10: */
		    }
		    *scale *= scaloc;
		}
		i__2 = k + l * c_dim1;
		c__[i__2].r = x11.r, c__[i__2].i = x11.i;

/* L20: */
	    }
/* L30: */
	}

    } else if (! notrna && notrnb) {

/*        Solve    A' *X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        upper-left corner column by column by */

/*            A'(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                   K-1                         L-1 */
/*          R(K,L) = SUM [A'(I,K)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)] */
/*                   I=1                         J=1 */

	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = *m;
	    for (k = 1; k <= i__2; ++k) {

		i__3 = k - 1;
		zdotc_(&z__1, &i__3, &a[k * a_dim1 + 1], &c__1, &c__[l * 
			c_dim1 + 1], &c__1);
		suml.r = z__1.r, suml.i = z__1.i;
		i__3 = l - 1;
		zdotu_(&z__1, &i__3, &c__[k + c_dim1], ldc, &b[l * b_dim1 + 1]
, &c__1);
		sumr.r = z__1.r, sumr.i = z__1.i;
		i__3 = k + l * c_dim1;
		z__3.r = sgn * sumr.r, z__3.i = sgn * sumr.i;
		z__2.r = suml.r + z__3.r, z__2.i = suml.i + z__3.i;
		z__1.r = c__[i__3].r - z__2.r, z__1.i = c__[i__3].i - z__2.i;
		vec.r = z__1.r, vec.i = z__1.i;

		scaloc = 1.;
		d_cnjg(&z__2, &a[k + k * a_dim1]);
		i__3 = l + l * b_dim1;
		z__3.r = sgn * b[i__3].r, z__3.i = sgn * b[i__3].i;
		z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
		a11.r = z__1.r, a11.i = z__1.i;
		da11 = (d__1 = a11.r, abs(d__1)) + (d__2 = d_imag(&a11), abs(
			d__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.;
		    da11 = smin;
		    *info = 1;
		}
		db = (d__1 = vec.r, abs(d__1)) + (d__2 = d_imag(&vec), abs(
			d__2));
		if (da11 < 1. && db > 1.) {
		    if (db > bignum * da11) {
			scaloc = 1. / db;
		    }
		}

		z__3.r = scaloc, z__3.i = 0.;
		z__2.r = vec.r * z__3.r - vec.i * z__3.i, z__2.i = vec.r * 
			z__3.i + vec.i * z__3.r;
		zladiv_(&z__1, &z__2, &a11);
		x11.r = z__1.r, x11.i = z__1.i;

		if (scaloc != 1.) {
		    i__3 = *n;
		    for (j = 1; j <= i__3; ++j) {
			zdscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L40: */
		    }
		    *scale *= scaloc;
		}
		i__3 = k + l * c_dim1;
		c__[i__3].r = x11.r, c__[i__3].i = x11.i;

/* L50: */
	    }
/* L60: */
	}

    } else if (! notrna && ! notrnb) {

/*        Solve    A'*X + ISGN*X*B' = C. */

/*        The (K,L)th block of X is determined starting from */
/*        upper-right corner column by column by */

/*            A'(K,K)*X(K,L) + ISGN*X(K,L)*B'(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    K-1 */
/*           R(K,L) = SUM [A'(I,K)*X(I,L)] + */
/*                    I=1 */
/*                           N */
/*                     ISGN*SUM [X(K,J)*B'(L,J)]. */
/*                          J=L+1 */

	for (l = *n; l >= 1; --l) {
	    i__1 = *m;
	    for (k = 1; k <= i__1; ++k) {

		i__2 = k - 1;
		zdotc_(&z__1, &i__2, &a[k * a_dim1 + 1], &c__1, &c__[l * 
			c_dim1 + 1], &c__1);
		suml.r = z__1.r, suml.i = z__1.i;
		i__2 = *n - l;
/* Computing MIN */
		i__3 = l + 1;
/* Computing MIN */
		i__4 = l + 1;
		zdotc_(&z__1, &i__2, &c__[k + min(i__3, *n)* c_dim1], ldc, &b[
			l + min(i__4, *n)* b_dim1], ldb);
		sumr.r = z__1.r, sumr.i = z__1.i;
		i__2 = k + l * c_dim1;
		d_cnjg(&z__4, &sumr);
		z__3.r = sgn * z__4.r, z__3.i = sgn * z__4.i;
		z__2.r = suml.r + z__3.r, z__2.i = suml.i + z__3.i;
		z__1.r = c__[i__2].r - z__2.r, z__1.i = c__[i__2].i - z__2.i;
		vec.r = z__1.r, vec.i = z__1.i;

		scaloc = 1.;
		i__2 = k + k * a_dim1;
		i__3 = l + l * b_dim1;
		z__3.r = sgn * b[i__3].r, z__3.i = sgn * b[i__3].i;
		z__2.r = a[i__2].r + z__3.r, z__2.i = a[i__2].i + z__3.i;
		d_cnjg(&z__1, &z__2);
		a11.r = z__1.r, a11.i = z__1.i;
		da11 = (d__1 = a11.r, abs(d__1)) + (d__2 = d_imag(&a11), abs(
			d__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.;
		    da11 = smin;
		    *info = 1;
		}
		db = (d__1 = vec.r, abs(d__1)) + (d__2 = d_imag(&vec), abs(
			d__2));
		if (da11 < 1. && db > 1.) {
		    if (db > bignum * da11) {
			scaloc = 1. / db;
		    }
		}

		z__3.r = scaloc, z__3.i = 0.;
		z__2.r = vec.r * z__3.r - vec.i * z__3.i, z__2.i = vec.r * 
			z__3.i + vec.i * z__3.r;
		zladiv_(&z__1, &z__2, &a11);
		x11.r = z__1.r, x11.i = z__1.i;

		if (scaloc != 1.) {
		    i__2 = *n;
		    for (j = 1; j <= i__2; ++j) {
			zdscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L70: */
		    }
		    *scale *= scaloc;
		}
		i__2 = k + l * c_dim1;
		c__[i__2].r = x11.r, c__[i__2].i = x11.i;

/* L80: */
	    }
/* L90: */
	}

    } else if (notrna && ! notrnb) {

/*        Solve    A*X + ISGN*X*B' = C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-left corner column by column by */

/*           A(K,K)*X(K,L) + ISGN*X(K,L)*B'(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    M                          N */
/*          R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B'(L,J)] */
/*                  I=K+1                      J=L+1 */

	for (l = *n; l >= 1; --l) {
	    for (k = *m; k >= 1; --k) {

		i__1 = *m - k;
/* Computing MIN */
		i__2 = k + 1;
/* Computing MIN */
		i__3 = k + 1;
		zdotu_(&z__1, &i__1, &a[k + min(i__2, *m)* a_dim1], lda, &c__[
			min(i__3, *m)+ l * c_dim1], &c__1);
		suml.r = z__1.r, suml.i = z__1.i;
		i__1 = *n - l;
/* Computing MIN */
		i__2 = l + 1;
/* Computing MIN */
		i__3 = l + 1;
		zdotc_(&z__1, &i__1, &c__[k + min(i__2, *n)* c_dim1], ldc, &b[
			l + min(i__3, *n)* b_dim1], ldb);
		sumr.r = z__1.r, sumr.i = z__1.i;
		i__1 = k + l * c_dim1;
		d_cnjg(&z__4, &sumr);
		z__3.r = sgn * z__4.r, z__3.i = sgn * z__4.i;
		z__2.r = suml.r + z__3.r, z__2.i = suml.i + z__3.i;
		z__1.r = c__[i__1].r - z__2.r, z__1.i = c__[i__1].i - z__2.i;
		vec.r = z__1.r, vec.i = z__1.i;

		scaloc = 1.;
		i__1 = k + k * a_dim1;
		d_cnjg(&z__3, &b[l + l * b_dim1]);
		z__2.r = sgn * z__3.r, z__2.i = sgn * z__3.i;
		z__1.r = a[i__1].r + z__2.r, z__1.i = a[i__1].i + z__2.i;
		a11.r = z__1.r, a11.i = z__1.i;
		da11 = (d__1 = a11.r, abs(d__1)) + (d__2 = d_imag(&a11), abs(
			d__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.;
		    da11 = smin;
		    *info = 1;
		}
		db = (d__1 = vec.r, abs(d__1)) + (d__2 = d_imag(&vec), abs(
			d__2));
		if (da11 < 1. && db > 1.) {
		    if (db > bignum * da11) {
			scaloc = 1. / db;
		    }
		}

		z__3.r = scaloc, z__3.i = 0.;
		z__2.r = vec.r * z__3.r - vec.i * z__3.i, z__2.i = vec.r * 
			z__3.i + vec.i * z__3.r;
		zladiv_(&z__1, &z__2, &a11);
		x11.r = z__1.r, x11.i = z__1.i;

		if (scaloc != 1.) {
		    i__1 = *n;
		    for (j = 1; j <= i__1; ++j) {
			zdscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L100: */
		    }
		    *scale *= scaloc;
		}
		i__1 = k + l * c_dim1;
		c__[i__1].r = x11.r, c__[i__1].i = x11.i;

/* L110: */
	    }
/* L120: */
	}

    }

    return 0;

/*     End of ZTRSYL */

} /* ztrsyl_ */
