/* zlagtm.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zlagtm_(char *trans, integer *n, integer *nrhs, 
	doublereal *alpha, doublecomplex *dl, doublecomplex *d__, 
	doublecomplex *du, doublecomplex *x, integer *ldx, doublereal *beta, 
	doublecomplex *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, x_dim1, x_offset, i__1, i__2, i__3, i__4, i__5, 
	    i__6, i__7, i__8, i__9, i__10;
    doublecomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7, z__8, z__9;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j;
    extern logical lsame_(char *, char *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLAGTM performs a matrix-vector product of the form */

/*     B := alpha * A * X + beta * B */

/*  where A is a tridiagonal matrix of order N, B and X are N by NRHS */
/*  matrices, and alpha and beta are real scalars, each of which may be */
/*  0., 1., or -1. */

/*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the operation applied to A. */
/*          = 'N':  No transpose, B := alpha * A * X + beta * B */
/*          = 'T':  Transpose,    B := alpha * A**T * X + beta * B */
/*          = 'C':  Conjugate transpose, B := alpha * A**H * X + beta * B */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices X and B. */

/*  ALPHA   (input) DOUBLE PRECISION */
/*          The scalar alpha.  ALPHA must be 0., 1., or -1.; otherwise, */
/*          it is assumed to be 0. */

/*  DL      (input) COMPLEX*16 array, dimension (N-1) */
/*          The (n-1) sub-diagonal elements of T. */

/*  D       (input) COMPLEX*16 array, dimension (N) */
/*          The diagonal elements of T. */

/*  DU      (input) COMPLEX*16 array, dimension (N-1) */
/*          The (n-1) super-diagonal elements of T. */

/*  X       (input) COMPLEX*16 array, dimension (LDX,NRHS) */
/*          The N by NRHS matrix X. */
/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X.  LDX >= max(N,1). */

/*  BETA    (input) DOUBLE PRECISION */
/*          The scalar beta.  BETA must be 0., 1., or -1.; otherwise, */
/*          it is assumed to be 1. */

/*  B       (input/output) COMPLEX*16 array, dimension (LDB,NRHS) */
/*          On entry, the N by NRHS matrix B. */
/*          On exit, B is overwritten by the matrix expression */
/*          B := alpha * A * X + beta * B. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(N,1). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (*n == 0) {
	return 0;
    }

/*     Multiply B by BETA if BETA.NE.1. */

    if (*beta == 0.) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		b[i__3].r = 0., b[i__3].i = 0.;
/* L10: */
	    }
/* L20: */
	}
    } else if (*beta == -1.) {
	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		i__3 = i__ + j * b_dim1;
		i__4 = i__ + j * b_dim1;
		z__1.r = -b[i__4].r, z__1.i = -b[i__4].i;
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L30: */
	    }
/* L40: */
	}
    }

    if (*alpha == 1.) {
	if (lsame_(trans, "N")) {

/*           Compute B := B + A*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__2.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__2.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__1.r = b[i__3].r + z__2.r, z__1.i = b[i__3].i + z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__3.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__3.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    i__5 = j * x_dim1 + 2;
		    z__4.r = du[1].r * x[i__5].r - du[1].i * x[i__5].i, 
			    z__4.i = du[1].r * x[i__5].i + du[1].i * x[i__5]
			    .r;
		    z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    i__4 = *n - 1;
		    i__5 = *n - 1 + j * x_dim1;
		    z__3.r = dl[i__4].r * x[i__5].r - dl[i__4].i * x[i__5].i, 
			    z__3.i = dl[i__4].r * x[i__5].i + dl[i__4].i * x[
			    i__5].r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    i__6 = *n;
		    i__7 = *n + j * x_dim1;
		    z__4.r = d__[i__6].r * x[i__7].r - d__[i__6].i * x[i__7]
			    .i, z__4.i = d__[i__6].r * x[i__7].i + d__[i__6]
			    .i * x[i__7].r;
		    z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			i__5 = i__ - 1;
			i__6 = i__ - 1 + j * x_dim1;
			z__4.r = dl[i__5].r * x[i__6].r - dl[i__5].i * x[i__6]
				.i, z__4.i = dl[i__5].r * x[i__6].i + dl[i__5]
				.i * x[i__6].r;
			z__3.r = b[i__4].r + z__4.r, z__3.i = b[i__4].i + 
				z__4.i;
			i__7 = i__;
			i__8 = i__ + j * x_dim1;
			z__5.r = d__[i__7].r * x[i__8].r - d__[i__7].i * x[
				i__8].i, z__5.i = d__[i__7].r * x[i__8].i + 
				d__[i__7].i * x[i__8].r;
			z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
			i__9 = i__;
			i__10 = i__ + 1 + j * x_dim1;
			z__6.r = du[i__9].r * x[i__10].r - du[i__9].i * x[
				i__10].i, z__6.i = du[i__9].r * x[i__10].i + 
				du[i__9].i * x[i__10].r;
			z__1.r = z__2.r + z__6.r, z__1.i = z__2.i + z__6.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L50: */
		    }
		}
/* L60: */
	    }
	} else if (lsame_(trans, "T")) {

/*           Compute B := B + A**T * X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__2.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__2.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__1.r = b[i__3].r + z__2.r, z__1.i = b[i__3].i + z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__3.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__3.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    i__5 = j * x_dim1 + 2;
		    z__4.r = dl[1].r * x[i__5].r - dl[1].i * x[i__5].i, 
			    z__4.i = dl[1].r * x[i__5].i + dl[1].i * x[i__5]
			    .r;
		    z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    i__4 = *n - 1;
		    i__5 = *n - 1 + j * x_dim1;
		    z__3.r = du[i__4].r * x[i__5].r - du[i__4].i * x[i__5].i, 
			    z__3.i = du[i__4].r * x[i__5].i + du[i__4].i * x[
			    i__5].r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    i__6 = *n;
		    i__7 = *n + j * x_dim1;
		    z__4.r = d__[i__6].r * x[i__7].r - d__[i__6].i * x[i__7]
			    .i, z__4.i = d__[i__6].r * x[i__7].i + d__[i__6]
			    .i * x[i__7].r;
		    z__1.r = z__2.r + z__4.r, z__1.i = z__2.i + z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			i__5 = i__ - 1;
			i__6 = i__ - 1 + j * x_dim1;
			z__4.r = du[i__5].r * x[i__6].r - du[i__5].i * x[i__6]
				.i, z__4.i = du[i__5].r * x[i__6].i + du[i__5]
				.i * x[i__6].r;
			z__3.r = b[i__4].r + z__4.r, z__3.i = b[i__4].i + 
				z__4.i;
			i__7 = i__;
			i__8 = i__ + j * x_dim1;
			z__5.r = d__[i__7].r * x[i__8].r - d__[i__7].i * x[
				i__8].i, z__5.i = d__[i__7].r * x[i__8].i + 
				d__[i__7].i * x[i__8].r;
			z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
			i__9 = i__;
			i__10 = i__ + 1 + j * x_dim1;
			z__6.r = dl[i__9].r * x[i__10].r - dl[i__9].i * x[
				i__10].i, z__6.i = dl[i__9].r * x[i__10].i + 
				dl[i__9].i * x[i__10].r;
			z__1.r = z__2.r + z__6.r, z__1.i = z__2.i + z__6.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L70: */
		    }
		}
/* L80: */
	    }
	} else if (lsame_(trans, "C")) {

/*           Compute B := B + A**H * X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    d_cnjg(&z__3, &d__[1]);
		    i__4 = j * x_dim1 + 1;
		    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, z__2.i =
			     z__3.r * x[i__4].i + z__3.i * x[i__4].r;
		    z__1.r = b[i__3].r + z__2.r, z__1.i = b[i__3].i + z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    d_cnjg(&z__4, &d__[1]);
		    i__4 = j * x_dim1 + 1;
		    z__3.r = z__4.r * x[i__4].r - z__4.i * x[i__4].i, z__3.i =
			     z__4.r * x[i__4].i + z__4.i * x[i__4].r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    d_cnjg(&z__6, &dl[1]);
		    i__5 = j * x_dim1 + 2;
		    z__5.r = z__6.r * x[i__5].r - z__6.i * x[i__5].i, z__5.i =
			     z__6.r * x[i__5].i + z__6.i * x[i__5].r;
		    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i + z__5.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    d_cnjg(&z__4, &du[*n - 1]);
		    i__4 = *n - 1 + j * x_dim1;
		    z__3.r = z__4.r * x[i__4].r - z__4.i * x[i__4].i, z__3.i =
			     z__4.r * x[i__4].i + z__4.i * x[i__4].r;
		    z__2.r = b[i__3].r + z__3.r, z__2.i = b[i__3].i + z__3.i;
		    d_cnjg(&z__6, &d__[*n]);
		    i__5 = *n + j * x_dim1;
		    z__5.r = z__6.r * x[i__5].r - z__6.i * x[i__5].i, z__5.i =
			     z__6.r * x[i__5].i + z__6.i * x[i__5].r;
		    z__1.r = z__2.r + z__5.r, z__1.i = z__2.i + z__5.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			d_cnjg(&z__5, &du[i__ - 1]);
			i__5 = i__ - 1 + j * x_dim1;
			z__4.r = z__5.r * x[i__5].r - z__5.i * x[i__5].i, 
				z__4.i = z__5.r * x[i__5].i + z__5.i * x[i__5]
				.r;
			z__3.r = b[i__4].r + z__4.r, z__3.i = b[i__4].i + 
				z__4.i;
			d_cnjg(&z__7, &d__[i__]);
			i__6 = i__ + j * x_dim1;
			z__6.r = z__7.r * x[i__6].r - z__7.i * x[i__6].i, 
				z__6.i = z__7.r * x[i__6].i + z__7.i * x[i__6]
				.r;
			z__2.r = z__3.r + z__6.r, z__2.i = z__3.i + z__6.i;
			d_cnjg(&z__9, &dl[i__]);
			i__7 = i__ + 1 + j * x_dim1;
			z__8.r = z__9.r * x[i__7].r - z__9.i * x[i__7].i, 
				z__8.i = z__9.r * x[i__7].i + z__9.i * x[i__7]
				.r;
			z__1.r = z__2.r + z__8.r, z__1.i = z__2.i + z__8.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L90: */
		    }
		}
/* L100: */
	    }
	}
    } else if (*alpha == -1.) {
	if (lsame_(trans, "N")) {

/*           Compute B := B - A*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__2.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__2.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__1.r = b[i__3].r - z__2.r, z__1.i = b[i__3].i - z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__3.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__3.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    i__5 = j * x_dim1 + 2;
		    z__4.r = du[1].r * x[i__5].r - du[1].i * x[i__5].i, 
			    z__4.i = du[1].r * x[i__5].i + du[1].i * x[i__5]
			    .r;
		    z__1.r = z__2.r - z__4.r, z__1.i = z__2.i - z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    i__4 = *n - 1;
		    i__5 = *n - 1 + j * x_dim1;
		    z__3.r = dl[i__4].r * x[i__5].r - dl[i__4].i * x[i__5].i, 
			    z__3.i = dl[i__4].r * x[i__5].i + dl[i__4].i * x[
			    i__5].r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    i__6 = *n;
		    i__7 = *n + j * x_dim1;
		    z__4.r = d__[i__6].r * x[i__7].r - d__[i__6].i * x[i__7]
			    .i, z__4.i = d__[i__6].r * x[i__7].i + d__[i__6]
			    .i * x[i__7].r;
		    z__1.r = z__2.r - z__4.r, z__1.i = z__2.i - z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			i__5 = i__ - 1;
			i__6 = i__ - 1 + j * x_dim1;
			z__4.r = dl[i__5].r * x[i__6].r - dl[i__5].i * x[i__6]
				.i, z__4.i = dl[i__5].r * x[i__6].i + dl[i__5]
				.i * x[i__6].r;
			z__3.r = b[i__4].r - z__4.r, z__3.i = b[i__4].i - 
				z__4.i;
			i__7 = i__;
			i__8 = i__ + j * x_dim1;
			z__5.r = d__[i__7].r * x[i__8].r - d__[i__7].i * x[
				i__8].i, z__5.i = d__[i__7].r * x[i__8].i + 
				d__[i__7].i * x[i__8].r;
			z__2.r = z__3.r - z__5.r, z__2.i = z__3.i - z__5.i;
			i__9 = i__;
			i__10 = i__ + 1 + j * x_dim1;
			z__6.r = du[i__9].r * x[i__10].r - du[i__9].i * x[
				i__10].i, z__6.i = du[i__9].r * x[i__10].i + 
				du[i__9].i * x[i__10].r;
			z__1.r = z__2.r - z__6.r, z__1.i = z__2.i - z__6.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L110: */
		    }
		}
/* L120: */
	    }
	} else if (lsame_(trans, "T")) {

/*           Compute B := B - A'*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__2.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__2.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__1.r = b[i__3].r - z__2.r, z__1.i = b[i__3].i - z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    i__4 = j * x_dim1 + 1;
		    z__3.r = d__[1].r * x[i__4].r - d__[1].i * x[i__4].i, 
			    z__3.i = d__[1].r * x[i__4].i + d__[1].i * x[i__4]
			    .r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    i__5 = j * x_dim1 + 2;
		    z__4.r = dl[1].r * x[i__5].r - dl[1].i * x[i__5].i, 
			    z__4.i = dl[1].r * x[i__5].i + dl[1].i * x[i__5]
			    .r;
		    z__1.r = z__2.r - z__4.r, z__1.i = z__2.i - z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    i__4 = *n - 1;
		    i__5 = *n - 1 + j * x_dim1;
		    z__3.r = du[i__4].r * x[i__5].r - du[i__4].i * x[i__5].i, 
			    z__3.i = du[i__4].r * x[i__5].i + du[i__4].i * x[
			    i__5].r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    i__6 = *n;
		    i__7 = *n + j * x_dim1;
		    z__4.r = d__[i__6].r * x[i__7].r - d__[i__6].i * x[i__7]
			    .i, z__4.i = d__[i__6].r * x[i__7].i + d__[i__6]
			    .i * x[i__7].r;
		    z__1.r = z__2.r - z__4.r, z__1.i = z__2.i - z__4.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			i__5 = i__ - 1;
			i__6 = i__ - 1 + j * x_dim1;
			z__4.r = du[i__5].r * x[i__6].r - du[i__5].i * x[i__6]
				.i, z__4.i = du[i__5].r * x[i__6].i + du[i__5]
				.i * x[i__6].r;
			z__3.r = b[i__4].r - z__4.r, z__3.i = b[i__4].i - 
				z__4.i;
			i__7 = i__;
			i__8 = i__ + j * x_dim1;
			z__5.r = d__[i__7].r * x[i__8].r - d__[i__7].i * x[
				i__8].i, z__5.i = d__[i__7].r * x[i__8].i + 
				d__[i__7].i * x[i__8].r;
			z__2.r = z__3.r - z__5.r, z__2.i = z__3.i - z__5.i;
			i__9 = i__;
			i__10 = i__ + 1 + j * x_dim1;
			z__6.r = dl[i__9].r * x[i__10].r - dl[i__9].i * x[
				i__10].i, z__6.i = dl[i__9].r * x[i__10].i + 
				dl[i__9].i * x[i__10].r;
			z__1.r = z__2.r - z__6.r, z__1.i = z__2.i - z__6.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L130: */
		    }
		}
/* L140: */
	    }
	} else if (lsame_(trans, "C")) {

/*           Compute B := B - A'*X */

	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		if (*n == 1) {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    d_cnjg(&z__3, &d__[1]);
		    i__4 = j * x_dim1 + 1;
		    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, z__2.i =
			     z__3.r * x[i__4].i + z__3.i * x[i__4].r;
		    z__1.r = b[i__3].r - z__2.r, z__1.i = b[i__3].i - z__2.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		} else {
		    i__2 = j * b_dim1 + 1;
		    i__3 = j * b_dim1 + 1;
		    d_cnjg(&z__4, &d__[1]);
		    i__4 = j * x_dim1 + 1;
		    z__3.r = z__4.r * x[i__4].r - z__4.i * x[i__4].i, z__3.i =
			     z__4.r * x[i__4].i + z__4.i * x[i__4].r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    d_cnjg(&z__6, &dl[1]);
		    i__5 = j * x_dim1 + 2;
		    z__5.r = z__6.r * x[i__5].r - z__6.i * x[i__5].i, z__5.i =
			     z__6.r * x[i__5].i + z__6.i * x[i__5].r;
		    z__1.r = z__2.r - z__5.r, z__1.i = z__2.i - z__5.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n + j * b_dim1;
		    i__3 = *n + j * b_dim1;
		    d_cnjg(&z__4, &du[*n - 1]);
		    i__4 = *n - 1 + j * x_dim1;
		    z__3.r = z__4.r * x[i__4].r - z__4.i * x[i__4].i, z__3.i =
			     z__4.r * x[i__4].i + z__4.i * x[i__4].r;
		    z__2.r = b[i__3].r - z__3.r, z__2.i = b[i__3].i - z__3.i;
		    d_cnjg(&z__6, &d__[*n]);
		    i__5 = *n + j * x_dim1;
		    z__5.r = z__6.r * x[i__5].r - z__6.i * x[i__5].i, z__5.i =
			     z__6.r * x[i__5].i + z__6.i * x[i__5].r;
		    z__1.r = z__2.r - z__5.r, z__1.i = z__2.i - z__5.i;
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = *n - 1;
		    for (i__ = 2; i__ <= i__2; ++i__) {
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + j * b_dim1;
			d_cnjg(&z__5, &du[i__ - 1]);
			i__5 = i__ - 1 + j * x_dim1;
			z__4.r = z__5.r * x[i__5].r - z__5.i * x[i__5].i, 
				z__4.i = z__5.r * x[i__5].i + z__5.i * x[i__5]
				.r;
			z__3.r = b[i__4].r - z__4.r, z__3.i = b[i__4].i - 
				z__4.i;
			d_cnjg(&z__7, &d__[i__]);
			i__6 = i__ + j * x_dim1;
			z__6.r = z__7.r * x[i__6].r - z__7.i * x[i__6].i, 
				z__6.i = z__7.r * x[i__6].i + z__7.i * x[i__6]
				.r;
			z__2.r = z__3.r - z__6.r, z__2.i = z__3.i - z__6.i;
			d_cnjg(&z__9, &dl[i__]);
			i__7 = i__ + 1 + j * x_dim1;
			z__8.r = z__9.r * x[i__7].r - z__9.i * x[i__7].i, 
				z__8.i = z__9.r * x[i__7].i + z__9.i * x[i__7]
				.r;
			z__1.r = z__2.r - z__8.r, z__1.i = z__2.i - z__8.i;
			b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L150: */
		    }
		}
/* L160: */
	    }
	}
    }
    return 0;

/*     End of ZLAGTM */

} /* zlagtm_ */
