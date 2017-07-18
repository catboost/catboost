/* cgtts2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int cgtts2_(integer *itrans, integer *n, integer *nrhs, 
	complex *dl, complex *d__, complex *du, complex *du2, integer *ipiv, 
	complex *b, integer *ldb)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5, i__6, i__7, i__8;
    complex q__1, q__2, q__3, q__4, q__5, q__6, q__7, q__8;

    /* Builtin functions */
    void c_div(complex *, complex *, complex *), r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j;
    complex temp;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CGTTS2 solves one of the systems of equations */
/*     A * X = B,  A**T * X = B,  or  A**H * X = B, */
/*  with a tridiagonal matrix A using the LU factorization computed */
/*  by CGTTRF. */

/*  Arguments */
/*  ========= */

/*  ITRANS  (input) INTEGER */
/*          Specifies the form of the system of equations. */
/*          = 0:  A * X = B     (No transpose) */
/*          = 1:  A**T * X = B  (Transpose) */
/*          = 2:  A**H * X = B  (Conjugate transpose) */

/*  N       (input) INTEGER */
/*          The order of the matrix A. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  DL      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) multipliers that define the matrix L from the */
/*          LU factorization of A. */

/*  D       (input) COMPLEX array, dimension (N) */
/*          The n diagonal elements of the upper triangular matrix U from */
/*          the LU factorization of A. */

/*  DU      (input) COMPLEX array, dimension (N-1) */
/*          The (n-1) elements of the first super-diagonal of U. */

/*  DU2     (input) COMPLEX array, dimension (N-2) */
/*          The (n-2) elements of the second super-diagonal of U. */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          The pivot indices; for 1 <= i <= n, row i of the matrix was */
/*          interchanged with row IPIV(i).  IPIV(i) will always be either */
/*          i or i+1; IPIV(i) = i indicates a row interchange was not */
/*          required. */

/*  B       (input/output) COMPLEX array, dimension (LDB,NRHS) */
/*          On entry, the matrix of right hand side vectors B. */
/*          On exit, B is overwritten by the solution vectors X. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    --du2;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

    if (*itrans == 0) {

/*        Solve A*X = B using the LU factorization of A, */
/*        overwriting each right hand side vector with its solution. */

	if (*nrhs <= 1) {
	    j = 1;
L10:

/*           Solve L*x = b. */

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		if (ipiv[i__] == i__) {
		    i__2 = i__ + 1 + j * b_dim1;
		    i__3 = i__ + 1 + j * b_dim1;
		    i__4 = i__;
		    i__5 = i__ + j * b_dim1;
		    q__2.r = dl[i__4].r * b[i__5].r - dl[i__4].i * b[i__5].i, 
			    q__2.i = dl[i__4].r * b[i__5].i + dl[i__4].i * b[
			    i__5].r;
		    q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - q__2.i;
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		} else {
		    i__2 = i__ + j * b_dim1;
		    temp.r = b[i__2].r, temp.i = b[i__2].i;
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__ + 1 + j * b_dim1;
		    b[i__2].r = b[i__3].r, b[i__2].i = b[i__3].i;
		    i__2 = i__ + 1 + j * b_dim1;
		    i__3 = i__;
		    i__4 = i__ + j * b_dim1;
		    q__2.r = dl[i__3].r * b[i__4].r - dl[i__3].i * b[i__4].i, 
			    q__2.i = dl[i__3].r * b[i__4].i + dl[i__3].i * b[
			    i__4].r;
		    q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		}
/* L20: */
	    }

/*           Solve U*x = b. */

	    i__1 = *n + j * b_dim1;
	    c_div(&q__1, &b[*n + j * b_dim1], &d__[*n]);
	    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    if (*n > 1) {
		i__1 = *n - 1 + j * b_dim1;
		i__2 = *n - 1 + j * b_dim1;
		i__3 = *n - 1;
		i__4 = *n + j * b_dim1;
		q__3.r = du[i__3].r * b[i__4].r - du[i__3].i * b[i__4].i, 
			q__3.i = du[i__3].r * b[i__4].i + du[i__3].i * b[i__4]
			.r;
		q__2.r = b[i__2].r - q__3.r, q__2.i = b[i__2].i - q__3.i;
		c_div(&q__1, &q__2, &d__[*n - 1]);
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    }
	    for (i__ = *n - 2; i__ >= 1; --i__) {
		i__1 = i__ + j * b_dim1;
		i__2 = i__ + j * b_dim1;
		i__3 = i__;
		i__4 = i__ + 1 + j * b_dim1;
		q__4.r = du[i__3].r * b[i__4].r - du[i__3].i * b[i__4].i, 
			q__4.i = du[i__3].r * b[i__4].i + du[i__3].i * b[i__4]
			.r;
		q__3.r = b[i__2].r - q__4.r, q__3.i = b[i__2].i - q__4.i;
		i__5 = i__;
		i__6 = i__ + 2 + j * b_dim1;
		q__5.r = du2[i__5].r * b[i__6].r - du2[i__5].i * b[i__6].i, 
			q__5.i = du2[i__5].r * b[i__6].i + du2[i__5].i * b[
			i__6].r;
		q__2.r = q__3.r - q__5.r, q__2.i = q__3.i - q__5.i;
		c_div(&q__1, &q__2, &d__[i__]);
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
/* L30: */
	    }
	    if (j < *nrhs) {
		++j;
		goto L10;
	    }
	} else {
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {

/*           Solve L*x = b. */

		i__2 = *n - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (ipiv[i__] == i__) {
			i__3 = i__ + 1 + j * b_dim1;
			i__4 = i__ + 1 + j * b_dim1;
			i__5 = i__;
			i__6 = i__ + j * b_dim1;
			q__2.r = dl[i__5].r * b[i__6].r - dl[i__5].i * b[i__6]
				.i, q__2.i = dl[i__5].r * b[i__6].i + dl[i__5]
				.i * b[i__6].r;
			q__1.r = b[i__4].r - q__2.r, q__1.i = b[i__4].i - 
				q__2.i;
			b[i__3].r = q__1.r, b[i__3].i = q__1.i;
		    } else {
			i__3 = i__ + j * b_dim1;
			temp.r = b[i__3].r, temp.i = b[i__3].i;
			i__3 = i__ + j * b_dim1;
			i__4 = i__ + 1 + j * b_dim1;
			b[i__3].r = b[i__4].r, b[i__3].i = b[i__4].i;
			i__3 = i__ + 1 + j * b_dim1;
			i__4 = i__;
			i__5 = i__ + j * b_dim1;
			q__2.r = dl[i__4].r * b[i__5].r - dl[i__4].i * b[i__5]
				.i, q__2.i = dl[i__4].r * b[i__5].i + dl[i__4]
				.i * b[i__5].r;
			q__1.r = temp.r - q__2.r, q__1.i = temp.i - q__2.i;
			b[i__3].r = q__1.r, b[i__3].i = q__1.i;
		    }
/* L40: */
		}

/*           Solve U*x = b. */

		i__2 = *n + j * b_dim1;
		c_div(&q__1, &b[*n + j * b_dim1], &d__[*n]);
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		if (*n > 1) {
		    i__2 = *n - 1 + j * b_dim1;
		    i__3 = *n - 1 + j * b_dim1;
		    i__4 = *n - 1;
		    i__5 = *n + j * b_dim1;
		    q__3.r = du[i__4].r * b[i__5].r - du[i__4].i * b[i__5].i, 
			    q__3.i = du[i__4].r * b[i__5].i + du[i__4].i * b[
			    i__5].r;
		    q__2.r = b[i__3].r - q__3.r, q__2.i = b[i__3].i - q__3.i;
		    c_div(&q__1, &q__2, &d__[*n - 1]);
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		}
		for (i__ = *n - 2; i__ >= 1; --i__) {
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__;
		    i__5 = i__ + 1 + j * b_dim1;
		    q__4.r = du[i__4].r * b[i__5].r - du[i__4].i * b[i__5].i, 
			    q__4.i = du[i__4].r * b[i__5].i + du[i__4].i * b[
			    i__5].r;
		    q__3.r = b[i__3].r - q__4.r, q__3.i = b[i__3].i - q__4.i;
		    i__6 = i__;
		    i__7 = i__ + 2 + j * b_dim1;
		    q__5.r = du2[i__6].r * b[i__7].r - du2[i__6].i * b[i__7]
			    .i, q__5.i = du2[i__6].r * b[i__7].i + du2[i__6]
			    .i * b[i__7].r;
		    q__2.r = q__3.r - q__5.r, q__2.i = q__3.i - q__5.i;
		    c_div(&q__1, &q__2, &d__[i__]);
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L50: */
		}
/* L60: */
	    }
	}
    } else if (*itrans == 1) {

/*        Solve A**T * X = B. */

	if (*nrhs <= 1) {
	    j = 1;
L70:

/*           Solve U**T * x = b. */

	    i__1 = j * b_dim1 + 1;
	    c_div(&q__1, &b[j * b_dim1 + 1], &d__[1]);
	    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    if (*n > 1) {
		i__1 = j * b_dim1 + 2;
		i__2 = j * b_dim1 + 2;
		i__3 = j * b_dim1 + 1;
		q__3.r = du[1].r * b[i__3].r - du[1].i * b[i__3].i, q__3.i = 
			du[1].r * b[i__3].i + du[1].i * b[i__3].r;
		q__2.r = b[i__2].r - q__3.r, q__2.i = b[i__2].i - q__3.i;
		c_div(&q__1, &q__2, &d__[2]);
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    }
	    i__1 = *n;
	    for (i__ = 3; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		i__4 = i__ - 1;
		i__5 = i__ - 1 + j * b_dim1;
		q__4.r = du[i__4].r * b[i__5].r - du[i__4].i * b[i__5].i, 
			q__4.i = du[i__4].r * b[i__5].i + du[i__4].i * b[i__5]
			.r;
		q__3.r = b[i__3].r - q__4.r, q__3.i = b[i__3].i - q__4.i;
		i__6 = i__ - 2;
		i__7 = i__ - 2 + j * b_dim1;
		q__5.r = du2[i__6].r * b[i__7].r - du2[i__6].i * b[i__7].i, 
			q__5.i = du2[i__6].r * b[i__7].i + du2[i__6].i * b[
			i__7].r;
		q__2.r = q__3.r - q__5.r, q__2.i = q__3.i - q__5.i;
		c_div(&q__1, &q__2, &d__[i__]);
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L80: */
	    }

/*           Solve L**T * x = b. */

	    for (i__ = *n - 1; i__ >= 1; --i__) {
		if (ipiv[i__] == i__) {
		    i__1 = i__ + j * b_dim1;
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__;
		    i__4 = i__ + 1 + j * b_dim1;
		    q__2.r = dl[i__3].r * b[i__4].r - dl[i__3].i * b[i__4].i, 
			    q__2.i = dl[i__3].r * b[i__4].i + dl[i__3].i * b[
			    i__4].r;
		    q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
		} else {
		    i__1 = i__ + 1 + j * b_dim1;
		    temp.r = b[i__1].r, temp.i = b[i__1].i;
		    i__1 = i__ + 1 + j * b_dim1;
		    i__2 = i__ + j * b_dim1;
		    i__3 = i__;
		    q__2.r = dl[i__3].r * temp.r - dl[i__3].i * temp.i, 
			    q__2.i = dl[i__3].r * temp.i + dl[i__3].i * 
			    temp.r;
		    q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
		    i__1 = i__ + j * b_dim1;
		    b[i__1].r = temp.r, b[i__1].i = temp.i;
		}
/* L90: */
	    }
	    if (j < *nrhs) {
		++j;
		goto L70;
	    }
	} else {
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {

/*           Solve U**T * x = b. */

		i__2 = j * b_dim1 + 1;
		c_div(&q__1, &b[j * b_dim1 + 1], &d__[1]);
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		if (*n > 1) {
		    i__2 = j * b_dim1 + 2;
		    i__3 = j * b_dim1 + 2;
		    i__4 = j * b_dim1 + 1;
		    q__3.r = du[1].r * b[i__4].r - du[1].i * b[i__4].i, 
			    q__3.i = du[1].r * b[i__4].i + du[1].i * b[i__4]
			    .r;
		    q__2.r = b[i__3].r - q__3.r, q__2.i = b[i__3].i - q__3.i;
		    c_div(&q__1, &q__2, &d__[2]);
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		}
		i__2 = *n;
		for (i__ = 3; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ + j * b_dim1;
		    i__5 = i__ - 1;
		    i__6 = i__ - 1 + j * b_dim1;
		    q__4.r = du[i__5].r * b[i__6].r - du[i__5].i * b[i__6].i, 
			    q__4.i = du[i__5].r * b[i__6].i + du[i__5].i * b[
			    i__6].r;
		    q__3.r = b[i__4].r - q__4.r, q__3.i = b[i__4].i - q__4.i;
		    i__7 = i__ - 2;
		    i__8 = i__ - 2 + j * b_dim1;
		    q__5.r = du2[i__7].r * b[i__8].r - du2[i__7].i * b[i__8]
			    .i, q__5.i = du2[i__7].r * b[i__8].i + du2[i__7]
			    .i * b[i__8].r;
		    q__2.r = q__3.r - q__5.r, q__2.i = q__3.i - q__5.i;
		    c_div(&q__1, &q__2, &d__[i__]);
		    b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L100: */
		}

/*           Solve L**T * x = b. */

		for (i__ = *n - 1; i__ >= 1; --i__) {
		    if (ipiv[i__] == i__) {
			i__2 = i__ + j * b_dim1;
			i__3 = i__ + j * b_dim1;
			i__4 = i__;
			i__5 = i__ + 1 + j * b_dim1;
			q__2.r = dl[i__4].r * b[i__5].r - dl[i__4].i * b[i__5]
				.i, q__2.i = dl[i__4].r * b[i__5].i + dl[i__4]
				.i * b[i__5].r;
			q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - 
				q__2.i;
			b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		    } else {
			i__2 = i__ + 1 + j * b_dim1;
			temp.r = b[i__2].r, temp.i = b[i__2].i;
			i__2 = i__ + 1 + j * b_dim1;
			i__3 = i__ + j * b_dim1;
			i__4 = i__;
			q__2.r = dl[i__4].r * temp.r - dl[i__4].i * temp.i, 
				q__2.i = dl[i__4].r * temp.i + dl[i__4].i * 
				temp.r;
			q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - 
				q__2.i;
			b[i__2].r = q__1.r, b[i__2].i = q__1.i;
			i__2 = i__ + j * b_dim1;
			b[i__2].r = temp.r, b[i__2].i = temp.i;
		    }
/* L110: */
		}
/* L120: */
	    }
	}
    } else {

/*        Solve A**H * X = B. */

	if (*nrhs <= 1) {
	    j = 1;
L130:

/*           Solve U**H * x = b. */

	    i__1 = j * b_dim1 + 1;
	    r_cnjg(&q__2, &d__[1]);
	    c_div(&q__1, &b[j * b_dim1 + 1], &q__2);
	    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    if (*n > 1) {
		i__1 = j * b_dim1 + 2;
		i__2 = j * b_dim1 + 2;
		r_cnjg(&q__4, &du[1]);
		i__3 = j * b_dim1 + 1;
		q__3.r = q__4.r * b[i__3].r - q__4.i * b[i__3].i, q__3.i = 
			q__4.r * b[i__3].i + q__4.i * b[i__3].r;
		q__2.r = b[i__2].r - q__3.r, q__2.i = b[i__2].i - q__3.i;
		r_cnjg(&q__5, &d__[2]);
		c_div(&q__1, &q__2, &q__5);
		b[i__1].r = q__1.r, b[i__1].i = q__1.i;
	    }
	    i__1 = *n;
	    for (i__ = 3; i__ <= i__1; ++i__) {
		i__2 = i__ + j * b_dim1;
		i__3 = i__ + j * b_dim1;
		r_cnjg(&q__5, &du[i__ - 1]);
		i__4 = i__ - 1 + j * b_dim1;
		q__4.r = q__5.r * b[i__4].r - q__5.i * b[i__4].i, q__4.i = 
			q__5.r * b[i__4].i + q__5.i * b[i__4].r;
		q__3.r = b[i__3].r - q__4.r, q__3.i = b[i__3].i - q__4.i;
		r_cnjg(&q__7, &du2[i__ - 2]);
		i__5 = i__ - 2 + j * b_dim1;
		q__6.r = q__7.r * b[i__5].r - q__7.i * b[i__5].i, q__6.i = 
			q__7.r * b[i__5].i + q__7.i * b[i__5].r;
		q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		r_cnjg(&q__8, &d__[i__]);
		c_div(&q__1, &q__2, &q__8);
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
/* L140: */
	    }

/*           Solve L**H * x = b. */

	    for (i__ = *n - 1; i__ >= 1; --i__) {
		if (ipiv[i__] == i__) {
		    i__1 = i__ + j * b_dim1;
		    i__2 = i__ + j * b_dim1;
		    r_cnjg(&q__3, &dl[i__]);
		    i__3 = i__ + 1 + j * b_dim1;
		    q__2.r = q__3.r * b[i__3].r - q__3.i * b[i__3].i, q__2.i =
			     q__3.r * b[i__3].i + q__3.i * b[i__3].r;
		    q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
		} else {
		    i__1 = i__ + 1 + j * b_dim1;
		    temp.r = b[i__1].r, temp.i = b[i__1].i;
		    i__1 = i__ + 1 + j * b_dim1;
		    i__2 = i__ + j * b_dim1;
		    r_cnjg(&q__3, &dl[i__]);
		    q__2.r = q__3.r * temp.r - q__3.i * temp.i, q__2.i = 
			    q__3.r * temp.i + q__3.i * temp.r;
		    q__1.r = b[i__2].r - q__2.r, q__1.i = b[i__2].i - q__2.i;
		    b[i__1].r = q__1.r, b[i__1].i = q__1.i;
		    i__1 = i__ + j * b_dim1;
		    b[i__1].r = temp.r, b[i__1].i = temp.i;
		}
/* L150: */
	    }
	    if (j < *nrhs) {
		++j;
		goto L130;
	    }
	} else {
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {

/*           Solve U**H * x = b. */

		i__2 = j * b_dim1 + 1;
		r_cnjg(&q__2, &d__[1]);
		c_div(&q__1, &b[j * b_dim1 + 1], &q__2);
		b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		if (*n > 1) {
		    i__2 = j * b_dim1 + 2;
		    i__3 = j * b_dim1 + 2;
		    r_cnjg(&q__4, &du[1]);
		    i__4 = j * b_dim1 + 1;
		    q__3.r = q__4.r * b[i__4].r - q__4.i * b[i__4].i, q__3.i =
			     q__4.r * b[i__4].i + q__4.i * b[i__4].r;
		    q__2.r = b[i__3].r - q__3.r, q__2.i = b[i__3].i - q__3.i;
		    r_cnjg(&q__5, &d__[2]);
		    c_div(&q__1, &q__2, &q__5);
		    b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		}
		i__2 = *n;
		for (i__ = 3; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * b_dim1;
		    i__4 = i__ + j * b_dim1;
		    r_cnjg(&q__5, &du[i__ - 1]);
		    i__5 = i__ - 1 + j * b_dim1;
		    q__4.r = q__5.r * b[i__5].r - q__5.i * b[i__5].i, q__4.i =
			     q__5.r * b[i__5].i + q__5.i * b[i__5].r;
		    q__3.r = b[i__4].r - q__4.r, q__3.i = b[i__4].i - q__4.i;
		    r_cnjg(&q__7, &du2[i__ - 2]);
		    i__6 = i__ - 2 + j * b_dim1;
		    q__6.r = q__7.r * b[i__6].r - q__7.i * b[i__6].i, q__6.i =
			     q__7.r * b[i__6].i + q__7.i * b[i__6].r;
		    q__2.r = q__3.r - q__6.r, q__2.i = q__3.i - q__6.i;
		    r_cnjg(&q__8, &d__[i__]);
		    c_div(&q__1, &q__2, &q__8);
		    b[i__3].r = q__1.r, b[i__3].i = q__1.i;
/* L160: */
		}

/*           Solve L**H * x = b. */

		for (i__ = *n - 1; i__ >= 1; --i__) {
		    if (ipiv[i__] == i__) {
			i__2 = i__ + j * b_dim1;
			i__3 = i__ + j * b_dim1;
			r_cnjg(&q__3, &dl[i__]);
			i__4 = i__ + 1 + j * b_dim1;
			q__2.r = q__3.r * b[i__4].r - q__3.i * b[i__4].i, 
				q__2.i = q__3.r * b[i__4].i + q__3.i * b[i__4]
				.r;
			q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - 
				q__2.i;
			b[i__2].r = q__1.r, b[i__2].i = q__1.i;
		    } else {
			i__2 = i__ + 1 + j * b_dim1;
			temp.r = b[i__2].r, temp.i = b[i__2].i;
			i__2 = i__ + 1 + j * b_dim1;
			i__3 = i__ + j * b_dim1;
			r_cnjg(&q__3, &dl[i__]);
			q__2.r = q__3.r * temp.r - q__3.i * temp.i, q__2.i = 
				q__3.r * temp.i + q__3.i * temp.r;
			q__1.r = b[i__3].r - q__2.r, q__1.i = b[i__3].i - 
				q__2.i;
			b[i__2].r = q__1.r, b[i__2].i = q__1.i;
			i__2 = i__ + j * b_dim1;
			b[i__2].r = temp.r, b[i__2].i = temp.i;
		    }
/* L170: */
		}
/* L180: */
	    }
	}
    }

/*     End of CGTTS2 */

    return 0;
} /* cgtts2_ */
