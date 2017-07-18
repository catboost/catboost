/* zla_lin_berr.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int zla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	doublecomplex *res, doublereal *ayb, doublereal *berr)
{
    /* System generated locals */
    integer ayb_dim1, ayb_offset, res_dim1, res_offset, i__1, i__2, i__3, 
	    i__4;
    doublereal d__1, d__2, d__3;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    integer i__, j;
    doublereal tmp, safe1;
    extern doublereal dlamch_(char *);


/*     -- LAPACK routine (version 3.2.1)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- April 2009                                                   -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     ZLA_LIN_BERR computes componentwise relative backward error from */
/*     the formula */
/*         max(i) ( abs(R(i)) / ( abs(op(A_s))*abs(Y) + abs(B_s) )(i) ) */
/*     where abs(Z) is the componentwise absolute value of the matrix */
/*     or vector Z. */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     NZ      (input) INTEGER */
/*     We add (NZ+1)*SLAMCH( 'Safe minimum' ) to R(i) in the numerator to */
/*     guard against spuriously zero residuals. Default value is N. */

/*     NRHS    (input) INTEGER */
/*     The number of right hand sides, i.e., the number of columns */
/*     of the matrices AYB, RES, and BERR.  NRHS >= 0. */

/*     RES    (input) DOUBLE PRECISION array, dimension (N,NRHS) */
/*     The residual matrix, i.e., the matrix R in the relative backward */
/*     error formula above. */

/*     AYB    (input) DOUBLE PRECISION array, dimension (N, NRHS) */
/*     The denominator in the relative backward error formula above, i.e., */
/*     the matrix abs(op(A_s))*abs(Y) + abs(B_s). The matrices A, Y, and B */
/*     are from iterative refinement (see zla_gerfsx_extended.f). */

/*     RES    (output) COMPLEX*16 array, dimension (NRHS) */
/*     The componentwise relative backward error from the formula above. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Adding SAFE1 to the numerator guards against spuriously zero */
/*     residuals.  A similar safeguard is in the CLA_yyAMV routine used */
/*     to compute AYB. */

    /* Parameter adjustments */
    --berr;
    ayb_dim1 = *n;
    ayb_offset = 1 + ayb_dim1;
    ayb -= ayb_offset;
    res_dim1 = *n;
    res_offset = 1 + res_dim1;
    res -= res_offset;

    /* Function Body */
    safe1 = dlamch_("Safe minimum");
    safe1 = (*nz + 1) * safe1;
    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {
	berr[j] = 0.;
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (ayb[i__ + j * ayb_dim1] != 0.) {
		i__3 = i__ + j * res_dim1;
		d__3 = (d__1 = res[i__3].r, abs(d__1)) + (d__2 = d_imag(&res[
			i__ + j * res_dim1]), abs(d__2));
		z__3.r = d__3, z__3.i = 0.;
		z__2.r = safe1 + z__3.r, z__2.i = z__3.i;
		i__4 = i__ + j * ayb_dim1;
		z__1.r = z__2.r / ayb[i__4], z__1.i = z__2.i / ayb[i__4];
		tmp = z__1.r;
/* Computing MAX */
		d__1 = berr[j];
		berr[j] = max(d__1,tmp);
	    }

/*     If AYB is exactly 0.0 (and if computed by CLA_yyAMV), then we know */
/*     the true residual also must be exactly 0.0. */

	}
    }
    return 0;
} /* zla_lin_berr__ */
