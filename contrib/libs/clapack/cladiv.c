/* cladiv.f -- translated by f2c (version 20061008).
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

/* Complex */ VOID cladiv_(complex * ret_val, complex *x, complex *y)
{
    /* System generated locals */
    real r__1, r__2, r__3, r__4;
    complex q__1;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    real zi, zr;
    extern /* Subroutine */ int sladiv_(real *, real *, real *, real *, real *
, real *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLADIV := X / Y, where X and Y are complex.  The computation of X / Y */
/*  will not overflow on an intermediary step unless the results */
/*  overflows. */

/*  Arguments */
/*  ========= */

/*  X       (input) COMPLEX */
/*  Y       (input) COMPLEX */
/*          The complex scalars X and Y. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    r__1 = x->r;
    r__2 = r_imag(x);
    r__3 = y->r;
    r__4 = r_imag(y);
    sladiv_(&r__1, &r__2, &r__3, &r__4, &zr, &zi);
    q__1.r = zr, q__1.i = zi;
     ret_val->r = q__1.r,  ret_val->i = q__1.i;

    return ;

/*     End of CLADIV */

} /* cladiv_ */
