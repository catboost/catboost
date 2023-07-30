/* snrm2sub.f -- translated by f2c (version 20200916).
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

/*     snrm2sub.f */

/*     The program is a fortran wrapper for snrm2. */
/*     Witten by Keita Teranishi.  2/11/1998 */

/* Subroutine */ int snrm2sub_(integer *n, real *x, integer *incx, real *nrm2)
{
    extern doublereal snrm2_(integer *, real *, integer *);



    /* Parameter adjustments */
    --x;

    /* Function Body */
    *nrm2 = snrm2_(n, &x[1], incx);
    return 0;
} /* snrm2sub_ */

