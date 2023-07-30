/* sdsdotsub.f -- translated by f2c (version 20200916).
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

/*     sdsdotsub.f */

/*     The program is a fortran wrapper for sdsdot. */
/*     Witten by Keita Teranishi.  2/11/1998 */

/* Subroutine */ int sdsdotsub_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *dot)
{
    extern doublereal sdsdot_(integer *, real *, integer *, real *, integer *)
	    ;



    /* Parameter adjustments */
    --y;
    --x;

    /* Function Body */
    *dot = sdsdot_(n, &x[1], incx, &y[1], incy);
    return 0;
} /* sdsdotsub_ */

