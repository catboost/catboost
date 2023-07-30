/* zdotusub.f -- translated by f2c (version 20200916).
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

/*     zdotusub.f */

/*     The program is a fortran wrapper for zdotu. */
/*     Witten by Keita Teranishi.  2/11/1998 */

/* Subroutine */ int zdotusub_(integer *n, doublecomplex *x, integer *incx, 
	doublecomplex *y, integer *incy, doublecomplex *dotu)
{
    /* System generated locals */
    doublecomplex z__1;

    /* Local variables */
    extern /* Double Complex */ VOID zdotu_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);



    /* Parameter adjustments */
    --y;
    --x;

    /* Function Body */
    zdotu_(&z__1, n, &x[1], incx, &y[1], incy);
    dotu->r = z__1.r, dotu->i = z__1.i;
    return 0;
} /* zdotusub_ */

