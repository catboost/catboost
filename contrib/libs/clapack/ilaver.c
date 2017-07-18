/* ilaver.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ilaver_(integer *vers_major__, integer *vers_minor__, 
	integer *vers_patch__)
{

/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2008 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine return the Lapack version. */

/*  Arguments */
/*  ========= */

/*  VERS_MAJOR   (output) INTEGER */
/*      return the lapack major version */
/*  VERS_MINOR   (output) INTEGER */
/*      return the lapack minor version from the major version */
/*  VERS_PATCH   (output) INTEGER */
/*      return the lapack patch version from the minor version */

/*     .. Executable Statements .. */

    *vers_major__ = 3;
    *vers_minor__ = 2;
    *vers_patch__ = 0;
/*  ===================================================================== */

    return 0;
} /* ilaver_ */
