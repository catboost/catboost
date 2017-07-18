/* ilatrans.f -- translated by f2c (version 20061008).
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

integer ilatrans_(char *trans)
{
    /* System generated locals */
    integer ret_val;

    /* Local variables */
    extern logical lsame_(char *, char *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     October 2008 */
/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine translates from a character string specifying a */
/*  transposition operation to the relevant BLAST-specified integer */
/*  constant. */

/*  ILATRANS returns an INTEGER.  If ILATRANS < 0, then the input is not */
/*  a character indicating a transposition operator.  Otherwise ILATRANS */
/*  returns the constant value corresponding to TRANS. */

/*  Arguments */
/*  ========= */
/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the form of the system of equations: */
/*          = 'N':  No transpose */
/*          = 'T':  Transpose */
/*          = 'C':  Conjugate transpose */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    if (lsame_(trans, "N")) {
	ret_val = 111;
    } else if (lsame_(trans, "T")) {
	ret_val = 112;
    } else if (lsame_(trans, "C")) {
	ret_val = 113;
    } else {
	ret_val = -1;
    }
    return ret_val;

/*     End of ILATRANS */

} /* ilatrans_ */
