/* ilaprec.f -- translated by f2c (version 20061008).
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

integer ilaprec_(char *prec)
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

/*  This subroutine translated from a character string specifying an */
/*  intermediate precision to the relevant BLAST-specified integer */
/*  constant. */

/*  ILAPREC returns an INTEGER.  If ILAPREC < 0, then the input is not a */
/*  character indicating a supported intermediate precision.  Otherwise */
/*  ILAPREC returns the constant value corresponding to PREC. */

/*  Arguments */
/*  ========= */
/*  PREC   (input) CHARACTER*1 */
/*          Specifies the form of the system of equations: */
/*          = 'S':  Single */
/*          = 'D':  Double */
/*          = 'I':  Indigenous */
/*          = 'X', 'E':  Extra */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    if (lsame_(prec, "S")) {
	ret_val = 211;
    } else if (lsame_(prec, "D")) {
	ret_val = 212;
    } else if (lsame_(prec, "I")) {
	ret_val = 213;
    } else if (lsame_(prec, "X") || lsame_(prec, "E")) {
	ret_val = 214;
    } else {
	ret_val = -1;
    }
    return ret_val;

/*     End of ILAPREC */

} /* ilaprec_ */
