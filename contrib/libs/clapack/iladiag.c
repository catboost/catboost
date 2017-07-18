/* iladiag.f -- translated by f2c (version 20061008).
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

integer iladiag_(char *diag)
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

/*  This subroutine translated from a character string specifying if a */
/*  matrix has unit diagonal or not to the relevant BLAST-specified */
/*  integer constant. */

/*  ILADIAG returns an INTEGER.  If ILADIAG < 0, then the input is not a */
/*  character indicating a unit or non-unit diagonal.  Otherwise ILADIAG */
/*  returns the constant value corresponding to DIAG. */

/*  Arguments */
/*  ========= */
/*  DIAG    (input) CHARACTER*1 */
/*          = 'N':  A is non-unit triangular; */
/*          = 'U':  A is unit triangular. */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    if (lsame_(diag, "N")) {
	ret_val = 131;
    } else if (lsame_(diag, "U")) {
	ret_val = 132;
    } else {
	ret_val = -1;
    }
    return ret_val;

/*     End of ILADIAG */

} /* iladiag_ */
