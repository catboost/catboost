/* ilauplo.f -- translated by f2c (version 20061008).
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

integer ilauplo_(char *uplo)
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

/*  This subroutine translated from a character string specifying a */
/*  upper- or lower-triangular matrix to the relevant BLAST-specified */
/*  integer constant. */

/*  ILAUPLO returns an INTEGER.  If ILAUPLO < 0, then the input is not */
/*  a character indicating an upper- or lower-triangular matrix. */
/*  Otherwise ILAUPLO returns the constant value corresponding to UPLO. */

/*  Arguments */
/*  ========= */
/*  UPLO    (input) CHARACTER */
/*          = 'U':  A is upper triangular; */
/*          = 'L':  A is lower triangular. */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    if (lsame_(uplo, "U")) {
	ret_val = 121;
    } else if (lsame_(uplo, "L")) {
	ret_val = 122;
    } else {
	ret_val = -1;
    }
    return ret_val;

/*     End of ILAUPLO */

} /* ilauplo_ */
