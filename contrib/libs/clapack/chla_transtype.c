/* chla_transtype.f -- translated by f2c (version 20061008).
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

/* Character */ VOID chla_transtype__(char *ret_val, ftnlen ret_val_len, 
	integer *trans)
{

/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     October 2008 */
/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine translates from a BLAST-specified integer constant to */
/*  the character string specifying a transposition operation. */

/*  CHLA_TRANSTYPE returns an CHARACTER*1.  If CHLA_TRANSTYPE is 'X', */
/*  then input is not an integer indicating a transposition operator. */
/*  Otherwise CHLA_TRANSTYPE returns the constant value corresponding to */
/*  TRANS. */

/*  Arguments */
/*  ========= */
/*  TRANS   (input) INTEGER */
/*          Specifies the form of the system of equations: */
/*          = BLAS_NO_TRANS   = 111 :  No Transpose */
/*          = BLAS_TRANS      = 112 :  Transpose */
/*          = BLAS_CONJ_TRANS = 113 :  Conjugate Transpose */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Executable Statements .. */
    if (*trans == 111) {
	*(unsigned char *)ret_val = 'N';
    } else if (*trans == 112) {
	*(unsigned char *)ret_val = 'T';
    } else if (*trans == 113) {
	*(unsigned char *)ret_val = 'C';
    } else {
	*(unsigned char *)ret_val = 'X';
    }
    return ;

/*     End of CHLA_TRANSTYPE */

} /* chla_transtype__ */
