/* xerbla_array.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int xerbla_array__(char *srname_array__, integer *
	srname_len__, integer *info, ftnlen srname_array_len)
{
    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Builtin functions */
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer i_len(char *, ftnlen);

    /* Local variables */
    integer i__;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    char srname[32];


/*  -- LAPACK auxiliary routine (version 3.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     September 19, 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  XERBLA_ARRAY assists other languages in calling XERBLA, the LAPACK */
/*  and BLAS error handler.  Rather than taking a Fortran string argument */
/*  as the function's name, XERBLA_ARRAY takes an array of single */
/*  characters along with the array's length.  XERBLA_ARRAY then copies */
/*  up to 32 characters of that array into a Fortran string and passes */
/*  that to XERBLA.  If called with a non-positive SRNAME_LEN, */
/*  XERBLA_ARRAY will call XERBLA with a string of all blank characters. */

/*  Say some macro or other device makes XERBLA_ARRAY available to C99 */
/*  by a name lapack_xerbla and with a common Fortran calling convention. */
/*  Then a C99 program could invoke XERBLA via: */
/*     { */
/*       int flen = strlen(__func__); */
/*       lapack_xerbla(__func__, &flen, &info); */
/*     } */

/*  Providing XERBLA_ARRAY is not necessary for intercepting LAPACK */
/*  errors.  XERBLA_ARRAY calls XERBLA. */

/*  Arguments */
/*  ========= */

/*  SRNAME_ARRAY (input) CHARACTER(1) array, dimension (SRNAME_LEN) */
/*          The name of the routine which called XERBLA_ARRAY. */

/*  SRNAME_LEN (input) INTEGER */
/*          The length of the name in SRNAME_ARRAY. */

/*  INFO    (input) INTEGER */
/*          The position of the invalid parameter in the parameter list */
/*          of the calling routine. */

/* ===================================================================== */

/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    --srname_array__;

    /* Function Body */
    s_copy(srname, "", (ftnlen)32, (ftnlen)0);
/* Computing MIN */
    i__2 = *srname_len__, i__3 = i_len(srname, (ftnlen)32);
    i__1 = min(i__2,i__3);
    for (i__ = 1; i__ <= i__1; ++i__) {
	*(unsigned char *)&srname[i__ - 1] = *(unsigned char *)&
		srname_array__[i__];
    }
    xerbla_(srname, info);
    return 0;
} /* xerbla_array__ */
