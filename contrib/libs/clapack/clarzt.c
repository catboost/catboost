/* clarzt.f -- translated by f2c (version 20061008).
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

/* Table of constant values */

static complex c_b1 = {0.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int clarzt_(char *direct, char *storev, integer *n, integer *
	k, complex *v, integer *ldv, complex *tau, complex *t, integer *ldt)
{
    /* System generated locals */
    integer t_dim1, t_offset, v_dim1, v_offset, i__1, i__2;
    complex q__1;

    /* Local variables */
    integer i__, j, info;
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
, complex *, integer *, complex *, integer *, complex *, complex *
, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrmv_(char *, char *, char *, integer *, 
	    complex *, integer *, complex *, integer *), clacgv_(integer *, complex *, integer *), xerbla_(char *, 
	     integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLARZT forms the triangular factor T of a complex block reflector */
/*  H of order > n, which is defined as a product of k elementary */
/*  reflectors. */

/*  If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular; */

/*  If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular. */

/*  If STOREV = 'C', the vector which defines the elementary reflector */
/*  H(i) is stored in the i-th column of the array V, and */

/*     H  =  I - V * T * V' */

/*  If STOREV = 'R', the vector which defines the elementary reflector */
/*  H(i) is stored in the i-th row of the array V, and */

/*     H  =  I - V' * T * V */

/*  Currently, only STOREV = 'R' and DIRECT = 'B' are supported. */

/*  Arguments */
/*  ========= */

/*  DIRECT  (input) CHARACTER*1 */
/*          Specifies the order in which the elementary reflectors are */
/*          multiplied to form the block reflector: */
/*          = 'F': H = H(1) H(2) . . . H(k) (Forward, not supported yet) */
/*          = 'B': H = H(k) . . . H(2) H(1) (Backward) */

/*  STOREV  (input) CHARACTER*1 */
/*          Specifies how the vectors which define the elementary */
/*          reflectors are stored (see also Further Details): */
/*          = 'C': columnwise                        (not supported yet) */
/*          = 'R': rowwise */

/*  N       (input) INTEGER */
/*          The order of the block reflector H. N >= 0. */

/*  K       (input) INTEGER */
/*          The order of the triangular factor T (= the number of */
/*          elementary reflectors). K >= 1. */

/*  V       (input/output) COMPLEX array, dimension */
/*                               (LDV,K) if STOREV = 'C' */
/*                               (LDV,N) if STOREV = 'R' */
/*          The matrix V. See further details. */

/*  LDV     (input) INTEGER */
/*          The leading dimension of the array V. */
/*          If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K. */

/*  TAU     (input) COMPLEX array, dimension (K) */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i). */

/*  T       (output) COMPLEX array, dimension (LDT,K) */
/*          The k by k triangular factor T of the block reflector. */
/*          If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is */
/*          lower triangular. The rest of the array is not used. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= K. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA */

/*  The shape of the matrix V and the storage of the vectors which define */
/*  the H(i) is best illustrated by the following example with n = 5 and */
/*  k = 3. The elements equal to 1 are not stored; the corresponding */
/*  array elements are modified but restored on exit. The rest of the */
/*  array is not used. */

/*  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R': */

/*                                              ______V_____ */
/*         ( v1 v2 v3 )                        /            \ */
/*         ( v1 v2 v3 )                      ( v1 v1 v1 v1 v1 . . . . 1 ) */
/*     V = ( v1 v2 v3 )                      ( v2 v2 v2 v2 v2 . . . 1   ) */
/*         ( v1 v2 v3 )                      ( v3 v3 v3 v3 v3 . . 1     ) */
/*         ( v1 v2 v3 ) */
/*            .  .  . */
/*            .  .  . */
/*            1  .  . */
/*               1  . */
/*                  1 */

/*  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R': */

/*                                                        ______V_____ */
/*            1                                          /            \ */
/*            .  1                           ( 1 . . . . v1 v1 v1 v1 v1 ) */
/*            .  .  1                        ( . 1 . . . v2 v2 v2 v2 v2 ) */
/*            .  .  .                        ( . . 1 . . v3 v3 v3 v3 v3 ) */
/*            .  .  . */
/*         ( v1 v2 v3 ) */
/*         ( v1 v2 v3 ) */
/*     V = ( v1 v2 v3 ) */
/*         ( v1 v2 v3 ) */
/*         ( v1 v2 v3 ) */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Check for currently supported options */

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    /* Function Body */
    info = 0;
    if (! lsame_(direct, "B")) {
	info = -1;
    } else if (! lsame_(storev, "R")) {
	info = -2;
    }
    if (info != 0) {
	i__1 = -info;
	xerbla_("CLARZT", &i__1);
	return 0;
    }

    for (i__ = *k; i__ >= 1; --i__) {
	i__1 = i__;
	if (tau[i__1].r == 0.f && tau[i__1].i == 0.f) {

/*           H(i)  =  I */

	    i__1 = *k;
	    for (j = i__; j <= i__1; ++j) {
		i__2 = j + i__ * t_dim1;
		t[i__2].r = 0.f, t[i__2].i = 0.f;
/* L10: */
	    }
	} else {

/*           general case */

	    if (i__ < *k) {

/*              T(i+1:k,i) = - tau(i) * V(i+1:k,1:n) * V(i,1:n)' */

		clacgv_(n, &v[i__ + v_dim1], ldv);
		i__1 = *k - i__;
		i__2 = i__;
		q__1.r = -tau[i__2].r, q__1.i = -tau[i__2].i;
		cgemv_("No transpose", &i__1, n, &q__1, &v[i__ + 1 + v_dim1], 
			ldv, &v[i__ + v_dim1], ldv, &c_b1, &t[i__ + 1 + i__ * 
			t_dim1], &c__1);
		clacgv_(n, &v[i__ + v_dim1], ldv);

/*              T(i+1:k,i) = T(i+1:k,i+1:k) * T(i+1:k,i) */

		i__1 = *k - i__;
		ctrmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__ + 1 
			+ (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ * t_dim1]
, &c__1);
	    }
	    i__1 = i__ + i__ * t_dim1;
	    i__2 = i__;
	    t[i__1].r = tau[i__2].r, t[i__1].i = tau[i__2].i;
	}
/* L20: */
    }
    return 0;

/*     End of CLARZT */

} /* clarzt_ */
