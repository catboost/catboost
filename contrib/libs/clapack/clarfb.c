/* clarfb.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {1.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int clarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, complex *v, integer *ldv, 
	complex *t, integer *ldt, complex *c__, integer *ldc, complex *work, 
	integer *ldwork)
{
    /* System generated locals */
    integer c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1, 
	    work_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *);
    extern logical lsame_(char *, char *);
    integer lastc;
    extern /* Subroutine */ int ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), ctrmm_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *);
    integer lastv;
    extern integer ilaclc_(integer *, integer *, complex *, integer *);
    extern /* Subroutine */ int clacgv_(integer *, complex *, integer *);
    extern integer ilaclr_(integer *, integer *, complex *, integer *);
    char transt[1];


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLARFB applies a complex block reflector H or its transpose H' to a */
/*  complex M-by-N matrix C, from either the left or the right. */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'L': apply H or H' from the Left */
/*          = 'R': apply H or H' from the Right */

/*  TRANS   (input) CHARACTER*1 */
/*          = 'N': apply H (No transpose) */
/*          = 'C': apply H' (Conjugate transpose) */

/*  DIRECT  (input) CHARACTER*1 */
/*          Indicates how H is formed from a product of elementary */
/*          reflectors */
/*          = 'F': H = H(1) H(2) . . . H(k) (Forward) */
/*          = 'B': H = H(k) . . . H(2) H(1) (Backward) */

/*  STOREV  (input) CHARACTER*1 */
/*          Indicates how the vectors which define the elementary */
/*          reflectors are stored: */
/*          = 'C': Columnwise */
/*          = 'R': Rowwise */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix C. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix C. */

/*  K       (input) INTEGER */
/*          The order of the matrix T (= the number of elementary */
/*          reflectors whose product defines the block reflector). */

/*  V       (input) COMPLEX array, dimension */
/*                                (LDV,K) if STOREV = 'C' */
/*                                (LDV,M) if STOREV = 'R' and SIDE = 'L' */
/*                                (LDV,N) if STOREV = 'R' and SIDE = 'R' */
/*          The matrix V. See further details. */

/*  LDV     (input) INTEGER */
/*          The leading dimension of the array V. */
/*          If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M); */
/*          if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N); */
/*          if STOREV = 'R', LDV >= K. */

/*  T       (input) COMPLEX array, dimension (LDT,K) */
/*          The triangular K-by-K matrix T in the representation of the */
/*          block reflector. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= K. */

/*  C       (input/output) COMPLEX array, dimension (LDC,N) */
/*          On entry, the M-by-N matrix C. */
/*          On exit, C is overwritten by H*C or H'*C or C*H or C*H'. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the array C. LDC >= max(1,M). */

/*  WORK    (workspace) COMPLEX array, dimension (LDWORK,K) */

/*  LDWORK  (input) INTEGER */
/*          The leading dimension of the array WORK. */
/*          If SIDE = 'L', LDWORK >= max(1,N); */
/*          if SIDE = 'R', LDWORK >= max(1,M). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    work_dim1 = *ldwork;
    work_offset = 1 + work_dim1;
    work -= work_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (lsame_(trans, "N")) {
	*(unsigned char *)transt = 'C';
    } else {
	*(unsigned char *)transt = 'N';
    }

    if (lsame_(storev, "C")) {

	if (lsame_(direct, "F")) {

/*           Let  V =  ( V1 )    (first K rows) */
/*                     ( V2 ) */
/*           where  V1  is unit lower triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclr_(m, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK) */

/*              W := C1' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[j + c_dim1], ldc, &work[j * work_dim1 
			    + 1], &c__1);
		    clacgv_(&lastc, &work[j * work_dim1 + 1], &c__1);
/* L10: */
		}

/*              W := W * V1 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2'*V2 */

		    i__1 = lastv - *k;
		    cgemm_("Conjugate transpose", "No transpose", &lastc, k, &
			    i__1, &c_b1, &c__[*k + 1 + c_dim1], ldc, &v[*k + 
			    1 + v_dim1], ldv, &c_b1, &work[work_offset], 
			    ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Upper", transt, "Non-unit", &lastc, k, &c_b1, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (*m > *k) {

/*                 C2 := C2 - V2 * W' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &i__1, &
			    lastc, k, &q__1, &v[*k + 1 + v_dim1], ldv, &work[
			    work_offset], ldwork, &c_b1, &c__[*k + 1 + c_dim1]
, ldc);
		}

/*              W := W * V1' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[v_offset], ldv, &work[work_offset]
, ldwork)
			;

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i - 
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L20: */
		    }
/* L30: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclr_(n, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK) */

/*              W := C1 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[j * c_dim1 + 1], &c__1, &work[j * 
			    work_dim1 + 1], &c__1);
/* L40: */
		}

/*              W := W * V1 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C2 * V2 */

		    i__1 = lastv - *k;
		    cgemm_("No transpose", "No transpose", &lastc, k, &i__1, &
			    c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k + 1 
			    + v_dim1], ldv, &c_b1, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Upper", trans, "Non-unit", &lastc, k, &c_b1, 
			&t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (lastv > *k) {

/*                 C2 := C2 - W * V2' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &lastc, &
			    i__1, k, &q__1, &work[work_offset], ldwork, &v[*k 
			    + 1 + v_dim1], ldv, &c_b1, &c__[(*k + 1) * c_dim1 
			    + 1], ldc);
		}

/*              W := W * V1' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[v_offset], ldv, &work[work_offset]
, ldwork)
			;

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L50: */
		    }
/* L60: */
		}
	    }

	} else {

/*           Let  V =  ( V1 ) */
/*                     ( V2 )    (last K rows) */
/*           where  V2  is unit upper triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclr_(m, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V  =  (C1'*V1 + C2'*V2)  (stored in WORK) */

/*              W := C2' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[lastv - *k + j + c_dim1], ldc, &work[
			    j * work_dim1 + 1], &c__1);
		    clacgv_(&lastc, &work[j * work_dim1 + 1], &c__1);
/* L70: */
		}

/*              W := W * V2 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1'*V1 */

		    i__1 = lastv - *k;
		    cgemm_("Conjugate transpose", "No transpose", &lastc, k, &
			    i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], 
			    ldv, &c_b1, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Lower", transt, "Non-unit", &lastc, k, &c_b1, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V * W' */

		if (lastv > *k) {

/*                 C1 := C1 - V1 * W' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &i__1, &
			    lastc, k, &q__1, &v[v_offset], ldv, &work[
			    work_offset], ldwork, &c_b1, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[lastv - *k + 1 + v_dim1], ldv, &
			work[work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = lastv - *k + j + i__ * c_dim1;
			i__4 = lastv - *k + j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i - 
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L80: */
		    }
/* L90: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclr_(n, k, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK) */

/*              W := C2 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[(lastv - *k + j) * c_dim1 + 1], &c__1, 
			     &work[j * work_dim1 + 1], &c__1);
/* L100: */
		}

/*              W := W * V2 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[lastv - *k + 1 + v_dim1], ldv, &work[
			work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1 * V1 */

		    i__1 = lastv - *k;
		    cgemm_("No transpose", "No transpose", &lastc, k, &i__1, &
			    c_b1, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b1, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Lower", trans, "Non-unit", &lastc, k, &c_b1, 
			&t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V' */

		if (lastv > *k) {

/*                 C1 := C1 - W * V1' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "Conjugate transpose", &lastc, &
			    i__1, k, &q__1, &work[work_offset], ldwork, &v[
			    v_offset], ldv, &c_b1, &c__[c_offset], ldc);
		}

/*              W := W * V2' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[lastv - *k + 1 + v_dim1], ldv, &
			work[work_offset], ldwork);

/*              C2 := C2 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + (lastv - *k + j) * c_dim1;
			i__4 = i__ + (lastv - *k + j) * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L110: */
		    }
/* L120: */
		}
	    }
	}

    } else if (lsame_(storev, "R")) {

	if (lsame_(direct, "F")) {

/*           Let  V =  ( V1  V2 )    (V1: first K columns) */
/*           where  V1  is unit upper triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclc_(k, m, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK) */

/*              W := C1' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[j + c_dim1], ldc, &work[j * work_dim1 
			    + 1], &c__1);
		    clacgv_(&lastc, &work[j * work_dim1 + 1], &c__1);
/* L130: */
		}

/*              W := W * V1' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[v_offset], ldv, &work[work_offset]
, ldwork)
			;
		if (lastv > *k) {

/*                 W := W + C2'*V2' */

		    i__1 = lastv - *k;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    lastc, k, &i__1, &c_b1, &c__[*k + 1 + c_dim1], 
			    ldc, &v[(*k + 1) * v_dim1 + 1], ldv, &c_b1, &work[
			    work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Upper", transt, "Non-unit", &lastc, k, &c_b1, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (lastv > *k) {

/*                 C2 := C2 - V2' * W' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, &lastc, k, &q__1, &v[(*k + 1) * v_dim1 + 1], 
			     ldv, &work[work_offset], ldwork, &c_b1, &c__[*k 
			    + 1 + c_dim1], ldc);
		}

/*              W := W * V1 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ * c_dim1;
			i__4 = j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i - 
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L140: */
		    }
/* L150: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclc_(k, n, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK) */

/*              W := C1 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[j * c_dim1 + 1], &c__1, &work[j * 
			    work_dim1 + 1], &c__1);
/* L160: */
		}

/*              W := W * V1' */

		ctrmm_("Right", "Upper", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[v_offset], ldv, &work[work_offset]
, ldwork)
			;
		if (lastv > *k) {

/*                 W := W + C2 * V2' */

		    i__1 = lastv - *k;
		    cgemm_("No transpose", "Conjugate transpose", &lastc, k, &
			    i__1, &c_b1, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[
			    (*k + 1) * v_dim1 + 1], ldv, &c_b1, &work[
			    work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Upper", trans, "Non-unit", &lastc, k, &c_b1, 
			&t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (lastv > *k) {

/*                 C2 := C2 - W * V2 */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "No transpose", &lastc, &i__1, k, &
			    q__1, &work[work_offset], ldwork, &v[(*k + 1) * 
			    v_dim1 + 1], ldv, &c_b1, &c__[(*k + 1) * c_dim1 + 
			    1], ldc);
		}

/*              W := W * V1 */

		ctrmm_("Right", "Upper", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[v_offset], ldv, &work[work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + j * c_dim1;
			i__4 = i__ + j * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L170: */
		    }
/* L180: */
		}

	    }

	} else {

/*           Let  V =  ( V1  V2 )    (V2: last K columns) */
/*           where  V2  is unit lower triangular. */

	    if (lsame_(side, "L")) {

/*              Form  H * C  or  H' * C  where  C = ( C1 ) */
/*                                                  ( C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclc_(k, m, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclc_(&lastv, n, &c__[c_offset], ldc);

/*              W := C' * V'  =  (C1'*V1' + C2'*V2') (stored in WORK) */

/*              W := C2' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[lastv - *k + j + c_dim1], ldc, &work[
			    j * work_dim1 + 1], &c__1);
		    clacgv_(&lastc, &work[j * work_dim1 + 1], &c__1);
/* L190: */
		}

/*              W := W * V2' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[(lastv - *k + 1) * v_dim1 + 1], 
			ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1'*V1' */

		    i__1 = lastv - *k;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    lastc, k, &i__1, &c_b1, &c__[c_offset], ldc, &v[
			    v_offset], ldv, &c_b1, &work[work_offset], ldwork);
		}

/*              W := W * T'  or  W * T */

		ctrmm_("Right", "Lower", transt, "Non-unit", &lastc, k, &c_b1, 
			 &t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - V' * W' */

		if (lastv > *k) {

/*                 C1 := C1 - V1' * W' */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("Conjugate transpose", "Conjugate transpose", &
			    i__1, &lastc, k, &q__1, &v[v_offset], ldv, &work[
			    work_offset], ldwork, &c_b1, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C2 := C2 - W' */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = lastv - *k + j + i__ * c_dim1;
			i__4 = lastv - *k + j + i__ * c_dim1;
			r_cnjg(&q__2, &work[i__ + j * work_dim1]);
			q__1.r = c__[i__4].r - q__2.r, q__1.i = c__[i__4].i - 
				q__2.i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L200: */
		    }
/* L210: */
		}

	    } else if (lsame_(side, "R")) {

/*              Form  C * H  or  C * H'  where  C = ( C1  C2 ) */

/* Computing MAX */
		i__1 = *k, i__2 = ilaclc_(k, n, &v[v_offset], ldv);
		lastv = max(i__1,i__2);
		lastc = ilaclr_(m, &lastv, &c__[c_offset], ldc);

/*              W := C * V'  =  (C1*V1' + C2*V2')  (stored in WORK) */

/*              W := C2 */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    ccopy_(&lastc, &c__[(lastv - *k + j) * c_dim1 + 1], &c__1, 
			     &work[j * work_dim1 + 1], &c__1);
/* L220: */
		}

/*              W := W * V2' */

		ctrmm_("Right", "Lower", "Conjugate transpose", "Unit", &
			lastc, k, &c_b1, &v[(lastv - *k + 1) * v_dim1 + 1], 
			ldv, &work[work_offset], ldwork);
		if (lastv > *k) {

/*                 W := W + C1 * V1' */

		    i__1 = lastv - *k;
		    cgemm_("No transpose", "Conjugate transpose", &lastc, k, &
			    i__1, &c_b1, &c__[c_offset], ldc, &v[v_offset], 
			    ldv, &c_b1, &work[work_offset], ldwork);
		}

/*              W := W * T  or  W * T' */

		ctrmm_("Right", "Lower", trans, "Non-unit", &lastc, k, &c_b1, 
			&t[t_offset], ldt, &work[work_offset], ldwork);

/*              C := C - W * V */

		if (lastv > *k) {

/*                 C1 := C1 - W * V1 */

		    i__1 = lastv - *k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemm_("No transpose", "No transpose", &lastc, &i__1, k, &
			    q__1, &work[work_offset], ldwork, &v[v_offset], 
			    ldv, &c_b1, &c__[c_offset], ldc);
		}

/*              W := W * V2 */

		ctrmm_("Right", "Lower", "No transpose", "Unit", &lastc, k, &
			c_b1, &v[(lastv - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);

/*              C1 := C1 - W */

		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = lastc;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = i__ + (lastv - *k + j) * c_dim1;
			i__4 = i__ + (lastv - *k + j) * c_dim1;
			i__5 = i__ + j * work_dim1;
			q__1.r = c__[i__4].r - work[i__5].r, q__1.i = c__[
				i__4].i - work[i__5].i;
			c__[i__3].r = q__1.r, c__[i__3].i = q__1.i;
/* L230: */
		    }
/* L240: */
		}

	    }

	}
    }

    return 0;

/*     End of CLARFB */

} /* clarfb_ */
