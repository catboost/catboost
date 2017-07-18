/* claqps.f -- translated by f2c (version 20061008).
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
static complex c_b2 = {1.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int claqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, complex *a, integer *lda, integer *jpvt, complex *
	tau, real *vn1, real *vn2, complex *auxv, complex *f, integer *ldf)
{
    /* System generated locals */
    integer a_dim1, a_offset, f_dim1, f_offset, i__1, i__2, i__3;
    real r__1, r__2;
    complex q__1;

    /* Builtin functions */
    double sqrt(doublereal);
    void r_cnjg(complex *, complex *);
    double c_abs(complex *);
    integer i_nint(real *);

    /* Local variables */
    integer j, k, rk;
    complex akk;
    integer pvt;
    real temp, temp2, tol3z;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *), cgemv_(char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *, complex *, complex *, integer *), cswap_(
	    integer *, complex *, integer *, complex *, integer *);
    integer itemp;
    extern doublereal scnrm2_(integer *, complex *, integer *);
    extern /* Subroutine */ int clarfp_(integer *, complex *, complex *, 
	    integer *, complex *);
    extern doublereal slamch_(char *);
    integer lsticc;
    extern integer isamax_(integer *, real *, integer *);
    integer lastrk;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLAQPS computes a step of QR factorization with column pivoting */
/*  of a complex M-by-N matrix A by using Blas-3.  It tries to factorize */
/*  NB columns from A starting from the row OFFSET+1, and updates all */
/*  of the matrix with Blas-3 xGEMM. */

/*  In some cases, due to catastrophic cancellations, it cannot */
/*  factorize NB columns.  Hence, the actual number of factorized */
/*  columns is returned in KB. */

/*  Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A. M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A. N >= 0 */

/*  OFFSET  (input) INTEGER */
/*          The number of rows of A that have been factorized in */
/*          previous steps. */

/*  NB      (input) INTEGER */
/*          The number of columns to factorize. */

/*  KB      (output) INTEGER */
/*          The number of columns actually factorized. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, block A(OFFSET+1:M,1:KB) is the triangular */
/*          factor obtained and block A(1:OFFSET,1:N) has been */
/*          accordingly pivoted, but no factorized. */
/*          The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has */
/*          been updated. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  JPVT    (input/output) INTEGER array, dimension (N) */
/*          JPVT(I) = K <==> Column K of the full matrix A has been */
/*          permuted into position I in AP. */

/*  TAU     (output) COMPLEX array, dimension (KB) */
/*          The scalar factors of the elementary reflectors. */

/*  VN1     (input/output) REAL array, dimension (N) */
/*          The vector with the partial column norms. */

/*  VN2     (input/output) REAL array, dimension (N) */
/*          The vector with the exact column norms. */

/*  AUXV    (input/output) COMPLEX array, dimension (NB) */
/*          Auxiliar vector. */

/*  F       (input/output) COMPLEX array, dimension (LDF,NB) */
/*          Matrix F' = L*Y'*A. */

/*  LDF     (input) INTEGER */
/*          The leading dimension of the array F. LDF >= max(1,N). */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*    G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain */
/*    X. Sun, Computer Science Dept., Duke University, USA */

/*  Partial column norm updating strategy modified by */
/*    Z. Drmac and Z. Bujanovic, Dept. of Mathematics, */
/*    University of Zagreb, Croatia. */
/*    June 2006. */
/*  For more details see LAPACK Working Note 176. */
/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --jpvt;
    --tau;
    --vn1;
    --vn2;
    --auxv;
    f_dim1 = *ldf;
    f_offset = 1 + f_dim1;
    f -= f_offset;

    /* Function Body */
/* Computing MIN */
    i__1 = *m, i__2 = *n + *offset;
    lastrk = min(i__1,i__2);
    lsticc = 0;
    k = 0;
    tol3z = sqrt(slamch_("Epsilon"));

/*     Beginning of while loop. */

L10:
    if (k < *nb && lsticc == 0) {
	++k;
	rk = *offset + k;

/*        Determine ith pivot column and swap if necessary */

	i__1 = *n - k + 1;
	pvt = k - 1 + isamax_(&i__1, &vn1[k], &c__1);
	if (pvt != k) {
	    cswap_(m, &a[pvt * a_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &c__1);
	    i__1 = k - 1;
	    cswap_(&i__1, &f[pvt + f_dim1], ldf, &f[k + f_dim1], ldf);
	    itemp = jpvt[pvt];
	    jpvt[pvt] = jpvt[k];
	    jpvt[k] = itemp;
	    vn1[pvt] = vn1[k];
	    vn2[pvt] = vn2[k];
	}

/*        Apply previous Householder reflectors to column K: */
/*        A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'. */

	if (k > 1) {
	    i__1 = k - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = k + j * f_dim1;
		r_cnjg(&q__1, &f[k + j * f_dim1]);
		f[i__2].r = q__1.r, f[i__2].i = q__1.i;
/* L20: */
	    }
	    i__1 = *m - rk + 1;
	    i__2 = k - 1;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &i__1, &i__2, &q__1, &a[rk + a_dim1], lda, 
		    &f[k + f_dim1], ldf, &c_b2, &a[rk + k * a_dim1], &c__1);
	    i__1 = k - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = k + j * f_dim1;
		r_cnjg(&q__1, &f[k + j * f_dim1]);
		f[i__2].r = q__1.r, f[i__2].i = q__1.i;
/* L30: */
	    }
	}

/*        Generate elementary reflector H(k). */

	if (rk < *m) {
	    i__1 = *m - rk + 1;
	    clarfp_(&i__1, &a[rk + k * a_dim1], &a[rk + 1 + k * a_dim1], &
		    c__1, &tau[k]);
	} else {
	    clarfp_(&c__1, &a[rk + k * a_dim1], &a[rk + k * a_dim1], &c__1, &
		    tau[k]);
	}

	i__1 = rk + k * a_dim1;
	akk.r = a[i__1].r, akk.i = a[i__1].i;
	i__1 = rk + k * a_dim1;
	a[i__1].r = 1.f, a[i__1].i = 0.f;

/*        Compute Kth column of F: */

/*        Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K). */

	if (k < *n) {
	    i__1 = *m - rk + 1;
	    i__2 = *n - k;
	    cgemv_("Conjugate transpose", &i__1, &i__2, &tau[k], &a[rk + (k + 
		    1) * a_dim1], lda, &a[rk + k * a_dim1], &c__1, &c_b1, &f[
		    k + 1 + k * f_dim1], &c__1);
	}

/*        Padding F(1:K,K) with zeros. */

	i__1 = k;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j + k * f_dim1;
	    f[i__2].r = 0.f, f[i__2].i = 0.f;
/* L40: */
	}

/*        Incremental updating of F: */
/*        F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)' */
/*                    *A(RK:M,K). */

	if (k > 1) {
	    i__1 = *m - rk + 1;
	    i__2 = k - 1;
	    i__3 = k;
	    q__1.r = -tau[i__3].r, q__1.i = -tau[i__3].i;
	    cgemv_("Conjugate transpose", &i__1, &i__2, &q__1, &a[rk + a_dim1]
, lda, &a[rk + k * a_dim1], &c__1, &c_b1, &auxv[1], &c__1);

	    i__1 = k - 1;
	    cgemv_("No transpose", n, &i__1, &c_b2, &f[f_dim1 + 1], ldf, &
		    auxv[1], &c__1, &c_b2, &f[k * f_dim1 + 1], &c__1);
	}

/*        Update the current row of A: */
/*        A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'. */

	if (k < *n) {
	    i__1 = *n - k;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemm_("No transpose", "Conjugate transpose", &c__1, &i__1, &k, &
		    q__1, &a[rk + a_dim1], lda, &f[k + 1 + f_dim1], ldf, &
		    c_b2, &a[rk + (k + 1) * a_dim1], lda);
	}

/*        Update partial column norms. */

	if (rk < lastrk) {
	    i__1 = *n;
	    for (j = k + 1; j <= i__1; ++j) {
		if (vn1[j] != 0.f) {

/*                 NOTE: The following 4 lines follow from the analysis in */
/*                 Lapack Working Note 176. */

		    temp = c_abs(&a[rk + j * a_dim1]) / vn1[j];
/* Computing MAX */
		    r__1 = 0.f, r__2 = (temp + 1.f) * (1.f - temp);
		    temp = dmax(r__1,r__2);
/* Computing 2nd power */
		    r__1 = vn1[j] / vn2[j];
		    temp2 = temp * (r__1 * r__1);
		    if (temp2 <= tol3z) {
			vn2[j] = (real) lsticc;
			lsticc = j;
		    } else {
			vn1[j] *= sqrt(temp);
		    }
		}
/* L50: */
	    }
	}

	i__1 = rk + k * a_dim1;
	a[i__1].r = akk.r, a[i__1].i = akk.i;

/*        End of while loop. */

	goto L10;
    }
    *kb = k;
    rk = *offset + *kb;

/*     Apply the block reflector to the rest of the matrix: */
/*     A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - */
/*                         A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'. */

/* Computing MIN */
    i__1 = *n, i__2 = *m - *offset;
    if (*kb < min(i__1,i__2)) {
	i__1 = *m - rk;
	i__2 = *n - *kb;
	q__1.r = -1.f, q__1.i = -0.f;
	cgemm_("No transpose", "Conjugate transpose", &i__1, &i__2, kb, &q__1, 
		 &a[rk + 1 + a_dim1], lda, &f[*kb + 1 + f_dim1], ldf, &c_b2, &
		a[rk + 1 + (*kb + 1) * a_dim1], lda);
    }

/*     Recomputation of difficult columns. */

L60:
    if (lsticc > 0) {
	itemp = i_nint(&vn2[lsticc]);
	i__1 = *m - rk;
	vn1[lsticc] = scnrm2_(&i__1, &a[rk + 1 + lsticc * a_dim1], &c__1);

/*        NOTE: The computation of VN1( LSTICC ) relies on the fact that */
/*        SNRM2 does not fail on vectors with norm below the value of */
/*        SQRT(DLAMCH('S')) */

	vn2[lsticc] = vn1[lsticc];
	lsticc = itemp;
	goto L60;
    }

    return 0;

/*     End of CLAQPS */

} /* claqps_ */
