/* cpbtrf.f -- translated by f2c (version 20061008).
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
static integer c_n1 = -1;
static real c_b21 = -1.f;
static real c_b22 = 1.f;
static integer c__33 = 33;

/* Subroutine */ int cpbtrf_(char *uplo, integer *n, integer *kd, complex *ab, 
	 integer *ldab, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    complex q__1;

    /* Local variables */
    integer i__, j, i2, i3, ib, nb, ii, jj;
    complex work[1056]	/* was [33][32] */;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *), cherk_(char *, 
	    char *, integer *, integer *, real *, complex *, integer *, real *
, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int ctrsm_(char *, char *, char *, char *, 
	    integer *, integer *, complex *, complex *, integer *, complex *, 
	    integer *), cpbtf2_(char *, 
	    integer *, integer *, complex *, integer *, integer *), 
	    cpotf2_(char *, integer *, complex *, integer *, integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CPBTRF computes the Cholesky factorization of a complex Hermitian */
/*  positive definite band matrix A. */

/*  The factorization has the form */
/*     A = U**H * U,  if UPLO = 'U', or */
/*     A = L  * L**H,  if UPLO = 'L', */
/*  where U is an upper triangular matrix and L is lower triangular. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  KD      (input) INTEGER */
/*          The number of superdiagonals of the matrix A if UPLO = 'U', */
/*          or the number of subdiagonals if UPLO = 'L'.  KD >= 0. */

/*  AB      (input/output) COMPLEX array, dimension (LDAB,N) */
/*          On entry, the upper or lower triangle of the Hermitian band */
/*          matrix A, stored in the first KD+1 rows of the array.  The */
/*          j-th column of A is stored in the j-th column of the array AB */
/*          as follows: */
/*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd). */

/*          On exit, if INFO = 0, the triangular factor U or L from the */
/*          Cholesky factorization A = U**H*U or A = L*L**H of the band */
/*          matrix A, in the same storage format as A. */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KD+1. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, the leading minor of order i is not */
/*                positive definite, and the factorization could not be */
/*                completed. */

/*  Further Details */
/*  =============== */

/*  The band storage scheme is illustrated by the following example, when */
/*  N = 6, KD = 2, and UPLO = 'U': */

/*  On entry:                       On exit: */

/*      *    *   a13  a24  a35  a46      *    *   u13  u24  u35  u46 */
/*      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56 */
/*     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66 */

/*  Similarly, if UPLO = 'L' the format of A is as follows: */

/*  On entry:                       On exit: */

/*     a11  a22  a33  a44  a55  a66     l11  l22  l33  l44  l55  l66 */
/*     a21  a32  a43  a54  a65   *      l21  l32  l43  l54  l65   * */
/*     a31  a42  a53  a64   *    *      l31  l42  l53  l64   *    * */

/*  Array elements marked * are not used by the routine. */

/*  Contributed by */
/*  Peter Mayes and Giuseppe Radicati, IBM ECSEC, Rome, March 23, 1989 */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;

    /* Function Body */
    *info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*kd < 0) {
	*info = -3;
    } else if (*ldab < *kd + 1) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPBTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment */

    nb = ilaenv_(&c__1, "CPBTRF", uplo, n, kd, &c_n1, &c_n1);

/*     The block size must not exceed the semi-bandwidth KD, and must not */
/*     exceed the limit set by the size of the local array WORK. */

    nb = min(nb,32);

    if (nb <= 1 || nb > *kd) {

/*        Use unblocked code */

	cpbtf2_(uplo, n, kd, &ab[ab_offset], ldab, info);
    } else {

/*        Use blocked code */

	if (lsame_(uplo, "U")) {

/*           Compute the Cholesky factorization of a Hermitian band */
/*           matrix, given the upper triangle of the matrix in band */
/*           storage. */

/*           Zero the upper triangle of the work array. */

	    i__1 = nb;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = i__ + j * 33 - 34;
		    work[i__3].r = 0.f, work[i__3].i = 0.f;
/* L10: */
		}
/* L20: */
	    }

/*           Process the band matrix one diagonal block at a time. */

	    i__1 = *n;
	    i__2 = nb;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);

/*              Factorize the diagonal block */

		i__3 = *ldab - 1;
		cpotf2_(uplo, &ib, &ab[*kd + 1 + i__ * ab_dim1], &i__3, &ii);
		if (ii != 0) {
		    *info = i__ + ii - 1;
		    goto L150;
		}
		if (i__ + ib <= *n) {

/*                 Update the relevant part of the trailing submatrix. */
/*                 If A11 denotes the diagonal block which has just been */
/*                 factorized, then we need to update the remaining */
/*                 blocks in the diagram: */

/*                    A11   A12   A13 */
/*                          A22   A23 */
/*                                A33 */

/*                 The numbers of rows and columns in the partitioning */
/*                 are IB, I2, I3 respectively. The blocks A12, A22 and */
/*                 A23 are empty if IB = KD. The upper triangle of A13 */
/*                 lies outside the band. */

/* Computing MIN */
		    i__3 = *kd - ib, i__4 = *n - i__ - ib + 1;
		    i2 = min(i__3,i__4);
/* Computing MIN */
		    i__3 = ib, i__4 = *n - i__ - *kd + 1;
		    i3 = min(i__3,i__4);

		    if (i2 > 0) {

/*                    Update A12 */

			i__3 = *ldab - 1;
			i__4 = *ldab - 1;
			ctrsm_("Left", "Upper", "Conjugate transpose", "Non-"
				"unit", &ib, &i2, &c_b1, &ab[*kd + 1 + i__ * 
				ab_dim1], &i__3, &ab[*kd + 1 - ib + (i__ + ib)
				 * ab_dim1], &i__4);

/*                    Update A22 */

			i__3 = *ldab - 1;
			i__4 = *ldab - 1;
			cherk_("Upper", "Conjugate transpose", &i2, &ib, &
				c_b21, &ab[*kd + 1 - ib + (i__ + ib) * 
				ab_dim1], &i__3, &c_b22, &ab[*kd + 1 + (i__ + 
				ib) * ab_dim1], &i__4);
		    }

		    if (i3 > 0) {

/*                    Copy the lower triangle of A13 into the work array. */

			i__3 = i3;
			for (jj = 1; jj <= i__3; ++jj) {
			    i__4 = ib;
			    for (ii = jj; ii <= i__4; ++ii) {
				i__5 = ii + jj * 33 - 34;
				i__6 = ii - jj + 1 + (jj + i__ + *kd - 1) * 
					ab_dim1;
				work[i__5].r = ab[i__6].r, work[i__5].i = ab[
					i__6].i;
/* L30: */
			    }
/* L40: */
			}

/*                    Update A13 (in the work array). */

			i__3 = *ldab - 1;
			ctrsm_("Left", "Upper", "Conjugate transpose", "Non-"
				"unit", &ib, &i3, &c_b1, &ab[*kd + 1 + i__ * 
				ab_dim1], &i__3, work, &c__33);

/*                    Update A23 */

			if (i2 > 0) {
			    q__1.r = -1.f, q__1.i = -0.f;
			    i__3 = *ldab - 1;
			    i__4 = *ldab - 1;
			    cgemm_("Conjugate transpose", "No transpose", &i2, 
				     &i3, &ib, &q__1, &ab[*kd + 1 - ib + (i__ 
				    + ib) * ab_dim1], &i__3, work, &c__33, &
				    c_b1, &ab[ib + 1 + (i__ + *kd) * ab_dim1], 
				     &i__4);
			}

/*                    Update A33 */

			i__3 = *ldab - 1;
			cherk_("Upper", "Conjugate transpose", &i3, &ib, &
				c_b21, work, &c__33, &c_b22, &ab[*kd + 1 + (
				i__ + *kd) * ab_dim1], &i__3);

/*                    Copy the lower triangle of A13 back into place. */

			i__3 = i3;
			for (jj = 1; jj <= i__3; ++jj) {
			    i__4 = ib;
			    for (ii = jj; ii <= i__4; ++ii) {
				i__5 = ii - jj + 1 + (jj + i__ + *kd - 1) * 
					ab_dim1;
				i__6 = ii + jj * 33 - 34;
				ab[i__5].r = work[i__6].r, ab[i__5].i = work[
					i__6].i;
/* L50: */
			    }
/* L60: */
			}
		    }
		}
/* L70: */
	    }
	} else {

/*           Compute the Cholesky factorization of a Hermitian band */
/*           matrix, given the lower triangle of the matrix in band */
/*           storage. */

/*           Zero the lower triangle of the work array. */

	    i__2 = nb;
	    for (j = 1; j <= i__2; ++j) {
		i__1 = nb;
		for (i__ = j + 1; i__ <= i__1; ++i__) {
		    i__3 = i__ + j * 33 - 34;
		    work[i__3].r = 0.f, work[i__3].i = 0.f;
/* L80: */
		}
/* L90: */
	    }

/*           Process the band matrix one diagonal block at a time. */

	    i__2 = *n;
	    i__1 = nb;
	    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);

/*              Factorize the diagonal block */

		i__3 = *ldab - 1;
		cpotf2_(uplo, &ib, &ab[i__ * ab_dim1 + 1], &i__3, &ii);
		if (ii != 0) {
		    *info = i__ + ii - 1;
		    goto L150;
		}
		if (i__ + ib <= *n) {

/*                 Update the relevant part of the trailing submatrix. */
/*                 If A11 denotes the diagonal block which has just been */
/*                 factorized, then we need to update the remaining */
/*                 blocks in the diagram: */

/*                    A11 */
/*                    A21   A22 */
/*                    A31   A32   A33 */

/*                 The numbers of rows and columns in the partitioning */
/*                 are IB, I2, I3 respectively. The blocks A21, A22 and */
/*                 A32 are empty if IB = KD. The lower triangle of A31 */
/*                 lies outside the band. */

/* Computing MIN */
		    i__3 = *kd - ib, i__4 = *n - i__ - ib + 1;
		    i2 = min(i__3,i__4);
/* Computing MIN */
		    i__3 = ib, i__4 = *n - i__ - *kd + 1;
		    i3 = min(i__3,i__4);

		    if (i2 > 0) {

/*                    Update A21 */

			i__3 = *ldab - 1;
			i__4 = *ldab - 1;
			ctrsm_("Right", "Lower", "Conjugate transpose", "Non"
				"-unit", &i2, &ib, &c_b1, &ab[i__ * ab_dim1 + 
				1], &i__3, &ab[ib + 1 + i__ * ab_dim1], &i__4);

/*                    Update A22 */

			i__3 = *ldab - 1;
			i__4 = *ldab - 1;
			cherk_("Lower", "No transpose", &i2, &ib, &c_b21, &ab[
				ib + 1 + i__ * ab_dim1], &i__3, &c_b22, &ab[(
				i__ + ib) * ab_dim1 + 1], &i__4);
		    }

		    if (i3 > 0) {

/*                    Copy the upper triangle of A31 into the work array. */

			i__3 = ib;
			for (jj = 1; jj <= i__3; ++jj) {
			    i__4 = min(jj,i3);
			    for (ii = 1; ii <= i__4; ++ii) {
				i__5 = ii + jj * 33 - 34;
				i__6 = *kd + 1 - jj + ii + (jj + i__ - 1) * 
					ab_dim1;
				work[i__5].r = ab[i__6].r, work[i__5].i = ab[
					i__6].i;
/* L100: */
			    }
/* L110: */
			}

/*                    Update A31 (in the work array). */

			i__3 = *ldab - 1;
			ctrsm_("Right", "Lower", "Conjugate transpose", "Non"
				"-unit", &i3, &ib, &c_b1, &ab[i__ * ab_dim1 + 
				1], &i__3, work, &c__33);

/*                    Update A32 */

			if (i2 > 0) {
			    q__1.r = -1.f, q__1.i = -0.f;
			    i__3 = *ldab - 1;
			    i__4 = *ldab - 1;
			    cgemm_("No transpose", "Conjugate transpose", &i3, 
				     &i2, &ib, &q__1, work, &c__33, &ab[ib + 
				    1 + i__ * ab_dim1], &i__3, &c_b1, &ab[*kd 
				    + 1 - ib + (i__ + ib) * ab_dim1], &i__4);
			}

/*                    Update A33 */

			i__3 = *ldab - 1;
			cherk_("Lower", "No transpose", &i3, &ib, &c_b21, 
				work, &c__33, &c_b22, &ab[(i__ + *kd) * 
				ab_dim1 + 1], &i__3);

/*                    Copy the upper triangle of A31 back into place. */

			i__3 = ib;
			for (jj = 1; jj <= i__3; ++jj) {
			    i__4 = min(jj,i3);
			    for (ii = 1; ii <= i__4; ++ii) {
				i__5 = *kd + 1 - jj + ii + (jj + i__ - 1) * 
					ab_dim1;
				i__6 = ii + jj * 33 - 34;
				ab[i__5].r = work[i__6].r, ab[i__5].i = work[
					i__6].i;
/* L120: */
			    }
/* L130: */
			}
		    }
		}
/* L140: */
	    }
	}
    }
    return 0;

L150:
    return 0;

/*     End of CPBTRF */

} /* cpbtrf_ */
