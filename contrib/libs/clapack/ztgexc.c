/* ztgexc.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int ztgexc_(logical *wantq, logical *wantz, integer *n, 
	doublecomplex *a, integer *lda, doublecomplex *b, integer *ldb, 
	doublecomplex *q, integer *ldq, doublecomplex *z__, integer *ldz, 
	integer *ifst, integer *ilst, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, q_dim1, q_offset, z_dim1, 
	    z_offset, i__1;

    /* Local variables */
    integer here;
    extern /* Subroutine */ int ztgex2_(logical *, logical *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, integer *, 
	     integer *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTGEXC reorders the generalized Schur decomposition of a complex */
/*  matrix pair (A,B), using an unitary equivalence transformation */
/*  (A, B) := Q * (A, B) * Z', so that the diagonal block of (A, B) with */
/*  row index IFST is moved to row ILST. */

/*  (A, B) must be in generalized Schur canonical form, that is, A and */
/*  B are both upper triangular. */

/*  Optionally, the matrices Q and Z of generalized Schur vectors are */
/*  updated. */

/*         Q(in) * A(in) * Z(in)' = Q(out) * A(out) * Z(out)' */
/*         Q(in) * B(in) * Z(in)' = Q(out) * B(out) * Z(out)' */

/*  Arguments */
/*  ========= */

/*  WANTQ   (input) LOGICAL */
/*          .TRUE. : update the left transformation matrix Q; */
/*          .FALSE.: do not update Q. */

/*  WANTZ   (input) LOGICAL */
/*          .TRUE. : update the right transformation matrix Z; */
/*          .FALSE.: do not update Z. */

/*  N       (input) INTEGER */
/*          The order of the matrices A and B. N >= 0. */

/*  A       (input/output) COMPLEX*16 array, dimension (LDA,N) */
/*          On entry, the upper triangular matrix A in the pair (A, B). */
/*          On exit, the updated matrix A. */

/*  LDA     (input)  INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,N). */

/*  B       (input/output) COMPLEX*16 array, dimension (LDB,N) */
/*          On entry, the upper triangular matrix B in the pair (A, B). */
/*          On exit, the updated matrix B. */

/*  LDB     (input)  INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,N). */

/*  Q       (input/output) COMPLEX*16 array, dimension (LDZ,N) */
/*          On entry, if WANTQ = .TRUE., the unitary matrix Q. */
/*          On exit, the updated matrix Q. */
/*          If WANTQ = .FALSE., Q is not referenced. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q. LDQ >= 1; */
/*          If WANTQ = .TRUE., LDQ >= N. */

/*  Z       (input/output) COMPLEX*16 array, dimension (LDZ,N) */
/*          On entry, if WANTZ = .TRUE., the unitary matrix Z. */
/*          On exit, the updated matrix Z. */
/*          If WANTZ = .FALSE., Z is not referenced. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z. LDZ >= 1; */
/*          If WANTZ = .TRUE., LDZ >= N. */

/*  IFST    (input) INTEGER */
/*  ILST    (input/output) INTEGER */
/*          Specify the reordering of the diagonal blocks of (A, B). */
/*          The block with row index IFST is moved to row ILST, by a */
/*          sequence of swapping between adjacent blocks. */

/*  INFO    (output) INTEGER */
/*           =0:  Successful exit. */
/*           <0:  if INFO = -i, the i-th argument had an illegal value. */
/*           =1:  The transformed matrix pair (A, B) would be too far */
/*                from generalized Schur form; the problem is ill- */
/*                conditioned. (A, B) may have been partially reordered, */
/*                and ILST points to the first row of the current */
/*                position of the block being moved. */


/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/*     Umea University, S-901 87 Umea, Sweden. */

/*  [1] B. Kagstrom; A Direct Method for Reordering Eigenvalues in the */
/*      Generalized Real Schur Form of a Regular Matrix Pair (A, B), in */
/*      M.S. Moonen et al (eds), Linear Algebra for Large Scale and */
/*      Real-Time Applications, Kluwer Academic Publ. 1993, pp 195-218. */

/*  [2] B. Kagstrom and P. Poromaa; Computing Eigenspaces with Specified */
/*      Eigenvalues of a Regular Matrix Pair (A, B) and Condition */
/*      Estimation: Theory, Algorithms and Software, Report */
/*      UMINF - 94.04, Department of Computing Science, Umea University, */
/*      S-901 87 Umea, Sweden, 1994. Also as LAPACK Working Note 87. */
/*      To appear in Numerical Algorithms, 1996. */

/*  [3] B. Kagstrom and P. Poromaa, LAPACK-Style Algorithms and Software */
/*      for Solving the Generalized Sylvester Equation and Estimating the */
/*      Separation between Regular Matrix Pairs, Report UMINF - 93.23, */
/*      Department of Computing Science, Umea University, S-901 87 Umea, */
/*      Sweden, December 1993, Revised April 1994, Also as LAPACK working */
/*      Note 75. To appear in ACM Trans. on Math. Software, Vol 22, No 1, */
/*      1996. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Decode and test input arguments. */
    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    } else if (*ldq < 1 || *wantq && *ldq < max(1,*n)) {
	*info = -9;
    } else if (*ldz < 1 || *wantz && *ldz < max(1,*n)) {
	*info = -11;
    } else if (*ifst < 1 || *ifst > *n) {
	*info = -12;
    } else if (*ilst < 1 || *ilst > *n) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTGEXC", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 1) {
	return 0;
    }
    if (*ifst == *ilst) {
	return 0;
    }

    if (*ifst < *ilst) {

	here = *ifst;

L10:

/*        Swap with next one below */

	ztgex2_(wantq, wantz, n, &a[a_offset], lda, &b[b_offset], ldb, &q[
		q_offset], ldq, &z__[z_offset], ldz, &here, info);
	if (*info != 0) {
	    *ilst = here;
	    return 0;
	}
	++here;
	if (here < *ilst) {
	    goto L10;
	}
	--here;
    } else {
	here = *ifst - 1;

L20:

/*        Swap with next one above */

	ztgex2_(wantq, wantz, n, &a[a_offset], lda, &b[b_offset], ldb, &q[
		q_offset], ldq, &z__[z_offset], ldz, &here, info);
	if (*info != 0) {
	    *ilst = here;
	    return 0;
	}
	--here;
	if (here >= *ilst) {
	    goto L20;
	}
	++here;
    }
    *ilst = here;
    return 0;

/*     End of ZTGEXC */

} /* ztgexc_ */
