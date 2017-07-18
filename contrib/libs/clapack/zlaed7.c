/* zlaed7.f -- translated by f2c (version 20061008).
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

static integer c__2 = 2;
static integer c__1 = 1;
static integer c_n1 = -1;

/* Subroutine */ int zlaed7_(integer *n, integer *cutpnt, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, doublereal *d__, 
	doublecomplex *q, integer *ldq, doublereal *rho, integer *indxq, 
	doublereal *qstore, integer *qptr, integer *prmptr, integer *perm, 
	integer *givptr, integer *givcol, doublereal *givnum, doublecomplex *
	work, doublereal *rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    integer i__, k, n1, n2, iq, iw, iz, ptr, indx, curr, indxc, indxp;
    extern /* Subroutine */ int dlaed9_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *), 
	    zlaed8_(integer *, integer *, integer *, doublecomplex *, integer 
	    *, doublereal *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublecomplex *, integer *, doublereal *, integer *, 
	     integer *, integer *, integer *, integer *, integer *, 
	    doublereal *, integer *), dlaeda_(integer *, integer *, integer *, 
	     integer *, integer *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *, 
	     integer *);
    integer idlmda;
    extern /* Subroutine */ int dlamrg_(integer *, integer *, doublereal *, 
	    integer *, integer *, integer *), xerbla_(char *, integer *), zlacrm_(integer *, integer *, doublecomplex *, integer *, 
	     doublereal *, integer *, doublecomplex *, integer *, doublereal *
);
    integer coltyp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLAED7 computes the updated eigensystem of a diagonal */
/*  matrix after modification by a rank-one symmetric matrix. This */
/*  routine is used only for the eigenproblem which requires all */
/*  eigenvalues and optionally eigenvectors of a dense or banded */
/*  Hermitian matrix that has been reduced to tridiagonal form. */

/*    T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out) */

/*    where Z = Q'u, u is a vector of length N with ones in the */
/*    CUTPNT and CUTPNT + 1 th elements and zeros elsewhere. */

/*     The eigenvectors of the original matrix are stored in Q, and the */
/*     eigenvalues are in D.  The algorithm consists of three stages: */

/*        The first stage consists of deflating the size of the problem */
/*        when there are multiple eigenvalues or if there is a zero in */
/*        the Z vector.  For each such occurence the dimension of the */
/*        secular equation problem is reduced by one.  This stage is */
/*        performed by the routine DLAED2. */

/*        The second stage consists of calculating the updated */
/*        eigenvalues. This is done by finding the roots of the secular */
/*        equation via the routine DLAED4 (as called by SLAED3). */
/*        This routine also calculates the eigenvectors of the current */
/*        problem. */

/*        The final stage consists of computing the updated eigenvectors */
/*        directly using the updated eigenvalues.  The eigenvectors for */
/*        the current problem are multiplied with the eigenvectors from */
/*        the overall problem. */

/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         The dimension of the symmetric tridiagonal matrix.  N >= 0. */

/*  CUTPNT (input) INTEGER */
/*         Contains the location of the last eigenvalue in the leading */
/*         sub-matrix.  min(1,N) <= CUTPNT <= N. */

/*  QSIZ   (input) INTEGER */
/*         The dimension of the unitary matrix used to reduce */
/*         the full matrix to tridiagonal form.  QSIZ >= N. */

/*  TLVLS  (input) INTEGER */
/*         The total number of merging levels in the overall divide and */
/*         conquer tree. */

/*  CURLVL (input) INTEGER */
/*         The current level in the overall merge routine, */
/*         0 <= curlvl <= tlvls. */

/*  CURPBM (input) INTEGER */
/*         The current problem in the current level in the overall */
/*         merge routine (counting from upper left to lower right). */

/*  D      (input/output) DOUBLE PRECISION array, dimension (N) */
/*         On entry, the eigenvalues of the rank-1-perturbed matrix. */
/*         On exit, the eigenvalues of the repaired matrix. */

/*  Q      (input/output) COMPLEX*16 array, dimension (LDQ,N) */
/*         On entry, the eigenvectors of the rank-1-perturbed matrix. */
/*         On exit, the eigenvectors of the repaired tridiagonal matrix. */

/*  LDQ    (input) INTEGER */
/*         The leading dimension of the array Q.  LDQ >= max(1,N). */

/*  RHO    (input) DOUBLE PRECISION */
/*         Contains the subdiagonal element used to create the rank-1 */
/*         modification. */

/*  INDXQ  (output) INTEGER array, dimension (N) */
/*         This contains the permutation which will reintegrate the */
/*         subproblem just solved back into sorted order, */
/*         ie. D( INDXQ( I = 1, N ) ) will be in ascending order. */

/*  IWORK  (workspace) INTEGER array, dimension (4*N) */

/*  RWORK  (workspace) DOUBLE PRECISION array, */
/*                                 dimension (3*N+2*QSIZ*N) */

/*  WORK   (workspace) COMPLEX*16 array, dimension (QSIZ*N) */

/*  QSTORE (input/output) DOUBLE PRECISION array, dimension (N**2+1) */
/*         Stores eigenvectors of submatrices encountered during */
/*         divide and conquer, packed together. QPTR points to */
/*         beginning of the submatrices. */

/*  QPTR   (input/output) INTEGER array, dimension (N+2) */
/*         List of indices pointing to beginning of submatrices stored */
/*         in QSTORE. The submatrices are numbered starting at the */
/*         bottom left of the divide and conquer tree, from left to */
/*         right and bottom to top. */

/*  PRMPTR (input) INTEGER array, dimension (N lg N) */
/*         Contains a list of pointers which indicate where in PERM a */
/*         level's permutation is stored.  PRMPTR(i+1) - PRMPTR(i) */
/*         indicates the size of the permutation and also the size of */
/*         the full, non-deflated problem. */

/*  PERM   (input) INTEGER array, dimension (N lg N) */
/*         Contains the permutations (from deflation and sorting) to be */
/*         applied to each eigenblock. */

/*  GIVPTR (input) INTEGER array, dimension (N lg N) */
/*         Contains a list of pointers which indicate where in GIVCOL a */
/*         level's Givens rotations are stored.  GIVPTR(i+1) - GIVPTR(i) */
/*         indicates the number of Givens rotations. */

/*  GIVCOL (input) INTEGER array, dimension (2, N lg N) */
/*         Each pair of numbers indicates a pair of columns to take place */
/*         in a Givens rotation. */

/*  GIVNUM (input) DOUBLE PRECISION array, dimension (2, N lg N) */
/*         Each number indicates the S value to be used in the */
/*         corresponding Givens rotation. */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  if INFO = 1, an eigenvalue did not converge */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --qstore;
    --qptr;
    --prmptr;
    --perm;
    --givptr;
    givcol -= 3;
    givnum -= 3;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

/*     IF( ICOMPQ.LT.0 .OR. ICOMPQ.GT.1 ) THEN */
/*        INFO = -1 */
/*     ELSE IF( N.LT.0 ) THEN */
    if (*n < 0) {
	*info = -1;
    } else if (min(1,*n) > *cutpnt || *n < *cutpnt) {
	*info = -2;
    } else if (*qsiz < *n) {
	*info = -3;
    } else if (*ldq < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLAED7", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     The following values are for bookkeeping purposes only.  They are */
/*     integer pointers which indicate the portion of the workspace */
/*     used by a particular array in DLAED2 and SLAED3. */

    iz = 1;
    idlmda = iz + *n;
    iw = idlmda + *n;
    iq = iw + *n;

    indx = 1;
    indxc = indx + *n;
    coltyp = indxc + *n;
    indxp = coltyp + *n;

/*     Form the z-vector which consists of the last row of Q_1 and the */
/*     first row of Q_2. */

    ptr = pow_ii(&c__2, tlvls) + 1;
    i__1 = *curlvl - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *tlvls - i__;
	ptr += pow_ii(&c__2, &i__2);
/* L10: */
    }
    curr = ptr + *curpbm;
    dlaeda_(n, tlvls, curlvl, curpbm, &prmptr[1], &perm[1], &givptr[1], &
	    givcol[3], &givnum[3], &qstore[1], &qptr[1], &rwork[iz], &rwork[
	    iz + *n], info);

/*     When solving the final problem, we no longer need the stored data, */
/*     so we will overwrite the data from this level onto the previously */
/*     used storage space. */

    if (*curlvl == *tlvls) {
	qptr[curr] = 1;
	prmptr[curr] = 1;
	givptr[curr] = 1;
    }

/*     Sort and Deflate eigenvalues. */

    zlaed8_(&k, n, qsiz, &q[q_offset], ldq, &d__[1], rho, cutpnt, &rwork[iz], 
	    &rwork[idlmda], &work[1], qsiz, &rwork[iw], &iwork[indxp], &iwork[
	    indx], &indxq[1], &perm[prmptr[curr]], &givptr[curr + 1], &givcol[
	    (givptr[curr] << 1) + 1], &givnum[(givptr[curr] << 1) + 1], info);
    prmptr[curr + 1] = prmptr[curr] + *n;
    givptr[curr + 1] += givptr[curr];

/*     Solve Secular Equation. */

    if (k != 0) {
	dlaed9_(&k, &c__1, &k, n, &d__[1], &rwork[iq], &k, rho, &rwork[idlmda]
, &rwork[iw], &qstore[qptr[curr]], &k, info);
	zlacrm_(qsiz, &k, &work[1], qsiz, &qstore[qptr[curr]], &k, &q[
		q_offset], ldq, &rwork[iq]);
/* Computing 2nd power */
	i__1 = k;
	qptr[curr + 1] = qptr[curr] + i__1 * i__1;
	if (*info != 0) {
	    return 0;
	}

/*     Prepare the INDXQ sorting premutation. */

	n1 = k;
	n2 = *n - k;
	dlamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &indxq[1]);
    } else {
	qptr[curr + 1] = qptr[curr];
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    indxq[i__] = i__;
/* L20: */
	}
    }

    return 0;

/*     End of ZLAED7 */

} /* zlaed7_ */
