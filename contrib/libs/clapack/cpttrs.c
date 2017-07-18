/* cpttrs.f -- translated by f2c (version 20061008).
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

static integer c__1 = 1;
static integer c_n1 = -1;

/* Subroutine */ int cpttrs_(char *uplo, integer *n, integer *nrhs, real *d__, 
	 complex *e, complex *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    integer j, jb, nb, iuplo;
    logical upper;
    extern /* Subroutine */ int cptts2_(integer *, integer *, integer *, real 
	    *, complex *, complex *, integer *), xerbla_(char *, integer *);
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

/*  CPTTRS solves a tridiagonal system of the form */
/*     A * X = B */
/*  using the factorization A = U'*D*U or A = L*D*L' computed by CPTTRF. */
/*  D is a diagonal matrix specified in the vector D, U (or L) is a unit */
/*  bidiagonal matrix whose superdiagonal (subdiagonal) is specified in */
/*  the vector E, and X and B are N by NRHS matrices. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies the form of the factorization and whether the */
/*          vector E is the superdiagonal of the upper bidiagonal factor */
/*          U or the subdiagonal of the lower bidiagonal factor L. */
/*          = 'U':  A = U'*D*U, E is the superdiagonal of U */
/*          = 'L':  A = L*D*L', E is the subdiagonal of L */

/*  N       (input) INTEGER */
/*          The order of the tridiagonal matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  D       (input) REAL array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          factorization A = U'*D*U or A = L*D*L'. */

/*  E       (input) COMPLEX array, dimension (N-1) */
/*          If UPLO = 'U', the (n-1) superdiagonal elements of the unit */
/*          bidiagonal factor U from the factorization A = U'*D*U. */
/*          If UPLO = 'L', the (n-1) subdiagonal elements of the unit */
/*          bidiagonal factor L from the factorization A = L*D*L'. */

/*  B       (input/output) REAL array, dimension (LDB,NRHS) */
/*          On entry, the right hand side vectors B for the system of */
/*          linear equations. */
/*          On exit, the solution vectors, X. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -k, the k-th argument had an illegal value */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments. */

    /* Parameter adjustments */
    --d__;
    --e;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = *(unsigned char *)uplo == 'U' || *(unsigned char *)uplo == 'u';
    if (! upper && ! (*(unsigned char *)uplo == 'L' || *(unsigned char *)uplo 
	    == 'l')) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CPTTRS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

/*     Determine the number of right-hand sides to solve at a time. */

    if (*nrhs == 1) {
	nb = 1;
    } else {
/* Computing MAX */
	i__1 = 1, i__2 = ilaenv_(&c__1, "CPTTRS", uplo, n, nrhs, &c_n1, &c_n1);
	nb = max(i__1,i__2);
    }

/*     Decode UPLO */

    if (upper) {
	iuplo = 1;
    } else {
	iuplo = 0;
    }

    if (nb >= *nrhs) {
	cptts2_(&iuplo, n, nrhs, &d__[1], &e[1], &b[b_offset], ldb);
    } else {
	i__1 = *nrhs;
	i__2 = nb;
	for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
/* Computing MIN */
	    i__3 = *nrhs - j + 1;
	    jb = min(i__3,nb);
	    cptts2_(&iuplo, n, &jb, &d__[1], &e[1], &b[j * b_dim1 + 1], ldb);
/* L10: */
	}
    }

    return 0;

/*     End of CPTTRS */

} /* cpttrs_ */
