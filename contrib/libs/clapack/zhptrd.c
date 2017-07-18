/* zhptrd.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b2 = {0.,0.};
static integer c__1 = 1;

/* Subroutine */ int zhptrd_(char *uplo, integer *n, doublecomplex *ap, 
	doublereal *d__, doublereal *e, doublecomplex *tau, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    doublereal d__1;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    integer i__, i1, ii, i1i1;
    doublecomplex taui;
    extern /* Subroutine */ int zhpr2_(char *, integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublecomplex *);
    doublecomplex alpha;
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    logical upper;
    extern /* Subroutine */ int zhpmv_(char *, integer *, doublecomplex *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    doublecomplex *, integer *), zaxpy_(integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *), xerbla_(char *, integer *), zlarfg_(integer *, 
	     doublecomplex *, doublecomplex *, integer *, doublecomplex *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZHPTRD reduces a complex Hermitian matrix A stored in packed form to */
/*  real symmetric tridiagonal form T by a unitary similarity */
/*  transformation: Q**H * A * Q = T. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  AP      (input/output) COMPLEX*16 array, dimension (N*(N+1)/2) */
/*          On entry, the upper or lower triangle of the Hermitian matrix */
/*          A, packed columnwise in a linear array.  The j-th column of A */
/*          is stored in the array AP as follows: */
/*          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/*          if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n. */
/*          On exit, if UPLO = 'U', the diagonal and first superdiagonal */
/*          of A are overwritten by the corresponding elements of the */
/*          tridiagonal matrix T, and the elements above the first */
/*          superdiagonal, with the array TAU, represent the unitary */
/*          matrix Q as a product of elementary reflectors; if UPLO */
/*          = 'L', the diagonal and first subdiagonal of A are over- */
/*          written by the corresponding elements of the tridiagonal */
/*          matrix T, and the elements below the first subdiagonal, with */
/*          the array TAU, represent the unitary matrix Q as a product */
/*          of elementary reflectors. See Further Details. */

/*  D       (output) DOUBLE PRECISION array, dimension (N) */
/*          The diagonal elements of the tridiagonal matrix T: */
/*          D(i) = A(i,i). */

/*  E       (output) DOUBLE PRECISION array, dimension (N-1) */
/*          The off-diagonal elements of the tridiagonal matrix T: */
/*          E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'. */

/*  TAU     (output) COMPLEX*16 array, dimension (N-1) */
/*          The scalar factors of the elementary reflectors (see Further */
/*          Details). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  If UPLO = 'U', the matrix Q is represented as a product of elementary */
/*  reflectors */

/*     Q = H(n-1) . . . H(2) H(1). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a complex scalar, and v is a complex vector with */
/*  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in AP, */
/*  overwriting A(1:i-1,i+1), and tau is stored in TAU(i). */

/*  If UPLO = 'L', the matrix Q is represented as a product of elementary */
/*  reflectors */

/*     Q = H(1) H(2) . . . H(n-1). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a complex scalar, and v is a complex vector with */
/*  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in AP, */
/*  overwriting A(i+2:n,i), and tau is stored in TAU(i). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters */

    /* Parameter adjustments */
    --tau;
    --e;
    --d__;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHPTRD", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 0) {
	return 0;
    }

    if (upper) {

/*        Reduce the upper triangle of A. */
/*        I1 is the index in AP of A(1,I+1). */

	i1 = *n * (*n - 1) / 2 + 1;
	i__1 = i1 + *n - 1;
	i__2 = i1 + *n - 1;
	d__1 = ap[i__2].r;
	ap[i__1].r = d__1, ap[i__1].i = 0.;
	for (i__ = *n - 1; i__ >= 1; --i__) {

/*           Generate elementary reflector H(i) = I - tau * v * v' */
/*           to annihilate A(1:i-1,i+1) */

	    i__1 = i1 + i__ - 1;
	    alpha.r = ap[i__1].r, alpha.i = ap[i__1].i;
	    zlarfg_(&i__, &alpha, &ap[i1], &c__1, &taui);
	    i__1 = i__;
	    e[i__1] = alpha.r;

	    if (taui.r != 0. || taui.i != 0.) {

/*              Apply H(i) from both sides to A(1:i,1:i) */

		i__1 = i1 + i__ - 1;
		ap[i__1].r = 1., ap[i__1].i = 0.;

/*              Compute  y := tau * A * v  storing y in TAU(1:i) */

		zhpmv_(uplo, &i__, &taui, &ap[1], &ap[i1], &c__1, &c_b2, &tau[
			1], &c__1);

/*              Compute  w := y - 1/2 * tau * (y'*v) * v */

		z__3.r = -.5, z__3.i = -0.;
		z__2.r = z__3.r * taui.r - z__3.i * taui.i, z__2.i = z__3.r * 
			taui.i + z__3.i * taui.r;
		zdotc_(&z__4, &i__, &tau[1], &c__1, &ap[i1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * 
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		zaxpy_(&i__, &alpha, &ap[i1], &c__1, &tau[1], &c__1);

/*              Apply the transformation as a rank-2 update: */
/*                 A := A - v * w' - w * v' */

		z__1.r = -1., z__1.i = -0.;
		zhpr2_(uplo, &i__, &z__1, &ap[i1], &c__1, &tau[1], &c__1, &ap[
			1]);

	    }
	    i__1 = i1 + i__ - 1;
	    i__2 = i__;
	    ap[i__1].r = e[i__2], ap[i__1].i = 0.;
	    i__1 = i__ + 1;
	    i__2 = i1 + i__;
	    d__[i__1] = ap[i__2].r;
	    i__1 = i__;
	    tau[i__1].r = taui.r, tau[i__1].i = taui.i;
	    i1 -= i__;
/* L10: */
	}
	d__[1] = ap[1].r;
    } else {

/*        Reduce the lower triangle of A. II is the index in AP of */
/*        A(i,i) and I1I1 is the index of A(i+1,i+1). */

	ii = 1;
	d__1 = ap[1].r;
	ap[1].r = d__1, ap[1].i = 0.;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i1i1 = ii + *n - i__ + 1;

/*           Generate elementary reflector H(i) = I - tau * v * v' */
/*           to annihilate A(i+2:n,i) */

	    i__2 = ii + 1;
	    alpha.r = ap[i__2].r, alpha.i = ap[i__2].i;
	    i__2 = *n - i__;
	    zlarfg_(&i__2, &alpha, &ap[ii + 2], &c__1, &taui);
	    i__2 = i__;
	    e[i__2] = alpha.r;

	    if (taui.r != 0. || taui.i != 0.) {

/*              Apply H(i) from both sides to A(i+1:n,i+1:n) */

		i__2 = ii + 1;
		ap[i__2].r = 1., ap[i__2].i = 0.;

/*              Compute  y := tau * A * v  storing y in TAU(i:n-1) */

		i__2 = *n - i__;
		zhpmv_(uplo, &i__2, &taui, &ap[i1i1], &ap[ii + 1], &c__1, &
			c_b2, &tau[i__], &c__1);

/*              Compute  w := y - 1/2 * tau * (y'*v) * v */

		z__3.r = -.5, z__3.i = -0.;
		z__2.r = z__3.r * taui.r - z__3.i * taui.i, z__2.i = z__3.r * 
			taui.i + z__3.i * taui.r;
		i__2 = *n - i__;
		zdotc_(&z__4, &i__2, &tau[i__], &c__1, &ap[ii + 1], &c__1);
		z__1.r = z__2.r * z__4.r - z__2.i * z__4.i, z__1.i = z__2.r * 
			z__4.i + z__2.i * z__4.r;
		alpha.r = z__1.r, alpha.i = z__1.i;
		i__2 = *n - i__;
		zaxpy_(&i__2, &alpha, &ap[ii + 1], &c__1, &tau[i__], &c__1);

/*              Apply the transformation as a rank-2 update: */
/*                 A := A - v * w' - w * v' */

		i__2 = *n - i__;
		z__1.r = -1., z__1.i = -0.;
		zhpr2_(uplo, &i__2, &z__1, &ap[ii + 1], &c__1, &tau[i__], &
			c__1, &ap[i1i1]);

	    }
	    i__2 = ii + 1;
	    i__3 = i__;
	    ap[i__2].r = e[i__3], ap[i__2].i = 0.;
	    i__2 = i__;
	    i__3 = ii;
	    d__[i__2] = ap[i__3].r;
	    i__2 = i__;
	    tau[i__2].r = taui.r, tau[i__2].i = taui.i;
	    ii = i1i1;
/* L20: */
	}
	i__1 = *n;
	i__2 = ii;
	d__[i__1] = ap[i__2].r;
    }

    return 0;

/*     End of ZHPTRD */

} /* zhptrd_ */
