/* zlatbs.f -- translated by f2c (version 20061008).
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
static doublereal c_b36 = .5;

/* Subroutine */ int zlatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, doublecomplex *ab, integer *ldab, 
	doublecomplex *x, doublereal *scale, doublereal *cnorm, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4, i__5;
    doublereal d__1, d__2, d__3, d__4;
    doublecomplex z__1, z__2, z__3, z__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer i__, j;
    doublereal xj, rec, tjj;
    integer jinc, jlen;
    doublereal xbnd;
    integer imax;
    doublereal tmax;
    doublecomplex tjjs;
    doublereal xmax, grow;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    integer maind;
    extern logical lsame_(char *, char *);
    doublereal tscal;
    doublecomplex uscal;
    integer jlast;
    doublecomplex csumj;
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    logical upper;
    extern /* Double Complex */ VOID zdotu_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    extern /* Subroutine */ int ztbsv_(char *, char *, char *, integer *, 
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *), zaxpy_(integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), dlabad_(
	    doublereal *, doublereal *);
    extern doublereal dlamch_(char *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *), zdscal_(
	    integer *, doublereal *, doublecomplex *, integer *);
    doublereal bignum;
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern /* Double Complex */ VOID zladiv_(doublecomplex *, doublecomplex *, 
	     doublecomplex *);
    logical notran;
    integer jfirst;
    extern doublereal dzasum_(integer *, doublecomplex *, integer *);
    doublereal smlnum;
    logical nounit;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLATBS solves one of the triangular systems */

/*     A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b, */

/*  with scaling to prevent overflow, where A is an upper or lower */
/*  triangular band matrix.  Here A' denotes the transpose of A, x and b */
/*  are n-element vectors, and s is a scaling factor, usually less than */
/*  or equal to 1, chosen so that the components of x will be less than */
/*  the overflow threshold.  If the unscaled problem will not cause */
/*  overflow, the Level 2 BLAS routine ZTBSV is called.  If the matrix A */
/*  is singular (A(j,j) = 0 for some j), then s is set to 0 and a */
/*  non-trivial solution to A*x = 0 is returned. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the matrix A is upper or lower triangular. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  TRANS   (input) CHARACTER*1 */
/*          Specifies the operation applied to A. */
/*          = 'N':  Solve A * x = s*b     (No transpose) */
/*          = 'T':  Solve A**T * x = s*b  (Transpose) */
/*          = 'C':  Solve A**H * x = s*b  (Conjugate transpose) */

/*  DIAG    (input) CHARACTER*1 */
/*          Specifies whether or not the matrix A is unit triangular. */
/*          = 'N':  Non-unit triangular */
/*          = 'U':  Unit triangular */

/*  NORMIN  (input) CHARACTER*1 */
/*          Specifies whether CNORM has been set or not. */
/*          = 'Y':  CNORM contains the column norms on entry */
/*          = 'N':  CNORM is not set on entry.  On exit, the norms will */
/*                  be computed and stored in CNORM. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  KD      (input) INTEGER */
/*          The number of subdiagonals or superdiagonals in the */
/*          triangular matrix A.  KD >= 0. */

/*  AB      (input) COMPLEX*16 array, dimension (LDAB,N) */
/*          The upper or lower triangular band matrix A, stored in the */
/*          first KD+1 rows of the array. The j-th column of A is stored */
/*          in the j-th column of the array AB as follows: */
/*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd). */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KD+1. */

/*  X       (input/output) COMPLEX*16 array, dimension (N) */
/*          On entry, the right hand side b of the triangular system. */
/*          On exit, X is overwritten by the solution vector x. */

/*  SCALE   (output) DOUBLE PRECISION */
/*          The scaling factor s for the triangular system */
/*             A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b. */
/*          If SCALE = 0, the matrix A is singular or badly scaled, and */
/*          the vector x is an exact or approximate solution to A*x = 0. */

/*  CNORM   (input or output) DOUBLE PRECISION array, dimension (N) */

/*          If NORMIN = 'Y', CNORM is an input argument and CNORM(j) */
/*          contains the norm of the off-diagonal part of the j-th column */
/*          of A.  If TRANS = 'N', CNORM(j) must be greater than or equal */
/*          to the infinity-norm, and if TRANS = 'T' or 'C', CNORM(j) */
/*          must be greater than or equal to the 1-norm. */

/*          If NORMIN = 'N', CNORM is an output argument and CNORM(j) */
/*          returns the 1-norm of the offdiagonal part of the j-th column */
/*          of A. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -k, the k-th argument had an illegal value */

/*  Further Details */
/*  ======= ======= */

/*  A rough bound on x is computed; if that is less than overflow, ZTBSV */
/*  is called, otherwise, specific code is used which checks for possible */
/*  overflow or divide-by-zero at every operation. */

/*  A columnwise scheme is used for solving A*x = b.  The basic algorithm */
/*  if A is lower triangular is */

/*       x[1:n] := b[1:n] */
/*       for j = 1, ..., n */
/*            x(j) := x(j) / A(j,j) */
/*            x[j+1:n] := x[j+1:n] - x(j) * A[j+1:n,j] */
/*       end */

/*  Define bounds on the components of x after j iterations of the loop: */
/*     M(j) = bound on x[1:j] */
/*     G(j) = bound on x[j+1:n] */
/*  Initially, let M(0) = 0 and G(0) = max{x(i), i=1,...,n}. */

/*  Then for iteration j+1 we have */
/*     M(j+1) <= G(j) / | A(j+1,j+1) | */
/*     G(j+1) <= G(j) + M(j+1) * | A[j+2:n,j+1] | */
/*            <= G(j) ( 1 + CNORM(j+1) / | A(j+1,j+1) | ) */

/*  where CNORM(j+1) is greater than or equal to the infinity-norm of */
/*  column j+1 of A, not counting the diagonal.  Hence */

/*     G(j) <= G(0) product ( 1 + CNORM(i) / | A(i,i) | ) */
/*                  1<=i<=j */
/*  and */

/*     |x(j)| <= ( G(0) / |A(j,j)| ) product ( 1 + CNORM(i) / |A(i,i)| ) */
/*                                   1<=i< j */

/*  Since |x(j)| <= M(j), we use the Level 2 BLAS routine ZTBSV if the */
/*  reciprocal of the largest M(j), j=1,..,n, is larger than */
/*  max(underflow, 1/overflow). */

/*  The bound on x(j) is also used to determine when a step in the */
/*  columnwise method can be performed without fear of overflow.  If */
/*  the computed bound is greater than a large constant, x is scaled to */
/*  prevent overflow, but if the bound overflows, x is set to 0, x(j) to */
/*  1, and scale to 0, and a non-trivial solution to A*x = 0 is found. */

/*  Similarly, a row-wise scheme is used to solve A**T *x = b  or */
/*  A**H *x = b.  The basic algorithm for A upper triangular is */

/*       for j = 1, ..., n */
/*            x(j) := ( b(j) - A[1:j-1,j]' * x[1:j-1] ) / A(j,j) */
/*       end */

/*  We simultaneously compute two bounds */
/*       G(j) = bound on ( b(i) - A[1:i-1,i]' * x[1:i-1] ), 1<=i<=j */
/*       M(j) = bound on x(i), 1<=i<=j */

/*  The initial values are G(0) = 0, M(0) = max{b(i), i=1,..,n}, and we */
/*  add the constraint G(j) >= G(j-1) and M(j) >= M(j-1) for j >= 1. */
/*  Then the bound on x(j) is */

/*       M(j) <= M(j-1) * ( 1 + CNORM(j) ) / | A(j,j) | */

/*            <= M(0) * product ( ( 1 + CNORM(i) ) / |A(i,i)| ) */
/*                      1<=i<=j */

/*  and we can safely call ZTBSV if 1/M(n) and 1/G(n) are both greater */
/*  than max(underflow, 1/overflow). */

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
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --x;
    --cnorm;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    notran = lsame_(trans, "N");
    nounit = lsame_(diag, "N");

/*     Test the input parameters. */

    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T") && ! 
	    lsame_(trans, "C")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (! lsame_(normin, "Y") && ! lsame_(normin, 
	     "N")) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*kd < 0) {
	*info = -6;
    } else if (*ldab < *kd + 1) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLATBS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine machine dependent parameters to control overflow. */

    smlnum = dlamch_("Safe minimum");
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    smlnum /= dlamch_("Precision");
    bignum = 1. / smlnum;
    *scale = 1.;

    if (lsame_(normin, "N")) {

/*        Compute the 1-norm of each column, not including the diagonal. */

	if (upper) {

/*           A is upper triangular. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		i__2 = *kd, i__3 = j - 1;
		jlen = min(i__2,i__3);
		cnorm[j] = dzasum_(&jlen, &ab[*kd + 1 - jlen + j * ab_dim1], &
			c__1);
/* L10: */
	    }
	} else {

/*           A is lower triangular. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		i__2 = *kd, i__3 = *n - j;
		jlen = min(i__2,i__3);
		if (jlen > 0) {
		    cnorm[j] = dzasum_(&jlen, &ab[j * ab_dim1 + 2], &c__1);
		} else {
		    cnorm[j] = 0.;
		}
/* L20: */
	    }
	}
    }

/*     Scale the column norms by TSCAL if the maximum element in CNORM is */
/*     greater than BIGNUM/2. */

    imax = idamax_(n, &cnorm[1], &c__1);
    tmax = cnorm[imax];
    if (tmax <= bignum * .5) {
	tscal = 1.;
    } else {
	tscal = .5 / (smlnum * tmax);
	dscal_(n, &tscal, &cnorm[1], &c__1);
    }

/*     Compute a bound on the computed solution vector to see if the */
/*     Level 2 BLAS routine ZTBSV can be used. */

    xmax = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__2 = j;
	d__3 = xmax, d__4 = (d__1 = x[i__2].r / 2., abs(d__1)) + (d__2 = 
		d_imag(&x[j]) / 2., abs(d__2));
	xmax = max(d__3,d__4);
/* L30: */
    }
    xbnd = xmax;
    if (notran) {

/*        Compute the growth in A * x = b. */

	if (upper) {
	    jfirst = *n;
	    jlast = 1;
	    jinc = -1;
	    maind = *kd + 1;
	} else {
	    jfirst = 1;
	    jlast = *n;
	    jinc = 1;
	    maind = 1;
	}

	if (tscal != 1.) {
	    grow = 0.;
	    goto L60;
	}

	if (nounit) {

/*           A is non-unit triangular. */

/*           Compute GROW = 1/G(j) and XBND = 1/M(j). */
/*           Initially, G(0) = max{x(i), i=1,...,n}. */

	    grow = .5 / max(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

		i__3 = maind + j * ab_dim1;
		tjjs.r = ab[i__3].r, tjjs.i = ab[i__3].i;
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));

		if (tjj >= smlnum) {

/*                 M(j) = G(j-1) / abs(A(j,j)) */

/* Computing MIN */
		    d__1 = xbnd, d__2 = min(1.,tjj) * grow;
		    xbnd = min(d__1,d__2);
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.;
		}

		if (tjj + cnorm[j] >= smlnum) {

/*                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) ) */

		    grow *= tjj / (tjj + cnorm[j]);
		} else {

/*                 G(j) could overflow, set GROW to 0. */

		    grow = 0.;
		}
/* L40: */
	    }
	    grow = xbnd;
	} else {

/*           A is unit triangular. */

/*           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}. */

/* Computing MIN */
	    d__1 = 1., d__2 = .5 / max(xbnd,smlnum);
	    grow = min(d__1,d__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

/*              G(j) = G(j-1)*( 1 + CNORM(j) ) */

		grow *= 1. / (cnorm[j] + 1.);
/* L50: */
	    }
	}
L60:

	;
    } else {

/*        Compute the growth in A**T * x = b  or  A**H * x = b. */

	if (upper) {
	    jfirst = 1;
	    jlast = *n;
	    jinc = 1;
	    maind = *kd + 1;
	} else {
	    jfirst = *n;
	    jlast = 1;
	    jinc = -1;
	    maind = 1;
	}

	if (tscal != 1.) {
	    grow = 0.;
	    goto L90;
	}

	if (nounit) {

/*           A is non-unit triangular. */

/*           Compute GROW = 1/G(j) and XBND = 1/M(j). */
/*           Initially, M(0) = max{x(i), i=1,...,n}. */

	    grow = .5 / max(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) ) */

		xj = cnorm[j] + 1.;
/* Computing MIN */
		d__1 = grow, d__2 = xbnd / xj;
		grow = min(d__1,d__2);

		i__3 = maind + j * ab_dim1;
		tjjs.r = ab[i__3].r, tjjs.i = ab[i__3].i;
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));

		if (tjj >= smlnum) {

/*                 M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j)) */

		    if (xj > tjj) {
			xbnd *= tjj / xj;
		    }
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.;
		}
/* L70: */
	    }
	    grow = min(grow,xbnd);
	} else {

/*           A is unit triangular. */

/*           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}. */

/* Computing MIN */
	    d__1 = 1., d__2 = .5 / max(xbnd,smlnum);
	    grow = min(d__1,d__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = ( 1 + CNORM(j) )*G(j-1) */

		xj = cnorm[j] + 1.;
		grow /= xj;
/* L80: */
	    }
	}
L90:
	;
    }

    if (grow * tscal > smlnum) {

/*        Use the Level 2 BLAS solve if the reciprocal of the bound on */
/*        elements of X is not too small. */

	ztbsv_(uplo, trans, diag, n, kd, &ab[ab_offset], ldab, &x[1], &c__1);
    } else {

/*        Use a Level 1 BLAS solve, scaling intermediate results. */

	if (xmax > bignum * .5) {

/*           Scale X so that its components are less than or equal to */
/*           BIGNUM in absolute value. */

	    *scale = bignum * .5 / xmax;
	    zdscal_(n, scale, &x[1], &c__1);
	    xmax = bignum;
	} else {
	    xmax *= 2.;
	}

	if (notran) {

/*           Solve A * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) / A(j,j), scaling x if necessary. */

		i__3 = j;
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]), 
			abs(d__2));
		if (nounit) {
		    i__3 = maind + j * ab_dim1;
		    z__1.r = tscal * ab[i__3].r, z__1.i = tscal * ab[i__3].i;
		    tjjs.r = z__1.r, tjjs.i = z__1.i;
		} else {
		    tjjs.r = tscal, tjjs.i = 0.;
		    if (tscal == 1.) {
			goto L110;
		    }
		}
		tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), abs(
			d__2));
		if (tjj > smlnum) {

/*                    abs(A(j,j)) > SMLNUM: */

		    if (tjj < 1.) {
			if (xj > tjj * bignum) {

/*                          Scale x by 1/b(j). */

			    rec = 1. / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    i__3 = j;
		    zladiv_(&z__1, &x[j], &tjjs);
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		} else if (tjj > 0.) {

/*                    0 < abs(A(j,j)) <= SMLNUM: */

		    if (xj > tjj * bignum) {

/*                       Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM */
/*                       to avoid overflow when dividing by A(j,j). */

			rec = tjj * bignum / xj;
			if (cnorm[j] > 1.) {

/*                          Scale by 1/CNORM(j) to avoid overflow when */
/*                          multiplying x(j) times column j. */

			    rec /= cnorm[j];
			}
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		    i__3 = j;
		    zladiv_(&z__1, &x[j], &tjjs);
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		} else {

/*                    A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                    scale = 0, and compute a solution to A*x = 0. */

		    i__3 = *n;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			x[i__4].r = 0., x[i__4].i = 0.;
/* L100: */
		    }
		    i__3 = j;
		    x[i__3].r = 1., x[i__3].i = 0.;
		    xj = 1.;
		    *scale = 0.;
		    xmax = 0.;
		}
L110:

/*              Scale x if necessary to avoid overflow when adding a */
/*              multiple of column j of A. */

		if (xj > 1.) {
		    rec = 1. / xj;
		    if (cnorm[j] > (bignum - xmax) * rec) {

/*                    Scale x by 1/(2*abs(x(j))). */

			rec *= .5;
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
		    }
		} else if (xj * cnorm[j] > bignum - xmax) {

/*                 Scale x by 1/2. */

		    zdscal_(n, &c_b36, &x[1], &c__1);
		    *scale *= .5;
		}

		if (upper) {
		    if (j > 1) {

/*                    Compute the update */
/*                       x(max(1,j-kd):j-1) := x(max(1,j-kd):j-1) - */
/*                                             x(j)* A(max(1,j-kd):j-1,j) */

/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			i__3 = j;
			z__2.r = -x[i__3].r, z__2.i = -x[i__3].i;
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			zaxpy_(&jlen, &z__1, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			i__3 = j - 1;
			i__ = izamax_(&i__3, &x[1], &c__1);
			i__3 = i__;
			xmax = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(
				&x[i__]), abs(d__2));
		    }
		} else if (j < *n) {

/*                 Compute the update */
/*                    x(j+1:min(j+kd,n)) := x(j+1:min(j+kd,n)) - */
/*                                          x(j) * A(j+1:min(j+kd,n),j) */

/* Computing MIN */
		    i__3 = *kd, i__4 = *n - j;
		    jlen = min(i__3,i__4);
		    if (jlen > 0) {
			i__3 = j;
			z__2.r = -x[i__3].r, z__2.i = -x[i__3].i;
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			zaxpy_(&jlen, &z__1, &ab[j * ab_dim1 + 2], &c__1, &x[
				j + 1], &c__1);
		    }
		    i__3 = *n - j;
		    i__ = j + izamax_(&i__3, &x[j + 1], &c__1);
		    i__3 = i__;
		    xmax = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[
			    i__]), abs(d__2));
		}
/* L120: */
	    }

	} else if (lsame_(trans, "T")) {

/*           Solve A**T * x = b */

	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Compute x(j) = b(j) - sum A(k,j)*x(k). */
/*                                    k<>j */

		i__3 = j;
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]), 
			abs(d__2));
		uscal.r = tscal, uscal.i = 0.;
		rec = 1. / max(xmax,1.);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5;
		    if (nounit) {
			i__3 = maind + j * ab_dim1;
			z__1.r = tscal * ab[i__3].r, z__1.i = tscal * ab[i__3]
				.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), 
			    abs(d__2));
		    if (tjj > 1.) {

/*                       Divide by A(j,j) when scaling x if A(j,j) > 1. */

/* Computing MIN */
			d__1 = 1., d__2 = rec * tjj;
			rec = min(d__1,d__2);
			zladiv_(&z__1, &uscal, &tjjs);
			uscal.r = z__1.r, uscal.i = z__1.i;
		    }
		    if (rec < 1.) {
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0., csumj.i = 0.;
		if (uscal.r == 1. && uscal.i == 0.) {

/*                 If the scaling needed for A in the dot product is 1, */
/*                 call ZDOTU to perform the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			zdotu_(&z__1, &jlen, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			if (jlen > 1) {
			    zdotu_(&z__1, &jlen, &ab[j * ab_dim1 + 2], &c__1, 
				    &x[j + 1], &c__1);
			    csumj.r = z__1.r, csumj.i = z__1.i;
			}
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = *kd + i__ - jlen + j * ab_dim1;
			    z__3.r = ab[i__4].r * uscal.r - ab[i__4].i * 
				    uscal.i, z__3.i = ab[i__4].r * uscal.i + 
				    ab[i__4].i * uscal.r;
			    i__5 = j - jlen - 1 + i__;
			    z__2.r = z__3.r * x[i__5].r - z__3.i * x[i__5].i, 
				    z__2.i = z__3.r * x[i__5].i + z__3.i * x[
				    i__5].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i + 
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L130: */
			}
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + 1 + j * ab_dim1;
			    z__3.r = ab[i__4].r * uscal.r - ab[i__4].i * 
				    uscal.i, z__3.i = ab[i__4].r * uscal.i + 
				    ab[i__4].i * uscal.r;
			    i__5 = j + i__;
			    z__2.r = z__3.r * x[i__5].r - z__3.i * x[i__5].i, 
				    z__2.i = z__3.r * x[i__5].i + z__3.i * x[
				    i__5].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i + 
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L140: */
			}
		    }
		}

		z__1.r = tscal, z__1.i = 0.;
		if (uscal.r == z__1.r && uscal.i == z__1.i) {

/*                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
/*                 was not used to scale the dotproduct. */

		    i__3 = j;
		    i__4 = j;
		    z__1.r = x[i__4].r - csumj.r, z__1.i = x[i__4].i - 
			    csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		    if (nounit) {

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

			i__3 = maind + j * ab_dim1;
			z__1.r = tscal * ab[i__3].r, z__1.i = tscal * ab[i__3]
				.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
			if (tscal == 1.) {
			    goto L160;
			}
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), 
			    abs(d__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1. / xj;
				zdscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else if (tjj > 0.) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else {

/*                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                       scale = 0 and compute a solution to A**T *x = 0. */

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0., x[i__4].i = 0.;
/* L150: */
			}
			i__3 = j;
			x[i__3].r = 1., x[i__3].i = 0.;
			*scale = 0.;
			xmax = 0.;
		    }
L160:
		    ;
		} else {

/*                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
/*                 product has already been divided by 1/A(j,j). */

		    i__3 = j;
		    zladiv_(&z__2, &x[j], &tjjs);
		    z__1.r = z__2.r - csumj.r, z__1.i = z__2.i - csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		}
/* Computing MAX */
		i__3 = j;
		d__3 = xmax, d__4 = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = 
			d_imag(&x[j]), abs(d__2));
		xmax = max(d__3,d__4);
/* L170: */
	    }

	} else {

/*           Solve A**H * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) - sum A(k,j)*x(k). */
/*                                    k<>j */

		i__3 = j;
		xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j]), 
			abs(d__2));
		uscal.r = tscal, uscal.i = 0.;
		rec = 1. / max(xmax,1.);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5;
		    if (nounit) {
			d_cnjg(&z__2, &ab[maind + j * ab_dim1]);
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), 
			    abs(d__2));
		    if (tjj > 1.) {

/*                       Divide by A(j,j) when scaling x if A(j,j) > 1. */

/* Computing MIN */
			d__1 = 1., d__2 = rec * tjj;
			rec = min(d__1,d__2);
			zladiv_(&z__1, &uscal, &tjjs);
			uscal.r = z__1.r, uscal.i = z__1.i;
		    }
		    if (rec < 1.) {
			zdscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0., csumj.i = 0.;
		if (uscal.r == 1. && uscal.i == 0.) {

/*                 If the scaling needed for A in the dot product is 1, */
/*                 call ZDOTC to perform the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			zdotc_(&z__1, &jlen, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			csumj.r = z__1.r, csumj.i = z__1.i;
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			if (jlen > 1) {
			    zdotc_(&z__1, &jlen, &ab[j * ab_dim1 + 2], &c__1, 
				    &x[j + 1], &c__1);
			    csumj.r = z__1.r, csumj.i = z__1.i;
			}
		    }
		} else {

/*                 Otherwise, use in-line code for the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    d_cnjg(&z__4, &ab[*kd + i__ - jlen + j * ab_dim1])
				    ;
			    z__3.r = z__4.r * uscal.r - z__4.i * uscal.i, 
				    z__3.i = z__4.r * uscal.i + z__4.i * 
				    uscal.r;
			    i__4 = j - jlen - 1 + i__;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, 
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i + 
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L180: */
			}
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    d_cnjg(&z__4, &ab[i__ + 1 + j * ab_dim1]);
			    z__3.r = z__4.r * uscal.r - z__4.i * uscal.i, 
				    z__3.i = z__4.r * uscal.i + z__4.i * 
				    uscal.r;
			    i__4 = j + i__;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, 
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = csumj.r + z__2.r, z__1.i = csumj.i + 
				    z__2.i;
			    csumj.r = z__1.r, csumj.i = z__1.i;
/* L190: */
			}
		    }
		}

		z__1.r = tscal, z__1.i = 0.;
		if (uscal.r == z__1.r && uscal.i == z__1.i) {

/*                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
/*                 was not used to scale the dotproduct. */

		    i__3 = j;
		    i__4 = j;
		    z__1.r = x[i__4].r - csumj.r, z__1.i = x[i__4].i - 
			    csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    i__3 = j;
		    xj = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = d_imag(&x[j])
			    , abs(d__2));
		    if (nounit) {

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

			d_cnjg(&z__2, &ab[maind + j * ab_dim1]);
			z__1.r = tscal * z__2.r, z__1.i = tscal * z__2.i;
			tjjs.r = z__1.r, tjjs.i = z__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.;
			if (tscal == 1.) {
			    goto L210;
			}
		    }
		    tjj = (d__1 = tjjs.r, abs(d__1)) + (d__2 = d_imag(&tjjs), 
			    abs(d__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1. / xj;
				zdscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else if (tjj > 0.) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    zdscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			zladiv_(&z__1, &x[j], &tjjs);
			x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		    } else {

/*                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                       scale = 0 and compute a solution to A**H *x = 0. */

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0., x[i__4].i = 0.;
/* L200: */
			}
			i__3 = j;
			x[i__3].r = 1., x[i__3].i = 0.;
			*scale = 0.;
			xmax = 0.;
		    }
L210:
		    ;
		} else {

/*                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
/*                 product has already been divided by 1/A(j,j). */

		    i__3 = j;
		    zladiv_(&z__2, &x[j], &tjjs);
		    z__1.r = z__2.r - csumj.r, z__1.i = z__2.i - csumj.i;
		    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
		}
/* Computing MAX */
		i__3 = j;
		d__3 = xmax, d__4 = (d__1 = x[i__3].r, abs(d__1)) + (d__2 = 
			d_imag(&x[j]), abs(d__2));
		xmax = max(d__3,d__4);
/* L220: */
	    }
	}
	*scale /= tscal;
    }

/*     Scale the column norms by 1/TSCAL for return. */

    if (tscal != 1.) {
	d__1 = 1. / tscal;
	dscal_(n, &d__1, &cnorm[1], &c__1);
    }

    return 0;

/*     End of ZLATBS */

} /* zlatbs_ */
