/* clatbs.f -- translated by f2c (version 20061008).
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
static real c_b36 = .5f;

/* Subroutine */ int clatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, complex *ab, integer *ldab, complex *
	x, real *scale, real *cnorm, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    double r_imag(complex *);
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer i__, j;
    real xj, rec, tjj;
    integer jinc, jlen;
    real xbnd;
    integer imax;
    real tmax;
    complex tjjs;
    real xmax, grow;
    integer maind;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    real tscal;
    complex uscal;
    integer jlast;
    extern /* Complex */ VOID cdotu_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    complex csumj;
    extern /* Subroutine */ int ctbsv_(char *, char *, char *, integer *, 
	    integer *, complex *, integer *, complex *, integer *), caxpy_(integer *, complex *, complex *, integer *
, complex *, integer *);
    logical upper;
    extern /* Subroutine */ int slabad_(real *, real *);
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Complex */ VOID cladiv_(complex *, complex *, complex *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer 
	    *), xerbla_(char *, integer *);
    real bignum;
    extern integer isamax_(integer *, real *, integer *);
    extern doublereal scasum_(integer *, complex *, integer *);
    logical notran;
    integer jfirst;
    real smlnum;
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

/*  CLATBS solves one of the triangular systems */

/*     A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b, */

/*  with scaling to prevent overflow, where A is an upper or lower */
/*  triangular band matrix.  Here A' denotes the transpose of A, x and b */
/*  are n-element vectors, and s is a scaling factor, usually less than */
/*  or equal to 1, chosen so that the components of x will be less than */
/*  the overflow threshold.  If the unscaled problem will not cause */
/*  overflow, the Level 2 BLAS routine CTBSV is called.  If the matrix A */
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

/*  AB      (input) COMPLEX array, dimension (LDAB,N) */
/*          The upper or lower triangular band matrix A, stored in the */
/*          first KD+1 rows of the array. The j-th column of A is stored */
/*          in the j-th column of the array AB as follows: */
/*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j; */
/*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd). */

/*  LDAB    (input) INTEGER */
/*          The leading dimension of the array AB.  LDAB >= KD+1. */

/*  X       (input/output) COMPLEX array, dimension (N) */
/*          On entry, the right hand side b of the triangular system. */
/*          On exit, X is overwritten by the solution vector x. */

/*  SCALE   (output) REAL */
/*          The scaling factor s for the triangular system */
/*             A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b. */
/*          If SCALE = 0, the matrix A is singular or badly scaled, and */
/*          the vector x is an exact or approximate solution to A*x = 0. */

/*  CNORM   (input or output) REAL array, dimension (N) */

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

/*  A rough bound on x is computed; if that is less than overflow, CTBSV */
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

/*  Since |x(j)| <= M(j), we use the Level 2 BLAS routine CTBSV if the */
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

/*  and we can safely call CTBSV if 1/M(n) and 1/G(n) are both greater */
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
	xerbla_("CLATBS", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine machine dependent parameters to control overflow. */

    smlnum = slamch_("Safe minimum");
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);
    smlnum /= slamch_("Precision");
    bignum = 1.f / smlnum;
    *scale = 1.f;

    if (lsame_(normin, "N")) {

/*        Compute the 1-norm of each column, not including the diagonal. */

	if (upper) {

/*           A is upper triangular. */

	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
/* Computing MIN */
		i__2 = *kd, i__3 = j - 1;
		jlen = min(i__2,i__3);
		cnorm[j] = scasum_(&jlen, &ab[*kd + 1 - jlen + j * ab_dim1], &
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
		    cnorm[j] = scasum_(&jlen, &ab[j * ab_dim1 + 2], &c__1);
		} else {
		    cnorm[j] = 0.f;
		}
/* L20: */
	    }
	}
    }

/*     Scale the column norms by TSCAL if the maximum element in CNORM is */
/*     greater than BIGNUM/2. */

    imax = isamax_(n, &cnorm[1], &c__1);
    tmax = cnorm[imax];
    if (tmax <= bignum * .5f) {
	tscal = 1.f;
    } else {
	tscal = .5f / (smlnum * tmax);
	sscal_(n, &tscal, &cnorm[1], &c__1);
    }

/*     Compute a bound on the computed solution vector to see if the */
/*     Level 2 BLAS routine CTBSV can be used. */

    xmax = 0.f;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	i__2 = j;
	r__3 = xmax, r__4 = (r__1 = x[i__2].r / 2.f, dabs(r__1)) + (r__2 = 
		r_imag(&x[j]) / 2.f, dabs(r__2));
	xmax = dmax(r__3,r__4);
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

	if (tscal != 1.f) {
	    grow = 0.f;
	    goto L60;
	}

	if (nounit) {

/*           A is non-unit triangular. */

/*           Compute GROW = 1/G(j) and XBND = 1/M(j). */
/*           Initially, G(0) = max{x(i), i=1,...,n}. */

	    grow = .5f / dmax(xbnd,smlnum);
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
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs), 
			dabs(r__2));

		if (tjj >= smlnum) {

/*                 M(j) = G(j-1) / abs(A(j,j)) */

/* Computing MIN */
		    r__1 = xbnd, r__2 = dmin(1.f,tjj) * grow;
		    xbnd = dmin(r__1,r__2);
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.f;
		}

		if (tjj + cnorm[j] >= smlnum) {

/*                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) ) */

		    grow *= tjj / (tjj + cnorm[j]);
		} else {

/*                 G(j) could overflow, set GROW to 0. */

		    grow = 0.f;
		}
/* L40: */
	    }
	    grow = xbnd;
	} else {

/*           A is unit triangular. */

/*           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}. */

/* Computing MIN */
	    r__1 = 1.f, r__2 = .5f / dmax(xbnd,smlnum);
	    grow = dmin(r__1,r__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L60;
		}

/*              G(j) = G(j-1)*( 1 + CNORM(j) ) */

		grow *= 1.f / (cnorm[j] + 1.f);
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

	if (tscal != 1.f) {
	    grow = 0.f;
	    goto L90;
	}

	if (nounit) {

/*           A is non-unit triangular. */

/*           Compute GROW = 1/G(j) and XBND = 1/M(j). */
/*           Initially, M(0) = max{x(i), i=1,...,n}. */

	    grow = .5f / dmax(xbnd,smlnum);
	    xbnd = grow;
	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) ) */

		xj = cnorm[j] + 1.f;
/* Computing MIN */
		r__1 = grow, r__2 = xbnd / xj;
		grow = dmin(r__1,r__2);

		i__3 = maind + j * ab_dim1;
		tjjs.r = ab[i__3].r, tjjs.i = ab[i__3].i;
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs), 
			dabs(r__2));

		if (tjj >= smlnum) {

/*                 M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j)) */

		    if (xj > tjj) {
			xbnd *= tjj / xj;
		    }
		} else {

/*                 M(j) could overflow, set XBND to 0. */

		    xbnd = 0.f;
		}
/* L70: */
	    }
	    grow = dmin(grow,xbnd);
	} else {

/*           A is unit triangular. */

/*           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}. */

/* Computing MIN */
	    r__1 = 1.f, r__2 = .5f / dmax(xbnd,smlnum);
	    grow = dmin(r__1,r__2);
	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Exit the loop if the growth factor is too small. */

		if (grow <= smlnum) {
		    goto L90;
		}

/*              G(j) = ( 1 + CNORM(j) )*G(j-1) */

		xj = cnorm[j] + 1.f;
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

	ctbsv_(uplo, trans, diag, n, kd, &ab[ab_offset], ldab, &x[1], &c__1);
    } else {

/*        Use a Level 1 BLAS solve, scaling intermediate results. */

	if (xmax > bignum * .5f) {

/*           Scale X so that its components are less than or equal to */
/*           BIGNUM in absolute value. */

	    *scale = bignum * .5f / xmax;
	    csscal_(n, scale, &x[1], &c__1);
	    xmax = bignum;
	} else {
	    xmax *= 2.f;
	}

	if (notran) {

/*           Solve A * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) / A(j,j), scaling x if necessary. */

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]), 
			dabs(r__2));
		if (nounit) {
		    i__3 = maind + j * ab_dim1;
		    q__1.r = tscal * ab[i__3].r, q__1.i = tscal * ab[i__3].i;
		    tjjs.r = q__1.r, tjjs.i = q__1.i;
		} else {
		    tjjs.r = tscal, tjjs.i = 0.f;
		    if (tscal == 1.f) {
			goto L105;
		    }
		}
		tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs), 
			dabs(r__2));
		if (tjj > smlnum) {

/*                    abs(A(j,j)) > SMLNUM: */

		    if (tjj < 1.f) {
			if (xj > tjj * bignum) {

/*                          Scale x by 1/b(j). */

			    rec = 1.f / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    i__3 = j;
		    cladiv_(&q__1, &x[j], &tjjs);
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		} else if (tjj > 0.f) {

/*                    0 < abs(A(j,j)) <= SMLNUM: */

		    if (xj > tjj * bignum) {

/*                       Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM */
/*                       to avoid overflow when dividing by A(j,j). */

			rec = tjj * bignum / xj;
			if (cnorm[j] > 1.f) {

/*                          Scale by 1/CNORM(j) to avoid overflow when */
/*                          multiplying x(j) times column j. */

			    rec /= cnorm[j];
			}
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		    i__3 = j;
		    cladiv_(&q__1, &x[j], &tjjs);
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		} else {

/*                    A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                    scale = 0, and compute a solution to A*x = 0. */

		    i__3 = *n;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			i__4 = i__;
			x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L100: */
		    }
		    i__3 = j;
		    x[i__3].r = 1.f, x[i__3].i = 0.f;
		    xj = 1.f;
		    *scale = 0.f;
		    xmax = 0.f;
		}
L105:

/*              Scale x if necessary to avoid overflow when adding a */
/*              multiple of column j of A. */

		if (xj > 1.f) {
		    rec = 1.f / xj;
		    if (cnorm[j] > (bignum - xmax) * rec) {

/*                    Scale x by 1/(2*abs(x(j))). */

			rec *= .5f;
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
		    }
		} else if (xj * cnorm[j] > bignum - xmax) {

/*                 Scale x by 1/2. */

		    csscal_(n, &c_b36, &x[1], &c__1);
		    *scale *= .5f;
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
			q__2.r = -x[i__3].r, q__2.i = -x[i__3].i;
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			caxpy_(&jlen, &q__1, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			i__3 = j - 1;
			i__ = icamax_(&i__3, &x[1], &c__1);
			i__3 = i__;
			xmax = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = 
				r_imag(&x[i__]), dabs(r__2));
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
			q__2.r = -x[i__3].r, q__2.i = -x[i__3].i;
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			caxpy_(&jlen, &q__1, &ab[j * ab_dim1 + 2], &c__1, &x[
				j + 1], &c__1);
		    }
		    i__3 = *n - j;
		    i__ = j + icamax_(&i__3, &x[j + 1], &c__1);
		    i__3 = i__;
		    xmax = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[
			    i__]), dabs(r__2));
		}
/* L110: */
	    }

	} else if (lsame_(trans, "T")) {

/*           Solve A**T * x = b */

	    i__2 = jlast;
	    i__1 = jinc;
	    for (j = jfirst; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {

/*              Compute x(j) = b(j) - sum A(k,j)*x(k). */
/*                                    k<>j */

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]), 
			dabs(r__2));
		uscal.r = tscal, uscal.i = 0.f;
		rec = 1.f / dmax(xmax,1.f);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5f;
		    if (nounit) {
			i__3 = maind + j * ab_dim1;
			q__1.r = tscal * ab[i__3].r, q__1.i = tscal * ab[i__3]
				.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > 1.f) {

/*                       Divide by A(j,j) when scaling x if A(j,j) > 1. */

/* Computing MIN */
			r__1 = 1.f, r__2 = rec * tjj;
			rec = dmin(r__1,r__2);
			cladiv_(&q__1, &uscal, &tjjs);
			uscal.r = q__1.r, uscal.i = q__1.i;
		    }
		    if (rec < 1.f) {
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0.f, csumj.i = 0.f;
		if (uscal.r == 1.f && uscal.i == 0.f) {

/*                 If the scaling needed for A in the dot product is 1, */
/*                 call CDOTU to perform the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			cdotu_(&q__1, &jlen, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			if (jlen > 1) {
			    cdotu_(&q__1, &jlen, &ab[j * ab_dim1 + 2], &c__1, 
				    &x[j + 1], &c__1);
			    csumj.r = q__1.r, csumj.i = q__1.i;
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
			    q__3.r = ab[i__4].r * uscal.r - ab[i__4].i * 
				    uscal.i, q__3.i = ab[i__4].r * uscal.i + 
				    ab[i__4].i * uscal.r;
			    i__5 = j - jlen - 1 + i__;
			    q__2.r = q__3.r * x[i__5].r - q__3.i * x[i__5].i, 
				    q__2.i = q__3.r * x[i__5].i + q__3.i * x[
				    i__5].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i + 
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L120: */
			}
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__ + 1 + j * ab_dim1;
			    q__3.r = ab[i__4].r * uscal.r - ab[i__4].i * 
				    uscal.i, q__3.i = ab[i__4].r * uscal.i + 
				    ab[i__4].i * uscal.r;
			    i__5 = j + i__;
			    q__2.r = q__3.r * x[i__5].r - q__3.i * x[i__5].i, 
				    q__2.i = q__3.r * x[i__5].i + q__3.i * x[
				    i__5].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i + 
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L130: */
			}
		    }
		}

		q__1.r = tscal, q__1.i = 0.f;
		if (uscal.r == q__1.r && uscal.i == q__1.i) {

/*                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
/*                 was not used to scale the dotproduct. */

		    i__3 = j;
		    i__4 = j;
		    q__1.r = x[i__4].r - csumj.r, q__1.i = x[i__4].i - 
			    csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		    if (nounit) {

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

			i__3 = maind + j * ab_dim1;
			q__1.r = tscal * ab[i__3].r, q__1.i = tscal * ab[i__3]
				.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
			if (tscal == 1.f) {
			    goto L145;
			}
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.f) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1.f / xj;
				csscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else if (tjj > 0.f) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else {

/*                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                       scale = 0 and compute a solution to A**T *x = 0. */

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L140: */
			}
			i__3 = j;
			x[i__3].r = 1.f, x[i__3].i = 0.f;
			*scale = 0.f;
			xmax = 0.f;
		    }
L145:
		    ;
		} else {

/*                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
/*                 product has already been divided by 1/A(j,j). */

		    i__3 = j;
		    cladiv_(&q__2, &x[j], &tjjs);
		    q__1.r = q__2.r - csumj.r, q__1.i = q__2.i - csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		}
/* Computing MAX */
		i__3 = j;
		r__3 = xmax, r__4 = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&x[j]), dabs(r__2));
		xmax = dmax(r__3,r__4);
/* L150: */
	    }

	} else {

/*           Solve A**H * x = b */

	    i__1 = jlast;
	    i__2 = jinc;
	    for (j = jfirst; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {

/*              Compute x(j) = b(j) - sum A(k,j)*x(k). */
/*                                    k<>j */

		i__3 = j;
		xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]), 
			dabs(r__2));
		uscal.r = tscal, uscal.i = 0.f;
		rec = 1.f / dmax(xmax,1.f);
		if (cnorm[j] > (bignum - xj) * rec) {

/*                 If x(j) could overflow, scale x by 1/(2*XMAX). */

		    rec *= .5f;
		    if (nounit) {
			r_cnjg(&q__2, &ab[maind + j * ab_dim1]);
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > 1.f) {

/*                       Divide by A(j,j) when scaling x if A(j,j) > 1. */

/* Computing MIN */
			r__1 = 1.f, r__2 = rec * tjj;
			rec = dmin(r__1,r__2);
			cladiv_(&q__1, &uscal, &tjjs);
			uscal.r = q__1.r, uscal.i = q__1.i;
		    }
		    if (rec < 1.f) {
			csscal_(n, &rec, &x[1], &c__1);
			*scale *= rec;
			xmax *= rec;
		    }
		}

		csumj.r = 0.f, csumj.i = 0.f;
		if (uscal.r == 1.f && uscal.i == 0.f) {

/*                 If the scaling needed for A in the dot product is 1, */
/*                 call CDOTC to perform the dot product. */

		    if (upper) {
/* Computing MIN */
			i__3 = *kd, i__4 = j - 1;
			jlen = min(i__3,i__4);
			cdotc_(&q__1, &jlen, &ab[*kd + 1 - jlen + j * ab_dim1]
, &c__1, &x[j - jlen], &c__1);
			csumj.r = q__1.r, csumj.i = q__1.i;
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			if (jlen > 1) {
			    cdotc_(&q__1, &jlen, &ab[j * ab_dim1 + 2], &c__1, 
				    &x[j + 1], &c__1);
			    csumj.r = q__1.r, csumj.i = q__1.i;
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
			    r_cnjg(&q__4, &ab[*kd + i__ - jlen + j * ab_dim1])
				    ;
			    q__3.r = q__4.r * uscal.r - q__4.i * uscal.i, 
				    q__3.i = q__4.r * uscal.i + q__4.i * 
				    uscal.r;
			    i__4 = j - jlen - 1 + i__;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i, 
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i + 
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L160: */
			}
		    } else {
/* Computing MIN */
			i__3 = *kd, i__4 = *n - j;
			jlen = min(i__3,i__4);
			i__3 = jlen;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    r_cnjg(&q__4, &ab[i__ + 1 + j * ab_dim1]);
			    q__3.r = q__4.r * uscal.r - q__4.i * uscal.i, 
				    q__3.i = q__4.r * uscal.i + q__4.i * 
				    uscal.r;
			    i__4 = j + i__;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i, 
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = csumj.r + q__2.r, q__1.i = csumj.i + 
				    q__2.i;
			    csumj.r = q__1.r, csumj.i = q__1.i;
/* L170: */
			}
		    }
		}

		q__1.r = tscal, q__1.i = 0.f;
		if (uscal.r == q__1.r && uscal.i == q__1.i) {

/*                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j) */
/*                 was not used to scale the dotproduct. */

		    i__3 = j;
		    i__4 = j;
		    q__1.r = x[i__4].r - csumj.r, q__1.i = x[i__4].i - 
			    csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    i__3 = j;
		    xj = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = r_imag(&x[j]
			    ), dabs(r__2));
		    if (nounit) {

/*                    Compute x(j) = x(j) / A(j,j), scaling if necessary. */

			r_cnjg(&q__2, &ab[maind + j * ab_dim1]);
			q__1.r = tscal * q__2.r, q__1.i = tscal * q__2.i;
			tjjs.r = q__1.r, tjjs.i = q__1.i;
		    } else {
			tjjs.r = tscal, tjjs.i = 0.f;
			if (tscal == 1.f) {
			    goto L185;
			}
		    }
		    tjj = (r__1 = tjjs.r, dabs(r__1)) + (r__2 = r_imag(&tjjs),
			     dabs(r__2));
		    if (tjj > smlnum) {

/*                       abs(A(j,j)) > SMLNUM: */

			if (tjj < 1.f) {
			    if (xj > tjj * bignum) {

/*                             Scale X by 1/abs(x(j)). */

				rec = 1.f / xj;
				csscal_(n, &rec, &x[1], &c__1);
				*scale *= rec;
				xmax *= rec;
			    }
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else if (tjj > 0.f) {

/*                       0 < abs(A(j,j)) <= SMLNUM: */

			if (xj > tjj * bignum) {

/*                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM. */

			    rec = tjj * bignum / xj;
			    csscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
			i__3 = j;
			cladiv_(&q__1, &x[j], &tjjs);
			x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		    } else {

/*                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and */
/*                       scale = 0 and compute a solution to A**H *x = 0. */

			i__3 = *n;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    i__4 = i__;
			    x[i__4].r = 0.f, x[i__4].i = 0.f;
/* L180: */
			}
			i__3 = j;
			x[i__3].r = 1.f, x[i__3].i = 0.f;
			*scale = 0.f;
			xmax = 0.f;
		    }
L185:
		    ;
		} else {

/*                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot */
/*                 product has already been divided by 1/A(j,j). */

		    i__3 = j;
		    cladiv_(&q__2, &x[j], &tjjs);
		    q__1.r = q__2.r - csumj.r, q__1.i = q__2.i - csumj.i;
		    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
		}
/* Computing MAX */
		i__3 = j;
		r__3 = xmax, r__4 = (r__1 = x[i__3].r, dabs(r__1)) + (r__2 = 
			r_imag(&x[j]), dabs(r__2));
		xmax = dmax(r__3,r__4);
/* L190: */
	    }
	}
	*scale /= tscal;
    }

/*     Scale the column norms by 1/TSCAL for return. */

    if (tscal != 1.f) {
	r__1 = 1.f / tscal;
	sscal_(n, &r__1, &cnorm[1], &c__1);
    }

    return 0;

/*     End of CLATBS */

} /* clatbs_ */
