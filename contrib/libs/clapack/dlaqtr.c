/* dlaqtr.f -- translated by f2c (version 20061008).
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
static logical c_false = FALSE_;
static integer c__2 = 2;
static doublereal c_b21 = 1.;
static doublereal c_b25 = 0.;
static logical c_true = TRUE_;

/* Subroutine */ int dlaqtr_(logical *ltran, logical *lreal, integer *n, 
	doublereal *t, integer *ldt, doublereal *b, doublereal *w, doublereal 
	*scale, doublereal *x, doublereal *work, integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, i__1, i__2;
    doublereal d__1, d__2, d__3, d__4, d__5, d__6;

    /* Local variables */
    doublereal d__[4]	/* was [2][2] */;
    integer i__, j, k;
    doublereal v[4]	/* was [2][2] */, z__;
    integer j1, j2, n1, n2;
    doublereal si, xj, sr, rec, eps, tjj, tmp;
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    integer ierr;
    doublereal smin, xmax;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern doublereal dasum_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *);
    integer jnext;
    doublereal sminw, xnorm;
    extern /* Subroutine */ int dlaln2_(logical *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *, doublereal *, 
	     doublereal *, doublereal *, integer *, doublereal *, doublereal *
, doublereal *, integer *, doublereal *, doublereal *, integer *);
    extern doublereal dlamch_(char *), dlange_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *);
    extern integer idamax_(integer *, doublereal *, integer *);
    doublereal scaloc;
    extern /* Subroutine */ int dladiv_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *);
    doublereal bignum;
    logical notran;
    doublereal smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAQTR solves the real quasi-triangular system */

/*               op(T)*p = scale*c,               if LREAL = .TRUE. */

/*  or the complex quasi-triangular systems */

/*             op(T + iB)*(p+iq) = scale*(c+id),  if LREAL = .FALSE. */

/*  in real arithmetic, where T is upper quasi-triangular. */
/*  If LREAL = .FALSE., then the first diagonal block of T must be */
/*  1 by 1, B is the specially structured matrix */

/*                 B = [ b(1) b(2) ... b(n) ] */
/*                     [       w            ] */
/*                     [           w        ] */
/*                     [              .     ] */
/*                     [                 w  ] */

/*  op(A) = A or A', A' denotes the conjugate transpose of */
/*  matrix A. */

/*  On input, X = [ c ].  On output, X = [ p ]. */
/*                [ d ]                  [ q ] */

/*  This subroutine is designed for the condition number estimation */
/*  in routine DTRSNA. */

/*  Arguments */
/*  ========= */

/*  LTRAN   (input) LOGICAL */
/*          On entry, LTRAN specifies the option of conjugate transpose: */
/*             = .FALSE.,    op(T+i*B) = T+i*B, */
/*             = .TRUE.,     op(T+i*B) = (T+i*B)'. */

/*  LREAL   (input) LOGICAL */
/*          On entry, LREAL specifies the input matrix structure: */
/*             = .FALSE.,    the input is complex */
/*             = .TRUE.,     the input is real */

/*  N       (input) INTEGER */
/*          On entry, N specifies the order of T+i*B. N >= 0. */

/*  T       (input) DOUBLE PRECISION array, dimension (LDT,N) */
/*          On entry, T contains a matrix in Schur canonical form. */
/*          If LREAL = .FALSE., then the first diagonal block of T mu */
/*          be 1 by 1. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the matrix T. LDT >= max(1,N). */

/*  B       (input) DOUBLE PRECISION array, dimension (N) */
/*          On entry, B contains the elements to form the matrix */
/*          B as described above. */
/*          If LREAL = .TRUE., B is not referenced. */

/*  W       (input) DOUBLE PRECISION */
/*          On entry, W is the diagonal element of the matrix B. */
/*          If LREAL = .TRUE., W is not referenced. */

/*  SCALE   (output) DOUBLE PRECISION */
/*          On exit, SCALE is the scale factor. */

/*  X       (input/output) DOUBLE PRECISION array, dimension (2*N) */
/*          On entry, X contains the right hand side of the system. */
/*          On exit, X is overwritten by the solution. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          On exit, INFO is set to */
/*             0: successful exit. */
/*               1: the some diagonal 1 by 1 block has been perturbed by */
/*                  a small number SMIN to keep nonsingularity. */
/*               2: the some diagonal 2 by 2 block has been perturbed by */
/*                  a small number in DLALN2 to keep nonsingularity. */
/*          NOTE: In the interests of speed, this routine does not */
/*                check the inputs for errors. */

/* ===================================================================== */

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

/*     Do not test the input parameters for errors */

    /* Parameter adjustments */
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    --b;
    --x;
    --work;

    /* Function Body */
    notran = ! (*ltran);
    *info = 0;

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Set constants to control overflow */

    eps = dlamch_("P");
    smlnum = dlamch_("S") / eps;
    bignum = 1. / smlnum;

    xnorm = dlange_("M", n, n, &t[t_offset], ldt, d__);
    if (! (*lreal)) {
/* Computing MAX */
	d__1 = xnorm, d__2 = abs(*w), d__1 = max(d__1,d__2), d__2 = dlange_(
		"M", n, &c__1, &b[1], n, d__);
	xnorm = max(d__1,d__2);
    }
/* Computing MAX */
    d__1 = smlnum, d__2 = eps * xnorm;
    smin = max(d__1,d__2);

/*     Compute 1-norm of each column of strictly upper triangular */
/*     part of T to control overflow in triangular solver. */

    work[1] = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	i__2 = j - 1;
	work[j] = dasum_(&i__2, &t[j * t_dim1 + 1], &c__1);
/* L10: */
    }

    if (! (*lreal)) {
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    work[i__] += (d__1 = b[i__], abs(d__1));
/* L20: */
	}
    }

    n2 = *n << 1;
    n1 = *n;
    if (! (*lreal)) {
	n1 = n2;
    }
    k = idamax_(&n1, &x[1], &c__1);
    xmax = (d__1 = x[k], abs(d__1));
    *scale = 1.;

    if (xmax > bignum) {
	*scale = bignum / xmax;
	dscal_(&n1, scale, &x[1], &c__1);
	xmax = bignum;
    }

    if (*lreal) {

	if (notran) {

/*           Solve T*p = scale*c */

	    jnext = *n;
	    for (j = *n; j >= 1; --j) {
		if (j > jnext) {
		    goto L30;
		}
		j1 = j;
		j2 = j;
		jnext = j - 1;
		if (j > 1) {
		    if (t[j + (j - 1) * t_dim1] != 0.) {
			j1 = j - 1;
			jnext = j - 2;
		    }
		}

		if (j1 == j2) {

/*                 Meet 1 by 1 diagonal block */

/*                 Scale to avoid overflow when computing */
/*                     x(j) = b(j)/T(j,j) */

		    xj = (d__1 = x[j1], abs(d__1));
		    tjj = (d__1 = t[j1 + j1 * t_dim1], abs(d__1));
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < smin) {
			tmp = smin;
			tjj = smin;
			*info = 1;
		    }

		    if (xj == 0.) {
			goto L30;
		    }

		    if (tjj < 1.) {
			if (xj > bignum * tjj) {
			    rec = 1. / xj;
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    x[j1] /= tmp;
		    xj = (d__1 = x[j1], abs(d__1));

/*                 Scale x if necessary to avoid overflow when adding a */
/*                 multiple of column j1 of T. */

		    if (xj > 1.) {
			rec = 1. / xj;
			if (work[j1] > (bignum - xmax) * rec) {
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }
		    if (j1 > 1) {
			i__1 = j1 - 1;
			d__1 = -x[j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			k = idamax_(&i__1, &x[1], &c__1);
			xmax = (d__1 = x[k], abs(d__1));
		    }

		} else {

/*                 Meet 2 by 2 diagonal block */

/*                 Call 2 by 2 linear system solve, to take */
/*                 care of possible overflow by scaling factor. */

		    d__[0] = x[j1];
		    d__[1] = x[j2];
		    dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b21, &t[j1 + j1 
			    * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, &c_b25, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.) {
			dscal_(n, &scaloc, &x[1], &c__1);
			*scale *= scaloc;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];

/*                 Scale V(1,1) (= X(J1)) and/or V(2,1) (=X(J2)) */
/*                 to avoid overflow in updating right-hand side. */

/* Computing MAX */
		    d__1 = abs(v[0]), d__2 = abs(v[1]);
		    xj = max(d__1,d__2);
		    if (xj > 1.) {
			rec = 1. / xj;
/* Computing MAX */
			d__1 = work[j1], d__2 = work[j2];
			if (max(d__1,d__2) > (bignum - xmax) * rec) {
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

/*                 Update right-hand side */

		    if (j1 > 1) {
			i__1 = j1 - 1;
			d__1 = -x[j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			d__1 = -x[j2];
			daxpy_(&i__1, &d__1, &t[j2 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			k = idamax_(&i__1, &x[1], &c__1);
			xmax = (d__1 = x[k], abs(d__1));
		    }

		}

L30:
		;
	    }

	} else {

/*           Solve T'*p = scale*c */

	    jnext = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (j < jnext) {
		    goto L40;
		}
		j1 = j;
		j2 = j;
		jnext = j + 1;
		if (j < *n) {
		    if (t[j + 1 + j * t_dim1] != 0.) {
			j2 = j + 1;
			jnext = j + 2;
		    }
		}

		if (j1 == j2) {

/*                 1 by 1 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

		    xj = (d__1 = x[j1], abs(d__1));
		    if (xmax > 1.) {
			rec = 1. / xmax;
			if (work[j1] > (bignum - xj) * rec) {
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    x[j1] -= ddot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[1], &
			    c__1);

		    xj = (d__1 = x[j1], abs(d__1));
		    tjj = (d__1 = t[j1 + j1 * t_dim1], abs(d__1));
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < smin) {
			tmp = smin;
			tjj = smin;
			*info = 1;
		    }

		    if (tjj < 1.) {
			if (xj > bignum * tjj) {
			    rec = 1. / xj;
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    x[j1] /= tmp;
/* Computing MAX */
		    d__2 = xmax, d__3 = (d__1 = x[j1], abs(d__1));
		    xmax = max(d__2,d__3);

		} else {

/*                 2 by 2 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side elements by inner product. */

/* Computing MAX */
		    d__3 = (d__1 = x[j1], abs(d__1)), d__4 = (d__2 = x[j2], 
			    abs(d__2));
		    xj = max(d__3,d__4);
		    if (xmax > 1.) {
			rec = 1. / xmax;
/* Computing MAX */
			d__1 = work[j2], d__2 = work[j1];
			if (max(d__1,d__2) > (bignum - xj) * rec) {
			    dscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    d__[0] = x[j1] - ddot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[1] = x[j2] - ddot_(&i__2, &t[j2 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);

		    dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b21, &t[j1 + j1 *
			     t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &c_b25, 
			     &c_b25, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.) {
			dscal_(n, &scaloc, &x[1], &c__1);
			*scale *= scaloc;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
/* Computing MAX */
		    d__3 = (d__1 = x[j1], abs(d__1)), d__4 = (d__2 = x[j2], 
			    abs(d__2)), d__3 = max(d__3,d__4);
		    xmax = max(d__3,xmax);

		}
L40:
		;
	    }
	}

    } else {

/* Computing MAX */
	d__1 = eps * abs(*w);
	sminw = max(d__1,smin);
	if (notran) {

/*           Solve (T + iB)*(p+iq) = c+id */

	    jnext = *n;
	    for (j = *n; j >= 1; --j) {
		if (j > jnext) {
		    goto L70;
		}
		j1 = j;
		j2 = j;
		jnext = j - 1;
		if (j > 1) {
		    if (t[j + (j - 1) * t_dim1] != 0.) {
			j1 = j - 1;
			jnext = j - 2;
		    }
		}

		if (j1 == j2) {

/*                 1 by 1 diagonal block */

/*                 Scale if necessary to avoid overflow in division */

		    z__ = *w;
		    if (j1 == 1) {
			z__ = b[1];
		    }
		    xj = (d__1 = x[j1], abs(d__1)) + (d__2 = x[*n + j1], abs(
			    d__2));
		    tjj = (d__1 = t[j1 + j1 * t_dim1], abs(d__1)) + abs(z__);
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < sminw) {
			tmp = sminw;
			tjj = sminw;
			*info = 1;
		    }

		    if (xj == 0.) {
			goto L70;
		    }

		    if (tjj < 1.) {
			if (xj > bignum * tjj) {
			    rec = 1. / xj;
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    dladiv_(&x[j1], &x[*n + j1], &tmp, &z__, &sr, &si);
		    x[j1] = sr;
		    x[*n + j1] = si;
		    xj = (d__1 = x[j1], abs(d__1)) + (d__2 = x[*n + j1], abs(
			    d__2));

/*                 Scale x if necessary to avoid overflow when adding a */
/*                 multiple of column j1 of T. */

		    if (xj > 1.) {
			rec = 1. / xj;
			if (work[j1] > (bignum - xmax) * rec) {
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

		    if (j1 > 1) {
			i__1 = j1 - 1;
			d__1 = -x[j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			d__1 = -x[*n + j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);

			x[1] += b[j1] * x[*n + j1];
			x[*n + 1] -= b[j1] * x[j1];

			xmax = 0.;
			i__1 = j1 - 1;
			for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			    d__3 = xmax, d__4 = (d__1 = x[k], abs(d__1)) + (
				    d__2 = x[k + *n], abs(d__2));
			    xmax = max(d__3,d__4);
/* L50: */
			}
		    }

		} else {

/*                 Meet 2 by 2 diagonal block */

		    d__[0] = x[j1];
		    d__[1] = x[j2];
		    d__[2] = x[*n + j1];
		    d__[3] = x[*n + j2];
		    d__1 = -(*w);
		    dlaln2_(&c_false, &c__2, &c__2, &sminw, &c_b21, &t[j1 + 
			    j1 * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, &d__1, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.) {
			i__1 = *n << 1;
			dscal_(&i__1, &scaloc, &x[1], &c__1);
			*scale = scaloc * *scale;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
		    x[*n + j1] = v[2];
		    x[*n + j2] = v[3];

/*                 Scale X(J1), .... to avoid overflow in */
/*                 updating right hand side. */

/* Computing MAX */
		    d__1 = abs(v[0]) + abs(v[2]), d__2 = abs(v[1]) + abs(v[3])
			    ;
		    xj = max(d__1,d__2);
		    if (xj > 1.) {
			rec = 1. / xj;
/* Computing MAX */
			d__1 = work[j1], d__2 = work[j2];
			if (max(d__1,d__2) > (bignum - xmax) * rec) {
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

/*                 Update the right-hand side. */

		    if (j1 > 1) {
			i__1 = j1 - 1;
			d__1 = -x[j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			d__1 = -x[j2];
			daxpy_(&i__1, &d__1, &t[j2 * t_dim1 + 1], &c__1, &x[1]
, &c__1);

			i__1 = j1 - 1;
			d__1 = -x[*n + j1];
			daxpy_(&i__1, &d__1, &t[j1 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);
			i__1 = j1 - 1;
			d__1 = -x[*n + j2];
			daxpy_(&i__1, &d__1, &t[j2 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);

			x[1] = x[1] + b[j1] * x[*n + j1] + b[j2] * x[*n + j2];
			x[*n + 1] = x[*n + 1] - b[j1] * x[j1] - b[j2] * x[j2];

			xmax = 0.;
			i__1 = j1 - 1;
			for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			    d__3 = (d__1 = x[k], abs(d__1)) + (d__2 = x[k + *
				    n], abs(d__2));
			    xmax = max(d__3,xmax);
/* L60: */
			}
		    }

		}
L70:
		;
	    }

	} else {

/*           Solve (T + iB)'*(p+iq) = c+id */

	    jnext = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (j < jnext) {
		    goto L80;
		}
		j1 = j;
		j2 = j;
		jnext = j + 1;
		if (j < *n) {
		    if (t[j + 1 + j * t_dim1] != 0.) {
			j2 = j + 1;
			jnext = j + 2;
		    }
		}

		if (j1 == j2) {

/*                 1 by 1 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

		    xj = (d__1 = x[j1], abs(d__1)) + (d__2 = x[j1 + *n], abs(
			    d__2));
		    if (xmax > 1.) {
			rec = 1. / xmax;
			if (work[j1] > (bignum - xj) * rec) {
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    x[j1] -= ddot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[1], &
			    c__1);
		    i__2 = j1 - 1;
		    x[*n + j1] -= ddot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[
			    *n + 1], &c__1);
		    if (j1 > 1) {
			x[j1] -= b[j1] * x[*n + 1];
			x[*n + j1] += b[j1] * x[1];
		    }
		    xj = (d__1 = x[j1], abs(d__1)) + (d__2 = x[j1 + *n], abs(
			    d__2));

		    z__ = *w;
		    if (j1 == 1) {
			z__ = b[1];
		    }

/*                 Scale if necessary to avoid overflow in */
/*                 complex division */

		    tjj = (d__1 = t[j1 + j1 * t_dim1], abs(d__1)) + abs(z__);
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < sminw) {
			tmp = sminw;
			tjj = sminw;
			*info = 1;
		    }

		    if (tjj < 1.) {
			if (xj > bignum * tjj) {
			    rec = 1. / xj;
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    d__1 = -z__;
		    dladiv_(&x[j1], &x[*n + j1], &tmp, &d__1, &sr, &si);
		    x[j1] = sr;
		    x[j1 + *n] = si;
/* Computing MAX */
		    d__3 = (d__1 = x[j1], abs(d__1)) + (d__2 = x[j1 + *n], 
			    abs(d__2));
		    xmax = max(d__3,xmax);

		} else {

/*                 2 by 2 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

/* Computing MAX */
		    d__5 = (d__1 = x[j1], abs(d__1)) + (d__2 = x[*n + j1], 
			    abs(d__2)), d__6 = (d__3 = x[j2], abs(d__3)) + (
			    d__4 = x[*n + j2], abs(d__4));
		    xj = max(d__5,d__6);
		    if (xmax > 1.) {
			rec = 1. / xmax;
/* Computing MAX */
			d__1 = work[j1], d__2 = work[j2];
			if (max(d__1,d__2) > (bignum - xj) / xmax) {
			    dscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    d__[0] = x[j1] - ddot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[1] = x[j2] - ddot_(&i__2, &t[j2 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[2] = x[*n + j1] - ddot_(&i__2, &t[j1 * t_dim1 + 1], &
			    c__1, &x[*n + 1], &c__1);
		    i__2 = j1 - 1;
		    d__[3] = x[*n + j2] - ddot_(&i__2, &t[j2 * t_dim1 + 1], &
			    c__1, &x[*n + 1], &c__1);
		    d__[0] -= b[j1] * x[*n + 1];
		    d__[1] -= b[j2] * x[*n + 1];
		    d__[2] += b[j1] * x[1];
		    d__[3] += b[j2] * x[1];

		    dlaln2_(&c_true, &c__2, &c__2, &sminw, &c_b21, &t[j1 + j1 
			    * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, w, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.) {
			dscal_(&n2, &scaloc, &x[1], &c__1);
			*scale = scaloc * *scale;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
		    x[*n + j1] = v[2];
		    x[*n + j2] = v[3];
/* Computing MAX */
		    d__5 = (d__1 = x[j1], abs(d__1)) + (d__2 = x[*n + j1], 
			    abs(d__2)), d__6 = (d__3 = x[j2], abs(d__3)) + (
			    d__4 = x[*n + j2], abs(d__4)), d__5 = max(d__5,
			    d__6);
		    xmax = max(d__5,xmax);

		}

L80:
		;
	    }

	}

    }

    return 0;

/*     End of DLAQTR */

} /* dlaqtr_ */
