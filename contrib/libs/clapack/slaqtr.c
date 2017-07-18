/* slaqtr.f -- translated by f2c (version 20061008).
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
static real c_b21 = 1.f;
static real c_b25 = 0.f;
static logical c_true = TRUE_;

/* Subroutine */ int slaqtr_(logical *ltran, logical *lreal, integer *n, real 
	*t, integer *ldt, real *b, real *w, real *scale, real *x, real *work, 
	integer *info)
{
    /* System generated locals */
    integer t_dim1, t_offset, i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6;

    /* Local variables */
    real d__[4]	/* was [2][2] */;
    integer i__, j, k;
    real v[4]	/* was [2][2] */, z__;
    integer j1, j2, n1, n2;
    real si, xj, sr, rec, eps, tjj, tmp;
    integer ierr;
    real smin;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    real xmax;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    integer jnext;
    extern doublereal sasum_(integer *, real *, integer *);
    real sminw, xnorm;
    extern /* Subroutine */ int saxpy_(integer *, real *, real *, integer *, 
	    real *, integer *), slaln2_(logical *, integer *, integer *, real 
	    *, real *, real *, integer *, real *, real *, real *, integer *, 
	    real *, real *, real *, integer *, real *, real *, integer *);
    real scaloc;
    extern doublereal slamch_(char *), slange_(char *, integer *, 
	    integer *, real *, integer *, real *);
    real bignum;
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int sladiv_(real *, real *, real *, real *, real *
, real *);
    logical notran;
    real smlnum;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAQTR solves the real quasi-triangular system */

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
/*  in routine STRSNA. */

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

/*  T       (input) REAL array, dimension (LDT,N) */
/*          On entry, T contains a matrix in Schur canonical form. */
/*          If LREAL = .FALSE., then the first diagonal block of T must */
/*          be 1 by 1. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the matrix T. LDT >= max(1,N). */

/*  B       (input) REAL array, dimension (N) */
/*          On entry, B contains the elements to form the matrix */
/*          B as described above. */
/*          If LREAL = .TRUE., B is not referenced. */

/*  W       (input) REAL */
/*          On entry, W is the diagonal element of the matrix B. */
/*          If LREAL = .TRUE., W is not referenced. */

/*  SCALE   (output) REAL */
/*          On exit, SCALE is the scale factor. */

/*  X       (input/output) REAL array, dimension (2*N) */
/*          On entry, X contains the right hand side of the system. */
/*          On exit, X is overwritten by the solution. */

/*  WORK    (workspace) REAL array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          On exit, INFO is set to */
/*             0: successful exit. */
/*               1: the some diagonal 1 by 1 block has been perturbed by */
/*                  a small number SMIN to keep nonsingularity. */
/*               2: the some diagonal 2 by 2 block has been perturbed by */
/*                  a small number in SLALN2 to keep nonsingularity. */
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

    eps = slamch_("P");
    smlnum = slamch_("S") / eps;
    bignum = 1.f / smlnum;

    xnorm = slange_("M", n, n, &t[t_offset], ldt, d__);
    if (! (*lreal)) {
/* Computing MAX */
	r__1 = xnorm, r__2 = dabs(*w), r__1 = max(r__1,r__2), r__2 = slange_(
		"M", n, &c__1, &b[1], n, d__);
	xnorm = dmax(r__1,r__2);
    }
/* Computing MAX */
    r__1 = smlnum, r__2 = eps * xnorm;
    smin = dmax(r__1,r__2);

/*     Compute 1-norm of each column of strictly upper triangular */
/*     part of T to control overflow in triangular solver. */

    work[1] = 0.f;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	i__2 = j - 1;
	work[j] = sasum_(&i__2, &t[j * t_dim1 + 1], &c__1);
/* L10: */
    }

    if (! (*lreal)) {
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    work[i__] += (r__1 = b[i__], dabs(r__1));
/* L20: */
	}
    }

    n2 = *n << 1;
    n1 = *n;
    if (! (*lreal)) {
	n1 = n2;
    }
    k = isamax_(&n1, &x[1], &c__1);
    xmax = (r__1 = x[k], dabs(r__1));
    *scale = 1.f;

    if (xmax > bignum) {
	*scale = bignum / xmax;
	sscal_(&n1, scale, &x[1], &c__1);
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
		    if (t[j + (j - 1) * t_dim1] != 0.f) {
			j1 = j - 1;
			jnext = j - 2;
		    }
		}

		if (j1 == j2) {

/*                 Meet 1 by 1 diagonal block */

/*                 Scale to avoid overflow when computing */
/*                     x(j) = b(j)/T(j,j) */

		    xj = (r__1 = x[j1], dabs(r__1));
		    tjj = (r__1 = t[j1 + j1 * t_dim1], dabs(r__1));
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < smin) {
			tmp = smin;
			tjj = smin;
			*info = 1;
		    }

		    if (xj == 0.f) {
			goto L30;
		    }

		    if (tjj < 1.f) {
			if (xj > bignum * tjj) {
			    rec = 1.f / xj;
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    x[j1] /= tmp;
		    xj = (r__1 = x[j1], dabs(r__1));

/*                 Scale x if necessary to avoid overflow when adding a */
/*                 multiple of column j1 of T. */

		    if (xj > 1.f) {
			rec = 1.f / xj;
			if (work[j1] > (bignum - xmax) * rec) {
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }
		    if (j1 > 1) {
			i__1 = j1 - 1;
			r__1 = -x[j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			k = isamax_(&i__1, &x[1], &c__1);
			xmax = (r__1 = x[k], dabs(r__1));
		    }

		} else {

/*                 Meet 2 by 2 diagonal block */

/*                 Call 2 by 2 linear system solve, to take */
/*                 care of possible overflow by scaling factor. */

		    d__[0] = x[j1];
		    d__[1] = x[j2];
		    slaln2_(&c_false, &c__2, &c__1, &smin, &c_b21, &t[j1 + j1 
			    * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, &c_b25, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.f) {
			sscal_(n, &scaloc, &x[1], &c__1);
			*scale *= scaloc;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];

/*                 Scale V(1,1) (= X(J1)) and/or V(2,1) (=X(J2)) */
/*                 to avoid overflow in updating right-hand side. */

/* Computing MAX */
		    r__1 = dabs(v[0]), r__2 = dabs(v[1]);
		    xj = dmax(r__1,r__2);
		    if (xj > 1.f) {
			rec = 1.f / xj;
/* Computing MAX */
			r__1 = work[j1], r__2 = work[j2];
			if (dmax(r__1,r__2) > (bignum - xmax) * rec) {
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

/*                 Update right-hand side */

		    if (j1 > 1) {
			i__1 = j1 - 1;
			r__1 = -x[j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			r__1 = -x[j2];
			saxpy_(&i__1, &r__1, &t[j2 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			k = isamax_(&i__1, &x[1], &c__1);
			xmax = (r__1 = x[k], dabs(r__1));
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
		    if (t[j + 1 + j * t_dim1] != 0.f) {
			j2 = j + 1;
			jnext = j + 2;
		    }
		}

		if (j1 == j2) {

/*                 1 by 1 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

		    xj = (r__1 = x[j1], dabs(r__1));
		    if (xmax > 1.f) {
			rec = 1.f / xmax;
			if (work[j1] > (bignum - xj) * rec) {
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    x[j1] -= sdot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[1], &
			    c__1);

		    xj = (r__1 = x[j1], dabs(r__1));
		    tjj = (r__1 = t[j1 + j1 * t_dim1], dabs(r__1));
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < smin) {
			tmp = smin;
			tjj = smin;
			*info = 1;
		    }

		    if (tjj < 1.f) {
			if (xj > bignum * tjj) {
			    rec = 1.f / xj;
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    x[j1] /= tmp;
/* Computing MAX */
		    r__2 = xmax, r__3 = (r__1 = x[j1], dabs(r__1));
		    xmax = dmax(r__2,r__3);

		} else {

/*                 2 by 2 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side elements by inner product. */

/* Computing MAX */
		    r__3 = (r__1 = x[j1], dabs(r__1)), r__4 = (r__2 = x[j2], 
			    dabs(r__2));
		    xj = dmax(r__3,r__4);
		    if (xmax > 1.f) {
			rec = 1.f / xmax;
/* Computing MAX */
			r__1 = work[j2], r__2 = work[j1];
			if (dmax(r__1,r__2) > (bignum - xj) * rec) {
			    sscal_(n, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    d__[0] = x[j1] - sdot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[1] = x[j2] - sdot_(&i__2, &t[j2 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);

		    slaln2_(&c_true, &c__2, &c__1, &smin, &c_b21, &t[j1 + j1 *
			     t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &c_b25, 
			     &c_b25, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.f) {
			sscal_(n, &scaloc, &x[1], &c__1);
			*scale *= scaloc;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
/* Computing MAX */
		    r__3 = (r__1 = x[j1], dabs(r__1)), r__4 = (r__2 = x[j2], 
			    dabs(r__2)), r__3 = max(r__3,r__4);
		    xmax = dmax(r__3,xmax);

		}
L40:
		;
	    }
	}

    } else {

/* Computing MAX */
	r__1 = eps * dabs(*w);
	sminw = dmax(r__1,smin);
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
		    if (t[j + (j - 1) * t_dim1] != 0.f) {
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
		    xj = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[*n + j1], 
			    dabs(r__2));
		    tjj = (r__1 = t[j1 + j1 * t_dim1], dabs(r__1)) + dabs(z__)
			    ;
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < sminw) {
			tmp = sminw;
			tjj = sminw;
			*info = 1;
		    }

		    if (xj == 0.f) {
			goto L70;
		    }

		    if (tjj < 1.f) {
			if (xj > bignum * tjj) {
			    rec = 1.f / xj;
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    sladiv_(&x[j1], &x[*n + j1], &tmp, &z__, &sr, &si);
		    x[j1] = sr;
		    x[*n + j1] = si;
		    xj = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[*n + j1], 
			    dabs(r__2));

/*                 Scale x if necessary to avoid overflow when adding a */
/*                 multiple of column j1 of T. */

		    if (xj > 1.f) {
			rec = 1.f / xj;
			if (work[j1] > (bignum - xmax) * rec) {
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

		    if (j1 > 1) {
			i__1 = j1 - 1;
			r__1 = -x[j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			r__1 = -x[*n + j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);

			x[1] += b[j1] * x[*n + j1];
			x[*n + 1] -= b[j1] * x[j1];

			xmax = 0.f;
			i__1 = j1 - 1;
			for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			    r__3 = xmax, r__4 = (r__1 = x[k], dabs(r__1)) + (
				    r__2 = x[k + *n], dabs(r__2));
			    xmax = dmax(r__3,r__4);
/* L50: */
			}
		    }

		} else {

/*                 Meet 2 by 2 diagonal block */

		    d__[0] = x[j1];
		    d__[1] = x[j2];
		    d__[2] = x[*n + j1];
		    d__[3] = x[*n + j2];
		    r__1 = -(*w);
		    slaln2_(&c_false, &c__2, &c__2, &sminw, &c_b21, &t[j1 + 
			    j1 * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, &r__1, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.f) {
			i__1 = *n << 1;
			sscal_(&i__1, &scaloc, &x[1], &c__1);
			*scale = scaloc * *scale;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
		    x[*n + j1] = v[2];
		    x[*n + j2] = v[3];

/*                 Scale X(J1), .... to avoid overflow in */
/*                 updating right hand side. */

/* Computing MAX */
		    r__1 = dabs(v[0]) + dabs(v[2]), r__2 = dabs(v[1]) + dabs(
			    v[3]);
		    xj = dmax(r__1,r__2);
		    if (xj > 1.f) {
			rec = 1.f / xj;
/* Computing MAX */
			r__1 = work[j1], r__2 = work[j2];
			if (dmax(r__1,r__2) > (bignum - xmax) * rec) {
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			}
		    }

/*                 Update the right-hand side. */

		    if (j1 > 1) {
			i__1 = j1 - 1;
			r__1 = -x[j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[1]
, &c__1);
			i__1 = j1 - 1;
			r__1 = -x[j2];
			saxpy_(&i__1, &r__1, &t[j2 * t_dim1 + 1], &c__1, &x[1]
, &c__1);

			i__1 = j1 - 1;
			r__1 = -x[*n + j1];
			saxpy_(&i__1, &r__1, &t[j1 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);
			i__1 = j1 - 1;
			r__1 = -x[*n + j2];
			saxpy_(&i__1, &r__1, &t[j2 * t_dim1 + 1], &c__1, &x[*
				n + 1], &c__1);

			x[1] = x[1] + b[j1] * x[*n + j1] + b[j2] * x[*n + j2];
			x[*n + 1] = x[*n + 1] - b[j1] * x[j1] - b[j2] * x[j2];

			xmax = 0.f;
			i__1 = j1 - 1;
			for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
			    r__3 = (r__1 = x[k], dabs(r__1)) + (r__2 = x[k + *
				    n], dabs(r__2));
			    xmax = dmax(r__3,xmax);
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
		    if (t[j + 1 + j * t_dim1] != 0.f) {
			j2 = j + 1;
			jnext = j + 2;
		    }
		}

		if (j1 == j2) {

/*                 1 by 1 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

		    xj = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[j1 + *n], 
			    dabs(r__2));
		    if (xmax > 1.f) {
			rec = 1.f / xmax;
			if (work[j1] > (bignum - xj) * rec) {
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    x[j1] -= sdot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[1], &
			    c__1);
		    i__2 = j1 - 1;
		    x[*n + j1] -= sdot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, &x[
			    *n + 1], &c__1);
		    if (j1 > 1) {
			x[j1] -= b[j1] * x[*n + 1];
			x[*n + j1] += b[j1] * x[1];
		    }
		    xj = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[j1 + *n], 
			    dabs(r__2));

		    z__ = *w;
		    if (j1 == 1) {
			z__ = b[1];
		    }

/*                 Scale if necessary to avoid overflow in */
/*                 complex division */

		    tjj = (r__1 = t[j1 + j1 * t_dim1], dabs(r__1)) + dabs(z__)
			    ;
		    tmp = t[j1 + j1 * t_dim1];
		    if (tjj < sminw) {
			tmp = sminw;
			tjj = sminw;
			*info = 1;
		    }

		    if (tjj < 1.f) {
			if (xj > bignum * tjj) {
			    rec = 1.f / xj;
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }
		    r__1 = -z__;
		    sladiv_(&x[j1], &x[*n + j1], &tmp, &r__1, &sr, &si);
		    x[j1] = sr;
		    x[j1 + *n] = si;
/* Computing MAX */
		    r__3 = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[j1 + *n], 
			    dabs(r__2));
		    xmax = dmax(r__3,xmax);

		} else {

/*                 2 by 2 diagonal block */

/*                 Scale if necessary to avoid overflow in forming the */
/*                 right-hand side element by inner product. */

/* Computing MAX */
		    r__5 = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[*n + j1], 
			    dabs(r__2)), r__6 = (r__3 = x[j2], dabs(r__3)) + (
			    r__4 = x[*n + j2], dabs(r__4));
		    xj = dmax(r__5,r__6);
		    if (xmax > 1.f) {
			rec = 1.f / xmax;
/* Computing MAX */
			r__1 = work[j1], r__2 = work[j2];
			if (dmax(r__1,r__2) > (bignum - xj) / xmax) {
			    sscal_(&n2, &rec, &x[1], &c__1);
			    *scale *= rec;
			    xmax *= rec;
			}
		    }

		    i__2 = j1 - 1;
		    d__[0] = x[j1] - sdot_(&i__2, &t[j1 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[1] = x[j2] - sdot_(&i__2, &t[j2 * t_dim1 + 1], &c__1, 
			    &x[1], &c__1);
		    i__2 = j1 - 1;
		    d__[2] = x[*n + j1] - sdot_(&i__2, &t[j1 * t_dim1 + 1], &
			    c__1, &x[*n + 1], &c__1);
		    i__2 = j1 - 1;
		    d__[3] = x[*n + j2] - sdot_(&i__2, &t[j2 * t_dim1 + 1], &
			    c__1, &x[*n + 1], &c__1);
		    d__[0] -= b[j1] * x[*n + 1];
		    d__[1] -= b[j2] * x[*n + 1];
		    d__[2] += b[j1] * x[1];
		    d__[3] += b[j2] * x[1];

		    slaln2_(&c_true, &c__2, &c__2, &sminw, &c_b21, &t[j1 + j1 
			    * t_dim1], ldt, &c_b21, &c_b21, d__, &c__2, &
			    c_b25, w, v, &c__2, &scaloc, &xnorm, &ierr);
		    if (ierr != 0) {
			*info = 2;
		    }

		    if (scaloc != 1.f) {
			sscal_(&n2, &scaloc, &x[1], &c__1);
			*scale = scaloc * *scale;
		    }
		    x[j1] = v[0];
		    x[j2] = v[1];
		    x[*n + j1] = v[2];
		    x[*n + j2] = v[3];
/* Computing MAX */
		    r__5 = (r__1 = x[j1], dabs(r__1)) + (r__2 = x[*n + j1], 
			    dabs(r__2)), r__6 = (r__3 = x[j2], dabs(r__3)) + (
			    r__4 = x[*n + j2], dabs(r__4)), r__5 = max(r__5,
			    r__6);
		    xmax = dmax(r__5,xmax);

		}

L80:
		;
	    }

	}

    }

    return 0;

/*     End of SLAQTR */

} /* slaqtr_ */
