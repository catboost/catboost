/* dlaein.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int dlaein_(logical *rightv, logical *noinit, integer *n, 
	doublereal *h__, integer *ldh, doublereal *wr, doublereal *wi, 
	doublereal *vr, doublereal *vi, doublereal *b, integer *ldb, 
	doublereal *work, doublereal *eps3, doublereal *smlnum, doublereal *
	bignum, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, h_dim1, h_offset, i__1, i__2, i__3, i__4;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j;
    doublereal w, x, y;
    integer i1, i2, i3;
    doublereal w1, ei, ej, xi, xr, rec;
    integer its, ierr;
    doublereal temp, norm, vmax;
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal scale;
    extern doublereal dasum_(integer *, doublereal *, integer *);
    char trans[1];
    doublereal vcrit, rootn, vnorm;
    extern doublereal dlapy2_(doublereal *, doublereal *);
    doublereal absbii, absbjj;
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dladiv_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *), dlatrs_(
	    char *, char *, char *, char *, integer *, doublereal *, integer *
, doublereal *, doublereal *, doublereal *, integer *);
    char normin[1];
    doublereal nrmsml, growto;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAEIN uses inverse iteration to find a right or left eigenvector */
/*  corresponding to the eigenvalue (WR,WI) of a real upper Hessenberg */
/*  matrix H. */

/*  Arguments */
/*  ========= */

/*  RIGHTV   (input) LOGICAL */
/*          = .TRUE. : compute right eigenvector; */
/*          = .FALSE.: compute left eigenvector. */

/*  NOINIT   (input) LOGICAL */
/*          = .TRUE. : no initial vector supplied in (VR,VI). */
/*          = .FALSE.: initial vector supplied in (VR,VI). */

/*  N       (input) INTEGER */
/*          The order of the matrix H.  N >= 0. */

/*  H       (input) DOUBLE PRECISION array, dimension (LDH,N) */
/*          The upper Hessenberg matrix H. */

/*  LDH     (input) INTEGER */
/*          The leading dimension of the array H.  LDH >= max(1,N). */

/*  WR      (input) DOUBLE PRECISION */
/*  WI      (input) DOUBLE PRECISION */
/*          The real and imaginary parts of the eigenvalue of H whose */
/*          corresponding right or left eigenvector is to be computed. */

/*  VR      (input/output) DOUBLE PRECISION array, dimension (N) */
/*  VI      (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, if NOINIT = .FALSE. and WI = 0.0, VR must contain */
/*          a real starting vector for inverse iteration using the real */
/*          eigenvalue WR; if NOINIT = .FALSE. and WI.ne.0.0, VR and VI */
/*          must contain the real and imaginary parts of a complex */
/*          starting vector for inverse iteration using the complex */
/*          eigenvalue (WR,WI); otherwise VR and VI need not be set. */
/*          On exit, if WI = 0.0 (real eigenvalue), VR contains the */
/*          computed real eigenvector; if WI.ne.0.0 (complex eigenvalue), */
/*          VR and VI contain the real and imaginary parts of the */
/*          computed complex eigenvector. The eigenvector is normalized */
/*          so that the component of largest magnitude has magnitude 1; */
/*          here the magnitude of a complex number (x,y) is taken to be */
/*          |x| + |y|. */
/*          VI is not referenced if WI = 0.0. */

/*  B       (workspace) DOUBLE PRECISION array, dimension (LDB,N) */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= N+1. */

/*  WORK   (workspace) DOUBLE PRECISION array, dimension (N) */

/*  EPS3    (input) DOUBLE PRECISION */
/*          A small machine-dependent value which is used to perturb */
/*          close eigenvalues, and to replace zero pivots. */

/*  SMLNUM  (input) DOUBLE PRECISION */
/*          A machine-dependent value close to the underflow threshold. */

/*  BIGNUM  (input) DOUBLE PRECISION */
/*          A machine-dependent value close to the overflow threshold. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          = 1:  inverse iteration did not converge; VR is set to the */
/*                last iterate, and so is VI if WI.ne.0.0. */

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
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --vr;
    --vi;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;

/*     GROWTO is the threshold used in the acceptance test for an */
/*     eigenvector. */

    rootn = sqrt((doublereal) (*n));
    growto = .1 / rootn;
/* Computing MAX */
    d__1 = 1., d__2 = *eps3 * rootn;
    nrmsml = max(d__1,d__2) * *smlnum;

/*     Form B = H - (WR,WI)*I (except that the subdiagonal elements and */
/*     the imaginary parts of the diagonal elements are not stored). */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    b[i__ + j * b_dim1] = h__[i__ + j * h_dim1];
/* L10: */
	}
	b[j + j * b_dim1] = h__[j + j * h_dim1] - *wr;
/* L20: */
    }

    if (*wi == 0.) {

/*        Real eigenvalue. */

	if (*noinit) {

/*           Set initial vector. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		vr[i__] = *eps3;
/* L30: */
	    }
	} else {

/*           Scale supplied initial vector. */

	    vnorm = dnrm2_(n, &vr[1], &c__1);
	    d__1 = *eps3 * rootn / max(vnorm,nrmsml);
	    dscal_(n, &d__1, &vr[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		ei = h__[i__ + 1 + i__ * h_dim1];
		if ((d__1 = b[i__ + i__ * b_dim1], abs(d__1)) < abs(ei)) {

/*                 Interchange rows and eliminate. */

		    x = b[i__ + i__ * b_dim1] / ei;
		    b[i__ + i__ * b_dim1] = ei;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			temp = b[i__ + 1 + j * b_dim1];
			b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - x * 
				temp;
			b[i__ + j * b_dim1] = temp;
/* L40: */
		    }
		} else {

/*                 Eliminate without interchange. */

		    if (b[i__ + i__ * b_dim1] == 0.) {
			b[i__ + i__ * b_dim1] = *eps3;
		    }
		    x = ei / b[i__ + i__ * b_dim1];
		    if (x != 0.) {
			i__2 = *n;
			for (j = i__ + 1; j <= i__2; ++j) {
			    b[i__ + 1 + j * b_dim1] -= x * b[i__ + j * b_dim1]
				    ;
/* L50: */
			}
		    }
		}
/* L60: */
	    }
	    if (b[*n + *n * b_dim1] == 0.) {
		b[*n + *n * b_dim1] = *eps3;
	    }

	    *(unsigned char *)trans = 'N';

	} else {

/*           UL decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		if ((d__1 = b[j + j * b_dim1], abs(d__1)) < abs(ej)) {

/*                 Interchange columns and eliminate. */

		    x = b[j + j * b_dim1] / ej;
		    b[j + j * b_dim1] = ej;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			temp = b[i__ + (j - 1) * b_dim1];
			b[i__ + (j - 1) * b_dim1] = b[i__ + j * b_dim1] - x * 
				temp;
			b[i__ + j * b_dim1] = temp;
/* L70: */
		    }
		} else {

/*                 Eliminate without interchange. */

		    if (b[j + j * b_dim1] == 0.) {
			b[j + j * b_dim1] = *eps3;
		    }
		    x = ej / b[j + j * b_dim1];
		    if (x != 0.) {
			i__1 = j - 1;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + (j - 1) * b_dim1] -= x * b[i__ + j * 
				    b_dim1];
/* L80: */
			}
		    }
		}
/* L90: */
	    }
	    if (b[b_dim1 + 1] == 0.) {
		b[b_dim1 + 1] = *eps3;
	    }

	    *(unsigned char *)trans = 'T';

	}

	*(unsigned char *)normin = 'N';
	i__1 = *n;
	for (its = 1; its <= i__1; ++its) {

/*           Solve U*x = scale*v for a right eigenvector */
/*             or U'*x = scale*v for a left eigenvector, */
/*           overwriting x on v. */

	    dlatrs_("Upper", trans, "Nonunit", normin, n, &b[b_offset], ldb, &
		    vr[1], &scale, &work[1], &ierr);
	    *(unsigned char *)normin = 'Y';

/*           Test for sufficient growth in the norm of v. */

	    vnorm = dasum_(n, &vr[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L120;
	    }

/*           Choose new orthogonal starting vector and try again. */

	    temp = *eps3 / (rootn + 1.);
	    vr[1] = *eps3;
	    i__2 = *n;
	    for (i__ = 2; i__ <= i__2; ++i__) {
		vr[i__] = temp;
/* L100: */
	    }
	    vr[*n - its + 1] -= *eps3 * rootn;
/* L110: */
	}

/*        Failure to find eigenvector in N iterations. */

	*info = 1;

L120:

/*        Normalize eigenvector. */

	i__ = idamax_(n, &vr[1], &c__1);
	d__2 = 1. / (d__1 = vr[i__], abs(d__1));
	dscal_(n, &d__2, &vr[1], &c__1);
    } else {

/*        Complex eigenvalue. */

	if (*noinit) {

/*           Set initial vector. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		vr[i__] = *eps3;
		vi[i__] = 0.;
/* L130: */
	    }
	} else {

/*           Scale supplied initial vector. */

	    d__1 = dnrm2_(n, &vr[1], &c__1);
	    d__2 = dnrm2_(n, &vi[1], &c__1);
	    norm = dlapy2_(&d__1, &d__2);
	    rec = *eps3 * rootn / max(norm,nrmsml);
	    dscal_(n, &rec, &vr[1], &c__1);
	    dscal_(n, &rec, &vi[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

/*           The imaginary part of the (i,j)-th element of U is stored in */
/*           B(j+1,i). */

	    b[b_dim1 + 2] = -(*wi);
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		b[i__ + 1 + b_dim1] = 0.;
/* L140: */
	    }

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		absbii = dlapy2_(&b[i__ + i__ * b_dim1], &b[i__ + 1 + i__ * 
			b_dim1]);
		ei = h__[i__ + 1 + i__ * h_dim1];
		if (absbii < abs(ei)) {

/*                 Interchange rows and eliminate. */

		    xr = b[i__ + i__ * b_dim1] / ei;
		    xi = b[i__ + 1 + i__ * b_dim1] / ei;
		    b[i__ + i__ * b_dim1] = ei;
		    b[i__ + 1 + i__ * b_dim1] = 0.;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			temp = b[i__ + 1 + j * b_dim1];
			b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - xr * 
				temp;
			b[j + 1 + (i__ + 1) * b_dim1] = b[j + 1 + i__ * 
				b_dim1] - xi * temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.;
/* L150: */
		    }
		    b[i__ + 2 + i__ * b_dim1] = -(*wi);
		    b[i__ + 1 + (i__ + 1) * b_dim1] -= xi * *wi;
		    b[i__ + 2 + (i__ + 1) * b_dim1] += xr * *wi;
		} else {

/*                 Eliminate without interchanging rows. */

		    if (absbii == 0.) {
			b[i__ + i__ * b_dim1] = *eps3;
			b[i__ + 1 + i__ * b_dim1] = 0.;
			absbii = *eps3;
		    }
		    ei = ei / absbii / absbii;
		    xr = b[i__ + i__ * b_dim1] * ei;
		    xi = -b[i__ + 1 + i__ * b_dim1] * ei;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			b[i__ + 1 + j * b_dim1] = b[i__ + 1 + j * b_dim1] - 
				xr * b[i__ + j * b_dim1] + xi * b[j + 1 + i__ 
				* b_dim1];
			b[j + 1 + (i__ + 1) * b_dim1] = -xr * b[j + 1 + i__ * 
				b_dim1] - xi * b[i__ + j * b_dim1];
/* L160: */
		    }
		    b[i__ + 2 + (i__ + 1) * b_dim1] -= *wi;
		}

/*              Compute 1-norm of offdiagonal elements of i-th row. */

		i__2 = *n - i__;
		i__3 = *n - i__;
		work[i__] = dasum_(&i__2, &b[i__ + (i__ + 1) * b_dim1], ldb) 
			+ dasum_(&i__3, &b[i__ + 2 + i__ * b_dim1], &c__1);
/* L170: */
	    }
	    if (b[*n + *n * b_dim1] == 0. && b[*n + 1 + *n * b_dim1] == 0.) {
		b[*n + *n * b_dim1] = *eps3;
	    }
	    work[*n] = 0.;

	    i1 = *n;
	    i2 = 1;
	    i3 = -1;
	} else {

/*           UL decomposition with partial pivoting of conjg(B), */
/*           replacing zero pivots by EPS3. */

/*           The imaginary part of the (i,j)-th element of U is stored in */
/*           B(j+1,i). */

	    b[*n + 1 + *n * b_dim1] = *wi;
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		b[*n + 1 + j * b_dim1] = 0.;
/* L180: */
	    }

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		absbjj = dlapy2_(&b[j + j * b_dim1], &b[j + 1 + j * b_dim1]);
		if (absbjj < abs(ej)) {

/*                 Interchange columns and eliminate */

		    xr = b[j + j * b_dim1] / ej;
		    xi = b[j + 1 + j * b_dim1] / ej;
		    b[j + j * b_dim1] = ej;
		    b[j + 1 + j * b_dim1] = 0.;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			temp = b[i__ + (j - 1) * b_dim1];
			b[i__ + (j - 1) * b_dim1] = b[i__ + j * b_dim1] - xr *
				 temp;
			b[j + i__ * b_dim1] = b[j + 1 + i__ * b_dim1] - xi * 
				temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.;
/* L190: */
		    }
		    b[j + 1 + (j - 1) * b_dim1] = *wi;
		    b[j - 1 + (j - 1) * b_dim1] += xi * *wi;
		    b[j + (j - 1) * b_dim1] -= xr * *wi;
		} else {

/*                 Eliminate without interchange. */

		    if (absbjj == 0.) {
			b[j + j * b_dim1] = *eps3;
			b[j + 1 + j * b_dim1] = 0.;
			absbjj = *eps3;
		    }
		    ej = ej / absbjj / absbjj;
		    xr = b[j + j * b_dim1] * ej;
		    xi = -b[j + 1 + j * b_dim1] * ej;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			b[i__ + (j - 1) * b_dim1] = b[i__ + (j - 1) * b_dim1] 
				- xr * b[i__ + j * b_dim1] + xi * b[j + 1 + 
				i__ * b_dim1];
			b[j + i__ * b_dim1] = -xr * b[j + 1 + i__ * b_dim1] - 
				xi * b[i__ + j * b_dim1];
/* L200: */
		    }
		    b[j + (j - 1) * b_dim1] += *wi;
		}

/*              Compute 1-norm of offdiagonal elements of j-th column. */

		i__1 = j - 1;
		i__2 = j - 1;
		work[j] = dasum_(&i__1, &b[j * b_dim1 + 1], &c__1) + dasum_(&
			i__2, &b[j + 1 + b_dim1], ldb);
/* L210: */
	    }
	    if (b[b_dim1 + 1] == 0. && b[b_dim1 + 2] == 0.) {
		b[b_dim1 + 1] = *eps3;
	    }
	    work[1] = 0.;

	    i1 = 1;
	    i2 = *n;
	    i3 = 1;
	}

	i__1 = *n;
	for (its = 1; its <= i__1; ++its) {
	    scale = 1.;
	    vmax = 1.;
	    vcrit = *bignum;

/*           Solve U*(xr,xi) = scale*(vr,vi) for a right eigenvector, */
/*             or U'*(xr,xi) = scale*(vr,vi) for a left eigenvector, */
/*           overwriting (xr,xi) on (vr,vi). */

	    i__2 = i2;
	    i__3 = i3;
	    for (i__ = i1; i__3 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__3) 
		    {

		if (work[i__] > vcrit) {
		    rec = 1. / vmax;
		    dscal_(n, &rec, &vr[1], &c__1);
		    dscal_(n, &rec, &vi[1], &c__1);
		    scale *= rec;
		    vmax = 1.;
		    vcrit = *bignum;
		}

		xr = vr[i__];
		xi = vi[i__];
		if (*rightv) {
		    i__4 = *n;
		    for (j = i__ + 1; j <= i__4; ++j) {
			xr = xr - b[i__ + j * b_dim1] * vr[j] + b[j + 1 + i__ 
				* b_dim1] * vi[j];
			xi = xi - b[i__ + j * b_dim1] * vi[j] - b[j + 1 + i__ 
				* b_dim1] * vr[j];
/* L220: */
		    }
		} else {
		    i__4 = i__ - 1;
		    for (j = 1; j <= i__4; ++j) {
			xr = xr - b[j + i__ * b_dim1] * vr[j] + b[i__ + 1 + j 
				* b_dim1] * vi[j];
			xi = xi - b[j + i__ * b_dim1] * vi[j] - b[i__ + 1 + j 
				* b_dim1] * vr[j];
/* L230: */
		    }
		}

		w = (d__1 = b[i__ + i__ * b_dim1], abs(d__1)) + (d__2 = b[i__ 
			+ 1 + i__ * b_dim1], abs(d__2));
		if (w > *smlnum) {
		    if (w < 1.) {
			w1 = abs(xr) + abs(xi);
			if (w1 > w * *bignum) {
			    rec = 1. / w1;
			    dscal_(n, &rec, &vr[1], &c__1);
			    dscal_(n, &rec, &vi[1], &c__1);
			    xr = vr[i__];
			    xi = vi[i__];
			    scale *= rec;
			    vmax *= rec;
			}
		    }

/*                 Divide by diagonal element of B. */

		    dladiv_(&xr, &xi, &b[i__ + i__ * b_dim1], &b[i__ + 1 + 
			    i__ * b_dim1], &vr[i__], &vi[i__]);
/* Computing MAX */
		    d__3 = (d__1 = vr[i__], abs(d__1)) + (d__2 = vi[i__], abs(
			    d__2));
		    vmax = max(d__3,vmax);
		    vcrit = *bignum / vmax;
		} else {
		    i__4 = *n;
		    for (j = 1; j <= i__4; ++j) {
			vr[j] = 0.;
			vi[j] = 0.;
/* L240: */
		    }
		    vr[i__] = 1.;
		    vi[i__] = 1.;
		    scale = 0.;
		    vmax = 1.;
		    vcrit = *bignum;
		}
/* L250: */
	    }

/*           Test for sufficient growth in the norm of (VR,VI). */

	    vnorm = dasum_(n, &vr[1], &c__1) + dasum_(n, &vi[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L280;
	    }

/*           Choose a new orthogonal starting vector and try again. */

	    y = *eps3 / (rootn + 1.);
	    vr[1] = *eps3;
	    vi[1] = 0.;

	    i__3 = *n;
	    for (i__ = 2; i__ <= i__3; ++i__) {
		vr[i__] = y;
		vi[i__] = 0.;
/* L260: */
	    }
	    vr[*n - its + 1] -= *eps3 * rootn;
/* L270: */
	}

/*        Failure to find eigenvector in N iterations */

	*info = 1;

L280:

/*        Normalize eigenvector. */

	vnorm = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    d__3 = vnorm, d__4 = (d__1 = vr[i__], abs(d__1)) + (d__2 = vi[i__]
		    , abs(d__2));
	    vnorm = max(d__3,d__4);
/* L290: */
	}
	d__1 = 1. / vnorm;
	dscal_(n, &d__1, &vr[1], &c__1);
	d__1 = 1. / vnorm;
	dscal_(n, &d__1, &vi[1], &c__1);

    }

    return 0;

/*     End of DLAEIN */

} /* dlaein_ */
