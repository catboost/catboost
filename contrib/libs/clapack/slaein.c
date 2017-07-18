/* slaein.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slaein_(logical *rightv, logical *noinit, integer *n, 
	real *h__, integer *ldh, real *wr, real *wi, real *vr, real *vi, real 
	*b, integer *ldb, real *work, real *eps3, real *smlnum, real *bignum, 
	integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, h_dim1, h_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j;
    real w, x, y;
    integer i1, i2, i3;
    real w1, ei, ej, xi, xr, rec;
    integer its, ierr;
    real temp, norm, vmax;
    extern doublereal snrm2_(integer *, real *, integer *);
    real scale;
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    char trans[1];
    real vcrit;
    extern doublereal sasum_(integer *, real *, integer *);
    real rootn, vnorm;
    extern doublereal slapy2_(real *, real *);
    real absbii, absbjj;
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int sladiv_(real *, real *, real *, real *, real *
, real *);
    char normin[1];
    real nrmsml;
    extern /* Subroutine */ int slatrs_(char *, char *, char *, char *, 
	    integer *, real *, integer *, real *, real *, real *, integer *);
    real growto;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAEIN uses inverse iteration to find a right or left eigenvector */
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

/*  H       (input) REAL array, dimension (LDH,N) */
/*          The upper Hessenberg matrix H. */

/*  LDH     (input) INTEGER */
/*          The leading dimension of the array H.  LDH >= max(1,N). */

/*  WR      (input) REAL */
/*  WI      (input) REAL */
/*          The real and imaginary parts of the eigenvalue of H whose */
/*          corresponding right or left eigenvector is to be computed. */

/*  VR      (input/output) REAL array, dimension (N) */
/*  VI      (input/output) REAL array, dimension (N) */
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

/*  B       (workspace) REAL array, dimension (LDB,N) */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= N+1. */

/*  WORK   (workspace) REAL array, dimension (N) */

/*  EPS3    (input) REAL */
/*          A small machine-dependent value which is used to perturb */
/*          close eigenvalues, and to replace zero pivots. */

/*  SMLNUM  (input) REAL */
/*          A machine-dependent value close to the underflow threshold. */

/*  BIGNUM  (input) REAL */
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

    rootn = sqrt((real) (*n));
    growto = .1f / rootn;
/* Computing MAX */
    r__1 = 1.f, r__2 = *eps3 * rootn;
    nrmsml = dmax(r__1,r__2) * *smlnum;

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

    if (*wi == 0.f) {

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

	    vnorm = snrm2_(n, &vr[1], &c__1);
	    r__1 = *eps3 * rootn / dmax(vnorm,nrmsml);
	    sscal_(n, &r__1, &vr[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		ei = h__[i__ + 1 + i__ * h_dim1];
		if ((r__1 = b[i__ + i__ * b_dim1], dabs(r__1)) < dabs(ei)) {

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

		    if (b[i__ + i__ * b_dim1] == 0.f) {
			b[i__ + i__ * b_dim1] = *eps3;
		    }
		    x = ei / b[i__ + i__ * b_dim1];
		    if (x != 0.f) {
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
	    if (b[*n + *n * b_dim1] == 0.f) {
		b[*n + *n * b_dim1] = *eps3;
	    }

	    *(unsigned char *)trans = 'N';

	} else {

/*           UL decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		if ((r__1 = b[j + j * b_dim1], dabs(r__1)) < dabs(ej)) {

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

		    if (b[j + j * b_dim1] == 0.f) {
			b[j + j * b_dim1] = *eps3;
		    }
		    x = ej / b[j + j * b_dim1];
		    if (x != 0.f) {
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
	    if (b[b_dim1 + 1] == 0.f) {
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

	    slatrs_("Upper", trans, "Nonunit", normin, n, &b[b_offset], ldb, &
		    vr[1], &scale, &work[1], &ierr);
	    *(unsigned char *)normin = 'Y';

/*           Test for sufficient growth in the norm of v. */

	    vnorm = sasum_(n, &vr[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L120;
	    }

/*           Choose new orthogonal starting vector and try again. */

	    temp = *eps3 / (rootn + 1.f);
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

	i__ = isamax_(n, &vr[1], &c__1);
	r__2 = 1.f / (r__1 = vr[i__], dabs(r__1));
	sscal_(n, &r__2, &vr[1], &c__1);
    } else {

/*        Complex eigenvalue. */

	if (*noinit) {

/*           Set initial vector. */

	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		vr[i__] = *eps3;
		vi[i__] = 0.f;
/* L130: */
	    }
	} else {

/*           Scale supplied initial vector. */

	    r__1 = snrm2_(n, &vr[1], &c__1);
	    r__2 = snrm2_(n, &vi[1], &c__1);
	    norm = slapy2_(&r__1, &r__2);
	    rec = *eps3 * rootn / dmax(norm,nrmsml);
	    sscal_(n, &rec, &vr[1], &c__1);
	    sscal_(n, &rec, &vi[1], &c__1);
	}

	if (*rightv) {

/*           LU decomposition with partial pivoting of B, replacing zero */
/*           pivots by EPS3. */

/*           The imaginary part of the (i,j)-th element of U is stored in */
/*           B(j+1,i). */

	    b[b_dim1 + 2] = -(*wi);
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		b[i__ + 1 + b_dim1] = 0.f;
/* L140: */
	    }

	    i__1 = *n - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		absbii = slapy2_(&b[i__ + i__ * b_dim1], &b[i__ + 1 + i__ * 
			b_dim1]);
		ei = h__[i__ + 1 + i__ * h_dim1];
		if (absbii < dabs(ei)) {

/*                 Interchange rows and eliminate. */

		    xr = b[i__ + i__ * b_dim1] / ei;
		    xi = b[i__ + 1 + i__ * b_dim1] / ei;
		    b[i__ + i__ * b_dim1] = ei;
		    b[i__ + 1 + i__ * b_dim1] = 0.f;
		    i__2 = *n;
		    for (j = i__ + 1; j <= i__2; ++j) {
			temp = b[i__ + 1 + j * b_dim1];
			b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - xr * 
				temp;
			b[j + 1 + (i__ + 1) * b_dim1] = b[j + 1 + i__ * 
				b_dim1] - xi * temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.f;
/* L150: */
		    }
		    b[i__ + 2 + i__ * b_dim1] = -(*wi);
		    b[i__ + 1 + (i__ + 1) * b_dim1] -= xi * *wi;
		    b[i__ + 2 + (i__ + 1) * b_dim1] += xr * *wi;
		} else {

/*                 Eliminate without interchanging rows. */

		    if (absbii == 0.f) {
			b[i__ + i__ * b_dim1] = *eps3;
			b[i__ + 1 + i__ * b_dim1] = 0.f;
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
		work[i__] = sasum_(&i__2, &b[i__ + (i__ + 1) * b_dim1], ldb) 
			+ sasum_(&i__3, &b[i__ + 2 + i__ * b_dim1], &c__1);
/* L170: */
	    }
	    if (b[*n + *n * b_dim1] == 0.f && b[*n + 1 + *n * b_dim1] == 0.f) 
		    {
		b[*n + *n * b_dim1] = *eps3;
	    }
	    work[*n] = 0.f;

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
		b[*n + 1 + j * b_dim1] = 0.f;
/* L180: */
	    }

	    for (j = *n; j >= 2; --j) {
		ej = h__[j + (j - 1) * h_dim1];
		absbjj = slapy2_(&b[j + j * b_dim1], &b[j + 1 + j * b_dim1]);
		if (absbjj < dabs(ej)) {

/*                 Interchange columns and eliminate */

		    xr = b[j + j * b_dim1] / ej;
		    xi = b[j + 1 + j * b_dim1] / ej;
		    b[j + j * b_dim1] = ej;
		    b[j + 1 + j * b_dim1] = 0.f;
		    i__1 = j - 1;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			temp = b[i__ + (j - 1) * b_dim1];
			b[i__ + (j - 1) * b_dim1] = b[i__ + j * b_dim1] - xr *
				 temp;
			b[j + i__ * b_dim1] = b[j + 1 + i__ * b_dim1] - xi * 
				temp;
			b[i__ + j * b_dim1] = temp;
			b[j + 1 + i__ * b_dim1] = 0.f;
/* L190: */
		    }
		    b[j + 1 + (j - 1) * b_dim1] = *wi;
		    b[j - 1 + (j - 1) * b_dim1] += xi * *wi;
		    b[j + (j - 1) * b_dim1] -= xr * *wi;
		} else {

/*                 Eliminate without interchange. */

		    if (absbjj == 0.f) {
			b[j + j * b_dim1] = *eps3;
			b[j + 1 + j * b_dim1] = 0.f;
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
		work[j] = sasum_(&i__1, &b[j * b_dim1 + 1], &c__1) + sasum_(&
			i__2, &b[j + 1 + b_dim1], ldb);
/* L210: */
	    }
	    if (b[b_dim1 + 1] == 0.f && b[b_dim1 + 2] == 0.f) {
		b[b_dim1 + 1] = *eps3;
	    }
	    work[1] = 0.f;

	    i1 = 1;
	    i2 = *n;
	    i3 = 1;
	}

	i__1 = *n;
	for (its = 1; its <= i__1; ++its) {
	    scale = 1.f;
	    vmax = 1.f;
	    vcrit = *bignum;

/*           Solve U*(xr,xi) = scale*(vr,vi) for a right eigenvector, */
/*             or U'*(xr,xi) = scale*(vr,vi) for a left eigenvector, */
/*           overwriting (xr,xi) on (vr,vi). */

	    i__2 = i2;
	    i__3 = i3;
	    for (i__ = i1; i__3 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__3) 
		    {

		if (work[i__] > vcrit) {
		    rec = 1.f / vmax;
		    sscal_(n, &rec, &vr[1], &c__1);
		    sscal_(n, &rec, &vi[1], &c__1);
		    scale *= rec;
		    vmax = 1.f;
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

		w = (r__1 = b[i__ + i__ * b_dim1], dabs(r__1)) + (r__2 = b[
			i__ + 1 + i__ * b_dim1], dabs(r__2));
		if (w > *smlnum) {
		    if (w < 1.f) {
			w1 = dabs(xr) + dabs(xi);
			if (w1 > w * *bignum) {
			    rec = 1.f / w1;
			    sscal_(n, &rec, &vr[1], &c__1);
			    sscal_(n, &rec, &vi[1], &c__1);
			    xr = vr[i__];
			    xi = vi[i__];
			    scale *= rec;
			    vmax *= rec;
			}
		    }

/*                 Divide by diagonal element of B. */

		    sladiv_(&xr, &xi, &b[i__ + i__ * b_dim1], &b[i__ + 1 + 
			    i__ * b_dim1], &vr[i__], &vi[i__]);
/* Computing MAX */
		    r__3 = (r__1 = vr[i__], dabs(r__1)) + (r__2 = vi[i__], 
			    dabs(r__2));
		    vmax = dmax(r__3,vmax);
		    vcrit = *bignum / vmax;
		} else {
		    i__4 = *n;
		    for (j = 1; j <= i__4; ++j) {
			vr[j] = 0.f;
			vi[j] = 0.f;
/* L240: */
		    }
		    vr[i__] = 1.f;
		    vi[i__] = 1.f;
		    scale = 0.f;
		    vmax = 1.f;
		    vcrit = *bignum;
		}
/* L250: */
	    }

/*           Test for sufficient growth in the norm of (VR,VI). */

	    vnorm = sasum_(n, &vr[1], &c__1) + sasum_(n, &vi[1], &c__1);
	    if (vnorm >= growto * scale) {
		goto L280;
	    }

/*           Choose a new orthogonal starting vector and try again. */

	    y = *eps3 / (rootn + 1.f);
	    vr[1] = *eps3;
	    vi[1] = 0.f;

	    i__3 = *n;
	    for (i__ = 2; i__ <= i__3; ++i__) {
		vr[i__] = y;
		vi[i__] = 0.f;
/* L260: */
	    }
	    vr[*n - its + 1] -= *eps3 * rootn;
/* L270: */
	}

/*        Failure to find eigenvector in N iterations */

	*info = 1;

L280:

/*        Normalize eigenvector. */

	vnorm = 0.f;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	    r__3 = vnorm, r__4 = (r__1 = vr[i__], dabs(r__1)) + (r__2 = vi[
		    i__], dabs(r__2));
	    vnorm = dmax(r__3,r__4);
/* L290: */
	}
	r__1 = 1.f / vnorm;
	sscal_(n, &r__1, &vr[1], &c__1);
	r__1 = 1.f / vnorm;
	sscal_(n, &r__1, &vi[1], &c__1);

    }

    return 0;

/*     End of SLAEIN */

} /* slaein_ */
