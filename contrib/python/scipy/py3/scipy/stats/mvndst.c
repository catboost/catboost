/* mvndst.f -- translated by f2c (version 20200916).
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

/* Common Block Declarations */

struct {
    integer ivls;
} dkblck_;

#define dkblck_1 dkblck_

/* Table of constant values */

static doublereal c_b13 = 1.;
static doublereal c_b14 = 2.;

/* Note: The test program has been removed and a utility routine mvnun has been */
/* added.  RTK 2004-08-10 */

/* Copyright 2000 by Alan Genz. */
/* Copyright 2004-2005 by Enthought, Inc. */

/* The subroutine MVNUN is copyrighted by Enthought, Inc. */
/* The rest of the file is copyrighted by Alan Genz and has kindly been offered */
/* to the Scipy project under it's BSD-style license. */

/* This file contains a short test program and MVNDST, a subroutine */
/* for computing multivariate normal distribution function values. */
/* The file is self contained and should compile without errors on (77) */
/* standard Fortran compilers. The test program demonstrates the use of */
/* MVNDST for computing MVN distribution values for a five dimensional */
/* example problem, with three different integration limit combinations. */

/*          Alan Genz */
/*          Department of Mathematics */
/*          Washington State University */
/*          Pullman, WA 99164-3113 */
/*          Email : alangenz@wsu.edu */

/* Subroutine */ int mvnun_(integer *d__, integer *n, doublereal *lower, 
	doublereal *upper, doublereal *means, doublereal *covar, integer *
	maxpts, doublereal *abseps, doublereal *releps, doublereal *value, 
	integer *inform__)
{
    /* System generated locals */
    integer covar_dim1, covar_offset, means_dim1, means_offset, i__1, i__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, inf;
    doublereal rho[(*d__)*((*d__)-1)/2];
    integer infin[(*d__)];
    doublereal stdev[(*d__)], error;
    static integer tmpinf;
    static real tmpval;
    doublereal nlower[(*d__)], nupper[(*d__)];
    extern /* Subroutine */ int mvndst_(integer *, doublereal *, doublereal *,
	     integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *);

/*  Parameters */

/*   d       integer, dimensionality of the data */
/*   n       integer, the number of data points */
/*   lower   double(2), the lower integration limits */
/*   upper   double(2), the upper integration limits */
/*   means   double(n), the mean of each kernel */
/*   covar   double(2,2), the covariance matrix */
/*   maxpts  integer, the maximum number of points to evaluate at */
/*   abseps  double, absolute error tolerance */
/*   releps  double, relative error tolerance */
/*   value   double intent(out), integral value */
/*   inform  integer intent(out), */
/*               if inform == 0: error < eps */
/*               elif inform == 1: error > eps, all maxpts used */
    /* Parameter adjustments */
    covar_dim1 = *d__;
    covar_offset = 1 + covar_dim1;
    covar -= covar_offset;
    --upper;
    --lower;
    means_dim1 = *d__;
    means_offset = 1 + means_dim1;
    means -= means_offset;

    /* Function Body */
    inf = 0;
    inf = (integer) (1. / inf);
    i__1 = *d__;
    for (i__ = 1; i__ <= i__1; ++i__) {
	stdev[i__ - 1] = sqrt(covar[i__ + i__ * covar_dim1]);
	if (upper[i__] == (doublereal) inf && lower[i__] == (doublereal) (
		-inf)) {
	    infin[i__ - 1] = -1;
	} else if (lower[i__] == (doublereal) (-inf)) {
	    infin[i__ - 1] = 0;
	} else if (upper[i__] == (doublereal) inf) {
	    infin[i__ - 1] = 1;
	} else {
	    infin[i__ - 1] = 2;
	}
    }
    i__1 = *d__;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    rho[j + (i__ - 2) * (i__ - 1) / 2 - 1] = covar[i__ + j * 
		    covar_dim1] / stdev[i__ - 1] / stdev[j - 1];
	}
    }
    *value = 0.;
    *inform__ = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *d__;
	for (j = 1; j <= i__2; ++j) {
	    nlower[j - 1] = (lower[j] - means[j + i__ * means_dim1]) / stdev[
		    j - 1];
	    nupper[j - 1] = (upper[j] - means[j + i__ * means_dim1]) / stdev[
		    j - 1];
	}
	mvndst_(d__, nlower, nupper, infin, rho, maxpts, abseps, releps, &
		error, &tmpval, &tmpinf);
	*value += tmpval;
	if (tmpinf == 1) {
	    *inform__ = 1;
	}
    }
    *value /= *n;
    return 0;
} /* mvnun_ */

/* Subroutine */ int mvnun_weighted__(integer *d__, integer *n, doublereal *
	lower, doublereal *upper, doublereal *means, doublereal *weights, 
	doublereal *covar, integer *maxpts, doublereal *abseps, doublereal *
	releps, doublereal *value, integer *inform__)
{
    /* System generated locals */
    integer covar_dim1, covar_offset, means_dim1, means_offset, i__1, i__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static integer i__, j, inf;
    doublereal rho[(*d__)*((*d__)-1)/2];
    integer infin[(*d__)];
    doublereal stdev[(*d__)], error;
    static integer tmpinf;
    static real tmpval;
    doublereal nlower[(*d__)], nupper[(*d__)];
    extern /* Subroutine */ int mvndst_(integer *, doublereal *, doublereal *,
	     integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *);

/*  Parameters */

/*   d       integer, dimensionality of the data */
/*   n       integer, the number of data points */
/*   lower   double(2), the lower integration limits */
/*   upper   double(2), the upper integration limits */
/*   means   double(n), the mean of each kernel */
/*   weights double(n), the weight of each kernel */
/*   covar   double(2,2), the covariance matrix */
/*   maxpts  integer, the maximum number of points to evaluate at */
/*   abseps  double, absolute error tolerance */
/*   releps  double, relative error tolerance */
/*   value   double intent(out), integral value */
/*   inform  integer intent(out), */
/*               if inform == 0: error < eps */
/*               elif inform == 1: error > eps, all maxpts used */
    /* Parameter adjustments */
    covar_dim1 = *d__;
    covar_offset = 1 + covar_dim1;
    covar -= covar_offset;
    --upper;
    --lower;
    --weights;
    means_dim1 = *d__;
    means_offset = 1 + means_dim1;
    means -= means_offset;

    /* Function Body */
    inf = 0;
    inf = (integer) (1. / inf);
    i__1 = *d__;
    for (i__ = 1; i__ <= i__1; ++i__) {
	stdev[i__ - 1] = sqrt(covar[i__ + i__ * covar_dim1]);
	if (upper[i__] == (doublereal) inf && lower[i__] == (doublereal) (
		-inf)) {
	    infin[i__ - 1] = -1;
	} else if (lower[i__] == (doublereal) (-inf)) {
	    infin[i__ - 1] = 0;
	} else if (upper[i__] == (doublereal) inf) {
	    infin[i__ - 1] = 1;
	} else {
	    infin[i__ - 1] = 2;
	}
    }
    i__1 = *d__;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    rho[j + (i__ - 2) * (i__ - 1) / 2 - 1] = covar[i__ + j * 
		    covar_dim1] / stdev[i__ - 1] / stdev[j - 1];
	}
    }
    *value = 0.;
    *inform__ = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *d__;
	for (j = 1; j <= i__2; ++j) {
	    nlower[j - 1] = (lower[j] - means[j + i__ * means_dim1]) / stdev[
		    j - 1];
	    nupper[j - 1] = (upper[j] - means[j + i__ * means_dim1]) / stdev[
		    j - 1];
	}
	mvndst_(d__, nlower, nupper, infin, rho, maxpts, abseps, releps, &
		error, &tmpval, &tmpinf);
	*value += tmpval * weights[i__];
	if (tmpinf == 1) {
	    *inform__ = 1;
	}
    }
    return 0;
} /* mvnun_weighted__ */

/* Subroutine */ int mvndst_(integer *n, doublereal *lower, doublereal *upper,
	 integer *infin, doublereal *correl, integer *maxpts, doublereal *
	abseps, doublereal *releps, doublereal *error, doublereal *value, 
	integer *inform__)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static doublereal d__, e;
    static integer infis;
    extern /* Subroutine */ int dkbvrc_(integer *, integer *, integer *, D_fp,
	     doublereal *, doublereal *, doublereal *, doublereal *, integer *
	    );
    extern doublereal mvndfn_(), mvndnt_(integer *, doublereal *, doublereal *
	    , doublereal *, integer *, integer *, doublereal *, doublereal *);


/*     A subroutine for computing multivariate normal probabilities. */
/*     This subroutine uses an algorithm given in the paper */
/*     "Numerical Computation of Multivariate Normal Probabilities", in */
/*     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by */
/*          Alan Genz */
/*          Department of Mathematics */
/*          Washington State University */
/*          Pullman, WA 99164-3113 */
/*          Email : AlanGenz@wsu.edu */

/*  Parameters */

/*     N      INTEGER, the number of variables. */
/*     LOWER  REAL, array of lower integration limits. */
/*     UPPER  REAL, array of upper integration limits. */
/*     INFIN  INTEGER, array of integration limits flags: */
/*            if INFIN(I) < 0, Ith limits are (-infinity, infinity); */
/*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)]; */
/*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity); */
/*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)]. */
/*     CORREL REAL, array of correlation coefficients; the correlation */
/*            coefficient in row I column J of the correlation matrix */
/*            should be stored in CORREL( J + ((I-2)*(I-1))/2 ), for J < I. */
/*            THe correlation matrix must be positive semidefinite. */
/*     MAXPTS INTEGER, maximum number of function values allowed. This */
/*            parameter can be used to limit the time. A sensible */
/*            strategy is to start with MAXPTS = 1000*N, and then */
/*            increase MAXPTS if ERROR is too large. */
/*     ABSEPS REAL absolute error tolerance. */
/*     RELEPS REAL relative error tolerance. */
/*     ERROR  REAL estimated absolute error, with 99% confidence level. */
/*     VALUE  REAL estimated value for the integral */
/*     INFORM INTEGER, termination status parameter: */
/*            if INFORM = 0, normal completion with ERROR < EPS; */
/*            if INFORM = 1, completion with ERROR > EPS and MAXPTS */
/*                           function values used; increase MAXPTS to */
/*                           decrease ERROR; */
/*            if INFORM = 2, N > 500 or N < 1. */

    /* Parameter adjustments */
    --correl;
    --infin;
    --upper;
    --lower;

    /* Function Body */
    if (*n > 500 || *n < 1) {
	*inform__ = 2;
	*value = 0.;
	*error = 1.;
    } else {
	*inform__ = (integer) mvndnt_(n, &correl[1], &lower[1], &upper[1], &
		infin[1], &infis, &d__, &e);
	if (*n - infis == 0) {
	    *value = 1.;
	    *error = 0.;
	} else if (*n - infis == 1) {
	    *value = e - d__;
	    *error = 2e-16;
	} else {

/*        Call the lattice rule integration subroutine */

	    dkblck_1.ivls = 0;
	    i__1 = *n - infis - 1;
	    dkbvrc_(&i__1, &dkblck_1.ivls, maxpts, (D_fp)mvndfn_, abseps, 
		    releps, error, value, inform__);
	}
    }
    return 0;
} /* mvndst_ */

doublereal mvndfn_0_(int n__, integer *n, doublereal *w, doublereal *correl, 
	doublereal *lower, doublereal *upper, integer *infin, integer *infis, 
	doublereal *d__, doublereal *e)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static doublereal a[500], b[500];
    static integer i__, j;
    static doublereal y[500], ai, bi, di, ei;
    static integer ij, ik;
    static doublereal cov[125250], sum;
    static integer infa, infb, infi[500];
    extern doublereal bvnmvn_(doublereal *, doublereal *, integer *, 
	    doublereal *), phinvs_(doublereal *);
    extern /* Subroutine */ int mvnlms_(doublereal *, doublereal *, integer *,
	     doublereal *, doublereal *), covsrt_(integer *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *);


/*     Integrand subroutine */

    /* Parameter adjustments */
    if (w) {
	--w;
	}
    if (correl) {
	--correl;
	}
    if (lower) {
	--lower;
	}
    if (upper) {
	--upper;
	}
    if (infin) {
	--infin;
	}

    /* Function Body */
    switch(n__) {
	case 1: goto L_mvndnt;
	}

    ret_val = 1.;
    infa = 0;
    infb = 0;
    ik = 1;
    ij = 0;
    i__1 = *n + 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sum = 0.;
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    ++ij;
	    if (j < ik) {
		sum += cov[ij - 1] * y[j - 1];
	    }
	}
	if (infi[i__ - 1] != 0) {
	    if (infa == 1) {
/* Computing MAX */
		d__1 = ai, d__2 = a[i__ - 1] - sum;
		ai = max(d__1,d__2);
	    } else {
		ai = a[i__ - 1] - sum;
		infa = 1;
	    }
	}
	if (infi[i__ - 1] != 1) {
	    if (infb == 1) {
/* Computing MIN */
		d__1 = bi, d__2 = b[i__ - 1] - sum;
		bi = min(d__1,d__2);
	    } else {
		bi = b[i__ - 1] - sum;
		infb = 1;
	    }
	}
	++ij;
	if (i__ == *n + 1 || cov[ij + ik] > 0.) {
	    i__2 = (infa << 1) + infb - 1;
	    mvnlms_(&ai, &bi, &i__2, &di, &ei);
	    if (di >= ei) {
		ret_val = 0.;
		return ret_val;
	    } else {
		ret_val *= ei - di;
		if (i__ <= *n) {
		    d__1 = di + w[ik] * (ei - di);
		    y[ik - 1] = phinvs_(&d__1);
		}
		++ik;
		infa = 0;
		infb = 0;
	    }
	}
    }
    return ret_val;

/*     Entry point for initialization. */


L_mvndnt:
    ret_val = 0.;

/*     Initialization and computation of covariance Cholesky factor. */

    covsrt_(n, &lower[1], &upper[1], &correl[1], &infin[1], y, infis, a, b, 
	    cov, infi);
    if (*n - *infis == 1) {
	mvnlms_(a, b, infi, d__, e);
    } else if (*n - *infis == 2) {
	if (abs(cov[2]) > 0.) {
/* Computing 2nd power */
	    d__1 = cov[1];
	    *d__ = sqrt(d__1 * d__1 + 1);
	    if (infi[1] != 0) {
		a[1] /= *d__;
	    }
	    if (infi[1] != 1) {
		b[1] /= *d__;
	    }
	    d__1 = cov[1] / *d__;
	    *e = bvnmvn_(a, b, infi, &d__1);
	    *d__ = 0.;
	} else {
	    if (infi[0] != 0) {
		if (infi[1] != 0) {
		    a[0] = max(a[0],a[1]);
		}
	    } else {
		if (infi[1] != 0) {
		    a[0] = a[1];
		}
	    }
	    if (infi[0] != 1) {
		if (infi[1] != 1) {
		    b[0] = min(b[0],b[1]);
		}
	    } else {
		if (infi[1] != 1) {
		    b[0] = b[1];
		}
	    }
	    if (infi[0] != infi[1]) {
		infi[0] = 2;
	    }
	    mvnlms_(a, b, infi, d__, e);
	}
	++(*infis);
    }
    return ret_val;
} /* mvndfn_ */

doublereal mvndfn_(integer *n, doublereal *w)
{
    return mvndfn_0_(0, n, w, (doublereal *)0, (doublereal *)0, (doublereal *)
	    0, (integer *)0, (integer *)0, (doublereal *)0, (doublereal *)0);
    }

doublereal mvndnt_(integer *n, doublereal *correl, doublereal *lower, 
	doublereal *upper, integer *infin, integer *infis, doublereal *d__, 
	doublereal *e)
{
    return mvndfn_0_(1, n, (doublereal *)0, correl, lower, upper, infin, 
	    infis, d__, e);
    }

/* Subroutine */ int mvnlms_(doublereal *a, doublereal *b, integer *infin, 
	doublereal *lower, doublereal *upper)
{
    extern doublereal mvnphi_(doublereal *);

    *lower = 0.;
    *upper = 1.;
    if (*infin >= 0) {
	if (*infin != 0) {
	    *lower = mvnphi_(a);
	}
	if (*infin != 1) {
	    *upper = mvnphi_(b);
	}
    }
    *upper = max(*upper,*lower);
    return 0;
} /* mvnlms_ */

/* Subroutine */ int covsrt_(integer *n, doublereal *lower, doublereal *upper,
	 doublereal *correl, integer *infin, doublereal *y, integer *infis, 
	doublereal *a, doublereal *b, doublereal *cov, integer *infi)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    doublereal d__1;

    /* Builtin functions */
    double sqrt(doublereal), exp(doublereal);

    /* Local variables */
    static doublereal d__, e;
    static integer i__, j, k, l, m;
    static doublereal aj, bj;
    static integer ii, ij, il;
    static doublereal yl, yu, sum, amin, bmin, dmin__, emin;
    static integer jmin;
    extern /* Subroutine */ int rcswp_(integer *, integer *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *);
    static doublereal sumsq, cvdiag;
    extern /* Subroutine */ int dkswap_(doublereal *, doublereal *), mvnlms_(
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *)
	    ;


/*     Subroutine to sort integration limits and determine Cholesky factor. */

    /* Parameter adjustments */
    --infi;
    --cov;
    --b;
    --a;
    --y;
    --infin;
    --correl;
    --upper;
    --lower;

    /* Function Body */
    ij = 0;
    ii = 0;
    *infis = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	a[i__] = 0.;
	b[i__] = 0.;
	infi[i__] = infin[i__];
	if (infi[i__] < 0) {
	    ++(*infis);
	} else {
	    if (infi[i__] != 0) {
		a[i__] = lower[i__];
	    }
	    if (infi[i__] != 1) {
		b[i__] = upper[i__];
	    }
	}
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    ++ij;
	    ++ii;
	    cov[ij] = correl[ii];
	}
	++ij;
	cov[ij] = 1.;
    }

/*     First move any doubly infinite limits to innermost positions. */

    if (*infis < *n) {
	i__1 = *n - *infis + 1;
	for (i__ = *n; i__ >= i__1; --i__) {
	    if (infi[i__] >= 0) {
		i__2 = i__ - 1;
		for (j = 1; j <= i__2; ++j) {
		    if (infi[j] < 0) {
			rcswp_(&j, &i__, &a[1], &b[1], &infi[1], n, &cov[1]);
			goto L10;
		    }
		}
	    }
L10:
	    ;
	}

/*     Sort remaining limits and determine Cholesky factor. */

	ii = 0;
	i__1 = *n - *infis;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*        Determine the integration limits for variable with minimum */
/*        expected probability and interchange that variable with Ith. */

	    dmin__ = 0.;
	    emin = 1.;
	    jmin = i__;
	    cvdiag = 0.;
	    ij = ii;
	    i__2 = *n - *infis;
	    for (j = i__; j <= i__2; ++j) {
		if (cov[ij + j] > 1e-10) {
		    sumsq = sqrt(cov[ij + j]);
		    sum = 0.;
		    i__3 = i__ - 1;
		    for (k = 1; k <= i__3; ++k) {
			sum += cov[ij + k] * y[k];
		    }
		    aj = (a[j] - sum) / sumsq;
		    bj = (b[j] - sum) / sumsq;
		    mvnlms_(&aj, &bj, &infi[j], &d__, &e);
		    if (emin + d__ >= e + dmin__) {
			jmin = j;
			amin = aj;
			bmin = bj;
			dmin__ = d__;
			emin = e;
			cvdiag = sumsq;
		    }
		}
		ij += j;
	    }
	    if (jmin > i__) {
		rcswp_(&i__, &jmin, &a[1], &b[1], &infi[1], n, &cov[1]);
	    }
	    cov[ii + i__] = cvdiag;

/*        Compute Ith column of Cholesky factor. */
/*        Compute expected value for Ith integration variable and */
/*         scale Ith covariance matrix row and limits. */

	    if (cvdiag > 0.) {
		il = ii + i__;
		i__2 = *n - *infis;
		for (l = i__ + 1; l <= i__2; ++l) {
		    cov[il + i__] /= cvdiag;
		    ij = ii + i__;
		    i__3 = l;
		    for (j = i__ + 1; j <= i__3; ++j) {
			cov[il + j] -= cov[il + i__] * cov[ij + i__];
			ij += j;
		    }
		    il += l;
		}
		if (emin > dmin__ + 1e-10) {
		    yl = 0.;
		    yu = 0.;
		    if (infi[i__] != 0) {
/* Computing 2nd power */
			d__1 = amin;
			yl = -exp(-(d__1 * d__1) / 2) / 2.506628274631001;
		    }
		    if (infi[i__] != 1) {
/* Computing 2nd power */
			d__1 = bmin;
			yu = -exp(-(d__1 * d__1) / 2) / 2.506628274631001;
		    }
		    y[i__] = (yu - yl) / (emin - dmin__);
		} else {
		    if (infi[i__] == 0) {
			y[i__] = bmin;
		    }
		    if (infi[i__] == 1) {
			y[i__] = amin;
		    }
		    if (infi[i__] == 2) {
			y[i__] = (amin + bmin) / 2;
		    }
		}
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    ++ii;
		    cov[ii] /= cvdiag;
		}
		a[i__] /= cvdiag;
		b[i__] /= cvdiag;
	    } else {
		il = ii + i__;
		i__2 = *n - *infis;
		for (l = i__ + 1; l <= i__2; ++l) {
		    cov[il + i__] = 0.;
		    il += l;
		}

/*        If the covariance matrix diagonal entry is zero, */
/*         permute limits and/or rows, if necessary. */


		for (j = i__ - 1; j >= 1; --j) {
		    if ((d__1 = cov[ii + j], abs(d__1)) > 1e-10) {
			a[i__] /= cov[ii + j];
			b[i__] /= cov[ii + j];
			if (cov[ii + j] < 0.) {
			    dkswap_(&a[i__], &b[i__]);
			    if (infi[i__] != 2) {
				infi[i__] = 1 - infi[i__];
			    }
			}
			i__2 = j;
			for (l = 1; l <= i__2; ++l) {
			    cov[ii + l] /= cov[ii + j];
			}
			i__2 = i__ - 1;
			for (l = j + 1; l <= i__2; ++l) {
			    if (cov[(l - 1) * l / 2 + j + 1] > 0.) {
				ij = ii;
				i__3 = l;
				for (k = i__ - 1; k >= i__3; --k) {
				    i__4 = k;
				    for (m = 1; m <= i__4; ++m) {
					dkswap_(&cov[ij - k + m], &cov[ij + m]
						);
				    }
				    dkswap_(&a[k], &a[k + 1]);
				    dkswap_(&b[k], &b[k + 1]);
				    m = infi[k];
				    infi[k] = infi[k + 1];
				    infi[k + 1] = m;
				    ij -= k;
				}
				goto L20;
			    }
			}
			goto L20;
		    }
		    cov[ii + j] = 0.;
		}
L20:
		ii += i__;
		y[i__] = 0.;
	    }
	}
    }
    return 0;
} /* covsrt_ */


/* Subroutine */ int dkswap_(doublereal *x, doublereal *y)
{
    static doublereal t;

    t = *x;
    *x = *y;
    *y = t;
    return 0;
} /* dkswap_ */


/* Subroutine */ int rcswp_(integer *p, integer *q, doublereal *a, doublereal 
	*b, integer *infin, integer *n, doublereal *c__)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__, j, ii, jj;
    extern /* Subroutine */ int dkswap_(doublereal *, doublereal *);


/*     Swaps rows and columns P and Q in situ, with P <= Q. */

    /* Parameter adjustments */
    --c__;
    --infin;
    --b;
    --a;

    /* Function Body */
    dkswap_(&a[*p], &a[*q]);
    dkswap_(&b[*p], &b[*q]);
    j = infin[*p];
    infin[*p] = infin[*q];
    infin[*q] = j;
    jj = *p * (*p - 1) / 2;
    ii = *q * (*q - 1) / 2;
    dkswap_(&c__[jj + *p], &c__[ii + *q]);
    i__1 = *p - 1;
    for (j = 1; j <= i__1; ++j) {
	dkswap_(&c__[jj + j], &c__[ii + j]);
    }
    jj += *p;
    i__1 = *q - 1;
    for (i__ = *p + 1; i__ <= i__1; ++i__) {
	dkswap_(&c__[jj + *p], &c__[ii + i__]);
	jj += i__;
    }
    ii += *q;
    i__1 = *n;
    for (i__ = *q + 1; i__ <= i__1; ++i__) {
	dkswap_(&c__[ii + *p], &c__[ii + *q]);
	ii += i__;
    }
    return 0;
} /* rcswp_ */


/* Subroutine */ int dkbvrc_(integer *ndim, integer *minvls, integer *maxvls, 
	D_fp functn, doublereal *abseps, doublereal *releps, doublereal *
	abserr, doublereal *finest, integer *inform__)
{
    /* Initialized data */

    static integer p[28] = { 31,47,73,113,173,263,397,593,907,1361,2053,3079,
	    4621,6947,10427,15641,23473,35221,52837,79259,118891,178349,
	    267523,401287,601942,902933,1354471,2031713 };
    static integer c__[2772]	/* was [28][99] */ = { 12,13,27,35,64,111,163,
	    246,347,505,794,1189,1763,2872,4309,6610,9861,10327,19540,34566,
	    31929,40701,103650,165843,130365,333459,500884,858339,9,11,28,27,
	    66,42,154,189,402,220,325,888,1018,3233,3758,6977,3647,7582,19926,
	    9579,49367,69087,125480,90647,236711,375354,566009,918142,9,17,10,
	    27,28,54,83,242,322,601,960,259,1500,1534,4034,1686,4073,7124,
	    11582,12654,10982,77576,59978,59925,110235,102417,399251,501970,
	    13,10,11,36,28,118,43,102,418,644,528,1082,432,2941,1963,3819,
	    2535,8214,11113,26856,3527,64590,46875,189541,125699,383544,
	    652979,234813,12,15,11,22,44,20,82,250,215,612,247,725,1332,2910,
	    730,2314,3430,9600,24585,37873,27066,39397,77172,67647,56483,
	    292630,355008,460565,12,15,20,29,44,31,92,250,220,160,247,811,
	    2203,393,642,5647,9865,10271,8726,38806,13226,33179,83021,74795,
	    93735,41147,430235,31996,12,15,11,29,55,31,150,102,339,206,338,
	    636,126,1796,1502,3953,2830,10193,17218,29501,56010,10858,126904,
	    68365,234469,374614,328722,753018,12,15,11,20,67,72,59,250,339,
	    206,366,965,2240,919,2246,3614,9328,10800,419,17271,18911,38935,
	    14541,167485,60549,48032,670680,256150,12,15,28,45,10,17,76,280,
	    339,206,847,497,1719,446,3834,5115,4320,9086,4918,3663,40574,
	    43129,56299,143918,1291,435453,405585,199809,12,15,13,5,10,94,76,
	    118,337,422,753,497,1284,919,1511,423,5913,2365,4918,10763,20767,
	    35468,43636,74912,93937,281493,405585,993599,12,22,13,5,10,14,47,
	    196,218,134,753,1490,878,919,1102,423,10365,4409,4918,18955,20767,
	    35468,11655,167289,245291,358168,424646,245149,12,15,28,5,10,14,
	    11,118,315,518,236,1490,1983,1117,1102,5408,8272,13812,15701,1298,
	    9686,5279,52680,75517,196061,114121,670180,794183,3,15,13,21,10,
	    11,11,191,315,134,334,392,266,103,1522,7426,3706,5661,17710,26560,
	    47603,61518,88549,8148,258647,346892,670180,121349,3,6,13,21,10,
	    14,100,215,315,134,334,1291,266,103,1522,423,6186,9344,4037,17132,
	    47603,61518,29804,172106,162489,238990,641587,150619,3,6,13,21,38,
	    14,131,121,315,518,461,508,266,103,3427,423,7806,9344,4037,17132,
	    11736,27945,101894,126159,176631,317313,215580,376952,12,6,14,21,
	    38,14,116,121,167,652,711,508,266,103,3427,487,7806,10362,15808,
	    4753,11736,70975,113675,35867,204895,164158,59048,809123,7,15,14,
	    21,10,94,116,49,167,382,652,1291,747,103,3928,6227,7806,9344,
	    11401,4753,41601,70975,48040,35867,73353,35497,633320,809123,7,15,
	    14,21,10,10,116,49,167,206,381,1291,747,103,915,2660,8610,9344,
	    19398,8713,12888,86478,113675,35867,172319,70530,81010,804319,12,
	    9,14,21,10,10,116,49,167,158,381,508,127,103,915,6227,2563,8585,
	    25950,18624,32948,86478,34987,121694,28881,70530,20789,67352,12,
	    13,14,21,10,10,116,49,361,441,381,1291,127,2311,3818,1221,11558,
	    11114,25950,13082,30801,20514,48308,52171,136787,434839,389250,
	    969594,12,2,14,21,10,10,116,49,201,179,652,508,2074,3117,3818,
	    3811,11558,13080,4454,6791,44243,20514,97926,95354,122081,24754,
	    389250,434796,12,2,14,21,49,14,138,49,124,441,381,508,127,1101,
	    3818,197,9421,13080,24987,1122,53351,73178,5475,113969,122081,
	    24754,638764,969594,12,2,14,21,49,14,138,49,124,56,381,867,2074,
	    3117,3818,4367,1181,13080,11719,19363,53351,73178,49449,113969,
	    275993,24754,638764,804319,12,13,14,21,49,14,138,49,124,559,381,
	    867,1400,3117,4782,351,9421,6949,8697,34695,16016,43098,6850,
	    76304,64673,393656,389250,391368,12,11,14,21,49,14,138,49,124,559,
	    381,867,1383,1101,4782,1281,1181,3436,1452,18770,35086,43098,
	    62545,123709,211587,118711,389250,761041,12,11,14,21,49,14,138,49,
	    124,56,381,867,1383,1101,4782,1221,1181,3436,1452,18770,35086,
	    4701,62545,123709,211587,118711,398094,754049,12,10,14,21,49,14,
	    138,49,124,56,381,934,1383,1101,3818,351,1181,3436,1452,18770,
	    32581,59979,9440,144615,211587,148227,80846,466264,3,15,14,21,49,
	    14,138,49,124,56,381,867,1383,1101,4782,351,9421,13213,1452,18770,
	    2464,59979,33242,123709,282859,271087,147776,754049,3,15,14,29,49,
	    11,138,171,124,56,226,867,1383,1101,3818,351,1181,6130,1452,15628,
	    2464,58556,9440,64958,282859,355831,147776,754049,3,15,14,17,49,
	    11,138,171,124,56,326,867,1383,2503,3818,7245,1181,6130,8697,
	    18770,49554,69916,33242,64958,211587,91034,296177,466264,12,15,14,
	    17,49,11,101,171,124,56,326,867,1383,2503,1327,1984,10574,8159,
	    8697,18770,2464,15170,9440,32377,242821,417029,398094,754049,7,15,
	    31,17,49,8,101,171,124,56,326,867,1383,2503,1327,2999,10574,8159,
	    6436,18770,2464,15170,33242,193002,256865,417029,398094,754049,7,
	    15,31,17,49,8,101,171,231,56,326,867,1383,2503,1327,2999,3534,
	    11595,21475,18770,49554,4832,9440,193002,256865,91034,147776,
	    282852,12,15,5,17,38,8,101,171,231,56,326,867,1383,2503,1327,2999,
	    3534,8159,6436,33766,49554,4832,62850,25023,256865,91034,147776,
	    429907,12,15,5,17,38,8,101,171,90,56,326,1284,1400,2503,1327,2999,
	    3534,3436,22913,20837,2464,43064,9440,40017,122203,417029,396313,
	    390017,12,15,5,17,31,8,101,171,90,56,326,1284,1383,2503,1327,2999,
	    3534,7096,6434,20837,81,71685,9440,141605,291915,91034,578233,
	    276645,12,6,31,17,4,8,101,171,90,56,126,1284,1383,2503,1327,2999,
	    3534,7096,18497,20837,27260,4832,9440,189165,122203,299843,578233,
	    994856,12,6,13,17,4,8,101,171,90,56,326,1284,1383,429,1387,3995,
	    2898,7096,11089,20837,10681,15170,90308,189165,291915,299843,
	    578233,250142,12,6,11,17,31,18,101,171,90,56,326,1284,1383,429,
	    1387,2063,2898,7096,11089,20837,2185,15170,90308,141605,291915,
	    413548,19482,144595,12,15,11,23,64,18,101,171,90,101,326,1284,
	    1383,429,1387,2063,2898,7096,11089,20837,2185,15170,90308,189165,
	    122203,413548,620706,907454,12,15,11,23,4,18,101,171,90,101,326,
	    1284,1383,429,1387,2063,3450,7096,11089,6545,2185,27679,47904,
	    189165,25639,308300,187095,689648,12,9,11,23,4,18,101,171,90,56,
	    326,1284,1383,429,1387,2063,2141,7096,3036,6545,2185,27679,47904,
	    141605,25639,413548,620706,687580,3,13,11,23,4,18,101,171,90,101,
	    326,1284,507,429,1387,1644,2141,7096,3036,6545,2185,27679,47904,
	    141605,291803,413548,187095,687580,3,2,11,23,64,113,101,171,90,
	    101,326,563,1073,429,1387,2063,2141,7096,14208,6545,2185,60826,
	    47904,141605,245397,413548,126467,687580,3,2,13,23,45,62,101,171,
	    90,101,326,563,1073,1702,1387,2077,2141,7096,14208,6545,2185,
	    60826,47904,189165,284047,308300,241663,687580,12,2,13,23,45,62,
	    101,171,90,101,326,563,1073,1702,1387,2512,2141,7096,14208,12138,
	    18086,6187,47904,127047,245397,308300,241663,978368,7,13,13,23,45,
	    45,101,171,90,101,326,563,1073,1702,2339,2512,2141,7096,14208,
	    12138,18086,6187,47904,127047,245397,308300,241663,687580,7,11,13,
	    23,45,45,101,171,90,101,195,1010,1990,184,2339,2512,2141,7096,
	    12906,12138,18086,4264,47904,127047,245397,413548,241663,552742,
	    12,11,13,23,45,113,101,171,48,101,195,1010,1990,184,2339,2077,
	    7055,7096,12906,12138,18086,4264,47904,127047,245397,308300,
	    241663,105195,12,10,13,23,45,113,101,171,48,101,55,1010,1990,184,
	    2339,2077,7055,7096,12906,12138,18086,4264,41143,127047,245397,
	    308300,241663,942843,12,15,13,23,66,113,101,171,48,193,55,208,
	    1990,184,2339,2077,7055,7096,12906,12138,17631,4264,41143,127047,
	    245397,308300,241663,768249,12,15,14,21,66,113,116,171,48,193,55,
	    838,1990,184,2339,2077,7055,7096,12906,12138,17631,4264,41143,
	    127047,245397,308300,241663,307142,12,15,14,27,66,113,116,171,90,
	    193,55,563,507,105,2339,754,7055,7096,12906,12138,18086,45567,
	    41143,127047,94241,308300,241663,307142,12,15,14,3,66,113,116,171,
	    90,193,55,563,507,105,2339,754,7055,4377,12906,12138,18086,32269,
	    41143,127047,66575,15311,241663,307142,12,15,14,3,66,113,116,171,
	    90,193,55,563,507,105,2339,754,7055,7096,12906,12138,18086,32269,
	    41143,127047,66575,15311,241663,307142,12,15,14,3,66,113,116,171,
	    90,193,55,759,507,105,2339,754,7055,4377,7614,12138,37335,32269,
	    41143,127047,217673,15311,241663,880619,12,15,14,24,66,113,116,
	    171,90,193,55,759,507,105,2339,754,7055,4377,7614,12138,37774,
	    32269,36114,127047,217673,15311,321632,880619,3,15,14,27,66,113,
	    100,171,90,101,55,564,507,105,2339,754,7055,4377,7614,12138,37774,
	    62060,36114,127047,217673,176255,23210,880619,3,15,14,27,66,113,
	    100,171,90,101,55,759,507,105,2339,754,7055,4377,7614,12138,37774,
	    62060,36114,127047,217673,176255,23210,880619,3,6,14,17,66,113,
	    100,171,90,101,55,759,507,105,3148,754,7055,4377,5021,30483,26401,
	    62060,36114,127047,217673,23613,394484,880619,12,6,14,29,66,113,
	    100,171,90,101,55,801,507,105,3148,754,7055,5410,5021,30483,26401,
	    62060,36114,127047,217673,23613,394484,880619,7,6,14,29,66,113,
	    100,171,90,101,55,801,1073,105,3148,754,7055,5410,5021,30483,
	    26401,62060,24997,127047,217673,23613,394484,880619,7,15,14,29,66,
	    113,138,161,90,101,55,801,1073,105,3148,754,7055,4377,5021,30483,
	    26401,62060,65162,127047,217673,23613,78101,117185,12,15,14,17,66,
	    113,138,161,90,101,55,801,1073,105,3148,754,2831,4377,5021,30483,
	    26401,62060,65162,127047,217673,23613,78101,117185,12,9,14,5,66,
	    113,138,161,90,101,55,759,1073,105,3148,754,8204,4377,5021,12138,
	    26401,62060,65162,127047,217673,23613,78101,117185,12,13,14,5,66,
	    63,138,161,90,101,55,759,1073,105,3148,754,8204,4377,10145,12138,
	    26401,62060,65162,127785,217673,172210,542095,117185,12,2,14,5,66,
	    63,138,161,90,101,55,759,1073,105,3148,754,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,204328,542095,117185,12,2,31,5,66,
	    53,101,161,90,101,55,759,1073,105,3148,754,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,204328,542095,117185,12,2,31,21,66,
	    63,101,161,90,101,195,759,1073,105,3148,754,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,204328,542095,117185,12,13,5,21,11,
	    67,101,161,90,101,195,563,1073,105,3148,754,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,204328,542095,117185,12,11,5,21,66,
	    67,101,14,90,101,195,563,1073,105,3148,754,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,121626,542095,117185,7,11,5,21,66,
	    67,101,14,90,101,195,563,1073,105,3148,1097,8204,4377,10145,12138,
	    26401,1803,65162,127785,217673,121626,542095,117185,3,10,11,21,66,
	    67,101,14,90,101,195,563,1073,105,3148,1097,8204,4377,10145,12138,
	    12982,1803,65162,127785,217673,121626,542095,117185,3,10,13,21,66,
	    67,101,14,90,101,195,563,1073,105,3148,754,8204,4377,10145,12138,
	    40398,1803,65162,127785,217673,121626,542095,60731,3,15,11,21,66,
	    67,101,14,90,101,195,563,1073,105,3148,754,8204,4377,10145,12138,
	    40398,1803,65162,127785,210249,121626,542095,60731,7,15,11,21,66,
	    67,101,14,243,101,132,563,1073,105,3148,754,8204,4377,10145,12138,
	    40398,1803,65162,80822,210249,200187,542095,60731,7,15,11,21,66,
	    67,101,14,243,101,132,563,1073,105,3148,754,8204,4377,10145,12138,
	    40398,1803,47650,80822,210249,200187,542095,60731,7,15,11,21,66,
	    67,101,14,243,101,132,226,1073,105,1776,248,8204,4377,10145,12138,
	    40398,1803,47650,80822,210249,200187,542095,60731,3,15,11,21,66,
	    67,101,14,243,122,132,226,22,105,1776,754,8204,4377,10145,12138,
	    40398,1803,47650,80822,210249,200187,542095,60731,3,15,11,21,45,
	    67,101,14,243,122,132,226,22,105,1776,1097,8204,4377,10145,12138,
	    3518,51108,47650,80822,210249,200187,542095,60731,3,15,11,21,11,
	    67,101,14,243,122,132,226,22,105,3354,1097,8204,4377,10145,12138,
	    3518,51108,47650,80822,210249,121551,542095,60731,3,15,13,21,7,67,
	    101,14,243,122,132,226,22,105,3354,1097,8204,4377,10145,12138,
	    3518,51108,47650,131661,210249,121551,542095,60731,3,6,13,21,3,67,
	    101,14,243,122,132,226,22,105,3354,1097,8204,4377,10145,12138,
	    37799,51108,47650,131661,210249,248492,542095,60731,3,2,11,21,2,
	    67,101,14,243,122,132,226,22,105,925,222,8204,4377,10145,9305,
	    37799,51108,40586,131661,210249,248492,542095,60731,3,3,13,17,2,
	    51,101,14,243,122,132,226,1073,105,3354,222,8204,4377,10145,11107,
	    37799,51108,40586,131661,94453,248492,277743,178309,3,2,5,17,2,51,
	    101,14,283,122,132,226,452,105,3354,222,8204,4377,10145,11107,
	    37799,51108,40586,131661,94453,248492,277743,178309,3,3,5,17,27,
	    51,38,14,283,122,387,226,452,784,925,222,8204,4377,10145,11107,
	    37799,51108,40586,131661,94453,248492,277743,178309,3,2,5,6,5,51,
	    38,10,283,122,387,226,452,784,925,754,8204,4377,10145,11107,37799,
	    51108,40586,131661,94453,248492,457259,178309,3,2,5,17,3,51,38,10,
	    283,122,387,226,452,784,925,1982,4688,4377,10145,11107,37799,
	    51108,40586,131661,94453,248492,457259,74373,3,2,14,17,3,12,38,10,
	    283,122,387,226,452,784,925,1982,4688,4377,4544,11107,37799,51108,
	    40586,131661,94453,248492,457259,74373,3,2,13,6,5,51,38,10,283,
	    122,387,226,452,784,925,1982,4688,4377,4544,11107,37799,51108,
	    38725,131661,94453,248492,457259,74373,3,2,5,3,5,12,38,10,283,122,
	    387,226,318,784,2133,1982,2831,4377,4544,11107,4721,55315,38725,
	    131661,94453,248492,457259,74373,3,2,5,6,2,51,38,10,283,122,387,
	    226,301,784,2133,1982,2831,4377,4544,11107,4721,55315,38725,
	    131661,94453,248492,457259,74373,3,2,5,6,2,5,38,103,283,122,387,
	    226,301,784,2133,1982,2831,4377,4544,11107,4721,54140,38725,
	    131661,94453,248492,457259,74373,3,2,5,3,2,3,3,10,16,122,387,226,
	    301,784,2133,1982,2831,440,4544,11107,4721,54140,88329,131661,
	    94453,13942,457259,74373,3,2,5,3,2,3,3,10,283,101,387,226,301,784,
	    2133,1982,2831,440,8394,11107,7067,54140,88329,131661,94453,13942,
	    457259,74373,3,2,5,3,2,2,3,10,16,101,387,226,86,784,2133,1982,
	    2831,1199,8394,11107,7067,54140,88329,131661,94453,13942,457259,
	    214965,3,2,5,3,2,2,3,10,283,101,387,226,86,784,2133,1982,2831,
	    1199,8394,9305,7067,54140,88329,7114,94453,13942,457259,214965,3,
	    2,5,3,2,5,3,5,283,101,387,226,15,784,2133,1982,2831,1199,8394,
	    9305,7067,13134,88329,131661,94453,13942,457259,214965 };

    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double d_mod(doublereal *, doublereal *), pow_dd(doublereal *, doublereal 
	    *), sqrt(doublereal);

    /* Local variables */
    static integer i__;
    static doublereal x[2000];
    static integer np;
    static doublereal vk[1000];
    static integer klimi;
    static doublereal value, difint, finval;
    extern /* Subroutine */ int dksmrc_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, D_fp, doublereal *);
    static doublereal varprd;
    static integer sampls;
    static doublereal varest, varsqr;
    static integer intvls;


/*  Automatic Multidimensional Integration Subroutine */

/*         AUTHOR: Alan Genz */
/*                 Department of Mathematics */
/*                 Washington State University */
/*                 Pulman, WA 99164-3113 */
/*                 Email: AlanGenz@wsu.edu */

/*         Last Change: 1/15/03 */

/*  KRBVRC computes an approximation to the integral */

/*      1  1     1 */
/*     I  I ... I       F(X)  dx(NDIM)...dx(2)dx(1) */
/*      0  0     0 */


/*  DKBVRC uses randomized Korobov rules for the first 100 variables. */
/*  The primary references are */
/*   "Randomization of Number Theoretic Methods for Multiple Integration" */
/*    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13, pp. 904-14, */
/*  and */
/*   "Optimal Parameters for Multidimensional Integration", */
/*    P. Keast, SIAM J Numer Anal, 10, pp.831-838. */
/*  If there are more than 100 variables, the remaining variables are */
/*  integrated using the rules described in the reference */
/*   "On a Number-Theoretical Integration Method" */
/*   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11. */

/* **************  Parameters ******************************************** */
/* ***** Input parameters */
/*  NDIM    Number of variables, must exceed 1, but not exceed 40 */
/*  MINVLS  Integer minimum number of function evaluations allowed. */
/*          MINVLS must not exceed MAXVLS.  If MINVLS < 0 then the */
/*          routine assumes a previous call has been made with */
/*          the same integrand and continues that calculation. */
/*  MAXVLS  Integer maximum number of function evaluations allowed. */
/*  FUNCTN  EXTERNALly declared user defined function to be integrated. */
/*          It must have parameters (NDIM,Z), where Z is a real array */
/*          of dimension NDIM. */

/*  ABSEPS  Required absolute accuracy. */
/*  RELEPS  Required relative accuracy. */
/* ***** Output parameters */
/*  MINVLS  Actual number of function evaluations used. */
/*  ABSERR  Estimated absolute accuracy of FINEST. */
/*  FINEST  Estimated value of integral. */
/*  INFORM  INFORM = 0 for normal exit, when */
/*                     ABSERR <= MAX(ABSEPS, RELEPS*ABS(FINEST)) */
/*                  and */
/*                     INTVLS <= MAXCLS. */
/*          INFORM = 1 If MAXVLS was too small to obtain the required */
/*          accuracy. In this case a value FINEST is returned with */
/*          estimated absolute accuracy ABSERR. */
/* *********************************************************************** */
    *inform__ = 1;
    intvls = 0;
    klimi = 100;
    if (*minvls >= 0) {
	*finest = 0.;
	varest = 0.;
	sampls = 8;
	for (i__ = min(*ndim,10); i__ <= 28; ++i__) {
	    np = i__;
	    if (*minvls < (sampls << 1) * p[i__ - 1]) {
		goto L10;
	    }
	}
/* Computing MAX */
	i__1 = 8, i__2 = *minvls / (p[np - 1] << 1);
	sampls = max(i__1,i__2);
    }
L10:
    vk[0] = 1. / p[np - 1];
    i__1 = *ndim;
    for (i__ = 2; i__ <= i__1; ++i__) {
	if (i__ <= 100) {
/* Computing MIN */
	    i__2 = *ndim - 1;
	    d__1 = c__[np + min(i__2,99) * 28 - 29] * vk[i__ - 2];
	    vk[i__ - 1] = d_mod(&d__1, &c_b13);
	} else {
	    d__1 = (doublereal) (i__ - 100) / (*ndim - 99);
	    vk[i__ - 1] = (doublereal) ((integer) (p[np - 1] * pow_dd(&c_b14, 
		    &d__1)));
	    d__1 = vk[i__ - 1] / p[np - 1];
	    vk[i__ - 1] = d_mod(&d__1, &c_b13);
	}
    }
    finval = 0.;
    varsqr = 0.;
    i__1 = sampls;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dksmrc_(ndim, &klimi, &value, &p[np - 1], vk, (D_fp)functn, x);
	difint = (value - finval) / i__;
	finval += difint;
/* Computing 2nd power */
	d__1 = difint;
	varsqr = (i__ - 2) * varsqr / i__ + d__1 * d__1;
    }
    intvls += (sampls << 1) * p[np - 1];
    varprd = varest * varsqr;
    *finest += (finval - *finest) / (varprd + 1);
    if (varsqr > 0.) {
	varest = (varprd + 1) / varsqr;
    }
    *abserr = sqrt(varsqr / (varprd + 1)) * 7 / 2;
/* Computing MAX */
    d__1 = *abseps, d__2 = abs(*finest) * *releps;
    if (*abserr > max(d__1,d__2)) {
	if (np < 28) {
	    ++np;
	} else {
/* Computing MIN */
	    i__1 = sampls * 3 / 2, i__2 = (*maxvls - intvls) / (p[np - 1] << 
		    1);
	    sampls = min(i__1,i__2);
	    sampls = max(8,sampls);
	}
	if (intvls + (sampls << 1) * p[np - 1] <= *maxvls) {
	    goto L10;
	}
    } else {
	*inform__ = 0;
    }
    *minvls = intvls;

/*    Optimal Parameters for Lattice Rules */


    return 0;
} /* dkbvrc_ */


/* Subroutine */ int dksmrc_(integer *ndim, integer *klim, doublereal *sumkro,
	 integer *prime, doublereal *vk, D_fp functn, doublereal *x)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double d_mod(doublereal *, doublereal *);

    /* Local variables */
    static integer j, k, nk, jp;
    static doublereal xt;
    extern doublereal mvnuni_(void);

    /* Parameter adjustments */
    --x;
    --vk;

    /* Function Body */
    *sumkro = 0.;
    nk = min(*ndim,*klim);
    i__1 = nk - 1;
    for (j = 1; j <= i__1; ++j) {
	jp = (integer) (j + mvnuni_() * (nk + 1 - j));
	xt = vk[j];
	vk[j] = vk[jp];
	vk[jp] = xt;
    }
    i__1 = *ndim;
    for (j = 1; j <= i__1; ++j) {
	x[*ndim + j] = mvnuni_();
    }
    i__1 = *prime;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *ndim;
	for (j = 1; j <= i__2; ++j) {
	    d__2 = k * vk[j] + x[*ndim + j];
	    x[j] = (d__1 = d_mod(&d__2, &c_b13) * 2 - 1, abs(d__1));
	}
	*sumkro += ((*functn)(ndim, &x[1]) - *sumkro) / ((k << 1) - 1);
	i__2 = *ndim;
	for (j = 1; j <= i__2; ++j) {
	    x[j] = 1 - x[j];
	}
	*sumkro += ((*functn)(ndim, &x[1]) - *sumkro) / (k << 1);
    }
    return 0;
} /* dksmrc_ */


doublereal mvnphi_(doublereal *z__)
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Builtin functions */
    double exp(doublereal);

    /* Local variables */
    static doublereal p, zabs, expntl;


/*     Normal distribution probabilities accurate to 1.e-15. */
/*     Z = no. of standard deviations from the mean. */

/*     Based upon algorithm 5666 for the error function, from: */
/*     Hart, J.F. et al, 'Computer Approximations', Wiley 1968 */

/*     Programmer: Alan Miller */

/*     Latest revision - 30 March 1986 */


    zabs = abs(*z__);

/*     |Z| > 37 */

    if (zabs > 37.) {
	p = 0.;
    } else {

/*     |Z| <= 37 */

/* Computing 2nd power */
	d__1 = zabs;
	expntl = exp(-(d__1 * d__1) / 2);

/*     |Z| < CUTOFF = 10/SQRT(2) */

	if (zabs < 7.071067811865475) {
	    p = expntl * ((((((zabs * .03526249659989109 + .7003830644436881) 
		    * zabs + 6.37396220353165) * zabs + 33.912866078383) * 
		    zabs + 112.0792914978709) * zabs + 221.2135961699311) * 
		    zabs + 220.2068679123761) / (((((((zabs * 
		    .08838834764831844 + 1.755667163182642) * zabs + 
		    16.06417757920695) * zabs + 86.78073220294608) * zabs + 
		    296.5642487796737) * zabs + 637.3336333788311) * zabs + 
		    793.8265125199484) * zabs + 440.4137358247522);

/*     |Z| >= CUTOFF. */

	} else {
	    p = expntl / (zabs + 1 / (zabs + 2 / (zabs + 3 / (zabs + 4 / (
		    zabs + .65))))) / 2.506628274631001;
	}
    }
    if (*z__ > 0.) {
	p = 1 - p;
    }
    ret_val = p;
    return ret_val;
} /* mvnphi_ */

doublereal phinvs_(doublereal *p)
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2;

    /* Builtin functions */
    double log(doublereal), sqrt(doublereal);

    /* Local variables */
    static doublereal q, r__;


/* 	ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3 */

/* 	Produces the normal deviate Z corresponding to a given lower */
/* 	tail area of P. */

/* 	The hash sums below are the sums of the mantissas of the */
/* 	coefficients.   They are included for use in checking */
/* 	transcription. */


/*     Coefficients for P close to 0.5 */

/*     HASH SUM AB    55.88319 28806 14901 4439 */

/*     Coefficients for P not close to 0, 0.5 or 1. */

/*     HASH SUM CD    49.33206 50330 16102 89036 */

/* 	Coefficients for P near 0 or 1. */

/*     HASH SUM EF    47.52583 31754 92896 71629 */

    q = (*p * 2 - 1) / 2;
    if (abs(q) <= .425) {
	r__ = .180625 - q * q;
	ret_val = q * (((((((r__ * 2509.0809287301226727 + 
		33430.575583588128105) * r__ + 67265.770927008700853) * r__ + 
		45921.953931549871457) * r__ + 13731.693765509461125) * r__ + 
		1971.5909503065514427) * r__ + 133.14166789178437745) * r__ + 
		3.387132872796366608) / (((((((r__ * 5226.495278852854561 + 
		28729.085735721942674) * r__ + 39307.89580009271061) * r__ + 
		21213.794301586595867) * r__ + 5394.1960214247511077) * r__ + 
		687.1870074920579083) * r__ + 42.313330701600911252) * r__ + 
		1);
    } else {
/* Computing MIN */
	d__1 = *p, d__2 = 1 - *p;
	r__ = min(d__1,d__2);
	if (r__ > 0.) {
	    r__ = sqrt(-log(r__));
	    if (r__ <= 5.) {
		r__ += -1.6;
		ret_val = (((((((r__ * 7.7454501427834140764e-4 + 
			.0227238449892691845833) * r__ + 
			.24178072517745061177) * r__ + 1.27045825245236838258)
			 * r__ + 3.64784832476320460504) * r__ + 
			5.7694972214606914055) * r__ + 4.6303378461565452959) 
			* r__ + 1.42343711074968357734) / (((((((r__ * 
			1.05075007164441684324e-9 + 5.475938084995344946e-4) *
			 r__ + .0151986665636164571966) * r__ + 
			.14810397642748007459) * r__ + .68976733498510000455) 
			* r__ + 1.6763848301838038494) * r__ + 
			2.05319162663775882187) * r__ + 1);
	    } else {
		r__ += -5.;
		ret_val = (((((((r__ * 2.01033439929228813265e-7 + 
			2.71155556874348757815e-5) * r__ + 
			.0012426609473880784386) * r__ + 
			.026532189526576123093) * r__ + .29656057182850489123)
			 * r__ + 1.7848265399172913358) * r__ + 
			5.4637849111641143699) * r__ + 6.6579046435011037772) 
			/ (((((((r__ * 2.04426310338993978564e-15 + 
			1.4215117583164458887e-7) * r__ + 
			1.8463183175100546818e-5) * r__ + 
			7.868691311456132591e-4) * r__ + 
			.0148753612908506148525) * r__ + 
			.13692988092273580531) * r__ + .59983220655588793769) 
			* r__ + 1);
	    }
	} else {
	    ret_val = 9.;
	}
	if (q < 0.) {
	    ret_val = -ret_val;
	}
    }
    return ret_val;
} /* phinvs_ */

doublereal bvnmvn_(doublereal *lower, doublereal *upper, integer *infin, 
	doublereal *correl)
{
    /* System generated locals */
    doublereal ret_val, d__1, d__2, d__3, d__4;

    /* Local variables */
    extern doublereal bvu_(doublereal *, doublereal *, doublereal *);


/*     A function for computing bivariate normal probabilities. */

/*  Parameters */

/*     LOWER  REAL, array of lower integration limits. */
/*     UPPER  REAL, array of upper integration limits. */
/*     INFIN  INTEGER, array of integration limits flags: */
/*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)]; */
/*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity); */
/*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)]. */
/*     CORREL REAL, correlation coefficient. */

    /* Parameter adjustments */
    --infin;
    --upper;
    --lower;

    /* Function Body */
    if (infin[1] == 2 && infin[2] == 2) {
	ret_val = bvu_(&lower[1], &lower[2], correl) - bvu_(&upper[1], &lower[
		2], correl) - bvu_(&lower[1], &upper[2], correl) + bvu_(&
		upper[1], &upper[2], correl);
    } else if (infin[1] == 2 && infin[2] == 1) {
	ret_val = bvu_(&lower[1], &lower[2], correl) - bvu_(&upper[1], &lower[
		2], correl);
    } else if (infin[1] == 1 && infin[2] == 2) {
	ret_val = bvu_(&lower[1], &lower[2], correl) - bvu_(&lower[1], &upper[
		2], correl);
    } else if (infin[1] == 2 && infin[2] == 0) {
	d__1 = -upper[1];
	d__2 = -upper[2];
	d__3 = -lower[1];
	d__4 = -upper[2];
	ret_val = bvu_(&d__1, &d__2, correl) - bvu_(&d__3, &d__4, correl);
    } else if (infin[1] == 0 && infin[2] == 2) {
	d__1 = -upper[1];
	d__2 = -upper[2];
	d__3 = -upper[1];
	d__4 = -lower[2];
	ret_val = bvu_(&d__1, &d__2, correl) - bvu_(&d__3, &d__4, correl);
    } else if (infin[1] == 1 && infin[2] == 0) {
	d__1 = -upper[2];
	d__2 = -(*correl);
	ret_val = bvu_(&lower[1], &d__1, &d__2);
    } else if (infin[1] == 0 && infin[2] == 1) {
	d__1 = -upper[1];
	d__2 = -(*correl);
	ret_val = bvu_(&d__1, &lower[2], &d__2);
    } else if (infin[1] == 1 && infin[2] == 1) {
	ret_val = bvu_(&lower[1], &lower[2], correl);
    } else if (infin[1] == 0 && infin[2] == 0) {
	d__1 = -upper[1];
	d__2 = -upper[2];
	ret_val = bvu_(&d__1, &d__2, correl);
    }
    return ret_val;
} /* bvnmvn_ */

doublereal bvu_(doublereal *sh, doublereal *sk, doublereal *r__)
{
    /* Initialized data */

    static struct {
	doublereal e_1[3];
	doublereal fill_2[7];
	doublereal e_3[6];
	doublereal fill_4[4];
	doublereal e_5[10];
	} equiv_114 = { .1713244923791705, .3607615730481384, 
		.4679139345726904, {0}, .04717533638651177, .1069393259953183,
		 .1600783285433464, .2031674267230659, .2334925365383547, 
		.2491470458134029, {0}, .01761400713915212, 
		.04060142980038694, .06267204833410906, .08327674157670475, 
		.1019301198172404, .1181945319615184, .1316886384491766, 
		.1420961093183821, .1491729864726037, .1527533871307259 };

#define w ((doublereal *)&equiv_114)

    static struct {
	doublereal e_1[3];
	doublereal fill_2[7];
	doublereal e_3[6];
	doublereal fill_4[4];
	doublereal e_5[10];
	} equiv_115 = { -.9324695142031522, -.6612093864662647, 
		-.238619186083197, {0}, -.9815606342467191, -.904117256370475,
		 -.769902674194305, -.5873179542866171, -.3678314989981802, 
		-.1252334085114692, {0}, -.9931285991850949, 
		-.9639719272779138, -.9122344282513259, -.8391169718222188, 
		-.7463319064601508, -.636053680726515, -.5108670019508271, 
		-.3737060887154196, -.2277858511416451, -.07652652113349733 };

#define x ((doublereal *)&equiv_115)


    /* System generated locals */
    integer i__1;
    doublereal ret_val, d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double asin(doublereal), sin(doublereal), exp(doublereal), sqrt(
	    doublereal);

    /* Local variables */
    static doublereal a, b, c__, d__, h__;
    static integer i__;
    static doublereal k;
    static integer lg;
    static doublereal as;
    static integer ng;
    static doublereal bs, hk, hs, sn, rs, xs, bvn, asr;
    extern doublereal mvnphi_(doublereal *);


/*     A function for computing bivariate normal probabilities. */

/*       Yihong Ge */
/*       Department of Computer Science and Electrical Engineering */
/*       Washington State University */
/*       Pullman, WA 99164-2752 */
/*     and */
/*       Alan Genz */
/*       Department of Mathematics */
/*       Washington State University */
/*       Pullman, WA 99164-3113 */
/*       Email : alangenz@wsu.edu */

/* BVN - calculate the probability that X is larger than SH and Y is */
/*       larger than SK. */

/* Parameters */

/*   SH  REAL, integration limit */
/*   SK  REAL, integration limit */
/*   R   REAL, correlation coefficient */
/*   LG  INTEGER, number of Gauss Rule Points and Weights */

/*     Gauss Legendre Points and Weights, N =  6 */
/*     Gauss Legendre Points and Weights, N = 12 */
/*     Gauss Legendre Points and Weights, N = 20 */
    if (abs(*r__) < .3f) {
	ng = 1;
	lg = 3;
    } else if (abs(*r__) < .75f) {
	ng = 2;
	lg = 6;
    } else {
	ng = 3;
	lg = 10;
    }
    h__ = *sh;
    k = *sk;
    hk = h__ * k;
    bvn = 0.;
    if (abs(*r__) < .925f) {
	hs = (h__ * h__ + k * k) / 2;
	asr = asin(*r__);
	i__1 = lg;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sn = sin(asr * (x[i__ + ng * 10 - 11] + 1) / 2);
	    bvn += w[i__ + ng * 10 - 11] * exp((sn * hk - hs) / (1 - sn * sn))
		    ;
	    sn = sin(asr * (-x[i__ + ng * 10 - 11] + 1) / 2);
	    bvn += w[i__ + ng * 10 - 11] * exp((sn * hk - hs) / (1 - sn * sn))
		    ;
	}
	d__1 = -h__;
	d__2 = -k;
	bvn = bvn * asr / 12.566370614359172 + mvnphi_(&d__1) * mvnphi_(&d__2)
		;
    } else {
	if (*r__ < 0.) {
	    k = -k;
	    hk = -hk;
	}
	if (abs(*r__) < 1.) {
	    as = (1 - *r__) * (*r__ + 1);
	    a = sqrt(as);
/* Computing 2nd power */
	    d__1 = h__ - k;
	    bs = d__1 * d__1;
	    c__ = (4 - hk) / 8;
	    d__ = (12 - hk) / 16;
	    bvn = a * exp(-(bs / as + hk) / 2) * (1 - c__ * (bs - as) * (1 - 
		    d__ * bs / 5) / 3 + c__ * d__ * as * as / 5);
	    if (hk > -160.) {
		b = sqrt(bs);
		d__1 = -b / a;
		bvn -= exp(-hk / 2) * sqrt(6.283185307179586) * mvnphi_(&d__1)
			 * b * (1 - c__ * bs * (1 - d__ * bs / 5) / 3);
	    }
	    a /= 2;
	    i__1 = lg;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
		d__1 = a * (x[i__ + ng * 10 - 11] + 1);
		xs = d__1 * d__1;
		rs = sqrt(1 - xs);
		bvn += a * w[i__ + ng * 10 - 11] * (exp(-bs / (xs * 2) - hk / 
			(rs + 1)) / rs - exp(-(bs / xs + hk) / 2) * (c__ * xs 
			* (d__ * xs + 1) + 1));
/* Computing 2nd power */
		d__1 = -x[i__ + ng * 10 - 11] + 1;
		xs = as * (d__1 * d__1) / 4;
		rs = sqrt(1 - xs);
		bvn += a * w[i__ + ng * 10 - 11] * exp(-(bs / xs + hk) / 2) * 
			(exp(-hk * (1 - rs) / ((rs + 1) * 2)) / rs - (c__ * 
			xs * (d__ * xs + 1) + 1));
	    }
	    bvn = -bvn / 6.283185307179586;
	}
	if (*r__ > 0.) {
	    d__1 = -max(h__,k);
	    bvn += mvnphi_(&d__1);
	}
	if (*r__ < 0.) {
/* Computing MAX */
	    d__3 = -h__;
	    d__4 = -k;
	    d__1 = 0., d__2 = mvnphi_(&d__3) - mvnphi_(&d__4);
	    bvn = -bvn + max(d__1,d__2);
	}
    }
    ret_val = bvn;
    return ret_val;
} /* bvu_ */

#undef x
#undef w


doublereal mvnuni_(void)
{
    /* Initialized data */

    static integer x10 = 15485857;
    static integer x11 = 17329489;
    static integer x12 = 36312197;
    static integer x20 = 55911127;
    static integer x21 = 75906931;
    static integer x22 = 96210113;

    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static integer h__, z__, p12, p13, p21, p23;


/*     Uniform (0,1) random number generator */

/*     Reference: */
/*     L'Ecuyer, Pierre (1996), */
/*     "Combined Multiple Recursive Random Number Generators" */
/*     Operations Research 44, pp. 816-822. */


/*                 INVMP1 = 1/(M1+1) */

/*     Component 1 */

    h__ = x10 / 11714;
    p13 = (x10 - h__ * 11714) * 183326 - h__ * 2883;
    h__ = x11 / 33921;
    p12 = (x11 - h__ * 33921) * 63308 - h__ * 12979;
    if (p13 < 0) {
	p13 += 2147483647;
    }
    if (p12 < 0) {
	p12 += 2147483647;
    }
    x10 = x11;
    x11 = x12;
    x12 = p12 - p13;
    if (x12 < 0) {
	x12 += 2147483647;
    }

/*     Component 2 */

    h__ = x20 / 3976;
    p23 = (x20 - h__ * 3976) * 539608 - h__ * 2071;
    h__ = x22 / 24919;
    p21 = (x22 - h__ * 24919) * 86098 - h__ * 7417;
    if (p23 < 0) {
	p23 += 2145483479;
    }
    if (p21 < 0) {
	p21 += 2145483479;
    }
    x20 = x21;
    x21 = x22;
    x22 = p21 - p23;
    if (x22 < 0) {
	x22 += 2145483479;
    }

/*     Combination */

    z__ = x12 - x22;
    if (z__ <= 0) {
	z__ += 2147483647;
    }
    ret_val = z__ * 4.656612873077392578125e-10;
    return ret_val;
} /* mvnuni_ */

