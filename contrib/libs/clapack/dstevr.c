/* dstevr.f -- translated by f2c (version 20061008).
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

static integer c__10 = 10;
static integer c__1 = 1;
static integer c__2 = 2;
static integer c__3 = 3;
static integer c__4 = 4;

/* Subroutine */ int dstevr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, jj;
    doublereal eps, vll, vuu, tmp1;
    integer imax;
    doublereal rmin, rmax;
    logical test;
    doublereal tnrm;
    integer itmp1;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal sigma;
    extern logical lsame_(char *, char *);
    char order[1];
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dswap_(integer *, doublereal *, integer 
	    *, doublereal *, integer *);
    integer lwmin;
    logical wantz;
    extern doublereal dlamch_(char *);
    logical alleig, indeig;
    integer iscale, ieeeok, indibl, indifl;
    logical valeig;
    doublereal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    integer indisp;
    extern /* Subroutine */ int dstein_(integer *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, integer *, integer *), 
	    dsterf_(integer *, doublereal *, doublereal *, integer *);
    integer indiwo;
    extern /* Subroutine */ int dstebz_(char *, char *, integer *, doublereal 
	    *, doublereal *, integer *, integer *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *, doublereal *, integer *, integer *), 
	    dstemr_(char *, char *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    logical *, doublereal *, integer *, integer *, integer *, integer 
	    *);
    integer liwmin;
    logical tryrac;
    integer nsplit;
    doublereal smlnum;
    logical lquery;


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DSTEVR computes selected eigenvalues and, optionally, eigenvectors */
/*  of a real symmetric tridiagonal matrix T.  Eigenvalues and */
/*  eigenvectors can be selected by specifying either a range of values */
/*  or a range of indices for the desired eigenvalues. */

/*  Whenever possible, DSTEVR calls DSTEMR to compute the */
/*  eigenspectrum using Relatively Robust Representations.  DSTEMR */
/*  computes eigenvalues by the dqds algorithm, while orthogonal */
/*  eigenvectors are computed from various "good" L D L^T representations */
/*  (also known as Relatively Robust Representations). Gram-Schmidt */
/*  orthogonalization is avoided as far as possible. More specifically, */
/*  the various steps of the algorithm are as follows. For the i-th */
/*  unreduced block of T, */
/*     (a) Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T */
/*          is a relatively robust representation, */
/*     (b) Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high */
/*         relative accuracy by the dqds algorithm, */
/*     (c) If there is a cluster of close eigenvalues, "choose" sigma_i */
/*         close to the cluster, and go to step (a), */
/*     (d) Given the approximate eigenvalue lambda_j of L_i D_i L_i^T, */
/*         compute the corresponding eigenvector by forming a */
/*         rank-revealing twisted factorization. */
/*  The desired accuracy of the output can be specified by the input */
/*  parameter ABSTOL. */

/*  For more details, see "A new O(n^2) algorithm for the symmetric */
/*  tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon, */
/*  Computer Science Division Technical Report No. UCB//CSD-97-971, */
/*  UC Berkeley, May 1997. */


/*  Note 1 : DSTEVR calls DSTEMR when the full spectrum is requested */
/*  on machines which conform to the ieee-754 floating point standard. */
/*  DSTEVR calls DSTEBZ and DSTEIN on non-ieee machines and */
/*  when partial spectrum requests are made. */

/*  Normal execution of DSTEMR may create NaNs and infinities and */
/*  hence may abort due to a floating point exception in environments */
/*  which do not handle NaNs and infinities in the ieee standard default */
/*  manner. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only; */
/*          = 'V':  Compute eigenvalues and eigenvectors. */

/*  RANGE   (input) CHARACTER*1 */
/*          = 'A': all eigenvalues will be found. */
/*          = 'V': all eigenvalues in the half-open interval (VL,VU] */
/*                 will be found. */
/*          = 'I': the IL-th through IU-th eigenvalues will be found. */
/* ********* For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and */
/* ********* DSTEIN are called */

/*  N       (input) INTEGER */
/*          The order of the matrix.  N >= 0. */

/*  D       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the n diagonal elements of the tridiagonal matrix */
/*          A. */
/*          On exit, D may be multiplied by a constant factor chosen */
/*          to avoid over/underflow in computing the eigenvalues. */

/*  E       (input/output) DOUBLE PRECISION array, dimension (max(1,N-1)) */
/*          On entry, the (n-1) subdiagonal elements of the tridiagonal */
/*          matrix A in elements 1 to N-1 of E. */
/*          On exit, E may be multiplied by a constant factor chosen */
/*          to avoid over/underflow in computing the eigenvalues. */

/*  VL      (input) DOUBLE PRECISION */
/*  VU      (input) DOUBLE PRECISION */
/*          If RANGE='V', the lower and upper bounds of the interval to */
/*          be searched for eigenvalues. VL < VU. */
/*          Not referenced if RANGE = 'A' or 'I'. */

/*  IL      (input) INTEGER */
/*  IU      (input) INTEGER */
/*          If RANGE='I', the indices (in ascending order) of the */
/*          smallest and largest eigenvalues to be returned. */
/*          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0. */
/*          Not referenced if RANGE = 'A' or 'V'. */

/*  ABSTOL  (input) DOUBLE PRECISION */
/*          The absolute error tolerance for the eigenvalues. */
/*          An approximate eigenvalue is accepted as converged */
/*          when it is determined to lie in an interval [a,b] */
/*          of width less than or equal to */

/*                  ABSTOL + EPS *   max( |a|,|b| ) , */

/*          where EPS is the machine precision.  If ABSTOL is less than */
/*          or equal to zero, then  EPS*|T|  will be used in its place, */
/*          where |T| is the 1-norm of the tridiagonal matrix obtained */
/*          by reducing A to tridiagonal form. */

/*          See "Computing Small Singular Values of Bidiagonal Matrices */
/*          with Guaranteed High Relative Accuracy," by Demmel and */
/*          Kahan, LAPACK Working Note #3. */

/*          If high relative accuracy is important, set ABSTOL to */
/*          DLAMCH( 'Safe minimum' ).  Doing so will guarantee that */
/*          eigenvalues are computed to high relative accuracy when */
/*          possible in future releases.  The current code does not */
/*          make any guarantees about high relative accuracy, but */
/*          future releases will. See J. Barlow and J. Demmel, */
/*          "Computing Accurate Eigensystems of Scaled Diagonally */
/*          Dominant Matrices", LAPACK Working Note #7, for a discussion */
/*          of which matrices define their eigenvalues to high relative */
/*          accuracy. */

/*  M       (output) INTEGER */
/*          The total number of eigenvalues found.  0 <= M <= N. */
/*          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1. */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          The first M elements contain the selected eigenvalues in */
/*          ascending order. */

/*  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M) ) */
/*          If JOBZ = 'V', then if INFO = 0, the first M columns of Z */
/*          contain the orthonormal eigenvectors of the matrix A */
/*          corresponding to the selected eigenvalues, with the i-th */
/*          column of Z holding the eigenvector associated with W(i). */
/*          Note: the user must ensure that at least max(1,M) columns are */
/*          supplied in the array Z; if RANGE = 'V', the exact value of M */
/*          is not known in advance and an upper bound must be used. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          JOBZ = 'V', LDZ >= max(1,N). */

/*  ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) ) */
/*          The support of the eigenvectors in Z, i.e., the indices */
/*          indicating the nonzero elements in Z. The i-th eigenvector */
/*          is nonzero only in elements ISUPPZ( 2*i-1 ) through */
/*          ISUPPZ( 2*i ). */
/* ********* Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1 */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal (and */
/*          minimal) LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK.  LWORK >= max(1,20*N). */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal sizes of the WORK and IWORK */
/*          arrays, returns these values as the first entries of the WORK */
/*          and IWORK arrays, and no error message related to LWORK or */
/*          LIWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          On exit, if INFO = 0, IWORK(1) returns the optimal (and */
/*          minimal) LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK.  LIWORK >= max(1,10*N). */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal sizes of the WORK and */
/*          IWORK arrays, returns these values as the first entries of */
/*          the WORK and IWORK arrays, and no error message related to */
/*          LWORK or LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  Internal error */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Inderjit Dhillon, IBM Almaden, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Ken Stanley, Computer Science Division, University of */
/*       California at Berkeley, USA */

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


/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    /* Function Body */
    ieeeok = ilaenv_(&c__10, "DSTEVR", "N", &c__1, &c__2, &c__3, &c__4);

    wantz = lsame_(jobz, "V");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");

    lquery = *lwork == -1 || *liwork == -1;
/* Computing MAX */
    i__1 = 1, i__2 = *n * 20;
    lwmin = max(i__1,i__2);
/* Computing MAX */
    i__1 = 1, i__2 = *n * 10;
    liwmin = max(i__1,i__2);


    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else {
	if (valeig) {
	    if (*n > 0 && *vu <= *vl) {
		*info = -7;
	    }
	} else if (indeig) {
	    if (*il < 1 || *il > max(1,*n)) {
		*info = -8;
	    } else if (*iu < min(*n,*il) || *iu > *n) {
		*info = -9;
	    }
	}
    }
    if (*info == 0) {
	if (*ldz < 1 || wantz && *ldz < *n) {
	    *info = -14;
	}
    }

    if (*info == 0) {
	work[1] = (doublereal) lwmin;
	iwork[1] = liwmin;

	if (*lwork < lwmin && ! lquery) {
	    *info = -17;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -19;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSTEVR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    *m = 0;
    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (alleig || indeig) {
	    *m = 1;
	    w[1] = d__[1];
	} else {
	    if (*vl < d__[1] && *vu >= d__[1]) {
		*m = 1;
		w[1] = d__[1];
	    }
	}
	if (wantz) {
	    z__[z_dim1 + 1] = 1.;
	}
	return 0;
    }

/*     Get machine constants. */

    safmin = dlamch_("Safe minimum");
    eps = dlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = sqrt(smlnum);
/* Computing MIN */
    d__1 = sqrt(bignum), d__2 = 1. / sqrt(sqrt(safmin));
    rmax = min(d__1,d__2);


/*     Scale matrix to allowable range, if necessary. */

    iscale = 0;
    vll = *vl;
    vuu = *vu;

    tnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (tnrm > 0. && tnrm < rmin) {
	iscale = 1;
	sigma = rmin / tnrm;
    } else if (tnrm > rmax) {
	iscale = 1;
	sigma = rmax / tnrm;
    }
    if (iscale == 1) {
	dscal_(n, &sigma, &d__[1], &c__1);
	i__1 = *n - 1;
	dscal_(&i__1, &sigma, &e[1], &c__1);
	if (valeig) {
	    vll = *vl * sigma;
	    vuu = *vu * sigma;
	}
    }
/*     Initialize indices into workspaces.  Note: These indices are used only */
/*     if DSTERF or DSTEMR fail. */
/*     IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and */
/*     stores the block indices of each of the M<=N eigenvalues. */
    indibl = 1;
/*     IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and */
/*     stores the starting and finishing indices of each block. */
    indisp = indibl + *n;
/*     IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors */
/*     that corresponding to eigenvectors that fail to converge in */
/*     DSTEIN.  This information is discarded; if any fail, the driver */
/*     returns INFO > 0. */
    indifl = indisp + *n;
/*     INDIWO is the offset of the remaining integer workspace. */
    indiwo = indisp + *n;

/*     If all eigenvalues are desired, then */
/*     call DSTERF or DSTEMR.  If this fails for some eigenvalue, then */
/*     try DSTEBZ. */


    test = FALSE_;
    if (indeig) {
	if (*il == 1 && *iu == *n) {
	    test = TRUE_;
	}
    }
    if ((alleig || test) && ieeeok == 1) {
	i__1 = *n - 1;
	dcopy_(&i__1, &e[1], &c__1, &work[1], &c__1);
	if (! wantz) {
	    dcopy_(n, &d__[1], &c__1, &w[1], &c__1);
	    dsterf_(n, &w[1], &work[1], info);
	} else {
	    dcopy_(n, &d__[1], &c__1, &work[*n + 1], &c__1);
	    if (*abstol <= *n * 2. * eps) {
		tryrac = TRUE_;
	    } else {
		tryrac = FALSE_;
	    }
	    i__1 = *lwork - (*n << 1);
	    dstemr_(jobz, "A", n, &work[*n + 1], &work[1], vl, vu, il, iu, m, 
		    &w[1], &z__[z_offset], ldz, n, &isuppz[1], &tryrac, &work[
		    (*n << 1) + 1], &i__1, &iwork[1], liwork, info);

	}
	if (*info == 0) {
	    *m = *n;
	    goto L10;
	}
	*info = 0;
    }

/*     Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN. */

    if (wantz) {
	*(unsigned char *)order = 'B';
    } else {
	*(unsigned char *)order = 'E';
    }
    dstebz_(range, order, n, &vll, &vuu, il, iu, abstol, &d__[1], &e[1], m, &
	    nsplit, &w[1], &iwork[indibl], &iwork[indisp], &work[1], &iwork[
	    indiwo], info);

    if (wantz) {
	dstein_(n, &d__[1], &e[1], m, &w[1], &iwork[indibl], &iwork[indisp], &
		z__[z_offset], ldz, &work[1], &iwork[indiwo], &iwork[indifl], 
		info);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

L10:
    if (iscale == 1) {
	if (*info == 0) {
	    imax = *m;
	} else {
	    imax = *info - 1;
	}
	d__1 = 1. / sigma;
	dscal_(&imax, &d__1, &w[1], &c__1);
    }

/*     If eigenvalues are not in order, then sort them, along with */
/*     eigenvectors. */

    if (wantz) {
	i__1 = *m - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__ = 0;
	    tmp1 = w[j];
	    i__2 = *m;
	    for (jj = j + 1; jj <= i__2; ++jj) {
		if (w[jj] < tmp1) {
		    i__ = jj;
		    tmp1 = w[jj];
		}
/* L20: */
	    }

	    if (i__ != 0) {
		itmp1 = iwork[i__];
		w[i__] = w[j];
		iwork[i__] = iwork[j];
		w[j] = tmp1;
		iwork[j] = itmp1;
		dswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * z_dim1 + 1], 
			 &c__1);
	    }
/* L30: */
	}
    }

/*      Causes problems with tests 19 & 20: */
/*      IF (wantz .and. INDEIG ) Z( 1,1) = Z(1,1) / 1.002 + .002 */


    work[1] = (doublereal) lwmin;
    iwork[1] = liwmin;
    return 0;

/*     End of DSTEVR */

} /* dstevr_ */
