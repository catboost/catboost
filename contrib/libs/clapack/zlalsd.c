/* zlalsd.f -- translated by f2c (version 20061008).
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

static doublecomplex c_b1 = {0.,0.};
static integer c__1 = 1;
static integer c__0 = 0;
static doublereal c_b10 = 1.;
static doublereal c_b35 = 0.;

/* Subroutine */ int zlalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, doublereal *d__, doublereal *e, doublecomplex *b, integer *ldb, 
	 doublereal *rcond, integer *rank, doublecomplex *work, doublereal *
	rwork, integer *iwork, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    doublereal d__1;
    doublecomplex z__1;

    /* Builtin functions */
    double d_imag(doublecomplex *), log(doublereal), d_sign(doublereal *, 
	    doublereal *);

    /* Local variables */
    integer c__, i__, j, k;
    doublereal r__;
    integer s, u, z__;
    doublereal cs;
    integer bx;
    doublereal sn;
    integer st, vt, nm1, st1;
    doublereal eps;
    integer iwk;
    doublereal tol;
    integer difl, difr;
    doublereal rcnd;
    integer jcol, irwb, perm, nsub, nlvl, sqre, bxst, jrow, irwu, jimag;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    integer jreal, irwib, poles, sizei, irwrb, nsize;
    extern /* Subroutine */ int zdrot_(integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublereal *, doublereal *), zcopy_(
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *)
	    ;
    integer irwvt, icmpq1, icmpq2;
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlasda_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     integer *), dlascl_(char *, integer *, integer *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dlasdq_(char *, integer *, integer *, integer 
	    *, integer *, integer *, doublereal *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, integer *), dlaset_(char *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *), dlartg_(doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), xerbla_(char *, integer *);
    integer givcol;
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int zlalsa_(integer *, integer *, integer *, 
	    integer *, doublecomplex *, integer *, doublecomplex *, integer *, 
	     doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, integer *, integer *), zlascl_(char *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, integer *, 
	    doublecomplex *, integer *, integer *), dlasrt_(char *, 
	    integer *, doublereal *, integer *), zlacpy_(char *, 
	    integer *, integer *, doublecomplex *, integer *, doublecomplex *, 
	     integer *), zlaset_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, doublecomplex *, integer *);
    doublereal orgnrm;
    integer givnum, givptr, nrwork, irwwrk, smlszp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLALSD uses the singular value decomposition of A to solve the least */
/*  squares problem of finding X to minimize the Euclidean norm of each */
/*  column of A*X-B, where A is N-by-N upper bidiagonal, and X and B */
/*  are N-by-NRHS. The solution X overwrites B. */

/*  The singular values of A smaller than RCOND times the largest */
/*  singular value are treated as zero in solving the least squares */
/*  problem; in this case a minimum norm solution is returned. */
/*  The actual singular values are returned in D in ascending order. */

/*  This code makes very mild assumptions about floating point */
/*  arithmetic. It will work on machines with a guard digit in */
/*  add/subtract, or on those binary machines without guard digits */
/*  which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2. */
/*  It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  UPLO   (input) CHARACTER*1 */
/*         = 'U': D and E define an upper bidiagonal matrix. */
/*         = 'L': D and E define a  lower bidiagonal matrix. */

/*  SMLSIZ (input) INTEGER */
/*         The maximum size of the subproblems at the bottom of the */
/*         computation tree. */

/*  N      (input) INTEGER */
/*         The dimension of the  bidiagonal matrix.  N >= 0. */

/*  NRHS   (input) INTEGER */
/*         The number of columns of B. NRHS must be at least 1. */

/*  D      (input/output) DOUBLE PRECISION array, dimension (N) */
/*         On entry D contains the main diagonal of the bidiagonal */
/*         matrix. On exit, if INFO = 0, D contains its singular values. */

/*  E      (input/output) DOUBLE PRECISION array, dimension (N-1) */
/*         Contains the super-diagonal entries of the bidiagonal matrix. */
/*         On exit, E has been destroyed. */

/*  B      (input/output) COMPLEX*16 array, dimension (LDB,NRHS) */
/*         On input, B contains the right hand sides of the least */
/*         squares problem. On output, B contains the solution X. */

/*  LDB    (input) INTEGER */
/*         The leading dimension of B in the calling subprogram. */
/*         LDB must be at least max(1,N). */

/*  RCOND  (input) DOUBLE PRECISION */
/*         The singular values of A less than or equal to RCOND times */
/*         the largest singular value are treated as zero in solving */
/*         the least squares problem. If RCOND is negative, */
/*         machine precision is used instead. */
/*         For example, if diag(S)*X=B were the least squares problem, */
/*         where diag(S) is a diagonal matrix of singular values, the */
/*         solution would be X(i) = B(i) / S(i) if S(i) is greater than */
/*         RCOND*max(S), and X(i) = 0 if S(i) is less than or equal to */
/*         RCOND*max(S). */

/*  RANK   (output) INTEGER */
/*         The number of singular values of A greater than RCOND times */
/*         the largest singular value. */

/*  WORK   (workspace) COMPLEX*16 array, dimension at least */
/*         (N * NRHS). */

/*  RWORK  (workspace) DOUBLE PRECISION array, dimension at least */
/*         (9*N + 2*N*SMLSIZ + 8*N*NLVL + 3*SMLSIZ*NRHS + (SMLSIZ+1)**2), */
/*         where */
/*         NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 ) */

/*  IWORK  (workspace) INTEGER array, dimension at least */
/*         (3*N*NLVL + 11*N). */

/*  INFO   (output) INTEGER */
/*         = 0:  successful exit. */
/*         < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*         > 0:  The algorithm failed to compute an singular value while */
/*               working on the submatrix lying in rows and columns */
/*               INFO/(N+1) through MOD(INFO,N+1). */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Ren-Cang Li, Computer Science Division, University of */
/*       California at Berkeley, USA */
/*     Osni Marques, LBNL/NERSC, USA */

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
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 1) {
	*info = -4;
    } else if (*ldb < 1 || *ldb < *n) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZLALSD", &i__1);
	return 0;
    }

    eps = dlamch_("Epsilon");

/*     Set up the tolerance. */

    if (*rcond <= 0. || *rcond >= 1.) {
	rcnd = eps;
    } else {
	rcnd = *rcond;
    }

    *rank = 0;

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	if (d__[1] == 0.) {
	    zlaset_("A", &c__1, nrhs, &c_b1, &c_b1, &b[b_offset], ldb);
	} else {
	    *rank = 1;
	    zlascl_("G", &c__0, &c__0, &d__[1], &c_b10, &c__1, nrhs, &b[
		    b_offset], ldb, info);
	    d__[1] = abs(d__[1]);
	}
	return 0;
    }

/*     Rotate the matrix if it is lower bidiagonal. */

    if (*(unsigned char *)uplo == 'L') {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (*nrhs == 1) {
		zdrot_(&c__1, &b[i__ + b_dim1], &c__1, &b[i__ + 1 + b_dim1], &
			c__1, &cs, &sn);
	    } else {
		rwork[(i__ << 1) - 1] = cs;
		rwork[i__ * 2] = sn;
	    }
/* L10: */
	}
	if (*nrhs > 1) {
	    i__1 = *nrhs;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *n - 1;
		for (j = 1; j <= i__2; ++j) {
		    cs = rwork[(j << 1) - 1];
		    sn = rwork[j * 2];
		    zdrot_(&c__1, &b[j + i__ * b_dim1], &c__1, &b[j + 1 + i__ 
			    * b_dim1], &c__1, &cs, &sn);
/* L20: */
		}
/* L30: */
	    }
	}
    }

/*     Scale. */

    nm1 = *n - 1;
    orgnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.) {
	zlaset_("A", n, nrhs, &c_b1, &c_b1, &b[b_offset], ldb);
	return 0;
    }

    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b10, n, &c__1, &d__[1], n, info);
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b10, &nm1, &c__1, &e[1], &nm1, 
	    info);

/*     If N is smaller than the minimum divide size SMLSIZ, then solve */
/*     the problem with another solver. */

    if (*n <= *smlsiz) {
	irwu = 1;
	irwvt = irwu + *n * *n;
	irwwrk = irwvt + *n * *n;
	irwrb = irwwrk;
	irwib = irwrb + *n * *nrhs;
	irwb = irwib + *n * *nrhs;
	dlaset_("A", n, n, &c_b35, &c_b10, &rwork[irwu], n);
	dlaset_("A", n, n, &c_b35, &c_b10, &rwork[irwvt], n);
	dlasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &rwork[irwvt], n, 
		&rwork[irwu], n, &rwork[irwwrk], &c__1, &rwork[irwwrk], info);
	if (*info != 0) {
	    return 0;
	}

/*        In the real version, B is passed to DLASDQ and multiplied */
/*        internally by Q'. Here B is complex and that product is */
/*        computed below in two steps (real and imaginary parts). */

	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		i__3 = jrow + jcol * b_dim1;
		rwork[j] = b[i__3].r;
/* L40: */
	    }
/* L50: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b10, &rwork[irwu], n, &rwork[irwb], n, 
		 &c_b35, &rwork[irwrb], n);
	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L60: */
	    }
/* L70: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b10, &rwork[irwu], n, &rwork[irwb], n, 
		 &c_b35, &rwork[irwib], n);
	jreal = irwrb - 1;
	jimag = irwib - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++jreal;
		++jimag;
		i__3 = jrow + jcol * b_dim1;
		i__4 = jreal;
		i__5 = jimag;
		z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L80: */
	    }
/* L90: */
	}

	tol = rcnd * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (d__[i__] <= tol) {
		zlaset_("A", &c__1, nrhs, &c_b1, &c_b1, &b[i__ + b_dim1], ldb);
	    } else {
		zlascl_("G", &c__0, &c__0, &d__[i__], &c_b10, &c__1, nrhs, &b[
			i__ + b_dim1], ldb, info);
		++(*rank);
	    }
/* L100: */
	}

/*        Since B is complex, the following call to DGEMM is performed */
/*        in two steps (real and imaginary parts). That is for V * B */
/*        (in the real version of the code V' is stored in WORK). */

/*        CALL DGEMM( 'T', 'N', N, NRHS, N, ONE, WORK, N, B, LDB, ZERO, */
/*    $               WORK( NWORK ), N ) */

	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		i__3 = jrow + jcol * b_dim1;
		rwork[j] = b[i__3].r;
/* L110: */
	    }
/* L120: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b10, &rwork[irwvt], n, &rwork[irwb], 
		n, &c_b35, &rwork[irwrb], n);
	j = irwb - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++j;
		rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L130: */
	    }
/* L140: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b10, &rwork[irwvt], n, &rwork[irwb], 
		n, &c_b35, &rwork[irwib], n);
	jreal = irwrb - 1;
	jimag = irwib - 1;
	i__1 = *nrhs;
	for (jcol = 1; jcol <= i__1; ++jcol) {
	    i__2 = *n;
	    for (jrow = 1; jrow <= i__2; ++jrow) {
		++jreal;
		++jimag;
		i__3 = jrow + jcol * b_dim1;
		i__4 = jreal;
		i__5 = jimag;
		z__1.r = rwork[i__4], z__1.i = rwork[i__5];
		b[i__3].r = z__1.r, b[i__3].i = z__1.i;
/* L150: */
	    }
/* L160: */
	}

/*        Unscale. */

	dlascl_("G", &c__0, &c__0, &c_b10, &orgnrm, n, &c__1, &d__[1], n, 
		info);
	dlasrt_("D", n, &d__[1], info);
	zlascl_("G", &c__0, &c__0, &orgnrm, &c_b10, n, nrhs, &b[b_offset], 
		ldb, info);

	return 0;
    }

/*     Book-keeping and setting up some constants. */

    nlvl = (integer) (log((doublereal) (*n) / (doublereal) (*smlsiz + 1)) / 
	    log(2.)) + 1;

    smlszp = *smlsiz + 1;

    u = 1;
    vt = *smlsiz * *n + 1;
    difl = vt + smlszp * *n;
    difr = difl + nlvl * *n;
    z__ = difr + (nlvl * *n << 1);
    c__ = z__ + nlvl * *n;
    s = c__ + *n;
    poles = s + *n;
    givnum = poles + (nlvl << 1) * *n;
    nrwork = givnum + (nlvl << 1) * *n;
    bx = 1;

    irwrb = nrwork;
    irwib = irwrb + *smlsiz * *nrhs;
    irwb = irwib + *smlsiz * *nrhs;

    sizei = *n + 1;
    k = sizei + *n;
    givptr = k + *n;
    perm = givptr + *n;
    givcol = perm + nlvl * *n;
    iwk = givcol + (nlvl * *n << 1);

    st = 1;
    sqre = 0;
    icmpq1 = 1;
    icmpq2 = 0;
    nsub = 0;

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = d__[i__], abs(d__1)) < eps) {
	    d__[i__] = d_sign(&eps, &d__[i__]);
	}
/* L170: */
    }

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = e[i__], abs(d__1)) < eps || i__ == nm1) {
	    ++nsub;
	    iwork[nsub] = st;

/*           Subproblem found. First determine its size and then */
/*           apply divide and conquer on it. */

	    if (i__ < nm1) {

/*              A subproblem with E(I) small for I < NM1. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else if ((d__1 = e[i__], abs(d__1)) >= eps) {

/*              A subproblem with E(NM1) not too small but I = NM1. */

		nsize = *n - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else {

/*              A subproblem with E(NM1) small. This implies an */
/*              1-by-1 subproblem at D(N), which is not solved */
/*              explicitly. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
		++nsub;
		iwork[nsub] = *n;
		iwork[sizei + nsub - 1] = 1;
		zcopy_(nrhs, &b[*n + b_dim1], ldb, &work[bx + nm1], n);
	    }
	    st1 = st - 1;
	    if (nsize == 1) {

/*              This is a 1-by-1 subproblem and is not solved */
/*              explicitly. */

		zcopy_(nrhs, &b[st + b_dim1], ldb, &work[bx + st1], n);
	    } else if (nsize <= *smlsiz) {

/*              This is a small subproblem and is solved by DLASDQ. */

		dlaset_("A", &nsize, &nsize, &c_b35, &c_b10, &rwork[vt + st1], 
			 n);
		dlaset_("A", &nsize, &nsize, &c_b35, &c_b10, &rwork[u + st1], 
			n);
		dlasdq_("U", &c__0, &nsize, &nsize, &nsize, &c__0, &d__[st], &
			e[st], &rwork[vt + st1], n, &rwork[u + st1], n, &
			rwork[nrwork], &c__1, &rwork[nrwork], info)
			;
		if (*info != 0) {
		    return 0;
		}

/*              In the real version, B is passed to DLASDQ and multiplied */
/*              internally by Q'. Here B is complex and that product is */
/*              computed below in two steps (real and imaginary parts). */

		j = irwb - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++j;
			i__4 = jrow + jcol * b_dim1;
			rwork[j] = b[i__4].r;
/* L180: */
		    }
/* L190: */
		}
		dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b10, &rwork[u + st1]
, n, &rwork[irwb], &nsize, &c_b35, &rwork[irwrb], &
			nsize);
		j = irwb - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++j;
			rwork[j] = d_imag(&b[jrow + jcol * b_dim1]);
/* L200: */
		    }
/* L210: */
		}
		dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b10, &rwork[u + st1]
, n, &rwork[irwb], &nsize, &c_b35, &rwork[irwib], &
			nsize);
		jreal = irwrb - 1;
		jimag = irwib - 1;
		i__2 = *nrhs;
		for (jcol = 1; jcol <= i__2; ++jcol) {
		    i__3 = st + nsize - 1;
		    for (jrow = st; jrow <= i__3; ++jrow) {
			++jreal;
			++jimag;
			i__4 = jrow + jcol * b_dim1;
			i__5 = jreal;
			i__6 = jimag;
			z__1.r = rwork[i__5], z__1.i = rwork[i__6];
			b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L220: */
		    }
/* L230: */
		}

		zlacpy_("A", &nsize, nrhs, &b[st + b_dim1], ldb, &work[bx + 
			st1], n);
	    } else {

/*              A large problem. Solve it using divide and conquer. */

		dlasda_(&icmpq1, smlsiz, &nsize, &sqre, &d__[st], &e[st], &
			rwork[u + st1], n, &rwork[vt + st1], &iwork[k + st1], 
			&rwork[difl + st1], &rwork[difr + st1], &rwork[z__ + 
			st1], &rwork[poles + st1], &iwork[givptr + st1], &
			iwork[givcol + st1], n, &iwork[perm + st1], &rwork[
			givnum + st1], &rwork[c__ + st1], &rwork[s + st1], &
			rwork[nrwork], &iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
		bxst = bx + st1;
		zlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &b[st + b_dim1], ldb, &
			work[bxst], n, &rwork[u + st1], n, &rwork[vt + st1], &
			iwork[k + st1], &rwork[difl + st1], &rwork[difr + st1]
, &rwork[z__ + st1], &rwork[poles + st1], &iwork[
			givptr + st1], &iwork[givcol + st1], n, &iwork[perm + 
			st1], &rwork[givnum + st1], &rwork[c__ + st1], &rwork[
			s + st1], &rwork[nrwork], &iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
	    }
	    st = i__ + 1;
	}
/* L240: */
    }

/*     Apply the singular values and treat the tiny ones as zero. */

    tol = rcnd * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Some of the elements in D can be negative because 1-by-1 */
/*        subproblems were not solved explicitly. */

	if ((d__1 = d__[i__], abs(d__1)) <= tol) {
	    zlaset_("A", &c__1, nrhs, &c_b1, &c_b1, &work[bx + i__ - 1], n);
	} else {
	    ++(*rank);
	    zlascl_("G", &c__0, &c__0, &d__[i__], &c_b10, &c__1, nrhs, &work[
		    bx + i__ - 1], n, info);
	}
	d__[i__] = (d__1 = d__[i__], abs(d__1));
/* L250: */
    }

/*     Now apply back the right singular vectors. */

    icmpq2 = 1;
    i__1 = nsub;
    for (i__ = 1; i__ <= i__1; ++i__) {
	st = iwork[i__];
	st1 = st - 1;
	nsize = iwork[sizei + i__ - 1];
	bxst = bx + st1;
	if (nsize == 1) {
	    zcopy_(nrhs, &work[bxst], n, &b[st + b_dim1], ldb);
	} else if (nsize <= *smlsiz) {

/*           Since B and BX are complex, the following call to DGEMM */
/*           is performed in two steps (real and imaginary parts). */

/*           CALL DGEMM( 'T', 'N', NSIZE, NRHS, NSIZE, ONE, */
/*    $                  RWORK( VT+ST1 ), N, RWORK( BXST ), N, ZERO, */
/*    $                  B( ST, 1 ), LDB ) */

	    j = bxst - *n - 1;
	    jreal = irwb - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		j += *n;
		i__3 = nsize;
		for (jrow = 1; jrow <= i__3; ++jrow) {
		    ++jreal;
		    i__4 = j + jrow;
		    rwork[jreal] = work[i__4].r;
/* L260: */
		}
/* L270: */
	    }
	    dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b10, &rwork[vt + st1], 
		    n, &rwork[irwb], &nsize, &c_b35, &rwork[irwrb], &nsize);
	    j = bxst - *n - 1;
	    jimag = irwb - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		j += *n;
		i__3 = nsize;
		for (jrow = 1; jrow <= i__3; ++jrow) {
		    ++jimag;
		    rwork[jimag] = d_imag(&work[j + jrow]);
/* L280: */
		}
/* L290: */
	    }
	    dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b10, &rwork[vt + st1], 
		    n, &rwork[irwb], &nsize, &c_b35, &rwork[irwib], &nsize);
	    jreal = irwrb - 1;
	    jimag = irwib - 1;
	    i__2 = *nrhs;
	    for (jcol = 1; jcol <= i__2; ++jcol) {
		i__3 = st + nsize - 1;
		for (jrow = st; jrow <= i__3; ++jrow) {
		    ++jreal;
		    ++jimag;
		    i__4 = jrow + jcol * b_dim1;
		    i__5 = jreal;
		    i__6 = jimag;
		    z__1.r = rwork[i__5], z__1.i = rwork[i__6];
		    b[i__4].r = z__1.r, b[i__4].i = z__1.i;
/* L300: */
		}
/* L310: */
	    }
	} else {
	    zlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &work[bxst], n, &b[st + 
		    b_dim1], ldb, &rwork[u + st1], n, &rwork[vt + st1], &
		    iwork[k + st1], &rwork[difl + st1], &rwork[difr + st1], &
		    rwork[z__ + st1], &rwork[poles + st1], &iwork[givptr + 
		    st1], &iwork[givcol + st1], n, &iwork[perm + st1], &rwork[
		    givnum + st1], &rwork[c__ + st1], &rwork[s + st1], &rwork[
		    nrwork], &iwork[iwk], info);
	    if (*info != 0) {
		return 0;
	    }
	}
/* L320: */
    }

/*     Unscale and sort the singular values. */

    dlascl_("G", &c__0, &c__0, &c_b10, &orgnrm, n, &c__1, &d__[1], n, info);
    dlasrt_("D", n, &d__[1], info);
    zlascl_("G", &c__0, &c__0, &orgnrm, &c_b10, n, nrhs, &b[b_offset], ldb, 
	    info);

    return 0;

/*     End of ZLALSD */

} /* zlalsd_ */
