/* dtrexc.f -- translated by f2c (version 20061008).
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
static integer c__2 = 2;

/* Subroutine */ int dtrexc_(char *compq, integer *n, doublereal *t, integer *
	ldt, doublereal *q, integer *ldq, integer *ifst, integer *ilst, 
	doublereal *work, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, t_dim1, t_offset, i__1;

    /* Local variables */
    integer nbf, nbl, here;
    extern logical lsame_(char *, char *);
    logical wantq;
    extern /* Subroutine */ int dlaexc_(logical *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, integer *, integer *, integer 
	    *, doublereal *, integer *), xerbla_(char *, integer *);
    integer nbnext;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTREXC reorders the real Schur factorization of a real matrix */
/*  A = Q*T*Q**T, so that the diagonal block of T with row index IFST is */
/*  moved to row ILST. */

/*  The real Schur form T is reordered by an orthogonal similarity */
/*  transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors */
/*  is updated by postmultiplying it with Z. */

/*  T must be in Schur canonical form (as returned by DHSEQR), that is, */
/*  block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each */
/*  2-by-2 diagonal block has its diagonal elements equal and its */
/*  off-diagonal elements of opposite sign. */

/*  Arguments */
/*  ========= */

/*  COMPQ   (input) CHARACTER*1 */
/*          = 'V':  update the matrix Q of Schur vectors; */
/*          = 'N':  do not update Q. */

/*  N       (input) INTEGER */
/*          The order of the matrix T. N >= 0. */

/*  T       (input/output) DOUBLE PRECISION array, dimension (LDT,N) */
/*          On entry, the upper quasi-triangular matrix T, in Schur */
/*          Schur canonical form. */
/*          On exit, the reordered upper quasi-triangular matrix, again */
/*          in Schur canonical form. */

/*  LDT     (input) INTEGER */
/*          The leading dimension of the array T. LDT >= max(1,N). */

/*  Q       (input/output) DOUBLE PRECISION array, dimension (LDQ,N) */
/*          On entry, if COMPQ = 'V', the matrix Q of Schur vectors. */
/*          On exit, if COMPQ = 'V', Q has been postmultiplied by the */
/*          orthogonal transformation matrix Z which reorders T. */
/*          If COMPQ = 'N', Q is not referenced. */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q.  LDQ >= max(1,N). */

/*  IFST    (input/output) INTEGER */
/*  ILST    (input/output) INTEGER */
/*          Specify the reordering of the diagonal blocks of T. */
/*          The block with row index IFST is moved to row ILST, by a */
/*          sequence of transpositions between adjacent blocks. */
/*          On exit, if IFST pointed on entry to the second row of a */
/*          2-by-2 block, it is changed to point to the first row; ILST */
/*          always points to the first row of the block in its final */
/*          position (which may differ from its input value by +1 or -1). */
/*          1 <= IFST <= N; 1 <= ILST <= N. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          = 1:  two adjacent blocks were too close to swap (the problem */
/*                is very ill-conditioned); T may have been partially */
/*                reordered, and ILST points to the first row of the */
/*                current position of the block being moved. */

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

/*     Decode and test the input arguments. */

    /* Parameter adjustments */
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --work;

    /* Function Body */
    *info = 0;
    wantq = lsame_(compq, "V");
    if (! wantq && ! lsame_(compq, "N")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldt < max(1,*n)) {
	*info = -4;
    } else if (*ldq < 1 || wantq && *ldq < max(1,*n)) {
	*info = -6;
    } else if (*ifst < 1 || *ifst > *n) {
	*info = -7;
    } else if (*ilst < 1 || *ilst > *n) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTREXC", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n <= 1) {
	return 0;
    }

/*     Determine the first row of specified block */
/*     and find out it is 1 by 1 or 2 by 2. */

    if (*ifst > 1) {
	if (t[*ifst + (*ifst - 1) * t_dim1] != 0.) {
	    --(*ifst);
	}
    }
    nbf = 1;
    if (*ifst < *n) {
	if (t[*ifst + 1 + *ifst * t_dim1] != 0.) {
	    nbf = 2;
	}
    }

/*     Determine the first row of the final block */
/*     and find out it is 1 by 1 or 2 by 2. */

    if (*ilst > 1) {
	if (t[*ilst + (*ilst - 1) * t_dim1] != 0.) {
	    --(*ilst);
	}
    }
    nbl = 1;
    if (*ilst < *n) {
	if (t[*ilst + 1 + *ilst * t_dim1] != 0.) {
	    nbl = 2;
	}
    }

    if (*ifst == *ilst) {
	return 0;
    }

    if (*ifst < *ilst) {

/*        Update ILST */

	if (nbf == 2 && nbl == 1) {
	    --(*ilst);
	}
	if (nbf == 1 && nbl == 2) {
	    ++(*ilst);
	}

	here = *ifst;

L10:

/*        Swap block with next one below */

	if (nbf == 1 || nbf == 2) {

/*           Current block either 1 by 1 or 2 by 2 */

	    nbnext = 1;
	    if (here + nbf + 1 <= *n) {
		if (t[here + nbf + 1 + (here + nbf) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &here, &
		    nbf, &nbnext, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    here += nbnext;

/*           Test if 2 by 2 block breaks into two 1 by 1 blocks */

	    if (nbf == 2) {
		if (t[here + 1 + here * t_dim1] == 0.) {
		    nbf = 3;
		}
	    }

	} else {

/*           Current block consists of two 1 by 1 blocks each of which */
/*           must be swapped individually */

	    nbnext = 1;
	    if (here + 3 <= *n) {
		if (t[here + 3 + (here + 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here + 1;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    c__1, &nbnext, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    if (nbnext == 1) {

/*              Swap two 1 by 1 blocks, no problems possible */

		dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			here, &c__1, &nbnext, &work[1], info);
		++here;
	    } else {

/*              Recompute NBNEXT in case 2 by 2 split */

		if (t[here + 2 + (here + 1) * t_dim1] == 0.) {
		    nbnext = 1;
		}
		if (nbnext == 2) {

/*                 2 by 2 Block did not split */

		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &nbnext, &work[1], info);
		    if (*info != 0) {
			*ilst = here;
			return 0;
		    }
		    here += 2;
		} else {

/*                 2 by 2 Block did split */

		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &c__1, &work[1], info);
		    i__1 = here + 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__1, &c__1, &work[1], info);
		    here += 2;
		}
	    }
	}
	if (here < *ilst) {
	    goto L10;
	}

    } else {

	here = *ifst;
L20:

/*        Swap block with next one above */

	if (nbf == 1 || nbf == 2) {

/*           Current block either 1 by 1 or 2 by 2 */

	    nbnext = 1;
	    if (here >= 3) {
		if (t[here - 1 + (here - 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here - nbnext;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    nbnext, &nbf, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    here -= nbnext;

/*           Test if 2 by 2 block breaks into two 1 by 1 blocks */

	    if (nbf == 2) {
		if (t[here + 1 + here * t_dim1] == 0.) {
		    nbf = 3;
		}
	    }

	} else {

/*           Current block consists of two 1 by 1 blocks each of which */
/*           must be swapped individually */

	    nbnext = 1;
	    if (here >= 3) {
		if (t[here - 1 + (here - 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here - nbnext;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    nbnext, &c__1, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    if (nbnext == 1) {

/*              Swap two 1 by 1 blocks, no problems possible */

		dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			here, &nbnext, &c__1, &work[1], info);
		--here;
	    } else {

/*              Recompute NBNEXT in case 2 by 2 split */

		if (t[here + (here - 1) * t_dim1] == 0.) {
		    nbnext = 1;
		}
		if (nbnext == 2) {

/*                 2 by 2 Block did not split */

		    i__1 = here - 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__2, &c__1, &work[1], info);
		    if (*info != 0) {
			*ilst = here;
			return 0;
		    }
		    here += -2;
		} else {

/*                 2 by 2 Block did split */

		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &c__1, &work[1], info);
		    i__1 = here - 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__1, &c__1, &work[1], info);
		    here += -2;
		}
	    }
	}
	if (here > *ilst) {
	    goto L20;
	}
    }
    *ilst = here;

    return 0;

/*     End of DTREXC */

} /* dtrexc_ */
