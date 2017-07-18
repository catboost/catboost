/* zla_herpvgrw.f -- translated by f2c (version 20061008).
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

doublereal zla_herpvgrw__(char *uplo, integer *n, integer *info, 
	doublecomplex *a, integer *lda, doublecomplex *af, integer *ldaf, 
	integer *ipiv, doublereal *work, ftnlen uplo_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, af_dim1, af_offset, i__1, i__2, i__3;
    doublereal ret_val, d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double d_imag(doublecomplex *);

    /* Local variables */
    integer i__, j, k, kp;
    doublereal tmp, amax, umax;
    extern logical lsame_(char *, char *);
    integer ncols;
    logical upper;
    doublereal rpvgrw;


/*     -- LAPACK routine (version 3.2.1)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- April 2009                                                   -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZLA_HERPVGRW computes the reciprocal pivot growth factor */
/*  norm(A)/norm(U). The "max absolute element" norm is used. If this is */
/*  much less than 1, the stability of the LU factorization of the */
/*  (equilibrated) matrix A could be poor. This also means that the */
/*  solution X, estimated condition numbers, and error bounds could be */
/*  unreliable. */

/*  Arguments */
/*  ========= */

/*     UPLO    (input) CHARACTER*1 */
/*       = 'U':  Upper triangle of A is stored; */
/*       = 'L':  Lower triangle of A is stored. */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     INFO    (input) INTEGER */
/*     The value of INFO returned from ZHETRF, .i.e., the pivot in */
/*     column INFO is exactly 0. */

/*     NCOLS   (input) INTEGER */
/*     The number of columns of the matrix A. NCOLS >= 0. */

/*     A       (input) COMPLEX*16 array, dimension (LDA,N) */
/*     On entry, the N-by-N matrix A. */

/*     LDA     (input) INTEGER */
/*     The leading dimension of the array A.  LDA >= max(1,N). */

/*     AF      (input) COMPLEX*16 array, dimension (LDAF,N) */
/*     The block diagonal matrix D and the multipliers used to */
/*     obtain the factor U or L as computed by ZHETRF. */

/*     LDAF    (input) INTEGER */
/*     The leading dimension of the array AF.  LDAF >= max(1,N). */

/*     IPIV    (input) INTEGER array, dimension (N) */
/*     Details of the interchanges and the block structure of D */
/*     as determined by ZHETRF. */

/*     WORK    (input) COMPLEX*16 array, dimension (2*N) */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    af_dim1 = *ldaf;
    af_offset = 1 + af_dim1;
    af -= af_offset;
    --ipiv;
    --work;

    /* Function Body */
    upper = lsame_("Upper", uplo);
    if (*info == 0) {
	if (upper) {
	    ncols = 1;
	} else {
	    ncols = *n;
	}
    } else {
	ncols = *info;
    }
    rpvgrw = 1.;
    i__1 = *n << 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = 0.;
    }

/*     Find the max magnitude entry of each column of A.  Compute the max */
/*     for all N columns so we can apply the pivot permutation while */
/*     looping below.  Assume a full factorization is the common case. */

    if (upper) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = j;
	    for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		i__3 = i__ + j * a_dim1;
		d__3 = (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[i__ 
			+ j * a_dim1]), abs(d__2)), d__4 = work[*n + i__];
		work[*n + i__] = max(d__3,d__4);
/* Computing MAX */
		i__3 = i__ + j * a_dim1;
		d__3 = (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[i__ 
			+ j * a_dim1]), abs(d__2)), d__4 = work[*n + j];
		work[*n + j] = max(d__3,d__4);
	    }
	}
    } else {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *n;
	    for (i__ = j; i__ <= i__2; ++i__) {
/* Computing MAX */
		i__3 = i__ + j * a_dim1;
		d__3 = (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[i__ 
			+ j * a_dim1]), abs(d__2)), d__4 = work[*n + i__];
		work[*n + i__] = max(d__3,d__4);
/* Computing MAX */
		i__3 = i__ + j * a_dim1;
		d__3 = (d__1 = a[i__3].r, abs(d__1)) + (d__2 = d_imag(&a[i__ 
			+ j * a_dim1]), abs(d__2)), d__4 = work[*n + j];
		work[*n + j] = max(d__3,d__4);
	    }
	}
    }

/*     Now find the max magnitude entry of each column of U or L.  Also */
/*     permute the magnitudes of A above so they're in the same order as */
/*     the factor. */

/*     The iteration orders and permutations were copied from zsytrs. */
/*     Calls to SSWAP would be severe overkill. */

    if (upper) {
	k = *n;
	while(k < ncols && k > 0) {
	    if (ipiv[k] > 0) {
/*              1x1 pivot */
		kp = ipiv[k];
		if (kp != k) {
		    tmp = work[*n + k];
		    work[*n + k] = work[*n + kp];
		    work[*n + kp] = tmp;
		}
		i__1 = k;
		for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		    i__2 = i__ + k * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + k * af_dim1]), abs(d__2)), d__4 = work[k]
			    ;
		    work[k] = max(d__3,d__4);
		}
		--k;
	    } else {
/*              2x2 pivot */
		kp = -ipiv[k];
		tmp = work[*n + k - 1];
		work[*n + k - 1] = work[*n + kp];
		work[*n + kp] = tmp;
		i__1 = k - 1;
		for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		    i__2 = i__ + k * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + k * af_dim1]), abs(d__2)), d__4 = work[k]
			    ;
		    work[k] = max(d__3,d__4);
/* Computing MAX */
		    i__2 = i__ + (k - 1) * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + (k - 1) * af_dim1]), abs(d__2)), d__4 = 
			    work[k - 1];
		    work[k - 1] = max(d__3,d__4);
		}
/* Computing MAX */
		i__1 = k + k * af_dim1;
		d__3 = (d__1 = af[i__1].r, abs(d__1)) + (d__2 = d_imag(&af[k 
			+ k * af_dim1]), abs(d__2)), d__4 = work[k];
		work[k] = max(d__3,d__4);
		k += -2;
	    }
	}
	k = ncols;
	while(k <= *n) {
	    if (ipiv[k] > 0) {
		kp = ipiv[k];
		if (kp != k) {
		    tmp = work[*n + k];
		    work[*n + k] = work[*n + kp];
		    work[*n + kp] = tmp;
		}
		++k;
	    } else {
		kp = -ipiv[k];
		tmp = work[*n + k];
		work[*n + k] = work[*n + kp];
		work[*n + kp] = tmp;
		k += 2;
	    }
	}
    } else {
	k = 1;
	while(k <= ncols) {
	    if (ipiv[k] > 0) {
/*              1x1 pivot */
		kp = ipiv[k];
		if (kp != k) {
		    tmp = work[*n + k];
		    work[*n + k] = work[*n + kp];
		    work[*n + kp] = tmp;
		}
		i__1 = *n;
		for (i__ = k; i__ <= i__1; ++i__) {
/* Computing MAX */
		    i__2 = i__ + k * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + k * af_dim1]), abs(d__2)), d__4 = work[k]
			    ;
		    work[k] = max(d__3,d__4);
		}
		++k;
	    } else {
/*              2x2 pivot */
		kp = -ipiv[k];
		tmp = work[*n + k + 1];
		work[*n + k + 1] = work[*n + kp];
		work[*n + kp] = tmp;
		i__1 = *n;
		for (i__ = k + 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		    i__2 = i__ + k * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + k * af_dim1]), abs(d__2)), d__4 = work[k]
			    ;
		    work[k] = max(d__3,d__4);
/* Computing MAX */
		    i__2 = i__ + (k + 1) * af_dim1;
		    d__3 = (d__1 = af[i__2].r, abs(d__1)) + (d__2 = d_imag(&
			    af[i__ + (k + 1) * af_dim1]), abs(d__2)), d__4 = 
			    work[k + 1];
		    work[k + 1] = max(d__3,d__4);
		}
/* Computing MAX */
		i__1 = k + k * af_dim1;
		d__3 = (d__1 = af[i__1].r, abs(d__1)) + (d__2 = d_imag(&af[k 
			+ k * af_dim1]), abs(d__2)), d__4 = work[k];
		work[k] = max(d__3,d__4);
		k += 2;
	    }
	}
	k = ncols;
	while(k >= 1) {
	    if (ipiv[k] > 0) {
		kp = ipiv[k];
		if (kp != k) {
		    tmp = work[*n + k];
		    work[*n + k] = work[*n + kp];
		    work[*n + kp] = tmp;
		}
		--k;
	    } else {
		kp = -ipiv[k];
		tmp = work[*n + k];
		work[*n + k] = work[*n + kp];
		work[*n + kp] = tmp;
		k += -2;
	    }
	}
    }

/*     Compute the *inverse* of the max element growth factor.  Dividing */
/*     by zero would imply the largest entry of the factor's column is */
/*     zero.  Than can happen when either the column of A is zero or */
/*     massive pivots made the factor underflow to zero.  Neither counts */
/*     as growth in itself, so simply ignore terms with zero */
/*     denominators. */

    if (upper) {
	i__1 = *n;
	for (i__ = ncols; i__ <= i__1; ++i__) {
	    umax = work[i__];
	    amax = work[*n + i__];
	    if (umax != 0.) {
/* Computing MIN */
		d__1 = amax / umax;
		rpvgrw = min(d__1,rpvgrw);
	    }
	}
    } else {
	i__1 = ncols;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    umax = work[i__];
	    amax = work[*n + i__];
	    if (umax != 0.) {
/* Computing MIN */
		d__1 = amax / umax;
		rpvgrw = min(d__1,rpvgrw);
	    }
	}
    }
    ret_val = rpvgrw;
    return ret_val;
} /* zla_herpvgrw__ */
