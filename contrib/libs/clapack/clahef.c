/* clahef.f -- translated by f2c (version 20061008).
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

static complex c_b1 = {1.f,0.f};
static integer c__1 = 1;

/* Subroutine */ int clahef_(char *uplo, integer *n, integer *nb, integer *kb, 
	 complex *a, integer *lda, integer *ipiv, complex *w, integer *ldw, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3, i__4, i__5;
    real r__1, r__2, r__3, r__4;
    complex q__1, q__2, q__3, q__4;

    /* Builtin functions */
    double sqrt(doublereal), r_imag(complex *);
    void r_cnjg(complex *, complex *), c_div(complex *, complex *, complex *);

    /* Local variables */
    integer j, k;
    real t, r1;
    complex d11, d21, d22;
    integer jb, jj, kk, jp, kp, kw, kkw, imax, jmax;
    real alpha;
    extern /* Subroutine */ int cgemm_(char *, char *, integer *, integer *, 
	    integer *, complex *, complex *, integer *, complex *, integer *, 
	    complex *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int cgemv_(char *, integer *, integer *, complex *
, complex *, integer *, complex *, integer *, complex *, complex *
, integer *), ccopy_(integer *, complex *, integer *, 
	    complex *, integer *), cswap_(integer *, complex *, integer *, 
	    complex *, integer *);
    integer kstep;
    real absakk;
    extern /* Subroutine */ int clacgv_(integer *, complex *, integer *);
    extern integer icamax_(integer *, complex *, integer *);
    extern /* Subroutine */ int csscal_(integer *, real *, complex *, integer 
	    *);
    real colmax, rowmax;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLAHEF computes a partial factorization of a complex Hermitian */
/*  matrix A using the Bunch-Kaufman diagonal pivoting method. The */
/*  partial factorization has the form: */

/*  A  =  ( I  U12 ) ( A11  0  ) (  I    0   )  if UPLO = 'U', or: */
/*        ( 0  U22 ) (  0   D  ) ( U12' U22' ) */

/*  A  =  ( L11  0 ) (  D   0  ) ( L11' L21' )  if UPLO = 'L' */
/*        ( L21  I ) (  0  A22 ) (  0    I   ) */

/*  where the order of D is at most NB. The actual order is returned in */
/*  the argument KB, and is either NB or NB-1, or N if N <= NB. */
/*  Note that U' denotes the conjugate transpose of U. */

/*  CLAHEF is an auxiliary routine called by CHETRF. It uses blocked code */
/*  (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or */
/*  A22 (if UPLO = 'L'). */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          Hermitian matrix A is stored: */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NB      (input) INTEGER */
/*          The maximum number of columns of the matrix A that should be */
/*          factored.  NB should be at least 2 to allow for 2-by-2 pivot */
/*          blocks. */

/*  KB      (output) INTEGER */
/*          The number of columns of A that were actually factored. */
/*          KB is either NB-1 or NB, or N if N <= NB. */

/*  A       (input/output) COMPLEX array, dimension (LDA,N) */
/*          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/*          n-by-n upper triangular part of A contains the upper */
/*          triangular part of the matrix A, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading n-by-n lower triangular part of A contains the lower */
/*          triangular part of the matrix A, and the strictly upper */
/*          triangular part of A is not referenced. */
/*          On exit, A contains details of the partial factorization. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  IPIV    (output) INTEGER array, dimension (N) */
/*          Details of the interchanges and the block structure of D. */
/*          If UPLO = 'U', only the last KB elements of IPIV are set; */
/*          if UPLO = 'L', only the first KB elements are set. */

/*          If IPIV(k) > 0, then rows and columns k and IPIV(k) were */
/*          interchanged and D(k,k) is a 1-by-1 diagonal block. */
/*          If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and */
/*          columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k) */
/*          is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) = */
/*          IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were */
/*          interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block. */

/*  W       (workspace) COMPLEX array, dimension (LDW,NB) */

/*  LDW     (input) INTEGER */
/*          The leading dimension of the array W.  LDW >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          > 0: if INFO = k, D(k,k) is exactly zero.  The factorization */
/*               has been completed, but the block diagonal matrix D is */
/*               exactly singular. */

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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;

    /* Function Body */
    *info = 0;

/*     Initialize ALPHA for use in choosing pivot block size. */

    alpha = (sqrt(17.f) + 1.f) / 8.f;

    if (lsame_(uplo, "U")) {

/*        Factorize the trailing columns of A using the upper triangle */
/*        of A and working backwards, and compute the matrix W = U12*D */
/*        for use in updating A11 (note that conjg(W) is actually stored) */

/*        K is the main loop index, decreasing from N in steps of 1 or 2 */

/*        KW is the column of W which corresponds to column K of A */

	k = *n;
L10:
	kw = *nb + k - *n;

/*        Exit from loop */

	if (k <= *n - *nb + 1 && *nb < *n || k < 1) {
	    goto L30;
	}

/*        Copy column K of A to column KW of W and update it */

	i__1 = k - 1;
	ccopy_(&i__1, &a[k * a_dim1 + 1], &c__1, &w[kw * w_dim1 + 1], &c__1);
	i__1 = k + kw * w_dim1;
	i__2 = k + k * a_dim1;
	r__1 = a[i__2].r;
	w[i__1].r = r__1, w[i__1].i = 0.f;
	if (k < *n) {
	    i__1 = *n - k;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemv_("No transpose", &k, &i__1, &q__1, &a[(k + 1) * a_dim1 + 1], 
		     lda, &w[k + (kw + 1) * w_dim1], ldw, &c_b1, &w[kw * 
		    w_dim1 + 1], &c__1);
	    i__1 = k + kw * w_dim1;
	    i__2 = k + kw * w_dim1;
	    r__1 = w[i__2].r;
	    w[i__1].r = r__1, w[i__1].i = 0.f;
	}

	kstep = 1;

/*        Determine rows and columns to be interchanged and whether */
/*        a 1-by-1 or 2-by-2 pivot block will be used */

	i__1 = k + kw * w_dim1;
	absakk = (r__1 = w[i__1].r, dabs(r__1));

/*        IMAX is the row-index of the largest off-diagonal element in */
/*        column K, and COLMAX is its absolute value */

	if (k > 1) {
	    i__1 = k - 1;
	    imax = icamax_(&i__1, &w[kw * w_dim1 + 1], &c__1);
	    i__1 = imax + kw * w_dim1;
	    colmax = (r__1 = w[i__1].r, dabs(r__1)) + (r__2 = r_imag(&w[imax 
		    + kw * w_dim1]), dabs(r__2));
	} else {
	    colmax = 0.f;
	}

	if (dmax(absakk,colmax) == 0.f) {

/*           Column K is zero: set INFO and continue */

	    if (*info == 0) {
		*info = k;
	    }
	    kp = k;
	    i__1 = k + k * a_dim1;
	    i__2 = k + k * a_dim1;
	    r__1 = a[i__2].r;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	} else {
	    if (absakk >= alpha * colmax) {

/*              no interchange, use 1-by-1 pivot block */

		kp = k;
	    } else {

/*              Copy column IMAX to column KW-1 of W and update it */

		i__1 = imax - 1;
		ccopy_(&i__1, &a[imax * a_dim1 + 1], &c__1, &w[(kw - 1) * 
			w_dim1 + 1], &c__1);
		i__1 = imax + (kw - 1) * w_dim1;
		i__2 = imax + imax * a_dim1;
		r__1 = a[i__2].r;
		w[i__1].r = r__1, w[i__1].i = 0.f;
		i__1 = k - imax;
		ccopy_(&i__1, &a[imax + (imax + 1) * a_dim1], lda, &w[imax + 
			1 + (kw - 1) * w_dim1], &c__1);
		i__1 = k - imax;
		clacgv_(&i__1, &w[imax + 1 + (kw - 1) * w_dim1], &c__1);
		if (k < *n) {
		    i__1 = *n - k;
		    q__1.r = -1.f, q__1.i = -0.f;
		    cgemv_("No transpose", &k, &i__1, &q__1, &a[(k + 1) * 
			    a_dim1 + 1], lda, &w[imax + (kw + 1) * w_dim1], 
			    ldw, &c_b1, &w[(kw - 1) * w_dim1 + 1], &c__1);
		    i__1 = imax + (kw - 1) * w_dim1;
		    i__2 = imax + (kw - 1) * w_dim1;
		    r__1 = w[i__2].r;
		    w[i__1].r = r__1, w[i__1].i = 0.f;
		}

/*              JMAX is the column-index of the largest off-diagonal */
/*              element in row IMAX, and ROWMAX is its absolute value */

		i__1 = k - imax;
		jmax = imax + icamax_(&i__1, &w[imax + 1 + (kw - 1) * w_dim1], 
			 &c__1);
		i__1 = jmax + (kw - 1) * w_dim1;
		rowmax = (r__1 = w[i__1].r, dabs(r__1)) + (r__2 = r_imag(&w[
			jmax + (kw - 1) * w_dim1]), dabs(r__2));
		if (imax > 1) {
		    i__1 = imax - 1;
		    jmax = icamax_(&i__1, &w[(kw - 1) * w_dim1 + 1], &c__1);
/* Computing MAX */
		    i__1 = jmax + (kw - 1) * w_dim1;
		    r__3 = rowmax, r__4 = (r__1 = w[i__1].r, dabs(r__1)) + (
			    r__2 = r_imag(&w[jmax + (kw - 1) * w_dim1]), dabs(
			    r__2));
		    rowmax = dmax(r__3,r__4);
		}

		if (absakk >= alpha * colmax * (colmax / rowmax)) {

/*                 no interchange, use 1-by-1 pivot block */

		    kp = k;
		} else /* if(complicated condition) */ {
		    i__1 = imax + (kw - 1) * w_dim1;
		    if ((r__1 = w[i__1].r, dabs(r__1)) >= alpha * rowmax) {

/*                 interchange rows and columns K and IMAX, use 1-by-1 */
/*                 pivot block */

			kp = imax;

/*                 copy column KW-1 of W to column KW */

			ccopy_(&k, &w[(kw - 1) * w_dim1 + 1], &c__1, &w[kw * 
				w_dim1 + 1], &c__1);
		    } else {

/*                 interchange rows and columns K-1 and IMAX, use 2-by-2 */
/*                 pivot block */

			kp = imax;
			kstep = 2;
		    }
		}
	    }

	    kk = k - kstep + 1;
	    kkw = *nb + kk - *n;

/*           Updated column KP is already stored in column KKW of W */

	    if (kp != kk) {

/*              Copy non-updated column KK to column KP */

		i__1 = kp + kp * a_dim1;
		i__2 = kk + kk * a_dim1;
		r__1 = a[i__2].r;
		a[i__1].r = r__1, a[i__1].i = 0.f;
		i__1 = kk - 1 - kp;
		ccopy_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + (kp + 
			1) * a_dim1], lda);
		i__1 = kk - 1 - kp;
		clacgv_(&i__1, &a[kp + (kp + 1) * a_dim1], lda);
		i__1 = kp - 1;
		ccopy_(&i__1, &a[kk * a_dim1 + 1], &c__1, &a[kp * a_dim1 + 1], 
			 &c__1);

/*              Interchange rows KK and KP in last KK columns of A and W */

		if (kk < *n) {
		    i__1 = *n - kk;
		    cswap_(&i__1, &a[kk + (kk + 1) * a_dim1], lda, &a[kp + (
			    kk + 1) * a_dim1], lda);
		}
		i__1 = *n - kk + 1;
		cswap_(&i__1, &w[kk + kkw * w_dim1], ldw, &w[kp + kkw * 
			w_dim1], ldw);
	    }

	    if (kstep == 1) {

/*              1-by-1 pivot block D(k): column KW of W now holds */

/*              W(k) = U(k)*D(k) */

/*              where U(k) is the k-th column of U */

/*              Store U(k) in column k of A */

		ccopy_(&k, &w[kw * w_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &
			c__1);
		i__1 = k + k * a_dim1;
		r1 = 1.f / a[i__1].r;
		i__1 = k - 1;
		csscal_(&i__1, &r1, &a[k * a_dim1 + 1], &c__1);

/*              Conjugate W(k) */

		i__1 = k - 1;
		clacgv_(&i__1, &w[kw * w_dim1 + 1], &c__1);
	    } else {

/*              2-by-2 pivot block D(k): columns KW and KW-1 of W now */
/*              hold */

/*              ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k) */

/*              where U(k) and U(k-1) are the k-th and (k-1)-th columns */
/*              of U */

		if (k > 2) {

/*                 Store U(k) and U(k-1) in columns k and k-1 of A */

		    i__1 = k - 1 + kw * w_dim1;
		    d21.r = w[i__1].r, d21.i = w[i__1].i;
		    r_cnjg(&q__2, &d21);
		    c_div(&q__1, &w[k + kw * w_dim1], &q__2);
		    d11.r = q__1.r, d11.i = q__1.i;
		    c_div(&q__1, &w[k - 1 + (kw - 1) * w_dim1], &d21);
		    d22.r = q__1.r, d22.i = q__1.i;
		    q__1.r = d11.r * d22.r - d11.i * d22.i, q__1.i = d11.r * 
			    d22.i + d11.i * d22.r;
		    t = 1.f / (q__1.r - 1.f);
		    q__2.r = t, q__2.i = 0.f;
		    c_div(&q__1, &q__2, &d21);
		    d21.r = q__1.r, d21.i = q__1.i;
		    i__1 = k - 2;
		    for (j = 1; j <= i__1; ++j) {
			i__2 = j + (k - 1) * a_dim1;
			i__3 = j + (kw - 1) * w_dim1;
			q__3.r = d11.r * w[i__3].r - d11.i * w[i__3].i, 
				q__3.i = d11.r * w[i__3].i + d11.i * w[i__3]
				.r;
			i__4 = j + kw * w_dim1;
			q__2.r = q__3.r - w[i__4].r, q__2.i = q__3.i - w[i__4]
				.i;
			q__1.r = d21.r * q__2.r - d21.i * q__2.i, q__1.i = 
				d21.r * q__2.i + d21.i * q__2.r;
			a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			i__2 = j + k * a_dim1;
			r_cnjg(&q__2, &d21);
			i__3 = j + kw * w_dim1;
			q__4.r = d22.r * w[i__3].r - d22.i * w[i__3].i, 
				q__4.i = d22.r * w[i__3].i + d22.i * w[i__3]
				.r;
			i__4 = j + (kw - 1) * w_dim1;
			q__3.r = q__4.r - w[i__4].r, q__3.i = q__4.i - w[i__4]
				.i;
			q__1.r = q__2.r * q__3.r - q__2.i * q__3.i, q__1.i = 
				q__2.r * q__3.i + q__2.i * q__3.r;
			a[i__2].r = q__1.r, a[i__2].i = q__1.i;
/* L20: */
		    }
		}

/*              Copy D(k) to A */

		i__1 = k - 1 + (k - 1) * a_dim1;
		i__2 = k - 1 + (kw - 1) * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
		i__1 = k - 1 + k * a_dim1;
		i__2 = k - 1 + kw * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
		i__1 = k + k * a_dim1;
		i__2 = k + kw * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;

/*              Conjugate W(k) and W(k-1) */

		i__1 = k - 1;
		clacgv_(&i__1, &w[kw * w_dim1 + 1], &c__1);
		i__1 = k - 2;
		clacgv_(&i__1, &w[(kw - 1) * w_dim1 + 1], &c__1);
	    }
	}

/*        Store details of the interchanges in IPIV */

	if (kstep == 1) {
	    ipiv[k] = kp;
	} else {
	    ipiv[k] = -kp;
	    ipiv[k - 1] = -kp;
	}

/*        Decrease K and return to the start of the main loop */

	k -= kstep;
	goto L10;

L30:

/*        Update the upper triangle of A11 (= A(1:k,1:k)) as */

/*        A11 := A11 - U12*D*U12' = A11 - U12*W' */

/*        computing blocks of NB columns at a time (note that conjg(W) is */
/*        actually stored) */

	i__1 = -(*nb);
	for (j = (k - 1) / *nb * *nb + 1; i__1 < 0 ? j >= 1 : j <= 1; j += 
		i__1) {
/* Computing MIN */
	    i__2 = *nb, i__3 = k - j + 1;
	    jb = min(i__2,i__3);

/*           Update the upper triangle of the diagonal block */

	    i__2 = j + jb - 1;
	    for (jj = j; jj <= i__2; ++jj) {
		i__3 = jj + jj * a_dim1;
		i__4 = jj + jj * a_dim1;
		r__1 = a[i__4].r;
		a[i__3].r = r__1, a[i__3].i = 0.f;
		i__3 = jj - j + 1;
		i__4 = *n - k;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__3, &i__4, &q__1, &a[j + (k + 1) * 
			a_dim1], lda, &w[jj + (kw + 1) * w_dim1], ldw, &c_b1, 
			&a[j + jj * a_dim1], &c__1);
		i__3 = jj + jj * a_dim1;
		i__4 = jj + jj * a_dim1;
		r__1 = a[i__4].r;
		a[i__3].r = r__1, a[i__3].i = 0.f;
/* L40: */
	    }

/*           Update the rectangular superdiagonal block */

	    i__2 = j - 1;
	    i__3 = *n - k;
	    q__1.r = -1.f, q__1.i = -0.f;
	    cgemm_("No transpose", "Transpose", &i__2, &jb, &i__3, &q__1, &a[(
		    k + 1) * a_dim1 + 1], lda, &w[j + (kw + 1) * w_dim1], ldw, 
		     &c_b1, &a[j * a_dim1 + 1], lda);
/* L50: */
	}

/*        Put U12 in standard form by partially undoing the interchanges */
/*        in columns k+1:n */

	j = k + 1;
L60:
	jj = j;
	jp = ipiv[j];
	if (jp < 0) {
	    jp = -jp;
	    ++j;
	}
	++j;
	if (jp != jj && j <= *n) {
	    i__1 = *n - j + 1;
	    cswap_(&i__1, &a[jp + j * a_dim1], lda, &a[jj + j * a_dim1], lda);
	}
	if (j <= *n) {
	    goto L60;
	}

/*        Set KB to the number of columns factorized */

	*kb = *n - k;

    } else {

/*        Factorize the leading columns of A using the lower triangle */
/*        of A and working forwards, and compute the matrix W = L21*D */
/*        for use in updating A22 (note that conjg(W) is actually stored) */

/*        K is the main loop index, increasing from 1 in steps of 1 or 2 */

	k = 1;
L70:

/*        Exit from loop */

	if (k >= *nb && *nb < *n || k > *n) {
	    goto L90;
	}

/*        Copy column K of A to column K of W and update it */

	i__1 = k + k * w_dim1;
	i__2 = k + k * a_dim1;
	r__1 = a[i__2].r;
	w[i__1].r = r__1, w[i__1].i = 0.f;
	if (k < *n) {
	    i__1 = *n - k;
	    ccopy_(&i__1, &a[k + 1 + k * a_dim1], &c__1, &w[k + 1 + k * 
		    w_dim1], &c__1);
	}
	i__1 = *n - k + 1;
	i__2 = k - 1;
	q__1.r = -1.f, q__1.i = -0.f;
	cgemv_("No transpose", &i__1, &i__2, &q__1, &a[k + a_dim1], lda, &w[k 
		+ w_dim1], ldw, &c_b1, &w[k + k * w_dim1], &c__1);
	i__1 = k + k * w_dim1;
	i__2 = k + k * w_dim1;
	r__1 = w[i__2].r;
	w[i__1].r = r__1, w[i__1].i = 0.f;

	kstep = 1;

/*        Determine rows and columns to be interchanged and whether */
/*        a 1-by-1 or 2-by-2 pivot block will be used */

	i__1 = k + k * w_dim1;
	absakk = (r__1 = w[i__1].r, dabs(r__1));

/*        IMAX is the row-index of the largest off-diagonal element in */
/*        column K, and COLMAX is its absolute value */

	if (k < *n) {
	    i__1 = *n - k;
	    imax = k + icamax_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
	    i__1 = imax + k * w_dim1;
	    colmax = (r__1 = w[i__1].r, dabs(r__1)) + (r__2 = r_imag(&w[imax 
		    + k * w_dim1]), dabs(r__2));
	} else {
	    colmax = 0.f;
	}

	if (dmax(absakk,colmax) == 0.f) {

/*           Column K is zero: set INFO and continue */

	    if (*info == 0) {
		*info = k;
	    }
	    kp = k;
	    i__1 = k + k * a_dim1;
	    i__2 = k + k * a_dim1;
	    r__1 = a[i__2].r;
	    a[i__1].r = r__1, a[i__1].i = 0.f;
	} else {
	    if (absakk >= alpha * colmax) {

/*              no interchange, use 1-by-1 pivot block */

		kp = k;
	    } else {

/*              Copy column IMAX to column K+1 of W and update it */

		i__1 = imax - k;
		ccopy_(&i__1, &a[imax + k * a_dim1], lda, &w[k + (k + 1) * 
			w_dim1], &c__1);
		i__1 = imax - k;
		clacgv_(&i__1, &w[k + (k + 1) * w_dim1], &c__1);
		i__1 = imax + (k + 1) * w_dim1;
		i__2 = imax + imax * a_dim1;
		r__1 = a[i__2].r;
		w[i__1].r = r__1, w[i__1].i = 0.f;
		if (imax < *n) {
		    i__1 = *n - imax;
		    ccopy_(&i__1, &a[imax + 1 + imax * a_dim1], &c__1, &w[
			    imax + 1 + (k + 1) * w_dim1], &c__1);
		}
		i__1 = *n - k + 1;
		i__2 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__1, &i__2, &q__1, &a[k + a_dim1], 
			lda, &w[imax + w_dim1], ldw, &c_b1, &w[k + (k + 1) * 
			w_dim1], &c__1);
		i__1 = imax + (k + 1) * w_dim1;
		i__2 = imax + (k + 1) * w_dim1;
		r__1 = w[i__2].r;
		w[i__1].r = r__1, w[i__1].i = 0.f;

/*              JMAX is the column-index of the largest off-diagonal */
/*              element in row IMAX, and ROWMAX is its absolute value */

		i__1 = imax - k;
		jmax = k - 1 + icamax_(&i__1, &w[k + (k + 1) * w_dim1], &c__1)
			;
		i__1 = jmax + (k + 1) * w_dim1;
		rowmax = (r__1 = w[i__1].r, dabs(r__1)) + (r__2 = r_imag(&w[
			jmax + (k + 1) * w_dim1]), dabs(r__2));
		if (imax < *n) {
		    i__1 = *n - imax;
		    jmax = imax + icamax_(&i__1, &w[imax + 1 + (k + 1) * 
			    w_dim1], &c__1);
/* Computing MAX */
		    i__1 = jmax + (k + 1) * w_dim1;
		    r__3 = rowmax, r__4 = (r__1 = w[i__1].r, dabs(r__1)) + (
			    r__2 = r_imag(&w[jmax + (k + 1) * w_dim1]), dabs(
			    r__2));
		    rowmax = dmax(r__3,r__4);
		}

		if (absakk >= alpha * colmax * (colmax / rowmax)) {

/*                 no interchange, use 1-by-1 pivot block */

		    kp = k;
		} else /* if(complicated condition) */ {
		    i__1 = imax + (k + 1) * w_dim1;
		    if ((r__1 = w[i__1].r, dabs(r__1)) >= alpha * rowmax) {

/*                 interchange rows and columns K and IMAX, use 1-by-1 */
/*                 pivot block */

			kp = imax;

/*                 copy column K+1 of W to column K */

			i__1 = *n - k + 1;
			ccopy_(&i__1, &w[k + (k + 1) * w_dim1], &c__1, &w[k + 
				k * w_dim1], &c__1);
		    } else {

/*                 interchange rows and columns K+1 and IMAX, use 2-by-2 */
/*                 pivot block */

			kp = imax;
			kstep = 2;
		    }
		}
	    }

	    kk = k + kstep - 1;

/*           Updated column KP is already stored in column KK of W */

	    if (kp != kk) {

/*              Copy non-updated column KK to column KP */

		i__1 = kp + kp * a_dim1;
		i__2 = kk + kk * a_dim1;
		r__1 = a[i__2].r;
		a[i__1].r = r__1, a[i__1].i = 0.f;
		i__1 = kp - kk - 1;
		ccopy_(&i__1, &a[kk + 1 + kk * a_dim1], &c__1, &a[kp + (kk + 
			1) * a_dim1], lda);
		i__1 = kp - kk - 1;
		clacgv_(&i__1, &a[kp + (kk + 1) * a_dim1], lda);
		if (kp < *n) {
		    i__1 = *n - kp;
		    ccopy_(&i__1, &a[kp + 1 + kk * a_dim1], &c__1, &a[kp + 1 
			    + kp * a_dim1], &c__1);
		}

/*              Interchange rows KK and KP in first KK columns of A and W */

		i__1 = kk - 1;
		cswap_(&i__1, &a[kk + a_dim1], lda, &a[kp + a_dim1], lda);
		cswap_(&kk, &w[kk + w_dim1], ldw, &w[kp + w_dim1], ldw);
	    }

	    if (kstep == 1) {

/*              1-by-1 pivot block D(k): column k of W now holds */

/*              W(k) = L(k)*D(k) */

/*              where L(k) is the k-th column of L */

/*              Store L(k) in column k of A */

		i__1 = *n - k + 1;
		ccopy_(&i__1, &w[k + k * w_dim1], &c__1, &a[k + k * a_dim1], &
			c__1);
		if (k < *n) {
		    i__1 = k + k * a_dim1;
		    r1 = 1.f / a[i__1].r;
		    i__1 = *n - k;
		    csscal_(&i__1, &r1, &a[k + 1 + k * a_dim1], &c__1);

/*                 Conjugate W(k) */

		    i__1 = *n - k;
		    clacgv_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
		}
	    } else {

/*              2-by-2 pivot block D(k): columns k and k+1 of W now hold */

/*              ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k) */

/*              where L(k) and L(k+1) are the k-th and (k+1)-th columns */
/*              of L */

		if (k < *n - 1) {

/*                 Store L(k) and L(k+1) in columns k and k+1 of A */

		    i__1 = k + 1 + k * w_dim1;
		    d21.r = w[i__1].r, d21.i = w[i__1].i;
		    c_div(&q__1, &w[k + 1 + (k + 1) * w_dim1], &d21);
		    d11.r = q__1.r, d11.i = q__1.i;
		    r_cnjg(&q__2, &d21);
		    c_div(&q__1, &w[k + k * w_dim1], &q__2);
		    d22.r = q__1.r, d22.i = q__1.i;
		    q__1.r = d11.r * d22.r - d11.i * d22.i, q__1.i = d11.r * 
			    d22.i + d11.i * d22.r;
		    t = 1.f / (q__1.r - 1.f);
		    q__2.r = t, q__2.i = 0.f;
		    c_div(&q__1, &q__2, &d21);
		    d21.r = q__1.r, d21.i = q__1.i;
		    i__1 = *n;
		    for (j = k + 2; j <= i__1; ++j) {
			i__2 = j + k * a_dim1;
			r_cnjg(&q__2, &d21);
			i__3 = j + k * w_dim1;
			q__4.r = d11.r * w[i__3].r - d11.i * w[i__3].i, 
				q__4.i = d11.r * w[i__3].i + d11.i * w[i__3]
				.r;
			i__4 = j + (k + 1) * w_dim1;
			q__3.r = q__4.r - w[i__4].r, q__3.i = q__4.i - w[i__4]
				.i;
			q__1.r = q__2.r * q__3.r - q__2.i * q__3.i, q__1.i = 
				q__2.r * q__3.i + q__2.i * q__3.r;
			a[i__2].r = q__1.r, a[i__2].i = q__1.i;
			i__2 = j + (k + 1) * a_dim1;
			i__3 = j + (k + 1) * w_dim1;
			q__3.r = d22.r * w[i__3].r - d22.i * w[i__3].i, 
				q__3.i = d22.r * w[i__3].i + d22.i * w[i__3]
				.r;
			i__4 = j + k * w_dim1;
			q__2.r = q__3.r - w[i__4].r, q__2.i = q__3.i - w[i__4]
				.i;
			q__1.r = d21.r * q__2.r - d21.i * q__2.i, q__1.i = 
				d21.r * q__2.i + d21.i * q__2.r;
			a[i__2].r = q__1.r, a[i__2].i = q__1.i;
/* L80: */
		    }
		}

/*              Copy D(k) to A */

		i__1 = k + k * a_dim1;
		i__2 = k + k * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
		i__1 = k + 1 + k * a_dim1;
		i__2 = k + 1 + k * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;
		i__1 = k + 1 + (k + 1) * a_dim1;
		i__2 = k + 1 + (k + 1) * w_dim1;
		a[i__1].r = w[i__2].r, a[i__1].i = w[i__2].i;

/*              Conjugate W(k) and W(k+1) */

		i__1 = *n - k;
		clacgv_(&i__1, &w[k + 1 + k * w_dim1], &c__1);
		i__1 = *n - k - 1;
		clacgv_(&i__1, &w[k + 2 + (k + 1) * w_dim1], &c__1);
	    }
	}

/*        Store details of the interchanges in IPIV */

	if (kstep == 1) {
	    ipiv[k] = kp;
	} else {
	    ipiv[k] = -kp;
	    ipiv[k + 1] = -kp;
	}

/*        Increase K and return to the start of the main loop */

	k += kstep;
	goto L70;

L90:

/*        Update the lower triangle of A22 (= A(k:n,k:n)) as */

/*        A22 := A22 - L21*D*L21' = A22 - L21*W' */

/*        computing blocks of NB columns at a time (note that conjg(W) is */
/*        actually stored) */

	i__1 = *n;
	i__2 = *nb;
	for (j = k; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
/* Computing MIN */
	    i__3 = *nb, i__4 = *n - j + 1;
	    jb = min(i__3,i__4);

/*           Update the lower triangle of the diagonal block */

	    i__3 = j + jb - 1;
	    for (jj = j; jj <= i__3; ++jj) {
		i__4 = jj + jj * a_dim1;
		i__5 = jj + jj * a_dim1;
		r__1 = a[i__5].r;
		a[i__4].r = r__1, a[i__4].i = 0.f;
		i__4 = j + jb - jj;
		i__5 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemv_("No transpose", &i__4, &i__5, &q__1, &a[jj + a_dim1], 
			lda, &w[jj + w_dim1], ldw, &c_b1, &a[jj + jj * a_dim1]
, &c__1);
		i__4 = jj + jj * a_dim1;
		i__5 = jj + jj * a_dim1;
		r__1 = a[i__5].r;
		a[i__4].r = r__1, a[i__4].i = 0.f;
/* L100: */
	    }

/*           Update the rectangular subdiagonal block */

	    if (j + jb <= *n) {
		i__3 = *n - j - jb + 1;
		i__4 = k - 1;
		q__1.r = -1.f, q__1.i = -0.f;
		cgemm_("No transpose", "Transpose", &i__3, &jb, &i__4, &q__1, 
			&a[j + jb + a_dim1], lda, &w[j + w_dim1], ldw, &c_b1, 
			&a[j + jb + j * a_dim1], lda);
	    }
/* L110: */
	}

/*        Put L21 in standard form by partially undoing the interchanges */
/*        in columns 1:k-1 */

	j = k - 1;
L120:
	jj = j;
	jp = ipiv[j];
	if (jp < 0) {
	    jp = -jp;
	    --j;
	}
	--j;
	if (jp != jj && j >= 1) {
	    cswap_(&j, &a[jp + a_dim1], lda, &a[jj + a_dim1], lda);
	}
	if (j >= 1) {
	    goto L120;
	}

/*        Set KB to the number of columns factorized */

	*kb = k - 1;

    }
    return 0;

/*     End of CLAHEF */

} /* clahef_ */
