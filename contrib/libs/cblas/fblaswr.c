#include "f2c.h"
#include "fblaswr.h"

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */

doublereal 
f2c_sdot(integer* N, 
         real* X, integer* incX, 
         real* Y, integer* incY)
{
    return sdot_(N, X, incX, Y, incY);
}

doublereal 
f2c_ddot(integer* N, 
         doublereal* X, integer* incX, 
         doublereal* Y, integer* incY)
{
    return ddot_(N, X, incX, Y, incY);
}


/*
 * Functions having prefixes Z and C only
 */

void
f2c_cdotu(complex* retval,
          integer* N, 
          complex* X, integer* incX, 
          complex* Y, integer* incY)
{
    cdotu_(retval, N, X, incX, Y, incY);
}

void
f2c_cdotc(complex* retval,
          integer* N, 
          complex* X, integer* incX, 
          complex* Y, integer* incY)
{
    cdotc_(retval, N, X, incX, Y, incY);
}

void
f2c_zdotu(doublecomplex* retval,
          integer* N, 
          doublecomplex* X, integer* incX, 
          doublecomplex* Y, integer* incY)
{
    zdotu_(retval, N, X, incX, Y, incY);
}

void
f2c_zdotc(doublecomplex* retval,
          integer* N, 
          doublecomplex* X, integer* incX, 
          doublecomplex* Y, integer* incY)
{
    zdotc_(retval, N, X, incX, Y, incY);
}


/*
 * Functions having prefixes S D SC DZ
 */

doublereal 
f2c_snrm2(integer* N, 
          real* X, integer* incX)
{
    return snrm2_(N, X, incX);
}

doublereal
f2c_sasum(integer* N, 
          real* X, integer* incX)
{
    return sasum_(N, X, incX);
}

doublereal 
f2c_dnrm2(integer* N, 
          doublereal* X, integer* incX)
{
    return dnrm2_(N, X, incX);
}

doublereal
f2c_dasum(integer* N, 
          doublereal* X, integer* incX)
{
    return dasum_(N, X, incX);
}

doublereal 
f2c_scnrm2(integer* N, 
           complex* X, integer* incX)
{
    return scnrm2_(N, X, incX);
}

doublereal
f2c_scasum(integer* N, 
           complex* X, integer* incX)
{
    return scasum_(N, X, incX);
}

doublereal 
f2c_dznrm2(integer* N, 
           doublecomplex* X, integer* incX)
{
    return dznrm2_(N, X, incX);
}

doublereal
f2c_dzasum(integer* N, 
           doublecomplex* X, integer* incX)
{
    return dzasum_(N, X, incX);
}


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
integer
f2c_isamax(integer* N,
           real* X, integer* incX)
{
    return isamax_(N, X, incX);
}

integer
f2c_idamax(integer* N,
           doublereal* X, integer* incX)
{
    return idamax_(N, X, incX);
}

integer
f2c_icamax(integer* N,
           complex* X, integer* incX)
{
    return icamax_(N, X, incX);
}

integer
f2c_izamax(integer* N,
           doublecomplex* X, integer* incX)
{
    return izamax_(N, X, incX);
}

/*
 * ===========================================================================
 * Prototypes for level 0 BLAS routines
 * ===========================================================================
 */
int
f2c_srotg(real* a,
	      real* b,
		  real* c,
		  real* s)
{
    srotg_(a, b, c, s);
    return 0;
}

int
f2c_crotg(complex* CA,
          complex* CB,
          complex* C,
          real* S)
{
    crotg_(CA, CB, C, S);
    return 0;
}

int
f2c_drotg(doublereal* a,
		  doublereal* b,
		  doublereal* c,
		  doublereal* s)
{
    drotg_(a, b, c, s);
    return 0;
}

int
f2c_zrotg(doublecomplex* CA,
          doublecomplex* CB,
          doublecomplex* C,
          doublereal* S)
{
    zrotg_(CA, CB, C, S);
    return 0;
}
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */

int
f2c_sswap(integer* N,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    sswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_scopy(integer* N,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    scopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_saxpy(integer* N,
          real* alpha,
          real* X, integer* incX,
          real* Y, integer* incY)
{
    saxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_dswap(integer* N,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    dswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_dcopy(integer* N,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    dcopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_daxpy(integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY)
{
    daxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_cswap(integer* N,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    cswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_ccopy(integer* N,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    ccopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_caxpy(integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY)
{
    caxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}

int
f2c_zswap(integer* N,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    zswap_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_zcopy(integer* N,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    zcopy_(N, X, incX, Y, incY);
    return 0;
}

int
f2c_zaxpy(integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY)
{
    zaxpy_(N, alpha, X, incX, Y, incY);
    return 0;
}


/*
 * Routines with S and D prefix only
 */

int
f2c_srot(integer* N,
         real* X, integer* incX,
         real* Y, integer* incY,
         real* c, real* s)
{
    srot_(N, X, incX, Y, incY, c, s);
    return 0;
}

int
f2c_drot(integer* N,
         doublereal* X, integer* incX,
         doublereal* Y, integer* incY,
         doublereal* c, doublereal* s)
{
    drot_(N, X, incX, Y, incY, c, s);
    return 0;
}


/*
 * Routines with S D C Z CS and ZD prefixes
 */

int
f2c_sscal(integer* N,
          real* alpha,
          real* X, integer* incX)
{
    sscal_(N, alpha, X, incX);
    return 0;
}

int
f2c_dscal(integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX)
{
    dscal_(N, alpha, X, incX);
    return 0;
}

int
f2c_cscal(integer* N,
          complex* alpha,
          complex* X, integer* incX)
{
    cscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_zscal(integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX)
{
    zscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_csscal(integer* N,
           real* alpha,
           complex* X, integer* incX)
{
    csscal_(N, alpha, X, incX);
    return 0;
}


int
f2c_zdscal(integer* N,
           doublereal* alpha,
           doublecomplex* X, integer* incX)
{
    zdscal_(N, alpha, X, incX);
    return 0;
}



/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
int
f2c_sgemv(char* trans, integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    sgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_sgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          real *alpha, 
          real *A, integer *lda, 
          real *X, integer *incX, 
          real *beta, 
          real *Y, integer *incY)
{
    sgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_strmv(char* uplo, char *trans, char* diag, integer *N,  
          real *A, integer *lda, 
          real *X, integer *incX)
{
    strmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_stbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          real* A, integer* lda,
          real* X, integer* incX)
{
    stbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_stpmv(char* uplo, char* trans, char* diag, integer* N, 
          real* Ap, 
          real* X, integer* incX)
{
    stpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_strsv(char* uplo, char* trans, char* diag, integer* N,
          real* A, integer* lda,
          real* X, integer* incX)
{
    strsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_stbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          real* A, integer* lda, 
          real* X, integer* incX)
{
    stbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_stpsv(char* uplo, char* trans, char* diag, integer* N, 
          real* Ap, 
          real* X, integer* incX)
{
    stpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_dgemv(char* trans, integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    dgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          doublereal *alpha, 
          doublereal *A, integer *lda, 
          doublereal *X, integer *incX, 
          doublereal *beta, 
          doublereal *Y, integer *incY)
{
    dgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dtrmv(char* uplo, char *trans, char* diag, integer *N,  
          doublereal *A, integer *lda, 
          doublereal *X, integer *incX)
{
    dtrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_dtbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX)
{
    dtbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_dtpmv(char* uplo, char* trans, char* diag, integer* N, 
          doublereal* Ap, 
          doublereal* X, integer* incX)
{
    dtpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_dtrsv(char* uplo, char* trans, char* diag, integer* N,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX)
{
    dtrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_dtbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublereal* A, integer* lda, 
          doublereal* X, integer* incX)
{
    dtbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_dtpsv(char* uplo, char* trans, char* diag, integer* N, 
          doublereal* Ap, 
          doublereal* X, integer* incX)
{
    dtpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_cgemv(char* trans, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    cgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_cgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          complex *alpha, 
          complex *A, integer *lda, 
          complex *X, integer *incX, 
          complex *beta, 
          complex *Y, integer *incY)
{
    cgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ctrmv(char* uplo, char *trans, char* diag, integer *N,  
          complex *A, integer *lda, 
          complex *X, integer *incX)
{
    ctrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ctbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          complex* A, integer* lda,
          complex* X, integer* incX)
{
    ctbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ctpmv(char* uplo, char* trans, char* diag, integer* N, 
          complex* Ap, 
          complex* X, integer* incX)
{
    ctpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_ctrsv(char* uplo, char* trans, char* diag, integer* N,
          complex* A, integer* lda,
          complex* X, integer* incX)
{
    ctrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ctbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          complex* A, integer* lda, 
          complex* X, integer* incX)
{
    ctbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ctpsv(char* uplo, char* trans, char* diag, integer* N, 
          complex* Ap, 
          complex* X, integer* incX)
{
    ctpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 



int
f2c_zgemv(char* trans, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    zgemv_(trans, M, N,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_zgbmv(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
          doublecomplex *alpha, 
          doublecomplex *A, integer *lda, 
          doublecomplex *X, integer *incX, 
          doublecomplex *beta, 
          doublecomplex *Y, integer *incY)
{
    zgbmv_(trans, M, N, KL, KU,
           alpha, A, lda, X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ztrmv(char* uplo, char *trans, char* diag, integer *N,  
          doublecomplex *A, integer *lda, 
          doublecomplex *X, integer *incX)
{
    ztrmv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ztbmv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX)
{
    ztbmv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ztpmv(char* uplo, char* trans, char* diag, integer* N, 
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX)
{
    ztpmv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
}

int
f2c_ztrsv(char* uplo, char* trans, char* diag, integer* N,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX)
{
    ztrsv_(uplo, trans, diag,
           N, A, lda, X, incX);
    return 0;
}

int
f2c_ztbsv(char* uplo, char* trans, char* diag, integer* N, integer* K,
          doublecomplex* A, integer* lda, 
          doublecomplex* X, integer* incX)
{
    ztbsv_(uplo, trans, diag,
           N, K, A, lda, X, incX);
    return 0;
}

int
f2c_ztpsv(char* uplo, char* trans, char* diag, integer* N, 
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX)
{
    ztpsv_(uplo, trans, diag,
           N, Ap, X, incX);
    return 0;
} 


/*
 * Routines with S and D prefixes only
 */

int
f2c_ssymv(char* uplo, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    ssymv_(uplo, N, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_ssbmv(char* uplo, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    ssbmv_(uplo, N, K, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_sspmv(char* uplo, integer* N,
          real* alpha,
          real* Ap,
          real* X, integer* incX,
          real* beta,
          real* Y, integer* incY)
{
    sspmv_(uplo, N, alpha, Ap,  
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_sger(integer* M, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* Y, integer* incY,
         real* A, integer* lda)
{
    sger_(M, N, alpha,
          X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_ssyr(char* uplo, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* A, integer* lda)
{
    ssyr_(uplo, N, alpha, X, incX, A, lda);
    return 0;
}

int
f2c_sspr(char* uplo, integer* N,
         real* alpha,
         real* X, integer* incX,
         real* Ap)
{
    sspr_(uplo, N, alpha, X, incX, Ap);
    return 0;
}

int
f2c_ssyr2(char* uplo, integer* N,
          real* alpha,
          real* X, integer* incX,
          real* Y, integer* incY,
          real* A, integer* lda)
{
    ssyr2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_sspr2(char* uplo, integer* N,
          real* alpha, 
          real* X, integer* incX,
          real* Y, integer* incY,
          real* A)
{
    sspr2_(uplo, N, alpha,
           X, incX, Y, incY, A);
    return 0;
}



int
f2c_dsymv(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    dsymv_(uplo, N, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int 
f2c_dsbmv(char* uplo, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    dsbmv_(uplo, N, K, alpha, A, lda, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_dspmv(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* Ap,
          doublereal* X, integer* incX,
          doublereal* beta,
          doublereal* Y, integer* incY)
{
    dspmv_(uplo, N, alpha, Ap,  
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_dger(integer* M, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* Y, integer* incY,
         doublereal* A, integer* lda)
{
    dger_(M, N, alpha,
          X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_dsyr(char* uplo, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* A, integer* lda)
{
    dsyr_(uplo, N, alpha, X, incX, A, lda);
    return 0;
}

int
f2c_dspr(char* uplo, integer* N,
         doublereal* alpha,
         doublereal* X, integer* incX,
         doublereal* Ap)
{
    dspr_(uplo, N, alpha, X, incX, Ap);
    return 0;
}

int
f2c_dsyr2(char* uplo, integer* N,
          doublereal* alpha,
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY,
          doublereal* A, integer* lda)
{
    dsyr2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_dspr2(char* uplo, integer* N,
          doublereal* alpha, 
          doublereal* X, integer* incX,
          doublereal* Y, integer* incY,
          doublereal* A)
{
    dspr2_(uplo, N, alpha,
           X, incX, Y, incY, A);
    return 0;
}



/*
 * Routines with C and Z prefixes only
 */

int
f2c_chemv(char* uplo, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    chemv_(uplo, N, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_chbmv(char* uplo, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    chbmv_(uplo, N, K, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_chpmv(char* uplo, integer* N, 
          complex* alpha,
          complex* Ap, 
          complex* X, integer* incX,
          complex* beta,
          complex* Y, integer* incY)
{
    chpmv_(uplo, N, alpha, Ap, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_cgeru(integer* M, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    cgeru_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_cgerc(integer* M, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    cgerc_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_cher(char* uplo, integer* N,
         real* alpha,
         complex* X, integer* incX,
         complex* A, integer* lda)
{
    cher_(uplo, N, alpha,
          X, incX, A, lda);
    return 0;
}

int
f2c_chpr(char* uplo, integer* N,
         real* alpha,
         complex* X, integer* incX,
         complex* Ap)
{
    chpr_(uplo, N, alpha,
          X, incX, Ap);
    return 0;
}

int
f2c_cher2(char* uplo, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* A, integer* lda)
{
    cher2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_chpr2(char* uplo, integer* N,
          complex* alpha,
          complex* X, integer* incX,
          complex* Y, integer* incY,
          complex* Ap)
{
    chpr2_(uplo, N, alpha,
           X, incX, Y, incY, Ap);
    return 0;
}



int
f2c_zhemv(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    zhemv_(uplo, N, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zhbmv(char* uplo, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    zhbmv_(uplo, N, K, alpha, A, lda,
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zhpmv(char* uplo, integer* N, 
          doublecomplex* alpha,
          doublecomplex* Ap, 
          doublecomplex* X, integer* incX,
          doublecomplex* beta,
          doublecomplex* Y, integer* incY)
{
    zhpmv_(uplo, N, alpha, Ap, 
           X, incX, beta, Y, incY);
    return 0;
}

int
f2c_zgeru(integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    zgeru_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zgerc(integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    zgerc_(M, N, alpha, 
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zher(char* uplo, integer* N,
         doublereal* alpha,
         doublecomplex* X, integer* incX,
         doublecomplex* A, integer* lda)
{
    zher_(uplo, N, alpha,
          X, incX, A, lda);
    return 0;
}

int
f2c_zhpr(char* uplo, integer* N,
         doublereal* alpha,
         doublecomplex* X, integer* incX,
         doublecomplex* Ap)
{
    zhpr_(uplo, N, alpha,
          X, incX, Ap);
    return 0;
}

int
f2c_zher2(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* A, integer* lda)
{
    zher2_(uplo, N, alpha,
           X, incX, Y, incY, A, lda);
    return 0;
}

int
f2c_zhpr2(char* uplo, integer* N,
          doublecomplex* alpha,
          doublecomplex* X, integer* incX,
          doublecomplex* Y, integer* incY,
          doublecomplex* Ap)
{
    zhpr2_(uplo, N, alpha,
           X, incX, Y, incY, Ap);
    return 0;
}



/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */

int
f2c_sgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb,
          real* beta,
          real* C, integer* ldc)
{
    sgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ssymm(char* side, char* uplo, integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb,
          real* beta,
          real* C, integer* ldc)
{
    ssymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ssyrk(char* uplo, char* trans, integer* N, integer* K,
          real* alpha,
          real* A, integer* lda,
          real* beta,
          real* C, integer* ldc)
{
    ssyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_ssyr2k(char* uplo, char* trans, integer* N, integer* K,
           real* alpha,
           real* A, integer* lda,
           real* B, integer* ldb,
           real* beta,
           real* C, integer* ldc)
{
    ssyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_strmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb)
{
    strmm_(side, uplo, 
           trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_strsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          real* alpha,
          real* A, integer* lda,
          real* B, integer* ldb)
{
    strsm_(side, uplo, 
           trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_dgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    dgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dsymm(char* side, char* uplo, integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    dsymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dsyrk(char* uplo, char* trans, integer* N, integer* K,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* beta,
          doublereal* C, integer* ldc)
{
    dsyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_dsyr2k(char* uplo, char* trans, integer* N, integer* K,
           doublereal* alpha,
           doublereal* A, integer* lda,
           doublereal* B, integer* ldb,
           doublereal* beta,
           doublereal* C, integer* ldc)
{
    dsyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_dtrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb)
{
    dtrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_dtrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          doublereal* alpha,
          doublereal* A, integer* lda,
          doublereal* B, integer* ldb)
{
    dtrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_cgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    cgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_csymm(char* side, char* uplo, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    csymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_csyrk(char* uplo, char* trans, integer* N, integer* K,
          complex* alpha,
          complex* A, integer* lda,
          complex* beta,
          complex* C, integer* ldc)
{
    csyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_csyr2k(char* uplo, char* trans, integer* N, integer* K,
           complex* alpha,
           complex* A, integer* lda,
           complex* B, integer* ldb,
           complex* beta,
           complex* C, integer* ldc)
{
    csyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ctrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb)
{
    ctrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_ctrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb)
{
    ctrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



int
f2c_zgemm(char* transA, char* transB, integer* M, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    zgemm_(transA, transB, M, N, K,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zsymm(char* side, char* uplo, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    zsymm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zsyrk(char* uplo, char* trans, integer* N, integer* K,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    zsyrk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_zsyr2k(char* uplo, char* trans, integer* N, integer* K,
           doublecomplex* alpha,
           doublecomplex* A, integer* lda,
           doublecomplex* B, integer* ldb,
           doublecomplex* beta,
           doublecomplex* C, integer* ldc)
{
    zsyr2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_ztrmm(char* side, char* uplo, char* trans, char* diag, 
          integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb)
{
    ztrmm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}

int 
f2c_ztrsm(char* side, char* uplo, char* trans, char* diag,
          integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb)
{
    ztrsm_(side, uplo, trans, diag, 
           M, N, alpha, A, lda, B, ldb);
    return 0;
}



/*
 * Routines with prefixes C and Z only
 */

int
f2c_chemm(char* side, char* uplo, integer* M, integer* N,
          complex* alpha,
          complex* A, integer* lda,
          complex* B, integer* ldb,
          complex* beta,
          complex* C, integer* ldc)
{
    chemm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_cherk(char* uplo, char* trans, integer* N, integer* K,
          real* alpha,
          complex* A, integer* lda,
          real* beta,
          complex* C, integer* ldc)
{
    cherk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_cher2k(char* uplo, char* trans, integer* N, integer* K,
           complex* alpha,
           complex* A, integer* lda,
           complex* B, integer* ldb,
           real* beta,
           complex* C, integer* ldc)
{
    cher2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}



int
f2c_zhemm(char* side, char* uplo, integer* M, integer* N,
          doublecomplex* alpha,
          doublecomplex* A, integer* lda,
          doublecomplex* B, integer* ldb,
          doublecomplex* beta,
          doublecomplex* C, integer* ldc)
{
    zhemm_(side, uplo, M, N,
           alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

int
f2c_zherk(char* uplo, char* trans, integer* N, integer* K,
          doublereal* alpha,
          doublecomplex* A, integer* lda,
          doublereal* beta,
          doublecomplex* C, integer* ldc)
{
    zherk_(uplo, trans, N, K,
           alpha, A, lda, beta, C, ldc);
    return 0;
}

int
f2c_zher2k(char* uplo, char* trans, integer* N, integer* K,
           doublecomplex* alpha,
           doublecomplex* A, integer* lda,
           doublecomplex* B, integer* ldb,
           doublereal* beta,
           doublecomplex* C, integer* ldc)
{
    zher2k_(uplo, trans, N, K,
            alpha, A, lda, B, ldb, beta, C, ldc);
    return 0;
}

