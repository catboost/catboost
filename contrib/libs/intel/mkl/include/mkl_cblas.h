/*******************************************************************************
* Copyright 1999-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) CBLAS interface
!******************************************************************************/

#ifndef __MKL_CBLAS_H__
#define __MKL_CBLAS_H__
#include <stddef.h>

#include "mkl_types.h"

#ifdef __cplusplus
#if __cplusplus > 199711L
#define NOTHROW noexcept
#else
#define NOTHROW throw()
#endif
#else
#define NOTHROW
#endif

#ifdef __cplusplus
extern "C" { /* Assume C declarations for C++ */
#endif /* __cplusplus */

#ifndef MKL_DECLSPEC
#ifdef _WIN32
#define MKL_DECLSPEC __declspec(dllexport)
#else
#define MKL_DECLSPEC
#endif
#endif
/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t /* this may vary between platforms */

enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
enum CBLAS_STORAGE {CblasPacked=151};
enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162};
enum CBLAS_OFFSET {CblasRowOffset=171, CblasColOffset=172, CblasFixOffset=173};

typedef enum CBLAS_LAYOUT CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO CBLAS_UPLO;
typedef enum CBLAS_DIAG CBLAS_DIAG;
typedef enum CBLAS_SIDE CBLAS_SIDE;
typedef enum CBLAS_STORAGE CBLAS_STORAGE;
typedef enum CBLAS_IDENTIFIER CBLAS_IDENTIFIER;
typedef enum CBLAS_OFFSET CBLAS_OFFSET;


typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */


double cblas_dcabs1(const void *z);
float cblas_scabs1(const void *c);

float cblas_sdot(const MKL_INT N, const float *X, const MKL_INT incX,
                 const float *Y, const MKL_INT incY) NOTHROW;
float cblas_sdoti(const MKL_INT N, const float *X, const MKL_INT *indx,
                  const float *Y) NOTHROW;
double cblas_ddot(const MKL_INT N, const double *X, const MKL_INT incX,
                  const double *Y, const MKL_INT incY) NOTHROW;
double cblas_ddoti(const MKL_INT N, const double *X, const MKL_INT *indx,
                   const double *Y);


double cblas_dsdot(const MKL_INT N, const float *X, const MKL_INT incX,
                   const float *Y, const MKL_INT incY) NOTHROW;

float cblas_sdsdot(const MKL_INT N, const float sb, const float *X,
                   const MKL_INT incX, const float *Y, const MKL_INT incY) NOTHROW;

/*
 * Functions having prefixes Z and C only
 */
void cblas_cdotu_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotu) NOTHROW;
void cblas_cdotui_sub(const MKL_INT N, const void *X, const MKL_INT *indx,
                      const void *Y, void *dotui);
void cblas_cdotc_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotc) NOTHROW;
void cblas_cdotci_sub(const MKL_INT N, const void *X, const MKL_INT *indx,
                      const void *Y, void *dotui);

void cblas_zdotu_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotu) NOTHROW;
void cblas_zdotui_sub(const MKL_INT N, const void *X, const MKL_INT *indx,
                      const void *Y, void *dotui);
void cblas_zdotc_sub(const MKL_INT N, const void *X, const MKL_INT incX,
                     const void *Y, const MKL_INT incY, void *dotc) NOTHROW;
void cblas_zdotci_sub(const MKL_INT N, const void *X, const MKL_INT *indx,
                      const void *Y, void *dotui);

/*
 * Functions having prefixes S D SC DZ
 */
float cblas_snrm2(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;
float cblas_sasum(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;

double cblas_dnrm2(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
double cblas_dasum(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;

float cblas_scnrm2(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;
float cblas_scasum(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

double cblas_dznrm2(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;
double cblas_dzasum(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX cblas_isamax(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_idamax(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_icamax(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_izamax(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_isamin(const MKL_INT N, const float *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_idamin(const MKL_INT N, const double *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_icamin(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;
CBLAS_INDEX cblas_izamin(const MKL_INT N, const void *X, const MKL_INT incX) NOTHROW;

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap(const MKL_INT N, float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY) NOTHROW;
void cblas_scopy(const MKL_INT N, const float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY) NOTHROW;
void cblas_saxpy(const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, float *Y, const MKL_INT incY) NOTHROW;
void cblas_saxpby(const MKL_INT N, const float alpha, const float *X,
                  const MKL_INT incX, const float beta, float *Y, const MKL_INT incY) NOTHROW;
void cblas_saxpyi(const MKL_INT N, const float alpha, const float *X,
                  const MKL_INT *indx, float *Y);
void cblas_sgthr(const MKL_INT N, const float *Y, float *X,
                 const MKL_INT *indx);
void cblas_sgthrz(const MKL_INT N, float *Y, float *X,
                  const MKL_INT *indx);
void cblas_ssctr(const MKL_INT N, const float *X, const MKL_INT *indx,
                 float *Y);
void cblas_srotg(float *a, float *b, float *c, float *s) NOTHROW;

void cblas_dswap(const MKL_INT N, double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY) NOTHROW;
void cblas_dcopy(const MKL_INT N, const double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY) NOTHROW;
void cblas_daxpy(const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, double *Y, const MKL_INT incY) NOTHROW;
void cblas_daxpby(const MKL_INT N, const double alpha, const double *X,
                  const MKL_INT incX, const double beta, double *Y, const MKL_INT incY) NOTHROW;
void cblas_daxpyi(const MKL_INT N, const double alpha, const double *X,
                  const MKL_INT *indx, double *Y);
void cblas_dgthr(const MKL_INT N, const double *Y, double *X,
                 const MKL_INT *indx);
void cblas_dgthrz(const MKL_INT N, double *Y, double *X,
                  const MKL_INT *indx);
void cblas_dsctr(const MKL_INT N, const double *X, const MKL_INT *indx,
                 double *Y);
void cblas_drotg(double *a, double *b, double *c, double *s) NOTHROW;

void cblas_cswap(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_ccopy(const MKL_INT N, const void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_caxpy(const MKL_INT N, const void *alpha, const void *X,
                 const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;
void cblas_caxpby(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_caxpyi(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT *indx, void *Y);
void cblas_cgthr(const MKL_INT N, const void *Y, void *X,
                 const MKL_INT *indx);
void cblas_cgthrz(const MKL_INT N, void *Y, void *X,
                  const MKL_INT *indx);
void cblas_csctr(const MKL_INT N, const void *X, const MKL_INT *indx,
                 void *Y);
void cblas_crotg(void *a, const void *b, float *c, void *s) NOTHROW;

void cblas_zswap(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_zcopy(const MKL_INT N, const void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_zaxpy(const MKL_INT N, const void *alpha, const void *X,
                 const MKL_INT incX, void *Y, const MKL_INT incY) NOTHROW;
void cblas_zaxpby(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW ;
void cblas_zaxpyi(const MKL_INT N, const void *alpha, const void *X,
                  const MKL_INT *indx, void *Y);
void cblas_zgthr(const MKL_INT N, const void *Y, void *X,
                 const MKL_INT *indx);
void cblas_zgthrz(const MKL_INT N, void *Y, void *X,
                  const MKL_INT *indx);
void cblas_zsctr(const MKL_INT N, const void *X, const MKL_INT *indx,
                 void *Y);
void cblas_zrotg(void *a, const void *b, double *c, void *s) NOTHROW;

/*
 * Routines with S and D prefix only
 */
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P) NOTHROW;

void cblas_sroti(const MKL_INT N, float *X, const MKL_INT *indx,
                 float *Y, const float c, const float s);
void cblas_srotm(const MKL_INT N, float *X, const MKL_INT incX,
                 float *Y, const MKL_INT incY, const float *P) NOTHROW;

void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P) NOTHROW;
void cblas_drotm(const MKL_INT N, double *X, const MKL_INT incX,
                 double *Y, const MKL_INT incY, const double *P) NOTHROW;
void cblas_droti(const MKL_INT N, double *X, const MKL_INT *indx,
                 double *Y, const double c, const double s);

/*
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal(const MKL_INT N, const float alpha, float *X, const MKL_INT incX) NOTHROW;
void cblas_dscal(const MKL_INT N, const double alpha, double *X, const MKL_INT incX) NOTHROW;
void cblas_cscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX) NOTHROW;
void cblas_zscal(const MKL_INT N, const void *alpha, void *X, const MKL_INT incX) NOTHROW;
void cblas_csscal(const MKL_INT N, const float alpha, void *X, const MKL_INT incX) NOTHROW;
void cblas_zdscal(const MKL_INT N, const double alpha, void *X, const MKL_INT incX) NOTHROW;

void cblas_srot(const MKL_INT N, float *X, const MKL_INT incX,
                float *Y, const MKL_INT incY, const float c, const float s) NOTHROW;
void cblas_drot(const MKL_INT N, double *X, const MKL_INT incX,
                double *Y, const MKL_INT incY, const double c, const double s) NOTHROW;
void cblas_crot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const float c, const void* s) NOTHROW;
void cblas_zrot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const double c, const void* s) NOTHROW;
void cblas_csrot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const float c, const float s) NOTHROW;
void cblas_zdrot(const MKL_INT N, void *X, const MKL_INT incX,
                 void *Y, const MKL_INT incY, const double c, const double s) NOTHROW;

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float *X, const MKL_INT incX, const float beta,
                 float *Y, const MKL_INT incY) NOTHROW;
void cblas_sgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const float alpha,
                 const float *A, const MKL_INT lda, const float *X,
                 const MKL_INT incX, const float beta, float *Y, const MKL_INT incY) NOTHROW;
void cblas_strmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;
void cblas_stbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;
void cblas_stpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *Ap, float *X, const MKL_INT incX) NOTHROW;
void cblas_strsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *A, const MKL_INT lda, float *X,
                 const MKL_INT incX) NOTHROW;
void cblas_stbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const float *A, const MKL_INT lda,
                 float *X, const MKL_INT incX) NOTHROW;
void cblas_stpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const float *Ap, float *X, const MKL_INT incX) NOTHROW;

void cblas_dgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double *X, const MKL_INT incX, const double beta,
                 double *Y, const MKL_INT incY) NOTHROW;
void cblas_dgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const double alpha,
                 const double *A, const MKL_INT lda, const double *X,
                 const MKL_INT incX, const double beta, double *Y, const MKL_INT incY) NOTHROW;
void cblas_dtrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;
void cblas_dtbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;
void cblas_dtpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *Ap, double *X, const MKL_INT incX) NOTHROW;
void cblas_dtrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *A, const MKL_INT lda, double *X,
                 const MKL_INT incX) NOTHROW;
void cblas_dtbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const double *A, const MKL_INT lda,
                 double *X, const MKL_INT incX) NOTHROW;
void cblas_dtpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const double *Ap, double *X, const MKL_INT incX) NOTHROW;

void cblas_cgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *X, const MKL_INT incX, const void *beta,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_cgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const void *alpha,
                 const void *A, const MKL_INT lda, const void *X,
                 const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_ctrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ctbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ctpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;
void cblas_ctrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                 const MKL_INT incX) NOTHROW;
void cblas_ctbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ctpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;

void cblas_zgemv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *X, const MKL_INT incX, const void *beta,
                 void *Y, const MKL_INT incY) NOTHROW;
void cblas_zgbmv(const CBLAS_LAYOUT Layout,
                 const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                 const MKL_INT KL, const MKL_INT KU, const void *alpha,
                 const void *A, const MKL_INT lda, const void *X,
                 const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_ztrmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ztbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ztpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;
void cblas_ztrsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *A, const MKL_INT lda, void *X,
                 const MKL_INT incX) NOTHROW;
void cblas_ztbsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
                 void *X, const MKL_INT incX) NOTHROW;
void cblas_ztpsv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                 const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) NOTHROW;


/*
 * Routines with S and D prefixes only
 */
void cblas_ssymv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *A,
                 const MKL_INT lda, const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;
void cblas_ssbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                 const MKL_INT lda, const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;
void cblas_sspmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *Ap,
                 const float *X, const MKL_INT incX,
                 const float beta, float *Y, const MKL_INT incY) NOTHROW;
void cblas_sger(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                const float alpha, const float *X, const MKL_INT incX,
                const float *Y, const MKL_INT incY, float *A, const MKL_INT lda) NOTHROW;
void cblas_ssyr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const float *X,
                const MKL_INT incX, float *A, const MKL_INT lda) NOTHROW;
void cblas_sspr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const float *X,
                const MKL_INT incX, float *Ap) NOTHROW;
void cblas_ssyr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, const float *Y, const MKL_INT incY, float *A,
                 const MKL_INT lda) NOTHROW;
void cblas_sspr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const float alpha, const float *X,
                 const MKL_INT incX, const float *Y, const MKL_INT incY, float *A) NOTHROW;

void cblas_dsymv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *A,
                 const MKL_INT lda, const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;
void cblas_dsbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const double alpha, const double *A,
                 const MKL_INT lda, const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;
void cblas_dspmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *Ap,
                 const double *X, const MKL_INT incX,
                 const double beta, double *Y, const MKL_INT incY) NOTHROW;
void cblas_dger(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                const double alpha, const double *X, const MKL_INT incX,
                const double *Y, const MKL_INT incY, double *A, const MKL_INT lda) NOTHROW;
void cblas_dsyr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const double *X,
                const MKL_INT incX, double *A, const MKL_INT lda) NOTHROW;
void cblas_dspr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const double *X,
                const MKL_INT incX, double *Ap) NOTHROW;
void cblas_dsyr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, const double *Y, const MKL_INT incY, double *A,
                 const MKL_INT lda) NOTHROW;
void cblas_dspr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const double alpha, const double *X,
                 const MKL_INT incX, const double *Y, const MKL_INT incY, double *A) NOTHROW;

/*
 * Routines with C and Z prefixes only
 */
void cblas_chemv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_chbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_chpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *Ap,
                 const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_cgeru(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_cgerc(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_cher(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const void *X, const MKL_INT incX,
                void *A, const MKL_INT lda) NOTHROW;
void cblas_chpr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const float alpha, const void *X,
                const MKL_INT incX, void *A) NOTHROW;
void cblas_cher2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_chpr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *Ap) NOTHROW;

void cblas_zhemv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_zhbmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_zhpmv(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const MKL_INT N, const void *alpha, const void *Ap,
                 const void *X, const MKL_INT incX,
                 const void *beta, void *Y, const MKL_INT incY) NOTHROW;
void cblas_zgeru(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_zgerc(const CBLAS_LAYOUT Layout, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_zher(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const void *X, const MKL_INT incX,
                void *A, const MKL_INT lda) NOTHROW;
void cblas_zhpr(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                const MKL_INT N, const double alpha, const void *X,
                const MKL_INT incX, void *A) NOTHROW;
void cblas_zher2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *A, const MKL_INT lda) NOTHROW;
void cblas_zhpr2(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT N,
                 const void *alpha, const void *X, const MKL_INT incX,
                 const void *Y, const MKL_INT incY, void *Ap) NOTHROW;

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const float alpha, const float *A,
                 const MKL_INT lda, const float *B, const MKL_INT ldb,
                 const float beta, float *C, const MKL_INT ldc) NOTHROW;
void cblas_sgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const float *alpha_Array, const float **A_Array,
                       const MKL_INT *lda_Array, const float **B_Array, const MKL_INT *ldb_Array,
                       const float *beta_Array, float **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_sgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const float alpha, const float *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const float *B, const MKL_INT ldb, const MKL_INT strideb,
                               const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;
void cblas_sgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const float alpha, const float *A, const MKL_INT lda,
                  const float *B, const MKL_INT ldb, const float beta,
                  float *C, const MKL_INT ldc) NOTHROW;
void cblas_ssymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float *B, const MKL_INT ldb, const float beta,
                 float *C, const MKL_INT ldc) NOTHROW;
void cblas_ssyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const float alpha, const float *A, const MKL_INT lda,
                 const float beta, float *C, const MKL_INT ldc) NOTHROW;
void cblas_ssyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const float alpha, const float *A, const MKL_INT lda, const MKL_INT stridea,
                 const float beta, float *C, const MKL_INT ldc, const MKL_INT stridec,
                 const MKL_INT batch_size) NOTHROW;
void cblas_ssyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                 const CBLAS_TRANSPOSE *Trans_array, const MKL_INT *N_array, const MKL_INT *K_array,
                 const float *alpha_array, const float **A_array, const MKL_INT *lda_array,
                 const float *beta_array, float **C_array, const MKL_INT *ldc_array,
                 const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_ssyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const float alpha, const float *A, const MKL_INT lda,
                  const float *B, const MKL_INT ldb, const float beta,
                  float *C, const MKL_INT ldc) NOTHROW;
void cblas_strmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 float *B, const MKL_INT ldb) NOTHROW;
void cblas_strsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const float alpha, const float *A, const MKL_INT lda,
                 float *B, const MKL_INT ldb) NOTHROW;
void cblas_strsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                       const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                       const MKL_INT *N_Array, const float *alpha_Array,
                       const float **A_Array, const MKL_INT *lda_Array,
                       float **B_Array, const MKL_INT *ldb_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_strsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                       const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_DIAG Diag, const MKL_INT M,
                       const MKL_INT N, const float alpha,
                       const float *A, const MKL_INT lda, const MKL_INT stridea,
                       float *B, const MKL_INT ldb, const MKL_INT strideb,
                       const MKL_INT batch_size) NOTHROW;

void cblas_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const double alpha, const double *A,
                 const MKL_INT lda, const double *B, const MKL_INT ldb,
                 const double beta, double *C, const MKL_INT ldc) NOTHROW;
void cblas_dgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const double *alpha_Array, const double **A_Array,
                       const MKL_INT *lda_Array, const double **B_Array, const MKL_INT* ldb_Array,
                       const double *beta_Array, double **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_dgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const double alpha, const double *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const double *B, const MKL_INT ldb, const MKL_INT strideb,
                               const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;
void cblas_dgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const double alpha, const double *A, const MKL_INT lda,
                  const double *B, const MKL_INT ldb, const double beta,
                  double *C, const MKL_INT ldc) NOTHROW;
void cblas_dsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double *B, const MKL_INT ldb, const double beta,
                 double *C, const MKL_INT ldc) NOTHROW;
void cblas_dsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const double alpha, const double *A, const MKL_INT lda,
                 const double beta, double *C, const MKL_INT ldc) NOTHROW;
void cblas_dsyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                 const CBLAS_TRANSPOSE *Trans_array, const MKL_INT *N_array, const MKL_INT *K_array,
                 const double *alpha_array, const double **A_array, const MKL_INT *lda_array,
                 const double *beta_array, double **C_array, const MKL_INT *ldc_array,
                 const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_dsyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const double alpha, const double *A, const MKL_INT lda, const MKL_INT stridea,
                 const double beta, double *C, const MKL_INT ldc, const MKL_INT stridec,
                 const MKL_INT batch_size) NOTHROW;
void cblas_dsyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const double alpha, const double *A, const MKL_INT lda,
                  const double *B, const MKL_INT ldb, const double beta,
                  double *C, const MKL_INT ldc) NOTHROW;
void cblas_dtrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 double *B, const MKL_INT ldb) NOTHROW;
void cblas_dtrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const double alpha, const double *A, const MKL_INT lda,
                 double *B, const MKL_INT ldb) NOTHROW;
void cblas_dtrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                       const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                       const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                       const MKL_INT *N_Array, const double *alpha_Array,
                       const double **A_Array, const MKL_INT *lda_Array,
                       double **B_Array, const MKL_INT *ldb_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_dtrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                       const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_DIAG Diag, const MKL_INT M,
                       const MKL_INT N, const double alpha,
                       const double *A, const MKL_INT lda, const MKL_INT stridea,
                       double *B, const MKL_INT ldb, const MKL_INT strideb,
                       const MKL_INT batch_size) NOTHROW;

void cblas_cgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *B, const MKL_INT ldb,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_cgemm3m(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                   const MKL_INT K, const void *alpha, const void *A,
                   const MKL_INT lda, const void *B, const MKL_INT ldb,
                   const void *beta, void *C, const MKL_INT ldc);
void cblas_cgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_cgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;
void cblas_cgemm3m_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                         const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                         const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                         const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                         const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                         const MKL_INT group_count, const MKL_INT *group_size);
void cblas_cgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;
void cblas_csymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;
void cblas_csyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_csyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                 const CBLAS_TRANSPOSE *Trans_array, const MKL_INT *N_array, const MKL_INT *K_array,
                 const void *alpha_array, const void **A_array, const MKL_INT *lda_array,
                 const void *beta_array, void **C_array, const MKL_INT *ldc_array,
                 const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_csyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                 const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                 const MKL_INT batch_size) NOTHROW;
void cblas_csyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;
void cblas_ctrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;
void cblas_ctrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;
void cblas_ctrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                       const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                       const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                       const MKL_INT *N_Array, const void *alpha_Array,
                       const void **A_Array, const MKL_INT *lda_Array,
                       void **B_Array, const MKL_INT *ldb_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_ctrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                       const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_DIAG Diag, const MKL_INT M,
                       const MKL_INT N, const void* alpha,
                       const void *A, const MKL_INT lda, const MKL_INT stridea,
                       void *B, const MKL_INT ldb, const MKL_INT strideb,
                       const MKL_INT batch_size) NOTHROW;

void cblas_zgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                 const MKL_INT K, const void *alpha, const void *A,
                 const MKL_INT lda, const void *B, const MKL_INT ldb,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_zgemm3m(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                   const MKL_INT K, const void *alpha, const void *A,
                   const MKL_INT lda, const void *B, const MKL_INT ldb,
                   const void *beta, void *C, const MKL_INT ldc);
void cblas_zgemm_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                       const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                       const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                       const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                       const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_zgemm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                               const MKL_INT K, const void *alpha, const void *A,
                               const MKL_INT lda, const MKL_INT stridea,
                               const void *B, const MKL_INT ldb, const MKL_INT strideb,
                               const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;
void cblas_zgemm3m_batch(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                         const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT *M_Array, const MKL_INT *N_Array,
                         const MKL_INT *K_Array, const void *alpha_Array, const void **A_Array,
                         const MKL_INT *lda_Array, const void **B_Array, const MKL_INT* ldb_Array,
                         const void *beta_Array, void **C_Array, const MKL_INT *ldc_Array,
                         const MKL_INT group_count, const MKL_INT *group_size);
void cblas_zgemmt(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                  const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;
void cblas_zsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;
void cblas_zsyrk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_zsyrk_batch(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                 const CBLAS_TRANSPOSE *Trans_array, const MKL_INT *N_array, const MKL_INT *K_array,
                 const void *alpha_array, const void **A_array, const MKL_INT *lda_array,
                 const void *beta_array, void **C_array, const MKL_INT *ldc_array,
                 const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_zsyrk_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                 const void *beta, void *C, const MKL_INT ldc, const MKL_INT stridec,
                 const MKL_INT batch_size) NOTHROW;
void cblas_zsyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const void *beta,
                  void *C, const MKL_INT ldc) NOTHROW;
void cblas_ztrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;
void cblas_ztrsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 void *B, const MKL_INT ldb) NOTHROW;
void cblas_ztrsm_batch(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                       const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                       const CBLAS_DIAG *Diag_Array, const MKL_INT *M_Array,
                       const MKL_INT *N_Array, const void *alpha_Array,
                       const void **A_Array, const MKL_INT *lda_Array,
                       void **B_Array, const MKL_INT *ldb_Array,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;
void cblas_ztrsm_batch_strided(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                       const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_DIAG Diag, const MKL_INT M,
                       const MKL_INT N, const void *alpha,
                       const void *A, const MKL_INT lda, const MKL_INT stridea,
                       void *B, const MKL_INT ldb, const MKL_INT strideb,
                       const MKL_INT batch_size) NOTHROW;

/*
 * Routines with prefixes C and Z only
 */
void cblas_chemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;
void cblas_cherk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const float alpha, const void *A, const MKL_INT lda,
                 const float beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_cher2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const float beta,
                  void *C, const MKL_INT ldc) NOTHROW;

void cblas_zhemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                 const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
                 const void *alpha, const void *A, const MKL_INT lda,
                 const void *B, const MKL_INT ldb, const void *beta,
                 void *C, const MKL_INT ldc) NOTHROW;
void cblas_zherk(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                 const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                 const double alpha, const void *A, const MKL_INT lda,
                 const double beta, void *C, const MKL_INT ldc) NOTHROW;
void cblas_zher2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const CBLAS_TRANSPOSE Trans, const MKL_INT N, const MKL_INT K,
                  const void *alpha, const void *A, const MKL_INT lda,
                  const void *B, const MKL_INT ldb, const double beta,
                  void *C, const MKL_INT ldc) NOTHROW;

/*
 * Routines with prefixes S and D only
 */
size_t cblas_sgemm_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                 const MKL_INT M, const MKL_INT N, const MKL_INT K);
void cblas_sgemm_pack(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                      const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N,
                      const MKL_INT K, const float alpha, const float *src,
                      const MKL_INT ld, float *dest);
void cblas_sgemm_compute(const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                         const MKL_INT TransB, const MKL_INT M, const MKL_INT N,
                         const MKL_INT K, const float *A,
                         const MKL_INT lda, const float *B, const MKL_INT ldb,
                         const float beta, float *C, const MKL_INT ldc);
size_t cblas_dgemm_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                 const MKL_INT M, const MKL_INT N, const MKL_INT K);
void cblas_dgemm_pack(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                      const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N,
                      const MKL_INT K, const double alpha, const double *src,
                      const MKL_INT ld, double *dest);
void cblas_dgemm_compute(const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                         const MKL_INT TransB, const MKL_INT M, const MKL_INT N,
                         const MKL_INT K, const double *A,
                         const MKL_INT lda, const double *B, const MKL_INT ldb,
                         const double beta, double *C, const MKL_INT ldc);

void cblas_hgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB,
                 const MKL_INT M, const MKL_INT N, const MKL_INT K,
                 const MKL_F16 alpha, const MKL_F16 *A, const MKL_INT lda,
                 const MKL_F16 *B, const MKL_INT ldb, const MKL_F16 beta,
                 MKL_F16 *C, const MKL_INT ldc) NOTHROW;
size_t cblas_hgemm_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                 const MKL_INT M, const MKL_INT N, const MKL_INT K);
void cblas_hgemm_pack(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                      const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N, const MKL_INT K,
                      const MKL_F16 alpha, const MKL_F16 *src, const MKL_INT ld, MKL_F16 *dest);
void cblas_hgemm_compute(const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                         const MKL_INT TransB,
                         const MKL_INT M, const MKL_INT N, const MKL_INT K,
                         const MKL_F16 *A, const MKL_INT lda,
                         const MKL_F16 *B, const MKL_INT ldb,
                         const MKL_F16 beta,
                         MKL_F16 *C, const MKL_INT ldc);

/*
 * Integer Routines
 */
void cblas_gemm_s16s16s32(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const CBLAS_OFFSET OffsetC,
                          const MKL_INT M, const MKL_INT N, const MKL_INT K,
                          const float alpha, const MKL_INT16 *A, const MKL_INT lda, const MKL_INT16 ao,
                          const MKL_INT16 *B, const MKL_INT ldb, const MKL_INT16 bo, const float beta,
                          MKL_INT32 *C, const MKL_INT ldc, const MKL_INT32 *cb);
void cblas_gemm_s8u8s32(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const CBLAS_OFFSET OffsetC,
                        const MKL_INT M, const MKL_INT N, const MKL_INT K,
                        const float alpha, const void *A, const MKL_INT lda, const MKL_INT8 ao,
                        const void *B, const MKL_INT ldb, const MKL_INT8 bo, const float beta,
                        MKL_INT32 *C, const MKL_INT ldc, const MKL_INT32 *cb) NOTHROW;
void cblas_gemm_bf16bf16f32(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB,
                            const MKL_INT M, const MKL_INT N, const MKL_INT K,
                            const float alpha, const MKL_BF16 *A, const MKL_INT lda,
                            const MKL_BF16 *B, const MKL_INT ldb, const float beta,
                            float *C, const MKL_INT ldc) NOTHROW;

size_t cblas_gemm_s8u8s32_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                        const MKL_INT M, const MKL_INT N, const MKL_INT K);
size_t cblas_gemm_s16s16s32_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                          const MKL_INT M, const MKL_INT N, const MKL_INT K);
size_t cblas_gemm_bf16bf16f32_pack_get_size(const CBLAS_IDENTIFIER identifier,
                                          const MKL_INT M, const MKL_INT N, const MKL_INT K);

void cblas_gemm_s8u8s32_pack (const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                              const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N, const MKL_INT K,
                              const void *src, const MKL_INT ld, void *dest);
void cblas_gemm_s16s16s32_pack(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N, const MKL_INT K,
                               const MKL_INT16 *src, const MKL_INT ld, MKL_INT16 *dest);
void cblas_gemm_bf16bf16f32_pack(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                               const CBLAS_TRANSPOSE Trans, const MKL_INT M, const MKL_INT N, const MKL_INT K,
                               const MKL_BF16 *src, const MKL_INT ld, MKL_BF16 *dest);

void cblas_gemm_s8u8s32_compute (const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                                 const MKL_INT TransB, const CBLAS_OFFSET offsetc,
                                 const MKL_INT M, const MKL_INT N, const MKL_INT K,
                                 const float alpha,
                                 const void *A, const MKL_INT lda, const MKL_INT8 ao,
                                 const void *B, const MKL_INT ldb, const MKL_INT8 bo,
                                 const float beta,
                                 MKL_INT32 *C, const MKL_INT ldc, const MKL_INT32 *co);
void cblas_gemm_s16s16s32_compute(const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                                  const MKL_INT TransB, const CBLAS_OFFSET offsetc,
                                  const MKL_INT M, const MKL_INT N, const MKL_INT K,
                                  const float alpha,
                                  const MKL_INT16 *A, const MKL_INT lda, const MKL_INT16 ao,
                                  const MKL_INT16 *B, const MKL_INT ldb, const MKL_INT16 bo,
                                  const float beta,
                                  MKL_INT32 *C, const MKL_INT ldc, const MKL_INT32 *co);
void cblas_gemm_bf16bf16f32_compute(const CBLAS_LAYOUT Layout, const MKL_INT TransA,
                                  const MKL_INT TransB,
                                  const MKL_INT M, const MKL_INT N, const MKL_INT K,
                                  const float alpha,
                                  const MKL_BF16 *A, const MKL_INT lda,
                                  const MKL_BF16 *B, const MKL_INT ldb,
                                  const float beta,
                                  float *C, const MKL_INT ldc);

/*
 * Jit routines
 */
#ifndef mkl_jit_create_dgemm
#define mkl_jit_create_dgemm mkl_cblas_jit_create_dgemm
#endif
mkl_jit_status_t mkl_cblas_jit_create_dgemm(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                            const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                            const double alpha, const MKL_INT lda, const MKL_INT ldb,
                                            const double beta, const MKL_INT ldc);

#ifndef mkl_jit_create_sgemm
#define mkl_jit_create_sgemm mkl_cblas_jit_create_sgemm
#endif
mkl_jit_status_t mkl_cblas_jit_create_sgemm(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                            const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                            const float alpha, const MKL_INT lda, const MKL_INT ldb,
                                            const float beta, const MKL_INT ldc);
#ifndef mkl_jit_create_cgemm
#define mkl_jit_create_cgemm mkl_cblas_jit_create_cgemm
#endif
mkl_jit_status_t mkl_cblas_jit_create_cgemm(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                            const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                            const void* alpha, const MKL_INT lda, const MKL_INT ldb,
                                            const void* beta, const MKL_INT ldc);

#ifndef mkl_jit_create_zgemm
#define mkl_jit_create_zgemm mkl_cblas_jit_create_zgemm
#endif
mkl_jit_status_t mkl_cblas_jit_create_zgemm(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                            const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                            const void* alpha, const MKL_INT lda, const MKL_INT ldb,
                                            const void* beta, const MKL_INT ldc);



dgemm_jit_kernel_t mkl_jit_get_dgemm_ptr(const void* jitter);
sgemm_jit_kernel_t mkl_jit_get_sgemm_ptr(const void* jitter);
cgemm_jit_kernel_t mkl_jit_get_cgemm_ptr(const void* jitter);
zgemm_jit_kernel_t mkl_jit_get_zgemm_ptr(const void* jitter);

mkl_jit_status_t mkl_jit_destroy(void* jitter);

/* Level1 BLAS batch API */

void cblas_saxpy_batch(const MKL_INT *n, const float *alpha,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_daxpy_batch(const MKL_INT *n, const double *alpha,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_caxpy_batch(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_zaxpy_batch(const MKL_INT *n, const void *alpha,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_saxpy_batch_strided(const MKL_INT N, const float alpha,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_daxpy_batch_strided(const MKL_INT N, const double alpha,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_caxpy_batch_strided(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_zaxpy_batch_strided(const MKL_INT N, const void *alpha,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_scopy_batch(const MKL_INT *n,
                       const float **x, const MKL_INT *incx,
                       float **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_dcopy_batch(const MKL_INT *n,
                       const double **x, const MKL_INT *incx,
                       double **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_ccopy_batch(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_zcopy_batch(const MKL_INT *n,
                       const void **x, const MKL_INT *incx,
                       void **y, const MKL_INT *incy,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_scopy_batch_strided(const MKL_INT N,
                               const float *X, const MKL_INT incX, const MKL_INT stridex,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_dcopy_batch_strided(const MKL_INT N,
                               const double *X, const MKL_INT incX, const MKL_INT stridex,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_ccopy_batch_strided(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_zcopy_batch_strided(const MKL_INT N,
                               const void *X, const MKL_INT incX, const MKL_INT stridex,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

/* Level2 BLAS batch API */

void cblas_sgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const float *alpha, const float **A, const MKL_INT *lda,
                       const float **X, const MKL_INT *incX, const float *beta,
                       float **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_sgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const float alpha, const float *A, const MKL_INT lda, const MKL_INT stridea,
                               const float *X, const MKL_INT incX, const MKL_INT stridex, const float beta,
                               float *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_dgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const double *alpha, const double **A, const MKL_INT *lda,
                       const double **X, const MKL_INT *incX, const double *beta,
                       double **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_dgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const double alpha, const double *A, const MKL_INT lda, const MKL_INT stridea,
                               const double *X, const MKL_INT incX, const MKL_INT stridex, const double beta,
                               double *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_cgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_cgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_zgemv_batch(const CBLAS_LAYOUT Layout,
                       const CBLAS_TRANSPOSE *TransA, const MKL_INT *M, const MKL_INT *N,
                       const void *alpha, const void **A, const MKL_INT *lda,
                       const void **X, const MKL_INT *incX, const void *beta,
                       void **Y, const MKL_INT *incY,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_zgemv_batch_strided(const CBLAS_LAYOUT Layout,
                               const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
                               const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
                               const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
                               void *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size) NOTHROW;

void cblas_sdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const float **a, const MKL_INT *lda,
                       const float **x, const MKL_INT *incx,
                       float **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_sdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const float *a, const MKL_INT lda, const MKL_INT stridea,
                               const float *x, const MKL_INT incx, const MKL_INT stridex,
                               float *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

void cblas_ddgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const double **a, const MKL_INT *lda,
                       const double **x, const MKL_INT *incx,
                       double **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_ddgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const double *a, const MKL_INT lda, const MKL_INT stridea,
                               const double *x, const MKL_INT incx, const MKL_INT stridex,
                               double *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

void cblas_cdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_cdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;

void cblas_zdgmm_batch(const CBLAS_LAYOUT layout,
                       const CBLAS_SIDE *side, const MKL_INT *m, const MKL_INT *n,
                       const void **a, const MKL_INT *lda,
                       const void **x, const MKL_INT *incx,
                       void **c, const MKL_INT *ldc,
                       const MKL_INT group_count, const MKL_INT *group_size) NOTHROW;

void cblas_zdgmm_batch_strided(const CBLAS_LAYOUT layout,
                               const CBLAS_SIDE side, const MKL_INT m, const MKL_INT n,
                               const void *a, const MKL_INT lda, const MKL_INT stridea,
                               const void *x, const MKL_INT incx, const MKL_INT stridex,
                               void *c, const MKL_INT ldc, const MKL_INT stridec,
                               const MKL_INT batch_size) NOTHROW;


#ifdef __cplusplus
}
#endif /* __cplusplus */

#include "mkl_cblas_64.h"

#endif /* __MKL_CBLAS_H__ */
