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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) CBLAS ILP64 interface
!******************************************************************************/

#ifndef __MKL_CBLAS_64_H__
#define __MKL_CBLAS_64_H__
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

#define CBLAS_INDEX_64 MKL_UINT64

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */


float cblas_sdot_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX,
                    const float *Y, const MKL_INT64 incY) NOTHROW;
float cblas_sdoti_64(const MKL_INT64 N, const float *X, const MKL_INT64 *indx,
                     const float *Y) NOTHROW;
double cblas_ddot_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX,
                     const double *Y, const MKL_INT64 incY) NOTHROW;
double cblas_ddoti_64(const MKL_INT64 N, const double *X, const MKL_INT64 *indx,
                      const double *Y);


double cblas_dsdot_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX,
                      const float *Y, const MKL_INT64 incY) NOTHROW;

float cblas_sdsdot_64(const MKL_INT64 N, const float sb, const float *X,
                      const MKL_INT64 incX, const float *Y, const MKL_INT64 incY) NOTHROW;

/*
 * Functions having prefixes Z and C only
 */
void cblas_cdotu_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                        const void *Y, const MKL_INT64 incY, void *dotu) NOTHROW;
void cblas_cdotui_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                         const void *Y, void *dotui);
void cblas_cdotc_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                        const void *Y, const MKL_INT64 incY, void *dotc) NOTHROW;
void cblas_cdotci_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                         const void *Y, void *dotui);

void cblas_zdotu_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                        const void *Y, const MKL_INT64 incY, void *dotu) NOTHROW;
void cblas_zdotui_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                         const void *Y, void *dotui);
void cblas_zdotc_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                        const void *Y, const MKL_INT64 incY, void *dotc) NOTHROW;
void cblas_zdotci_sub_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                         const void *Y, void *dotui);

/*
 * Functions having prefixes S D SC DZ
 */
float cblas_snrm2_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX) NOTHROW;
float cblas_sasum_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX) NOTHROW;

double cblas_dnrm2_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX) NOTHROW;
double cblas_dasum_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX) NOTHROW;

float cblas_scnrm2_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;
float cblas_scasum_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;

double cblas_dznrm2_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;
double cblas_dzasum_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX_64 cblas_isamax_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_idamax_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_icamax_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_izamax_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_isamin_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_idamin_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_icamin_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;
CBLAS_INDEX_64 cblas_izamin_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX) NOTHROW;

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap_64(const MKL_INT64 N, float *X, const MKL_INT64 incX,
                    float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_scopy_64(const MKL_INT64 N, const float *X, const MKL_INT64 incX,
                    float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_saxpy_64(const MKL_INT64 N, const float alpha, const float *X,
                    const MKL_INT64 incX, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_saxpby_64(const MKL_INT64 N, const float alpha, const float *X,
                     const MKL_INT64 incX, const float beta, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_saxpyi_64(const MKL_INT64 N, const float alpha, const float *X,
                     const MKL_INT64 *indx, float *Y);
void cblas_sgthr_64(const MKL_INT64 N, const float *Y, float *X,
                    const MKL_INT64 *indx);
void cblas_sgthrz_64(const MKL_INT64 N, float *Y, float *X,
                     const MKL_INT64 *indx);
void cblas_ssctr_64(const MKL_INT64 N, const float *X, const MKL_INT64 *indx,
                    float *Y);

void cblas_dswap_64(const MKL_INT64 N, double *X, const MKL_INT64 incX,
                    double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dcopy_64(const MKL_INT64 N, const double *X, const MKL_INT64 incX,
                    double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_daxpy_64(const MKL_INT64 N, const double alpha, const double *X,
                    const MKL_INT64 incX, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_daxpby_64(const MKL_INT64 N, const double alpha, const double *X,
                     const MKL_INT64 incX, const double beta, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_daxpyi_64(const MKL_INT64 N, const double alpha, const double *X,
                     const MKL_INT64 *indx, double *Y);
void cblas_dgthr_64(const MKL_INT64 N, const double *Y, double *X,
                    const MKL_INT64 *indx);
void cblas_dgthrz_64(const MKL_INT64 N, double *Y, double *X,
                     const MKL_INT64 *indx);
void cblas_dsctr_64(const MKL_INT64 N, const double *X, const MKL_INT64 *indx,
                    double *Y);

void cblas_cswap_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_ccopy_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_caxpy_64(const MKL_INT64 N, const void *alpha, const void *X,
                    const MKL_INT64 incX, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_caxpby_64(const MKL_INT64 N, const void *alpha, const void *X,
                     const MKL_INT64 incX, const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_caxpyi_64(const MKL_INT64 N, const void *alpha, const void *X,
                     const MKL_INT64 *indx, void *Y);
void cblas_cgthr_64(const MKL_INT64 N, const void *Y, void *X,
                    const MKL_INT64 *indx);
void cblas_cgthrz_64(const MKL_INT64 N, void *Y, void *X,
                     const MKL_INT64 *indx);
void cblas_csctr_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                    void *Y);

void cblas_zswap_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zcopy_64(const MKL_INT64 N, const void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zaxpy_64(const MKL_INT64 N, const void *alpha, const void *X,
                    const MKL_INT64 incX, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zaxpby_64(const MKL_INT64 N, const void *alpha, const void *X,
                     const MKL_INT64 incX, const void *beta, void *Y, const MKL_INT64 incY) NOTHROW ;
void cblas_zaxpyi_64(const MKL_INT64 N, const void *alpha, const void *X,
                     const MKL_INT64 *indx, void *Y);
void cblas_zgthr_64(const MKL_INT64 N, const void *Y, void *X,
                    const MKL_INT64 *indx);
void cblas_zgthrz_64(const MKL_INT64 N, void *Y, void *X,
                     const MKL_INT64 *indx);
void cblas_zsctr_64(const MKL_INT64 N, const void *X, const MKL_INT64 *indx,
                    void *Y);

/*
 * Routines with S and D prefix only
 */
void cblas_sroti_64(const MKL_INT64 N, float *X, const MKL_INT64 *indx,
                    float *Y, const float c, const float s);
void cblas_srotm_64(const MKL_INT64 N, float *X, const MKL_INT64 incX,
                    float *Y, const MKL_INT64 incY, const float *P) NOTHROW;
void cblas_drotm_64(const MKL_INT64 N, double *X, const MKL_INT64 incX,
                    double *Y, const MKL_INT64 incY, const double *P) NOTHROW;
void cblas_droti_64(const MKL_INT64 N, double *X, const MKL_INT64 *indx,
                    double *Y, const double c, const double s);

/*
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal_64(const MKL_INT64 N, const float alpha, float *X, const MKL_INT64 incX) NOTHROW;
void cblas_dscal_64(const MKL_INT64 N, const double alpha, double *X, const MKL_INT64 incX) NOTHROW;
void cblas_cscal_64(const MKL_INT64 N, const void *alpha, void *X, const MKL_INT64 incX) NOTHROW;
void cblas_zscal_64(const MKL_INT64 N, const void *alpha, void *X, const MKL_INT64 incX) NOTHROW;
void cblas_csscal_64(const MKL_INT64 N, const float alpha, void *X, const MKL_INT64 incX) NOTHROW;
void cblas_zdscal_64(const MKL_INT64 N, const double alpha, void *X, const MKL_INT64 incX) NOTHROW;

void cblas_srot_64(const MKL_INT64 N, float *X, const MKL_INT64 incX,
                   float *Y, const MKL_INT64 incY, const float c, const float s) NOTHROW;    
void cblas_drot_64(const MKL_INT64 N, double *X, const MKL_INT64 incX,
                   double *Y, const MKL_INT64 incY, const double c, const double s) NOTHROW;
void cblas_crot_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                   void *Y, const MKL_INT64 incY, const float c, const void* s) NOTHROW;
void cblas_zrot_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                   void *Y, const MKL_INT64 incY, const double c, const void* s) NOTHROW;
void cblas_csrot_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY, const float c, const float s) NOTHROW;
void cblas_zdrot_64(const MKL_INT64 N, void *X, const MKL_INT64 incX,
                    void *Y, const MKL_INT64 incY, const double c, const double s) NOTHROW;    
/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const float alpha, const float *A, const MKL_INT64 lda,
                    const float *X, const MKL_INT64 incX, const float beta,
                    float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_sgbmv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 KL, const MKL_INT64 KU, const float alpha,
                    const float *A, const MKL_INT64 lda, const float *X,
                    const MKL_INT64 incX, const float beta, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_strmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const float *A, const MKL_INT64 lda,
                    float *X, const MKL_INT64 incX) NOTHROW;
void cblas_stbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const float *A, const MKL_INT64 lda,
                    float *X, const MKL_INT64 incX) NOTHROW;
void cblas_stpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const float *Ap, float *X, const MKL_INT64 incX) NOTHROW;
void cblas_strsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const float *A, const MKL_INT64 lda, float *X,
                    const MKL_INT64 incX) NOTHROW;
void cblas_stbsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const float *A, const MKL_INT64 lda,
                    float *X, const MKL_INT64 incX) NOTHROW;
void cblas_stpsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const float *Ap, float *X, const MKL_INT64 incX) NOTHROW;

void cblas_dgemv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const double alpha, const double *A, const MKL_INT64 lda,
                    const double *X, const MKL_INT64 incX, const double beta,
                    double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dgbmv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 KL, const MKL_INT64 KU, const double alpha,
                    const double *A, const MKL_INT64 lda, const double *X,
                    const MKL_INT64 incX, const double beta, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dtrmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const double *A, const MKL_INT64 lda,
                    double *X, const MKL_INT64 incX) NOTHROW;
void cblas_dtbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const double *A, const MKL_INT64 lda,
                    double *X, const MKL_INT64 incX) NOTHROW;
void cblas_dtpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const double *Ap, double *X, const MKL_INT64 incX) NOTHROW;
void cblas_dtrsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const double *A, const MKL_INT64 lda, double *X,
                    const MKL_INT64 incX) NOTHROW;
void cblas_dtbsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const double *A, const MKL_INT64 lda,
                    double *X, const MKL_INT64 incX) NOTHROW;
void cblas_dtpsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const double *Ap, double *X, const MKL_INT64 incX) NOTHROW;

void cblas_cgemv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *X, const MKL_INT64 incX, const void *beta,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_cgbmv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 KL, const MKL_INT64 KU, const void *alpha,
                    const void *A, const MKL_INT64 lda, const void *X,
                    const MKL_INT64 incX, const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_ctrmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ctbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ctpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *Ap, void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ctrsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *A, const MKL_INT64 lda, void *X,
                    const MKL_INT64 incX) NOTHROW;
void cblas_ctbsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ctpsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *Ap, void *X, const MKL_INT64 incX) NOTHROW;

void cblas_zgemv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *X, const MKL_INT64 incX, const void *beta,
                    void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zgbmv_64(const CBLAS_LAYOUT Layout,
                    const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 KL, const MKL_INT64 KU, const void *alpha,
                    const void *A, const MKL_INT64 lda, const void *X,
                    const MKL_INT64 incX, const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_ztrmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ztbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ztpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *Ap, void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ztrsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *A, const MKL_INT64 lda, void *X,
                    const MKL_INT64 incX) NOTHROW;
void cblas_ztbsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const MKL_INT64 K, const void *A, const MKL_INT64 lda,
                    void *X, const MKL_INT64 incX) NOTHROW;
void cblas_ztpsv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                    const MKL_INT64 N, const void *Ap, void *X, const MKL_INT64 incX) NOTHROW;


/*
 * Routines with S and D prefixes only
 */
void cblas_ssymv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const float alpha, const float *A,
                    const MKL_INT64 lda, const float *X, const MKL_INT64 incX,
                    const float beta, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_ssbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const MKL_INT64 K, const float alpha, const float *A,
                    const MKL_INT64 lda, const float *X, const MKL_INT64 incX,
                    const float beta, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_sspmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const float alpha, const float *Ap,
                    const float *X, const MKL_INT64 incX,
                    const float beta, float *Y, const MKL_INT64 incY) NOTHROW;
void cblas_sger_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                   const float alpha, const float *X, const MKL_INT64 incX,
                   const float *Y, const MKL_INT64 incY, float *A, const MKL_INT64 lda) NOTHROW;
void cblas_ssyr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const float alpha, const float *X,
                   const MKL_INT64 incX, float *A, const MKL_INT64 lda) NOTHROW;
void cblas_sspr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const float alpha, const float *X,
                   const MKL_INT64 incX, float *Ap) NOTHROW;
void cblas_ssyr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const float alpha, const float *X,
                    const MKL_INT64 incX, const float *Y, const MKL_INT64 incY, float *A,
                    const MKL_INT64 lda) NOTHROW;
void cblas_sspr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const float alpha, const float *X,
                    const MKL_INT64 incX, const float *Y, const MKL_INT64 incY, float *A) NOTHROW;

void cblas_dsymv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const double alpha, const double *A,
                    const MKL_INT64 lda, const double *X, const MKL_INT64 incX,
                    const double beta, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dsbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const MKL_INT64 K, const double alpha, const double *A,
                    const MKL_INT64 lda, const double *X, const MKL_INT64 incX,
                    const double beta, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dspmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const double alpha, const double *Ap,
                    const double *X, const MKL_INT64 incX,
                    const double beta, double *Y, const MKL_INT64 incY) NOTHROW;
void cblas_dger_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                   const double alpha, const double *X, const MKL_INT64 incX,
                   const double *Y, const MKL_INT64 incY, double *A, const MKL_INT64 lda) NOTHROW;
void cblas_dsyr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const double alpha, const double *X,
                   const MKL_INT64 incX, double *A, const MKL_INT64 lda) NOTHROW;
void cblas_dspr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const double alpha, const double *X,
                   const MKL_INT64 incX, double *Ap) NOTHROW;
void cblas_dsyr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const double alpha, const double *X,
                    const MKL_INT64 incX, const double *Y, const MKL_INT64 incY, double *A,
                    const MKL_INT64 lda) NOTHROW;
void cblas_dspr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const double alpha, const double *X,
                    const MKL_INT64 incX, const double *Y, const MKL_INT64 incY, double *A) NOTHROW;

/*
 * Routines with C and Z prefixes only
 */
void cblas_chemv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_chbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const MKL_INT64 K, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_chpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const void *alpha, const void *Ap,
                    const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_cgeru_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_cgerc_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_cher_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const float alpha, const void *X, const MKL_INT64 incX,
                   void *A, const MKL_INT64 lda) NOTHROW;
void cblas_chpr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const float alpha, const void *X,
                   const MKL_INT64 incX, void *A) NOTHROW;
void cblas_cher2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_chpr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *Ap) NOTHROW;

void cblas_zhemv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zhbmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const MKL_INT64 K, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zhpmv_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const MKL_INT64 N, const void *alpha, const void *Ap,
                    const void *X, const MKL_INT64 incX,
                    const void *beta, void *Y, const MKL_INT64 incY) NOTHROW;
void cblas_zgeru_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_zgerc_64(const CBLAS_LAYOUT Layout, const MKL_INT64 M, const MKL_INT64 N,
                 const void *alpha, const void *X, const MKL_INT64 incX,
                 const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_zher_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const double alpha, const void *X, const MKL_INT64 incX,
                   void *A, const MKL_INT64 lda) NOTHROW;
void cblas_zhpr_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                   const MKL_INT64 N, const double alpha, const void *X,
                   const MKL_INT64 incX, void *A) NOTHROW;
void cblas_zher2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *A, const MKL_INT64 lda) NOTHROW;
void cblas_zhpr2_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo, const MKL_INT64 N,
                    const void *alpha, const void *X, const MKL_INT64 incX,
                    const void *Y, const MKL_INT64 incY, void *Ap) NOTHROW;

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 K, const float alpha, const float *A,
                    const MKL_INT64 lda, const float *B, const MKL_INT64 ldb,
                    const float beta, float *C, const MKL_INT64 ldc) NOTHROW;
void cblas_sgemm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                          const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                          const MKL_INT64 *K_Array, const float *alpha_Array, const float **A_Array,
                          const MKL_INT64 *lda_Array, const float **B_Array, const MKL_INT64 *ldb_Array,
                          const float *beta_Array, float **C_Array, const MKL_INT64 *ldc_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_sgemm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                                  const MKL_INT64 K, const float alpha, const float *A,
                                  const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const float *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const float beta, float *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_sgemmt_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                     const MKL_INT64 N, const MKL_INT64 K,
                     const float alpha, const float *A, const MKL_INT64 lda,
                     const float *B, const MKL_INT64 ldb, const float beta,
                     float *C, const MKL_INT64 ldc) NOTHROW;
void cblas_ssymm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const float alpha, const float *A, const MKL_INT64 lda,
                    const float *B, const MKL_INT64 ldb, const float beta,
                    float *C, const MKL_INT64 ldc) NOTHROW;
void cblas_ssyrk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const float alpha, const float *A, const MKL_INT64 lda,
                    const float beta, float *C, const MKL_INT64 ldc) NOTHROW;
void cblas_ssyrk_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                  const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                                  const float alpha, const float *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const float beta, float *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_ssyrk_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                          const CBLAS_TRANSPOSE *Trans_array, const MKL_INT64 *N_array, const MKL_INT64 *K_array,
                          const float *alpha_array, const float **A_array, const MKL_INT64 *lda_array,
                          const float *beta_array, float **C_array, const MKL_INT64 *ldc_array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_ssyr2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const float alpha, const float *A, const MKL_INT64 lda,
                     const float *B, const MKL_INT64 ldb, const float beta,
                     float *C, const MKL_INT64 ldc) NOTHROW;
void cblas_strmm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const float alpha, const float *A, const MKL_INT64 lda,
                    float *B, const MKL_INT64 ldb) NOTHROW;
void cblas_strsm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const float alpha, const float *A, const MKL_INT64 lda,
                    float *B, const MKL_INT64 ldb) NOTHROW;
void cblas_strsm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *TransA_Array,
                          const CBLAS_DIAG *Diag_Array, const MKL_INT64 *M_Array,
                          const MKL_INT64 *N_Array, const float *alpha_Array,
                          const float **A_Array, const MKL_INT64 *lda_Array,
                          float **B_Array, const MKL_INT64 *ldb_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_strsm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_DIAG Diag, const MKL_INT64 M,
                                  const MKL_INT64 N, const float alpha,
                                  const float *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  float *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_dgemm_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 K, const double alpha, const double *A,
                    const MKL_INT64 lda, const double *B, const MKL_INT64 ldb,
                    const double beta, double *C, const MKL_INT64 ldc) NOTHROW;
void cblas_dgemm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                          const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                          const MKL_INT64 *K_Array, const double *alpha_Array, const double **A_Array,
                          const MKL_INT64 *lda_Array, const double **B_Array, const MKL_INT64* ldb_Array,
                          const double *beta_Array, double **C_Array, const MKL_INT64 *ldc_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_dgemm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                                  const MKL_INT64 K, const double alpha, const double *A,
                                  const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const double *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const double beta, double *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_dgemmt_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                     const MKL_INT64 N, const MKL_INT64 K,
                     const double alpha, const double *A, const MKL_INT64 lda,
                     const double *B, const MKL_INT64 ldb, const double beta,
                     double *C, const MKL_INT64 ldc) NOTHROW;
void cblas_dsymm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const double alpha, const double *A, const MKL_INT64 lda,
                    const double *B, const MKL_INT64 ldb, const double beta,
                    double *C, const MKL_INT64 ldc) NOTHROW;
void cblas_dsyrk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const double alpha, const double *A, const MKL_INT64 lda,
                    const double beta, double *C, const MKL_INT64 ldc) NOTHROW;
void cblas_dsyrk_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                          const CBLAS_TRANSPOSE *Trans_array, const MKL_INT64 *N_array, const MKL_INT64 *K_array,
                          const double *alpha_array, const double **A_array, const MKL_INT64 *lda_array,
                          const double *beta_array, double **C_array, const MKL_INT64 *ldc_array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_dsyrk_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                  const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                                  const double alpha, const double *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const double beta, double *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_dsyr2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const double alpha, const double *A, const MKL_INT64 lda,
                     const double *B, const MKL_INT64 ldb, const double beta,
                     double *C, const MKL_INT64 ldc) NOTHROW;
void cblas_dtrmm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const double alpha, const double *A, const MKL_INT64 lda,
                    double *B, const MKL_INT64 ldb) NOTHROW;
void cblas_dtrsm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const double alpha, const double *A, const MKL_INT64 lda,
                    double *B, const MKL_INT64 ldb) NOTHROW;
void cblas_dtrsm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                          const CBLAS_DIAG *Diag_Array, const MKL_INT64 *M_Array,
                          const MKL_INT64 *N_Array, const double *alpha_Array,
                          const double **A_Array, const MKL_INT64 *lda_Array,
                          double **B_Array, const MKL_INT64 *ldb_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_dtrsm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_DIAG Diag, const MKL_INT64 M,
                                  const MKL_INT64 N, const double alpha,
                                  const double *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  double *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_cgemm_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 K, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *B, const MKL_INT64 ldb,
                    const void *beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_cgemm3m_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                      const MKL_INT64 K, const void *alpha, const void *A,
                      const MKL_INT64 lda, const void *B, const MKL_INT64 ldb,
                      const void *beta, void *C, const MKL_INT64 ldc);
void cblas_cgemm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                          const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                          const MKL_INT64 *K_Array, const void *alpha_Array, const void **A_Array,
                          const MKL_INT64 *lda_Array, const void **B_Array, const MKL_INT64* ldb_Array,
                          const void *beta_Array, void **C_Array, const MKL_INT64 *ldc_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_cgemm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                                  const MKL_INT64 K, const void *alpha, const void *A,
                                  const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const void *beta, void *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_cgemm3m_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                            const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                            const MKL_INT64 *K_Array, const void *alpha_Array, const void **A_Array,
                            const MKL_INT64 *lda_Array, const void **B_Array, const MKL_INT64* ldb_Array,
                            const void *beta_Array, void **C_Array, const MKL_INT64 *ldc_Array,
                            const MKL_INT64 group_count, const MKL_INT64 *group_size);
void cblas_cgemmt_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                     const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const void *beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_csymm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *B, const MKL_INT64 ldb, const void *beta,
                    void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_csyrk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_csyrk_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                          const CBLAS_TRANSPOSE *Trans_array, const MKL_INT64 *N_array, const MKL_INT64 *K_array,
                          const void *alpha_array, const void **A_array, const MKL_INT64 *lda_array,
                          const void *beta_array, void **C_array, const MKL_INT64 *ldc_array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_csyrk_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                  const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                                  const void *alpha, const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *beta, void *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_csyr2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const void *beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_ctrmm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    void *B, const MKL_INT64 ldb) NOTHROW;
void cblas_ctrsm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    void *B, const MKL_INT64 ldb) NOTHROW;
void cblas_ctrsm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                          const CBLAS_DIAG *Diag_Array, const MKL_INT64 *M_Array,
                          const MKL_INT64 *N_Array, const void *alpha_Array,
                          const void **A_Array, const MKL_INT64 *lda_Array,
                          void **B_Array, const MKL_INT64 *ldb_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_ctrsm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_DIAG Diag, const MKL_INT64 M,
                                  const MKL_INT64 N, const void* alpha,
                                  const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  void *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_zgemm_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                    const MKL_INT64 K, const void *alpha, const void *A,
                    const MKL_INT64 lda, const void *B, const MKL_INT64 ldb,
                    const void *beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zgemm3m_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                      const MKL_INT64 K, const void *alpha, const void *A,
                      const MKL_INT64 lda, const void *B, const MKL_INT64 ldb,
                      const void *beta, void *C, const MKL_INT64 ldc);
void cblas_zgemm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                          const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                          const MKL_INT64 *K_Array, const void *alpha_Array, const void **A_Array,
                          const MKL_INT64 *lda_Array, const void **B_Array, const MKL_INT64* ldb_Array,
                          const void *beta_Array, void **C_Array, const MKL_INT64 *ldc_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_zgemm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const MKL_INT64 M, const MKL_INT64 N,
                                  const MKL_INT64 K, const void *alpha, const void *A,
                                  const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const void *beta, void *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_zgemm3m_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE *TransA_Array,
                            const CBLAS_TRANSPOSE *TransB_Array, const MKL_INT64 *M_Array, const MKL_INT64 *N_Array,
                            const MKL_INT64 *K_Array, const void *alpha_Array, const void **A_Array,
                            const MKL_INT64 *lda_Array, const void **B_Array, const MKL_INT64* ldb_Array,
                            const void *beta_Array, void **C_Array, const MKL_INT64 *ldc_Array,
                            const MKL_INT64 group_count, const MKL_INT64 *group_size);
void cblas_zgemmt_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                     const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const void *beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zsymm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *B, const MKL_INT64 ldb, const void *beta,
                    void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zsyrk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zsyrk_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO *Uplo_array,
                          const CBLAS_TRANSPOSE *Trans_array, const MKL_INT64 *N_array, const MKL_INT64 *K_array,
                          const void *alpha_array, const void **A_array, const MKL_INT64 *lda_array,
                          const void *beta_array, void **C_array, const MKL_INT64 *ldc_array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_zsyrk_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                                  const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                                  const void *alpha, const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *beta, void *C, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;
void cblas_zsyr2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const void *beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_ztrmm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    void *B, const MKL_INT64 ldb) NOTHROW;
void cblas_ztrsm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_DIAG Diag, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    void *B, const MKL_INT64 ldb) NOTHROW;
void cblas_ztrsm_batch_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE *Side_Array,
                          const CBLAS_UPLO *Uplo_Array, const CBLAS_TRANSPOSE *Transa_Array,
                          const CBLAS_DIAG *Diag_Array, const MKL_INT64 *M_Array,
                          const MKL_INT64 *N_Array, const void *alpha_Array,
                          const void **A_Array, const MKL_INT64 *lda_Array,
                          void **B_Array, const MKL_INT64 *ldb_Array,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;
void cblas_ztrsm_batch_strided_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                                  const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_DIAG Diag, const MKL_INT64 M,
                                  const MKL_INT64 N, const void *alpha,
                                  const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  void *B, const MKL_INT64 ldb, const MKL_INT64 strideb,
                                  const MKL_INT64 batch_size) NOTHROW;

/*
 * Routines with prefixes C and Z only
 */
void cblas_chemm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *B, const MKL_INT64 ldb, const void *beta,
                    void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_cherk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const float alpha, const void *A, const MKL_INT64 lda,
                    const float beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_cher2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const float beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;

void cblas_zhemm_64(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                    const CBLAS_UPLO Uplo, const MKL_INT64 M, const MKL_INT64 N,
                    const void *alpha, const void *A, const MKL_INT64 lda,
                    const void *B, const MKL_INT64 ldb, const void *beta,
                    void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zherk_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                    const double alpha, const void *A, const MKL_INT64 lda,
                    const double beta, void *C, const MKL_INT64 ldc) NOTHROW;
void cblas_zher2k_64(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                     const CBLAS_TRANSPOSE Trans, const MKL_INT64 N, const MKL_INT64 K,
                     const void *alpha, const void *A, const MKL_INT64 lda,
                     const void *B, const MKL_INT64 ldb, const double beta,
                     void *C, const MKL_INT64 ldc) NOTHROW;

/*
 * Routines with prefixes S and D only
 */
MKL_UINT64 cblas_sgemm_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                        const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);
void cblas_sgemm_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                         const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N,
                         const MKL_INT64 K, const float alpha, const float *src,
                         const MKL_INT64 ld, float *dest);
void cblas_sgemm_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                            const MKL_INT64 TransB, const MKL_INT64 M, const MKL_INT64 N,
                            const MKL_INT64 K, const float *A,
                            const MKL_INT64 lda, const float *B, const MKL_INT64 ldb,
                            const float beta, float *C, const MKL_INT64 ldc);
MKL_UINT64 cblas_dgemm_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                        const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);
void cblas_dgemm_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                         const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N,
                         const MKL_INT64 K, const double alpha, const double *src,
                         const MKL_INT64 ld, double *dest);
void cblas_dgemm_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                            const MKL_INT64 TransB, const MKL_INT64 M, const MKL_INT64 N,
                            const MKL_INT64 K, const double *A,
                            const MKL_INT64 lda, const double *B, const MKL_INT64 ldb,
                            const double beta, double *C, const MKL_INT64 ldc);

void cblas_hgemm_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                    const CBLAS_TRANSPOSE TransB,
                    const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                    const MKL_F16 alpha, const MKL_F16 *A, const MKL_INT64 lda,
                    const MKL_F16 *B, const MKL_INT64 ldb, const MKL_F16 beta,
                    MKL_F16 *C, const MKL_INT64 ldc);
MKL_UINT64 cblas_hgemm_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                        const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);
void cblas_hgemm_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                         const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                         const MKL_F16 alpha, const MKL_F16 *src, const MKL_INT64 ld, MKL_F16 *dest);
void cblas_hgemm_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                            const MKL_INT64 TransB,
                            const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K, 
                            const MKL_F16 *A, const MKL_INT64 lda,
                            const MKL_F16 *B, const MKL_INT64 ldb,
                            const MKL_F16 beta,
                            MKL_F16 *C, const MKL_INT64 ldc);

/*
 * Integer Routines
 */
void cblas_gemm_s16s16s32_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, const CBLAS_OFFSET OffsetC,
                             const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                             const float alpha, const MKL_INT16 *A, const MKL_INT64 lda, const MKL_INT16 ao,
                             const MKL_INT16 *B, const MKL_INT64 ldb, const MKL_INT16 bo, const float beta,
                             MKL_INT32 *C, const MKL_INT64 ldc, const MKL_INT32 *cb);
void cblas_gemm_s8u8s32_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const CBLAS_OFFSET OffsetC,
                           const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                           const float alpha, const void *A, const MKL_INT64 lda, const MKL_INT8 ao,
                           const void *B, const MKL_INT64 ldb, const MKL_INT8 bo, const float beta,
                           MKL_INT32 *C, const MKL_INT64 ldc, const MKL_INT32 *cb);
void cblas_gemm_bf16bf16f32_64(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                               const CBLAS_TRANSPOSE TransB,
                               const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                               const float alpha, const MKL_BF16 *A, const MKL_INT64 lda,
                               const MKL_BF16 *B, const MKL_INT64 ldb, const float beta,
                               float *C, const MKL_INT64 ldc);

MKL_UINT64 cblas_gemm_s8u8s32_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                               const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);
MKL_UINT64 cblas_gemm_s16s16s32_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                                 const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);
MKL_UINT64 cblas_gemm_bf16bf16f32_pack_get_size_64(const CBLAS_IDENTIFIER identifier,
                                                   const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K);

void cblas_gemm_s8u8s32_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                                const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                const void *src, const MKL_INT64 ld, void *dest);
void cblas_gemm_s16s16s32_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                                  const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                  const MKL_INT16 *src, const MKL_INT64 ld, MKL_INT16 *dest);
void cblas_gemm_bf16bf16f32_pack_64(const CBLAS_LAYOUT Layout, const CBLAS_IDENTIFIER identifier,
                                    const CBLAS_TRANSPOSE Trans, const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                    const MKL_BF16 *src, const MKL_INT64 ld, MKL_BF16 *dest);

void cblas_gemm_s8u8s32_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                                   const MKL_INT64 TransB, const CBLAS_OFFSET offsetc,
                                   const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                   const float alpha,
                                   const void *A, const MKL_INT64 lda, const MKL_INT8 ao,
                                   const void *B, const MKL_INT64 ldb, const MKL_INT8 bo,
                                   const float beta,
                                   MKL_INT32 *C, const MKL_INT64 ldc, const MKL_INT32 *co);
void cblas_gemm_s16s16s32_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                                     const MKL_INT64 TransB, const CBLAS_OFFSET offsetc,
                                     const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                     const float alpha,
                                     const MKL_INT16 *A, const MKL_INT64 lda, const MKL_INT16 ao,
                                     const MKL_INT16 *B, const MKL_INT64 ldb, const MKL_INT16 bo,
                                     const float beta,
                                     MKL_INT32 *C, const MKL_INT64 ldc, const MKL_INT32 *co);
void cblas_gemm_bf16bf16f32_compute_64(const CBLAS_LAYOUT Layout, const MKL_INT64 TransA,
                                       const MKL_INT64 TransB,
                                       const MKL_INT64 M, const MKL_INT64 N, const MKL_INT64 K,
                                       const float alpha,
                                       const MKL_BF16 *A, const MKL_INT64 lda,
                                       const MKL_BF16 *B, const MKL_INT64 ldb,
                                       const float beta,
                                       float *C, const MKL_INT64 ldc);

/*
 * Jit routines
 */
#ifndef mkl_jit_create_dgemm_64
#define mkl_jit_create_dgemm_64 mkl_cblas_jit_create_dgemm_64
#endif
mkl_jit_status_t mkl_cblas_jit_create_dgemm_64(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                               const MKL_INT64 m, const MKL_INT64 n, const MKL_INT64 k,
                                               const double alpha, const MKL_INT64 lda, const MKL_INT64 ldb,
                                               const double beta, const MKL_INT64 ldc);

#ifndef mkl_jit_create_sgemm_64
#define mkl_jit_create_sgemm_64 mkl_cblas_jit_create_sgemm_64
#endif
mkl_jit_status_t mkl_cblas_jit_create_sgemm_64(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                               const MKL_INT64 m, const MKL_INT64 n, const MKL_INT64 k,
                                               const float alpha, const MKL_INT64 lda, const MKL_INT64 ldb,
                                               const float beta, const MKL_INT64 ldc);
#ifndef mkl_jit_create_cgemm_64
#define mkl_jit_create_cgemm_64 mkl_cblas_jit_create_cgemm_64
#endif
mkl_jit_status_t mkl_cblas_jit_create_cgemm_64(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                               const MKL_INT64 m, const MKL_INT64 n, const MKL_INT64 k,
                                               const void* alpha, const MKL_INT64 lda, const MKL_INT64 ldb,
                                               const void* beta, const MKL_INT64 ldc);

#ifndef mkl_jit_create_zgemm_64
#define mkl_jit_create_zgemm_64 mkl_cblas_jit_create_zgemm_64
#endif
mkl_jit_status_t mkl_cblas_jit_create_zgemm_64(void** jitter, const MKL_LAYOUT layout, const MKL_TRANSPOSE transa, const MKL_TRANSPOSE transb,
                                               const MKL_INT64 m, const MKL_INT64 n, const MKL_INT64 k,
                                               const void* alpha, const MKL_INT64 lda, const MKL_INT64 ldb,
                                               const void* beta, const MKL_INT64 ldc);

/* Level1 BLAS batch API */

void cblas_saxpy_batch_64(const MKL_INT64 *n, const float *alpha,
                          const float **x, const MKL_INT64 *incx,
                          float **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_daxpy_batch_64(const MKL_INT64 *n, const double *alpha,
                          const double **x, const MKL_INT64 *incx,
                          double **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_caxpy_batch_64(const MKL_INT64 *n, const void *alpha,
                          const void **x, const MKL_INT64 *incx,
                          void **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_zaxpy_batch_64(const MKL_INT64 *n, const void *alpha,
                          const void **x, const MKL_INT64 *incx,
                          void **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_saxpy_batch_strided_64(const MKL_INT64 N, const float alpha,
                                  const float *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  float *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_daxpy_batch_strided_64(const MKL_INT64 N, const double alpha,
                                  const double *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  double *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_caxpy_batch_strided_64(const MKL_INT64 N, const void *alpha,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_zaxpy_batch_strided_64(const MKL_INT64 N, const void *alpha,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_scopy_batch_64(const MKL_INT64 *n,
                          const float **x, const MKL_INT64 *incx,
                          float **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_dcopy_batch_64(const MKL_INT64 *n,
                          const double **x, const MKL_INT64 *incx,
                          double **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_ccopy_batch_64(const MKL_INT64 *n,
                          const void **x, const MKL_INT64 *incx,
                          void **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_zcopy_batch_64(const MKL_INT64 *n,
                          const void **x, const MKL_INT64 *incx,
                          void **y, const MKL_INT64 *incy,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_scopy_batch_strided_64(const MKL_INT64 N,
                                  const float *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  float *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_dcopy_batch_strided_64(const MKL_INT64 N,
                                  const double *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  double *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_ccopy_batch_strided_64(const MKL_INT64 N,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_zcopy_batch_strided_64(const MKL_INT64 N,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

/* Level2 BLAS batch API */

void cblas_sgemv_batch_64(const CBLAS_LAYOUT Layout,
                          const CBLAS_TRANSPOSE *TransA, const MKL_INT64 *M, const MKL_INT64 *N,
                          const float *alpha, const float **A, const MKL_INT64 *lda,
                          const float **X, const MKL_INT64 *incX, const float *beta,
                          float **Y, const MKL_INT64 *incY,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_sgemv_batch_strided_64(const CBLAS_LAYOUT Layout,
                                  const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                                  const float alpha, const float *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const float *X, const MKL_INT64 incX, const MKL_INT64 stridex, const float beta,
                                  float *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_dgemv_batch_64(const CBLAS_LAYOUT Layout,
                          const CBLAS_TRANSPOSE *TransA, const MKL_INT64 *M, const MKL_INT64 *N,
                          const double *alpha, const double **A, const MKL_INT64 *lda,
                          const double **X, const MKL_INT64 *incX, const double *beta,
                          double **Y, const MKL_INT64 *incY,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_dgemv_batch_strided_64(const CBLAS_LAYOUT Layout,
                                  const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                                  const double alpha, const double *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const double *X, const MKL_INT64 incX, const MKL_INT64 stridex, const double beta,
                                  double *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_cgemv_batch_64(const CBLAS_LAYOUT Layout,
                          const CBLAS_TRANSPOSE *TransA, const MKL_INT64 *M, const MKL_INT64 *N,
                          const void *alpha, const void **A, const MKL_INT64 *lda,
                          const void **X, const MKL_INT64 *incX, const void *beta,
                          void **Y, const MKL_INT64 *incY,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_cgemv_batch_strided_64(const CBLAS_LAYOUT Layout,
                                  const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                                  const void *alpha, const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex, const void *beta,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_zgemv_batch_64(const CBLAS_LAYOUT Layout,
                          const CBLAS_TRANSPOSE *TransA, const MKL_INT64 *M, const MKL_INT64 *N,
                          const void *alpha, const void **A, const MKL_INT64 *lda,
                          const void **X, const MKL_INT64 *incX, const void *beta,
                          void **Y, const MKL_INT64 *incY,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_zgemv_batch_strided_64(const CBLAS_LAYOUT Layout,
                                  const CBLAS_TRANSPOSE TransA, const MKL_INT64 M, const MKL_INT64 N,
                                  const void *alpha, const void *A, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *X, const MKL_INT64 incX, const MKL_INT64 stridex, const void *beta,
                                  void *Y, const MKL_INT64 incY, const MKL_INT64 stridey,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_sdgmm_batch_64(const CBLAS_LAYOUT layout,
                          const CBLAS_SIDE *side, const MKL_INT64 *m, const MKL_INT64 *n,
                          const float **a, const MKL_INT64 *lda,
                          const float **x, const MKL_INT64 *incx,
                          float **c, const MKL_INT64 *ldc,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_sdgmm_batch_strided_64(const CBLAS_LAYOUT layout,
                                  const CBLAS_SIDE side, const MKL_INT64 m, const MKL_INT64 n,
                                  const float *a, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const float *x, const MKL_INT64 incx, const MKL_INT64 stridex,
                                  float *c, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_ddgmm_batch_64(const CBLAS_LAYOUT layout,
                          const CBLAS_SIDE *side, const MKL_INT64 *m, const MKL_INT64 *n,
                          const double **a, const MKL_INT64 *lda,
                          const double **x, const MKL_INT64 *incx,
                          double **c, const MKL_INT64 *ldc,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_ddgmm_batch_strided_64(const CBLAS_LAYOUT layout,
                                  const CBLAS_SIDE side, const MKL_INT64 m, const MKL_INT64 n,
                                  const double *a, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const double *x, const MKL_INT64 incx, const MKL_INT64 stridex,
                                  double *c, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_cdgmm_batch_64(const CBLAS_LAYOUT layout,
                          const CBLAS_SIDE *side, const MKL_INT64 *m, const MKL_INT64 *n,
                          const void **a, const MKL_INT64 *lda,
                          const void **x, const MKL_INT64 *incx,
                          void **c, const MKL_INT64 *ldc,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_cdgmm_batch_strided_64(const CBLAS_LAYOUT layout,
                                  const CBLAS_SIDE side, const MKL_INT64 m, const MKL_INT64 n,
                                  const void *a, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *x, const MKL_INT64 incx, const MKL_INT64 stridex,
                                  void *c, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;

void cblas_zdgmm_batch_64(const CBLAS_LAYOUT layout,
                          const CBLAS_SIDE *side, const MKL_INT64 *m, const MKL_INT64 *n,
                          const void **a, const MKL_INT64 *lda,
                          const void **x, const MKL_INT64 *incx,
                          void **c, const MKL_INT64 *ldc,
                          const MKL_INT64 group_count, const MKL_INT64 *group_size) NOTHROW;

void cblas_zdgmm_batch_strided_64(const CBLAS_LAYOUT layout,
                                  const CBLAS_SIDE side, const MKL_INT64 m, const MKL_INT64 n,
                                  const void *a, const MKL_INT64 lda, const MKL_INT64 stridea,
                                  const void *x, const MKL_INT64 incx, const MKL_INT64 stridex,
                                  void *c, const MKL_INT64 ldc, const MKL_INT64 stridec,
                                  const MKL_INT64 batch_size) NOTHROW;


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_CBLAS_64_H__ */
