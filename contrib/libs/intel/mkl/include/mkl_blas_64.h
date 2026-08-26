/*******************************************************************************
* Copyright 2021-2022 Intel Corporation.
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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) interface for ILP64 BLAS routines
!      Note: ILP64 interfaces are not supported on IA-32 architecture
!******************************************************************************/

#ifndef _MKL_BLAS_64_H_
#define _MKL_BLAS_64_H_
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
extern "C" {
#endif /* __cplusplus */

/* Upper case declaration */

/* BLAS Level1 */

float SCABS1_64(const MKL_Complex8 *c);
float SASUM_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
void SAXPY_64(const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
void SAXPBY_64(const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SAXPYI_64(const MKL_INT64 *nz, const float *a, const float *x, const MKL_INT64 *indx,float *y);
float SCASUM_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
float SCNRM2_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void SCOPY_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
float SDOT_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
float SDSDOT_64(const MKL_INT64 *n, const float *sb, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
float SDOTI_64(const MKL_INT64 *nz, const float *x, const MKL_INT64 *indx, const float *y);
void SGTHR_64(const MKL_INT64 *nz, const float *y, float *x, const MKL_INT64 *indx);
void SGTHRZ_64(const MKL_INT64 *nz, float *y, float *x, const MKL_INT64 *indx);
float SNRM2_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
void SROT_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy, const float *c, const float *s) NOTHROW;
void SROTG_64(float *a,float *b,float *c,float *s) NOTHROW;
void SROTI_64(const MKL_INT64 *nz, float *x, const MKL_INT64 *indx, float *y, const float *c, const float *s);
void SROTM_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy, const float *param) NOTHROW;
void SROTMG_64(float *d1, float *d2, float *x1, const float *y1, float *param) NOTHROW;
void SSCAL_64(const MKL_INT64 *n, const float *a, float *x, const MKL_INT64 *incx) NOTHROW;
void SSCTR_64(const MKL_INT64 *nz, const float *x, const MKL_INT64 *indx, float *y);
void SSWAP_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 ISAMAX_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 ISAMIN_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;

void CAXPY_64(const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CAXPBY_64(const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy);
void CAXPYI_64(const MKL_INT64 *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT64 *indx, MKL_Complex8 *y);
void CCOPY_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CDOTC_64(MKL_Complex8 *pres, const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CDOTCI_64(MKL_Complex8 *pres, const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, const MKL_Complex8 *y);
void CDOTU_64(MKL_Complex8 *pres, const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CDOTUI_64(MKL_Complex8 *pres, const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, const MKL_Complex8 *y);
void CGTHR_64(const MKL_INT64 *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT64 *indx);
void CGTHRZ_64(const MKL_INT64 *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT64 *indx);
void CROT_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy, const float *c, const MKL_Complex8 *s) NOTHROW;    
void CROTG_64(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s) NOTHROW;
void CSCAL_64(const MKL_INT64 *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CSCTR_64(const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, MKL_Complex8 *y);
void CSROT_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy, const float *c, const float *s) NOTHROW;
void CSSCAL_64(const MKL_INT64 *n, const float *a, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CSWAP_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 ICAMAX_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 ICAMIN_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;

double DCABS1_64(const MKL_Complex16 *z);
double DASUM_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
void DAXPY_64(const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
void DAXPBY_64(const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx, const double *beta, double *y, const MKL_INT64 *incy);
void DAXPYI_64(const MKL_INT64 *nz, const double *a, const double *x, const MKL_INT64 *indx, double *y);
void DCOPY_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
double DDOT_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx, const double *y, const MKL_INT64 *incy) NOTHROW;
double DSDOT_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
double DDOTI_64(const MKL_INT64 *nz, const double *x, const MKL_INT64 *indx, const double *y);
void DGTHR_64(const MKL_INT64 *nz, const double *y, double *x, const MKL_INT64 *indx);
void DGTHRZ_64(const MKL_INT64 *nz, double *y, double *x, const MKL_INT64 *indx);
double DNRM2_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
void DROT_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy, const double *c, const double *s) NOTHROW;
void DROTG_64(double *a,double *b,double *c,double *s) NOTHROW;
void DROTI_64(const MKL_INT64 *nz, double *x, const MKL_INT64 *indx, double *y, const double *c, const double *s);
void DROTM_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy, const double *param) NOTHROW;
void DROTMG_64(double *d1, double *d2, double *x1, const double *y1, double *param) NOTHROW;
void DSCAL_64(const MKL_INT64 *n, const double *a, double *x, const MKL_INT64 *incx) NOTHROW;
void DSCTR_64(const MKL_INT64 *nz, const double *x, const MKL_INT64 *indx, double *y);
void DSWAP_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
double DZASUM_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
double DZNRM2_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 IDAMAX_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 IDAMIN_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;

void ZAXPY_64(const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZAXPBY_64(const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy);
void ZAXPYI_64(const MKL_INT64 *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT64 *indx, MKL_Complex16 *y);
void ZCOPY_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZDOTC_64(MKL_Complex16 *pres, const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZDOTCI_64(MKL_Complex16 *pres,const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, const MKL_Complex16 *y);
void ZDOTU_64(MKL_Complex16 *pres, const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZDOTUI_64(MKL_Complex16 *pres, const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, const MKL_Complex16 *y);
void ZDROT_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy, const double *c, const double *s) NOTHROW;
void ZDSCAL_64(const MKL_INT64 *n, const double *a, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZGTHR_64(const MKL_INT64 *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT64 *indx);
void ZGTHRZ_64(const MKL_INT64 *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT64 *indx);
void ZROT_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy, const double *c, const MKL_Complex16 *s) NOTHROW;    
void ZROTG_64(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s) NOTHROW;
void ZSCAL_64(const MKL_INT64 *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZSCTR_64(const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, MKL_Complex16 *y);
void ZSWAP_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 IZAMAX_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 IZAMIN_64(const MKL_INT64 *n,const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;

/* BLAS Level2 */

void SGBMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
           const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SGER_64(const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
          const float *y, const MKL_INT64 *incy, float *a, const MKL_INT64 *lda) NOTHROW;
void SSBMV_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SSPMV_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *ap,
           const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SSPR_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx, float *ap) NOTHROW;
void SSPR2_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
           const float *y, const MKL_INT64 *incy, float *ap) NOTHROW;
void SSYMV_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void SSYR_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
          float *a, const MKL_INT64 *lda) NOTHROW;
void SSYR2_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
           const float *y, const MKL_INT64 *incy, float *a, const MKL_INT64 *lda) NOTHROW;
void STBMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void STBSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void STPMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const float *ap, float *x, const MKL_INT64 *incx) NOTHROW;
void STPSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const float *ap, float *x, const MKL_INT64 *incx) NOTHROW;
void STRMV_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const float *a, const MKL_INT64 *lda, float *b, const MKL_INT64 *incx) NOTHROW;
void STRSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void SGEM2VU_64(const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
             const float *x1, const MKL_INT64 *incx1, const float *x2, const MKL_INT64 *incx2,
             const float *beta, float *y1, const MKL_INT64 *incy1, float *y2, const MKL_INT64 *incy2);

void CGBMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CGERC_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void CGERU_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void CHBMV_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CHEMV_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CHER_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx,
          MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void CHER2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void CHPMV_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void CHPR_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx,
          MKL_Complex8 *ap) NOTHROW;
void CHPR2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *ap) NOTHROW;
void CTBMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CTBSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CTPMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CTPSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CTRMV_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *b, const MKL_INT64 *incx) NOTHROW;
void CTRSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void CGEM2VC_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
             const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x1, const MKL_INT64 *incx1,
             const MKL_Complex8 *x2, const MKL_INT64 *incx2, const MKL_Complex8 *beta,
             MKL_Complex8 *y1, const MKL_INT64 *incy1, MKL_Complex8 *y2, const MKL_INT64 *incy2);
void SCGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
            const float *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
            const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy);

void DGBMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void DGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
           const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void DGER_64(const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
          const double *y, const MKL_INT64 *incy, double *a, const MKL_INT64 *lda) NOTHROW;
void DSBMV_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const double *alpha,
           const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void DSPMV_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *ap,
           const double *x, const MKL_INT64 *incx, const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void DSPR_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx, double *ap) NOTHROW;
void DSPR2_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
           const double *y, const MKL_INT64 *incy, double *ap) NOTHROW;
void DSYMV_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           const double *x, const MKL_INT64 *incx, const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void DSYR_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
          double *a, const MKL_INT64 *lda) NOTHROW;
void DSYR2_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
           const double *y, const MKL_INT64 *incy, double *a, const MKL_INT64 *lda) NOTHROW;
void DTBMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void DTBSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void DTPMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *ap, double *x, const MKL_INT64 *incx) NOTHROW;
void DTPSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *ap, double *x, const MKL_INT64 *incx) NOTHROW;
void DTRMV_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const double *a, const MKL_INT64 *lda, double *b, const MKL_INT64 *incx) NOTHROW;
void DTRSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void DGEM2VU_64(const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
             const double *a, const MKL_INT64 *lda, const double *x1, const MKL_INT64 *incx1,
             const double *x2, const MKL_INT64 *incx2, const double *beta,
             double *y1, const MKL_INT64 *incy1, double *y2, const MKL_INT64 *incy2);

void ZGBMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZGERC_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy,
           MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void ZGERU_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy,
           MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void ZHBMV_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZHEMV_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZHER_64(const char *uplo, const MKL_INT64 *n, const double *alpha,
          const MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void ZHER2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy,
           MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void ZHPMV_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void ZHPR_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const MKL_Complex16 *x,
          const MKL_INT64 *incx, MKL_Complex16 *ap) NOTHROW;
void ZHPR2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy,
           MKL_Complex16 *ap) NOTHROW;
void ZTBMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZTBSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZTPMV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZTPSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZTRMV_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *incx) NOTHROW;
void ZTRSV_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ZGEM2VC_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
             const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x1, const MKL_INT64 *incx1,
             const MKL_Complex16 *x2, const MKL_INT64 *incx2, const MKL_Complex16 *beta,
             MKL_Complex16 *y1, const MKL_INT64 *incy1, MKL_Complex16 *y2, const MKL_INT64 *incy2);
void DZGEMV_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
            const double *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
            const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy);

/* BLAS Level3 */

void SGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
           const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
size_t SGEMM_PACK_GET_SIZE_64(const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void SGEMM_PACK_64(const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                const float *alpha, const float *src, const MKL_INT64 *ld, float *dest);
void SGEMM_COMPUTE_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb, const float *beta, float *c, const MKL_INT64 *ldc);
void SGEMM_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT64 *lda_array, const float **b_array, const MKL_INT64 *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void SGEMM_BATCH_STRIDED_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const float *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const float *beta, float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;
void SGEMMT_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
            const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void SSYMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
           const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void SSYR2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
            const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void SSYRK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda,
           const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void SSYRK_BATCH_STRIDED_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const float *beta,
           float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void SSYRK_BATCH_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const float *alpha_array, const float **a_array, const MKL_INT64 *lda_array, const float *beta_array,
           float **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void STRMM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           float *b, const MKL_INT64 *ldb) NOTHROW;
void STRSM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           float *b, const MKL_INT64 *ldb) NOTHROW;
void STRSM_BATCH_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const float *alpha_array, const float **a_array,
                 const MKL_INT64 *lda_array, float **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void STRSM_BATCH_STRIDED_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 float *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void CGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;

void CGEMM_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array, const MKL_Complex8 **b_array, const MKL_INT64 *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void CGEMM_BATCH_STRIDED_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;
void SCGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const float *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc);
void CGEMM3M_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
             const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
             const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
             MKL_Complex8 *c, const MKL_INT64 *ldc);
void CGEMM3M_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                   const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array, const MKL_Complex8 **b_array, const MKL_INT64 *ldb_array,
                   const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size);
void CGEMMT_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb,
            const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CTRSM_BATCH_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                 const MKL_INT64 *lda_array, MKL_Complex8 **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void CTRSM_BATCH_STRIDED_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void CHEMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CHER2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const float *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CHERK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const float *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CSYMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *b, const MKL_INT64 *ldb,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CSYR2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb,
            const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CSYRK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void CSYRK_BATCH_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array,
           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void CSYRK_BATCH_STRIDED_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void CTRMM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda,
           MKL_Complex8 *b, const MKL_INT64 *ldb) NOTHROW;
void CTRSM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda,
           MKL_Complex8 *b, const MKL_INT64 *ldb) NOTHROW;

void DGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
           const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
size_t DGEMM_PACK_GET_SIZE_64(const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void DGEMM_PACK_64(const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                const double *alpha, const double *src, const MKL_INT64 *ld, double *dest);
void DGEMM_COMPUTE_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb, const double *beta, double *c, const MKL_INT64 *ldc);
void DGEMM_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT64 *lda_array, const double **b_array, const MKL_INT64 *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void DGEMM_BATCH_STRIDED_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const double *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const double *beta, double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;

void DGEMMT_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
            const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;

void DSYMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
           const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
void DSYR2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
            const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
void DSYRK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *beta,
           double *c, const MKL_INT64 *ldc) NOTHROW;
void DSYRK_BATCH_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const double *alpha_array, const double **a_array, const MKL_INT64 *lda_array,
           const double *beta_array, double **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void DSYRK_BATCH_STRIDED_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const double *beta,
           double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void DTRMM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           double *b, const MKL_INT64 *ldb) NOTHROW;
void DTRSM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           double *b, const MKL_INT64 *ldb) NOTHROW;
void DTRSM_BATCH_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const double *alpha_array, const double **a_array,
                 const MKL_INT64 *lda_array, double **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void DTRSM_BATCH_STRIDED_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 double *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void ZGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZGEMM_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array, const MKL_Complex16 **b_array, const MKL_INT64 *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ZGEMM_BATCH_STRIDED_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;

void DZGEMM_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const double *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc);
void ZGEMM3M_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
             const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
             const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
             MKL_Complex16 *c, const MKL_INT64 *ldc);
void ZGEMM3M_BATCH_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                   const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array, const MKL_Complex16 **b_array, const MKL_INT64 *ldb_array,
                   const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size);
void ZGEMMT_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZHEMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZHER2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const double *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZHERK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const double *beta, MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZSYMM_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZSYR2K_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZSYRK_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void ZSYRK_BATCH_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array,
           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ZSYRK_BATCH_STRIDED_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void ZTRMM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *ldb) NOTHROW;
void ZTRSM_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *ldb) NOTHROW;
void ZTRSM_BATCH_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                 const MKL_INT64 *lda_array, MKL_Complex16 **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ZTRSM_BATCH_STRIDED_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void GEMM_S8U8S32_64(const char *transa, const char *transb, const char *offsetc,
                  const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                  const float *alpha, const MKL_INT8 *a, const MKL_INT64 *lda, const MKL_INT8 *ao,
                  const MKL_UINT8 *b, const MKL_INT64 *ldb, const MKL_INT8 *bo,
                  const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);
void GEMM_S16S16S32_64(const char *transa, const char *transb, const char *offsetc,
                    const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                    const float *alpha, const MKL_INT16 *a, const MKL_INT64 *lda, const MKL_INT16 *ao,
                    const MKL_INT16 *b, const MKL_INT64 *ldb, const MKL_INT16 *bo,
                    const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);

size_t GEMM_S8U8S32_PACK_GET_SIZE_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
size_t GEMM_S16S16S32_PACK_GET_SIZE_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);

void GEMM_S8U8S32_PACK_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                        const void *src, const MKL_INT64 *ld, void *dest);
void GEMM_S16S16S32_PACK_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                          const MKL_INT16 *src, const MKL_INT64 *ld, MKL_INT16 *dest);
void GEMM_S8U8S32_COMPUTE_64 (const char *transa, const char *transb, const char *offsetc,
                           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                           const float *alpha,
                           const MKL_INT8 *a, const MKL_INT64 *lda, const MKL_INT8 *ao,
                           const MKL_UINT8 *b, const MKL_INT64 *ldb, const MKL_INT8 *bo,
                           const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);
void GEMM_S16S16S32_COMPUTE_64(const char *transa, const char *transb, const char *offsetc,
                            const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                            const float *alpha,
                            const MKL_INT16 *a, const MKL_INT64 *lda, const MKL_INT16 *ao,
                            const MKL_INT16 *b, const MKL_INT64 *ldb, const MKL_INT16 *bo,
                            const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);

void HGEMM_64(const char *transa, const char *transb,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_F16 *alpha, const MKL_F16 *a, const MKL_INT64 *lda,
           const MKL_F16 *b, const MKL_INT64 *ldb,
           const MKL_F16 *beta, MKL_F16 *c, const MKL_INT64 *ldc);
size_t HGEMM_PACK_GET_SIZE_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void HGEMM_PACK_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_F16 *alpha, const MKL_F16 *src, const MKL_INT64 *ld, MKL_F16 *dest);
void HGEMM_COMPUTE_64(const char *transa, const char *transb,
                   const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const MKL_F16 *a, const MKL_INT64 *lda,
                   const MKL_F16 *b, const MKL_INT64 *ldb,
                   const MKL_F16 *beta, MKL_F16 *c, const MKL_INT64 *ldc);

/* Lower case declaration */

/* BLAS Level1 */
float scabs1_64(const MKL_Complex8 *c);
float sasum_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
void saxpy_64(const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
void saxpby_64(const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void saxpyi_64(const MKL_INT64 *nz, const float *a, const float *x, const MKL_INT64 *indx, float *y);
float scasum_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
float scnrm2_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void scopy_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
float sdot_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
float sdoti_64(const MKL_INT64 *nz, const float *x, const MKL_INT64 *indx, const float *y);
float sdsdot_64(const MKL_INT64 *n, const float *sb, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
void sgthr_64(const MKL_INT64 *nz, const float *y, float *x, const MKL_INT64 *indx);
void sgthrz_64(const MKL_INT64 *nz, float *y, float *x, const MKL_INT64 *indx);
float snrm2_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
void srot_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy, const float *c, const float *s) NOTHROW;
void srotg_64(float *a,float *b,float *c,float *s) NOTHROW;
void sroti_64(const MKL_INT64 *nz, float *x, const MKL_INT64 *indx, float *y, const float *c, const float *s);
void srotm_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy, const float *param) NOTHROW;
void srotmg_64(float *d1, float *d2, float *x1, const float *y1, float *param) NOTHROW;
void sscal_64(const MKL_INT64 *n, const float *a, float *x, const MKL_INT64 *incx) NOTHROW;
void ssctr_64(const MKL_INT64 *nz, const float *x, const MKL_INT64 *indx, float *y);
void sswap_64(const MKL_INT64 *n, float *x, const MKL_INT64 *incx, float *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 isamax_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 isamin_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx) NOTHROW;

void caxpy_64(const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void caxpby_64(const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void caxpyi_64(const MKL_INT64 *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT64 *indx, MKL_Complex8 *y);
void ccopy_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cdotc_64(MKL_Complex8 *pres, const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cdotci_64(MKL_Complex8 *pres, const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, const MKL_Complex8 *y);
void cdotu_64(MKL_Complex8 *pres, const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cdotui_64(MKL_Complex8 *pres, const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, const MKL_Complex8 *y);
void cgthr_64(const MKL_INT64 *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT64 *indx);
void cgthrz_64(const MKL_INT64 *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT64 *indx);
void crot_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy, const float *c, const MKL_Complex8 *s) NOTHROW;    
void crotg_64(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s) NOTHROW;
void cscal_64(const MKL_INT64 *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void csctr_64(const MKL_INT64 *nz, const MKL_Complex8 *x, const MKL_INT64 *indx, MKL_Complex8 *y);
void csrot_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy, const float *c, const float *s) NOTHROW;
void csscal_64(const MKL_INT64 *n, const float *a, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void cswap_64(const MKL_INT64 *n, MKL_Complex8 *x, const MKL_INT64 *incx, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 icamax_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 icamin_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;

double dcabs1_64(const MKL_Complex16 *z);
double dasum_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
void daxpy_64(const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
void daxpby_64(const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx, const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void daxpyi_64(const MKL_INT64 *nz, const double *a, const double *x, const MKL_INT64 *indx, double *y);
void dcopy_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
double ddot_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx, const double *y, const MKL_INT64 *incy) NOTHROW;
double dsdot_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx, const float *y, const MKL_INT64 *incy) NOTHROW;
double ddoti_64(const MKL_INT64 *nz, const double *x, const MKL_INT64 *indx, const double *y);
void dgthr_64(const MKL_INT64 *nz, const double *y, double *x, const MKL_INT64 *indx);
void dgthrz_64(const MKL_INT64 *nz, double *y, double *x, const MKL_INT64 *indx);
double dnrm2_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
void drot_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy, const double *c, const double *s) NOTHROW;
void drotg_64(double *a, double *b, double *c, double *s) NOTHROW;
void droti_64(const MKL_INT64 *nz, double *x, const MKL_INT64 *indx, double *y, const double *c, const double *s);
void drotm_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy, const double *param) NOTHROW;
void drotmg_64(double *d1, double *d2, double *x1, const double *y1, double *param) NOTHROW;
void dscal_64(const MKL_INT64 *n, const double *a, double *x, const MKL_INT64 *incx) NOTHROW;
void dsctr_64(const MKL_INT64 *nz, const double *x, const MKL_INT64 *indx, double *y);
void dswap_64(const MKL_INT64 *n, double *x, const MKL_INT64 *incx, double *y, const MKL_INT64 *incy) NOTHROW;
double dzasum_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
double dznrm2_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 idamax_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 idamin_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx) NOTHROW;

void zaxpy_64(const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zaxpby_64(const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zaxpyi_64(const MKL_INT64 *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT64 *indx, MKL_Complex16 *y);
void zcopy_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zdotc_64(MKL_Complex16 *pres, const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zdotci_64(MKL_Complex16 *pres, const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, const MKL_Complex16 *y);
void zdotu_64(MKL_Complex16 *pres, const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zdotui_64(MKL_Complex16 *pres, const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, const MKL_Complex16 *y);
void zdrot_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy, const double *c, const double *s) NOTHROW;
void zdscal_64(const MKL_INT64 *n, const double *a, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void zgthr_64(const MKL_INT64 *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT64 *indx);
void zgthrz_64(const MKL_INT64 *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT64 *indx);
void zrot_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy, const double *c, const MKL_Complex16 *s) NOTHROW;    
void zrotg_64(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s) NOTHROW;
void zscal_64(const MKL_INT64 *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void zsctr_64(const MKL_INT64 *nz, const MKL_Complex16 *x, const MKL_INT64 *indx, MKL_Complex16 *y);
void zswap_64(const MKL_INT64 *n, MKL_Complex16 *x, const MKL_INT64 *incx, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
MKL_INT64 izamax_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
MKL_INT64 izamin_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;

/* blas level2 */

void sgbmv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void sgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
           const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void sger_64(const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
          const float *y, const MKL_INT64 *incy, float *a, const MKL_INT64 *lda) NOTHROW;
void ssbmv_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const float *alpha,
           const float *a, const MKL_INT64 *lda, const float *x, const MKL_INT64 *incx,
           const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void sspmv_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *ap,
           const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void sspr_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
          float *ap) NOTHROW;
void sspr2_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
           const float *y, const MKL_INT64 *incy, float *ap) NOTHROW;
void ssymv_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           const float *x, const MKL_INT64 *incx, const float *beta, float *y, const MKL_INT64 *incy) NOTHROW;
void ssyr_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
          float *a, const MKL_INT64 *lda) NOTHROW;
void ssyr2_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const float *x, const MKL_INT64 *incx,
           const float *y, const MKL_INT64 *incy, float *a, const MKL_INT64 *lda) NOTHROW;
void stbmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void stbsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void stpmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const float *ap,
           float *x, const MKL_INT64 *incx) NOTHROW;
void stpsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const float *ap,
           float *x, const MKL_INT64 *incx) NOTHROW;
void strmv_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n, const float *a,
           const MKL_INT64 *lda, float *b, const MKL_INT64 *incx) NOTHROW;
void strsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const float *a, const MKL_INT64 *lda, float *x, const MKL_INT64 *incx) NOTHROW;
void sgem2vu_64(const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
             const float *x1, const MKL_INT64 *incx1, const float *x2, const MKL_INT64 *incx2,
             const float *beta, float *y1, const MKL_INT64 *incy1, float *y2, const MKL_INT64 *incy2);

void cgbmv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cgerc_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void cgeru_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void chbmv_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void chemv_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void cher_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx,
          MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void cher2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *y, const MKL_INT64 *incy,
           MKL_Complex8 *a, const MKL_INT64 *lda) NOTHROW;
void chpmv_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
           const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT64 *incy) NOTHROW;
void chpr_64(const char *uplo, const MKL_INT64 *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx,
          MKL_Complex8 *ap) NOTHROW;
void chpr2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT64 *incx,
           const MKL_Complex8 *y, const MKL_INT64 *incy, MKL_Complex8 *ap) NOTHROW;
void ctbmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void ctbsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void ctpmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void ctpsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void ctrmv_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *b, const MKL_INT64 *incx) NOTHROW;
void ctrsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *x, const MKL_INT64 *incx) NOTHROW;
void cgem2vc_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
             const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_Complex8 *x1, const MKL_INT64 *incx1,
             const MKL_Complex8 *x2, const MKL_INT64 *incx2, const MKL_Complex8 *beta,
             MKL_Complex8 *y1, const MKL_INT64 *incy1, MKL_Complex8 *y2, const MKL_INT64 *incy2);
void scgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
            const float *a, const MKL_INT64 *lda, const MKL_Complex8 *x, const MKL_INT64 *incx,
            const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy);

void dgbmv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void dgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
           const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void dger_64(const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
          const double *y, const MKL_INT64 *incy, double *a, const MKL_INT64 *lda) NOTHROW;
void dsbmv_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const double *alpha,
           const double *a, const MKL_INT64 *lda, const double *x, const MKL_INT64 *incx,
           const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void dspmv_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *ap,
           const double *x, const MKL_INT64 *incx, const double *beta,
           double *y, const MKL_INT64 *incy) NOTHROW;
void dspr_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
          double *ap) NOTHROW;
void dspr2_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
           const double *y, const MKL_INT64 *incy, double *ap) NOTHROW;
void dsymv_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           const double *x, const MKL_INT64 *incx, const double *beta, double *y, const MKL_INT64 *incy) NOTHROW;
void dsyr_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
          double *a, const MKL_INT64 *lda) NOTHROW;
void dsyr2_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const double *x, const MKL_INT64 *incx,
           const double *y, const MKL_INT64 *incy, double *a, const MKL_INT64 *lda) NOTHROW;
void dtbmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void dtbsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void dtpmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *ap, double *x, const MKL_INT64 *incx) NOTHROW;
void dtpsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *ap, double *x, const MKL_INT64 *incx) NOTHROW;
void dtrmv_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const double *a, const MKL_INT64 *lda, double *b, const MKL_INT64 *incx) NOTHROW;
void dtrsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const double *a, const MKL_INT64 *lda, double *x, const MKL_INT64 *incx) NOTHROW;
void dgem2vu_64(const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
             const double *a, const MKL_INT64 *lda, const double *x1, const MKL_INT64 *incx1,
             const double *x2, const MKL_INT64 *incx2, const double *beta,
             double *y1, const MKL_INT64 *incy1, double *y2, const MKL_INT64 *incy2);

void zgbmv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *kl, const MKL_INT64 *ku,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zgerc_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *y, const MKL_INT64 *incy, MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void zgeru_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *y, const MKL_INT64 *incy, MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void zhbmv_64(const char *uplo, const MKL_INT64 *n, const MKL_INT64 *k, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zhemv_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zher_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx,
          MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void zher2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *y, const MKL_INT64 *incy,
           MKL_Complex16 *a, const MKL_INT64 *lda) NOTHROW;
void zhpmv_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
           const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT64 *incy) NOTHROW;
void zhpr_64(const char *uplo, const MKL_INT64 *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx,
          MKL_Complex16 *ap) NOTHROW;
void zhpr2_64(const char *uplo, const MKL_INT64 *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT64 *incx,
           const MKL_Complex16 *y, const MKL_INT64 *incy, MKL_Complex16 *ap) NOTHROW;
void ztbmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ztbsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ztpmv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ztpsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void ztrmv_64(const char *uplo, const char *transa, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *incx) NOTHROW;
void ztrsv_64(const char *uplo, const char *trans, const char *diag, const MKL_INT64 *n,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *x, const MKL_INT64 *incx) NOTHROW;
void zgem2vc_64(const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
             const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_Complex16 *x1, const MKL_INT64 *incx1,
             const MKL_Complex16 *x2, const MKL_INT64 *incx2, const MKL_Complex16 *beta,
             MKL_Complex16 *y1, const MKL_INT64 *incy1, MKL_Complex16 *y2, const MKL_INT64 *incy2);
void dzgemv_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
            const double *a, const MKL_INT64 *lda, const MKL_Complex16 *x, const MKL_INT64 *incx,
            const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy);

/* blas level3 */

void sgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
           const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
size_t sgemm_pack_get_size_64(const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void sgemm_pack_64(const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                const float *alpha, const float *src, const MKL_INT64 *ld, float *dest);
void sgemm_compute_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb, const float *beta, float *c, const MKL_INT64 *ldc);
void sgemm_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT64 *lda_array, const float **b_array, const MKL_INT64 *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void sgemm_batch_strided_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const float *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const float *beta, float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;

void sgemmt_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
            const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void ssymm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
           const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void ssyr2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const float *alpha, const float *a, const MKL_INT64 *lda, const float *b, const MKL_INT64 *ldb,
            const float *beta, float *c, const MKL_INT64 *ldc) NOTHROW;
void ssyrk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const float *beta,
           float *c, const MKL_INT64 *ldc) NOTHROW;
void ssyrk_batch_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const float *alpha_array, const float **a_array, const MKL_INT64 *lda_array, const float *beta_array,
           float **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ssyrk_batch_strided_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const float *beta,
           float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void strmm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           float *b, const MKL_INT64 *ldb) NOTHROW;
void strsm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha, const float *a, const MKL_INT64 *lda,
           float *b, const MKL_INT64 *ldb) NOTHROW;
void strsm_batch_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const float *alpha_array, const float **a_array,
                 const MKL_INT64 *lda_array, float **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void strsm_batch_strided_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const float *alpha, const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 float *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void cgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void cgemm_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array, const MKL_Complex8 **b_array, const MKL_INT64 *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void cgemm_batch_strided_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;

void scgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const float *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc);
void cgemm3m_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
             const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
             const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
             MKL_Complex8 *c, const MKL_INT64 *ldc);
void cgemm3m_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                   const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array, const MKL_Complex8 **b_array, const MKL_INT64 *ldb_array,
                   const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size);
void cgemmt_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void chemm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void cher2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const float *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void cherk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const float *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const float *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void csymm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void csyr2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
            const MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void csyrk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT64 *ldc) NOTHROW;
void csyrk_batch_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT64 *lda_array,
           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void csyrk_batch_strided_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void ctrmm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *b, const MKL_INT64 *ldb) NOTHROW;
void ctrsm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT64 *lda, MKL_Complex8 *b, const MKL_INT64 *ldb) NOTHROW;
void ctrsm_batch_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                 const MKL_INT64 *lda_array, MKL_Complex8 **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ctrsm_batch_strided_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 MKL_Complex8 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void dgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
           const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
size_t dgemm_pack_get_size_64(const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void dgemm_pack_64(const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                const double *alpha, const double *src, const MKL_INT64 *ld, double *dest);
void dgemm_compute_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb, const double *beta, double *c, const MKL_INT64 *ldc);
void dgemm_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT64 *lda_array, const double **b_array, const MKL_INT64 *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void dgemm_batch_strided_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const double *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const double *beta, double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;
void dgemmt_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
            const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
void dsymm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
           const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
void dsyr2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const double *alpha, const double *a, const MKL_INT64 *lda, const double *b, const MKL_INT64 *ldb,
            const double *beta, double *c, const MKL_INT64 *ldc) NOTHROW;
void dsyrk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const double *beta,
           double *c, const MKL_INT64 *ldc) NOTHROW;
void dsyrk_batch_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const double *alpha_array, const double **a_array, const MKL_INT64 *lda_array, const double *beta_array,
           double **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void dsyrk_batch_strided_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const double *beta,
           double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void dtrmm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           double *b, const MKL_INT64 *ldb) NOTHROW;
void dtrsm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha, const double *a, const MKL_INT64 *lda,
           double *b, const MKL_INT64 *ldb) NOTHROW;
void dtrsm_batch_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const double *alpha_array, const double **a_array,
                 const MKL_INT64 *lda_array, double **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void dtrsm_batch_strided_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const double *alpha, const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 double *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void zgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zgemm_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array, const MKL_Complex16 **b_array, const MKL_INT64 *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void zgemm_batch_strided_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                 const MKL_INT64 *batch_size) NOTHROW;
void dzgemm_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const double *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc);
void zgemm3m_64(const char *transa, const char *transb, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
             const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
             const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
             MKL_Complex16 *c, const MKL_INT64 *ldc);
void zgemm3m_batch_64(const char *transa_array, const char *transb_array, const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
                   const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array, const MKL_Complex16 **b_array, const MKL_INT64 *ldb_array,
                   const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size);
void zgemmt_64(const char *uplo, const char *transa, const char *transb, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zhemm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zher2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const double *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zherk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const double *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const double *beta, MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zsymm_64(const char *side, const char *uplo, const MKL_INT64 *m, const MKL_INT64 *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zsyr2k_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
            const MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zsyrk_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda,
           const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT64 *ldc) NOTHROW;
void zsyrk_batch_64(const char *uplo_array, const char *trans_array, const MKL_INT64 *n_array, const MKL_INT64 *k_array,
           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT64 *lda_array,
           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT64 *ldc_array, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void zsyrk_batch_strided_64(const char *uplo, const char *trans, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec, const MKL_INT64 *batch_size) NOTHROW;
void ztrmm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *ldb) NOTHROW;
void ztrsm_64(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT64 *lda, MKL_Complex16 *b, const MKL_INT64 *ldb) NOTHROW;
void ztrsm_batch_64(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT64 *m_array, const MKL_INT64 *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                 const MKL_INT64 *lda_array, MKL_Complex16 **b_array, const MKL_INT64 *ldb, const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void ztrsm_batch_strided_64(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                 MKL_Complex16 *b, const MKL_INT64 *ldb, const MKL_INT64 *strideb,
                 const MKL_INT64 *batch_size) NOTHROW;

void gemm_s16s16s32_64(const char *transa, const char *transb, const char *offsetc,
                    const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                    const float *alpha, const MKL_INT16 *a, const MKL_INT64 *lda, const MKL_INT16 *ao,
                    const MKL_INT16 *b, const MKL_INT64 *ldb, const MKL_INT16 *bo,
                    const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);
void gemm_s8u8s32_64(const char *transa, const char *transb, const char *offsetc,
                  const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                  const float *alpha, const MKL_INT8 *a, const MKL_INT64 *lda, const MKL_INT8 *ao,
                  const MKL_UINT8 *b, const MKL_INT64 *ldb, const MKL_INT8 *bo,
                  const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);

size_t gemm_s8u8s32_pack_get_size_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
size_t gemm_s16s16s32_pack_get_size_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void gemm_s8u8s32_pack_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                        const void *src, const MKL_INT64 *ld, void *dest);
void gemm_s16s16s32_pack_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                          const MKL_INT16 *src, const MKL_INT64 *ld, MKL_INT16 *dest);
void gemm_s8u8s32_compute_64 (const char *transa, const char *transb, const char *offsetc,
                           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                           const float *alpha,
                           const MKL_INT8 *a, const MKL_INT64 *lda, const MKL_INT8 *ao,
                           const MKL_UINT8 *b, const MKL_INT64 *ldb, const MKL_INT8 *bo,
                           const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);
void gemm_s16s16s32_compute_64(const char *transa, const char *transb, const char *offsetc,
                            const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                            const float *alpha,
                            const MKL_INT16 *a, const MKL_INT64 *lda, const MKL_INT16 *ao,
                            const MKL_INT16 *b, const MKL_INT64 *ldb, const MKL_INT16 *bo,
                            const float *beta, MKL_INT32 *c, const MKL_INT64 *ldc, const MKL_INT32 *co);

void hgemm_64(const char *transa, const char *transb,
           const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
           const MKL_F16 *alpha, const MKL_F16 *a, const MKL_INT64 *lda,
           const MKL_F16 *b, const MKL_INT64 *ldb,
           const MKL_F16 *beta, MKL_F16 *c, const MKL_INT64 *ldc);
size_t hgemm_pack_get_size_64 (const char *identifier, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k);
void hgemm_pack_64 (const char *identifier, const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                 const MKL_F16 *alpha, const MKL_F16 *src, const MKL_INT64 *ld, MKL_F16 *dest);
void hgemm_compute_64(const char *transa, const char *transb,
                   const MKL_INT64 *m, const MKL_INT64 *n, const MKL_INT64 *k,
                   const MKL_F16 *a, const MKL_INT64 *lda,
                   const MKL_F16 *b, const MKL_INT64 *ldb,
                   const MKL_F16 *beta, MKL_F16 *c, const MKL_INT64 *ldc);


/* Level1 BLAS batch API */

void SAXPY_BATCH_64(const MKL_INT64 *n, const float *alpha,
                 const float **x, const MKL_INT64 *incx,
                 float **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void saxpy_batch_64(const MKL_INT64 *n, const float *alpha,
                 const float **x, const MKL_INT64 *incx,
                 float **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void DAXPY_BATCH_64(const MKL_INT64 *n, const double *alpha,
                 const double **x, const MKL_INT64 *incx,
                 double **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void daxpy_batch_64(const MKL_INT64 *n, const double *alpha,
                 const double **x, const MKL_INT64 *incx,
                 double **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void CAXPY_BATCH_64(const MKL_INT64 *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void caxpy_batch_64(const MKL_INT64 *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void ZAXPY_BATCH_64(const MKL_INT64 *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;
void zaxpy_batch_64(const MKL_INT64 *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void SCOPY_BATCH_64(const MKL_INT64 *n, const float **x, const MKL_INT64 *incx,
                 float **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;
void scopy_batch_64(const MKL_INT64 *n, const float **x, const MKL_INT64 *incx,
                 float **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;

void DCOPY_BATCH_64(const MKL_INT64 *n, const double **x, const MKL_INT64 *incx,
                 double **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;
void dcopy_batch_64(const MKL_INT64 *n, const double **x, const MKL_INT64 *incx,
                 double **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;

void CCOPY_BATCH_64(const MKL_INT64 *n, const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;
void ccopy_batch_64(const MKL_INT64 *n, const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;

void ZCOPY_BATCH_64(const MKL_INT64 *n, const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;
void zcopy_batch_64(const MKL_INT64 *n, const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **y, const MKL_INT64 *incy, const MKL_INT64 *group_count,
                 const MKL_INT64 *group_size) NOTHROW;

void SAXPY_BATCH_STRIDED_64(const MKL_INT64 *n, const float *alpha,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         float *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;
void saxpy_batch_strided_64(const MKL_INT64 *n, const float *alpha,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         float *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void DAXPY_BATCH_STRIDED_64(const MKL_INT64 *n, const double *alpha,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         double *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;
void daxpy_batch_strided_64(const MKL_INT64 *n, const double *alpha,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         double *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void CAXPY_BATCH_STRIDED_64(const MKL_INT64 *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex8 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;
void caxpy_batch_strided_64(const MKL_INT64 *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex8 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void ZAXPY_BATCH_STRIDED_64(const MKL_INT64 *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex16 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;
void zaxpy_batch_strided_64(const MKL_INT64 *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex16 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void SCOPY_BATCH_STRIDED_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, float *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;
void scopy_batch_strided_64(const MKL_INT64 *n, const float *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, float *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;

void DCOPY_BATCH_STRIDED_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, double *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;
void dcopy_batch_strided_64(const MKL_INT64 *n, const double *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, double *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;

void CCOPY_BATCH_STRIDED_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, MKL_Complex8 *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;
void ccopy_batch_strided_64(const MKL_INT64 *n, const MKL_Complex8 *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, MKL_Complex8 *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;

void ZCOPY_BATCH_STRIDED_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, MKL_Complex16 *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;
void zcopy_batch_strided_64(const MKL_INT64 *n, const MKL_Complex16 *x, const MKL_INT64 *incx,
                         const MKL_INT64 *stridex, MKL_Complex16 *y, const MKL_INT64 *incy,
                         const MKL_INT64 *stridey, const MKL_INT64 *batch_size) NOTHROW;

/* Level2 BLAS batch API */
void sgemv_batch_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
                 const float **a, const MKL_INT64 *lda, const float **x, const MKL_INT64 *incx,
                 const float *beta, float **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void sgemv_batch_strided_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
                         const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const float *beta, float *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void dgemv_batch_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
                 const double **a, const MKL_INT64 *lda, const double **x, const MKL_INT64 *incx,
                 const double *beta, double **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void dgemv_batch_strided_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
                         const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const double *beta, double *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void cgemv_batch_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT64 *lda, const MKL_Complex8 **x, const MKL_INT64 *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void cgemv_batch_strided_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void zgemv_batch_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT64 *lda, const MKL_Complex16 **x, const MKL_INT64 *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void zgemv_batch_strided_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;


void SGEMV_BATCH_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
                 const float **a, const MKL_INT64 *lda, const float **x, const MKL_INT64 *incx,
                 const float *beta, float **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void SGEMV_BATCH_STRIDED_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const float *alpha,
                         const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const float *beta, float *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void DGEMV_BATCH_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
                 const double **a, const MKL_INT64 *lda, const double **x, const MKL_INT64 *incx,
                 const double *beta, double **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void DGEMV_BATCH_STRIDED_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const double *alpha,
                         const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const double *beta, double *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void CGEMV_BATCH_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT64 *lda, const MKL_Complex8 **x, const MKL_INT64 *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void CGEMV_BATCH_STRIDED_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void ZGEMV_BATCH_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT64 *lda, const MKL_Complex16 **x, const MKL_INT64 *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT64 *incy,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void ZGEMV_BATCH_STRIDED_64(const char *trans, const MKL_INT64 *m, const MKL_INT64 *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT64 *incy, const MKL_INT64 *stridey,
                         const MKL_INT64 *batch_size) NOTHROW;

void sdgmm_batch_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const float **a, const MKL_INT64 *lda,
                 const float **x, const MKL_INT64 *incx,
                 float **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void sdgmm_batch_strided_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void ddgmm_batch_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const double **a, const MKL_INT64 *lda,
                 const double **x, const MKL_INT64 *incx,
                 double **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void ddgmm_batch_strided_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void cdgmm_batch_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex8 **a, const MKL_INT64 *lda,
                 const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void cdgmm_batch_strided_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void zdgmm_batch_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex16 **a, const MKL_INT64 *lda,
                 const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void zdgmm_batch_strided_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void SDGMM_BATCH_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const float **a, const MKL_INT64 *lda,
                 const float **x, const MKL_INT64 *incx,
                 float **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void SDGMM_BATCH_STRIDED_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const float *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const float *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         float *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void DDGMM_BATCH_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const double **a, const MKL_INT64 *lda,
                 const double **x, const MKL_INT64 *incx,
                 double **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void DDGMM_BATCH_STRIDED_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const double *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const double *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         double *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void CDGMM_BATCH_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex8 **a, const MKL_INT64 *lda,
                 const MKL_Complex8 **x, const MKL_INT64 *incx,
                 MKL_Complex8 **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void CDGMM_BATCH_STRIDED_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const MKL_Complex8 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex8 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex8 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

void ZDGMM_BATCH_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                 const MKL_Complex16 **a, const MKL_INT64 *lda,
                 const MKL_Complex16 **x, const MKL_INT64 *incx,
                 MKL_Complex16 **c, const MKL_INT64 *ldc,
                 const MKL_INT64 *group_count, const MKL_INT64 *group_size) NOTHROW;

void ZDGMM_BATCH_STRIDED_64(const char *side, const MKL_INT64 *m, const MKL_INT64 *n,
                         const MKL_Complex16 *a, const MKL_INT64 *lda, const MKL_INT64 *stridea,
                         const MKL_Complex16 *x, const MKL_INT64 *incx, const MKL_INT64 *stridex,
                         MKL_Complex16 *c, const MKL_INT64 *ldc, const MKL_INT64 *stridec,
                         const MKL_INT64 *batch_size) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_BLAS_64_H_ */
