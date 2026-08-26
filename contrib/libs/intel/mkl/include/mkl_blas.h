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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) interface for BLAS routines
!******************************************************************************/

#ifndef _MKL_BLAS_H_
#define _MKL_BLAS_H_
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

void XERBLA(const char *srname, const int *info, const int lsrname);
int LSAME(const char *ca, const char *cb, const MKL_INT lca, const MKL_INT lcb);

/* BLAS Level1 */

float SCABS1(const MKL_Complex8 *c);
float SASUM(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
void SAXPY(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
void SAXPBY(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SAXPYI(const MKL_INT *nz, const float *a, const float *x, const MKL_INT *indx,float *y);
float SCASUM(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
float SCNRM2(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void SCOPY(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
float SDOT(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
float SDSDOT(const MKL_INT *n, const float *sb, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
float SDOTI(const MKL_INT *nz, const float *x, const MKL_INT *indx, const float *y);
void SGTHR(const MKL_INT *nz, const float *y, float *x, const MKL_INT *indx);
void SGTHRZ(const MKL_INT *nz, float *y, float *x, const MKL_INT *indx);
float SNRM2(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
void SROT(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *c, const float *s) NOTHROW;
void SROTG(float *a,float *b,float *c,float *s) NOTHROW;
void SROTI(const MKL_INT *nz, float *x, const MKL_INT *indx, float *y, const float *c, const float *s);
void SROTM(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *param) NOTHROW;
void SROTMG(float *d1, float *d2, float *x1, const float *y1, float *param) NOTHROW;
void SSCAL(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx) NOTHROW;
void SSCTR(const MKL_INT *nz, const float *x, const MKL_INT *indx, float *y);
void SSWAP(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
MKL_INT ISAMAX(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
MKL_INT ISAMIN(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

void CAXPY(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CAXPBY(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void CAXPYI(const MKL_INT *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void CCOPY(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CDOTC(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CDOTCI(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void CDOTU(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CDOTUI(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void CGTHR(const MKL_INT *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void CGTHRZ(const MKL_INT *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void CROT(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const MKL_Complex8 *s) NOTHROW;
void CROTG(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s) NOTHROW;
void CSCAL(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CSCTR(const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void CSROT(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const float *s) NOTHROW;
void CSSCAL(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CSWAP(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
MKL_INT ICAMAX(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
MKL_INT ICAMIN(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

double DCABS1(const MKL_Complex16 *z);
double DASUM(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
void DAXPY(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
void DAXPBY(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void DAXPYI(const MKL_INT *nz, const double *a, const double *x, const MKL_INT *indx, double *y);
void DCOPY(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
double DDOT(const MKL_INT *n, const double *x, const MKL_INT *incx, const double *y, const MKL_INT *incy) NOTHROW;
double DSDOT(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
double DDOTI(const MKL_INT *nz, const double *x, const MKL_INT *indx, const double *y);
void DGTHR(const MKL_INT *nz, const double *y, double *x, const MKL_INT *indx);
void DGTHRZ(const MKL_INT *nz, double *y, double *x, const MKL_INT *indx);
double DNRM2(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
void DROT(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *c, const double *s) NOTHROW;
void DROTG(double *a,double *b,double *c,double *s) NOTHROW;
void DROTI(const MKL_INT *nz, double *x, const MKL_INT *indx, double *y, const double *c, const double *s);
void DROTM(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *param) NOTHROW;
void DROTMG(double *d1, double *d2, double *x1, const double *y1, double *param) NOTHROW;
void DSCAL(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx) NOTHROW;
void DSCTR(const MKL_INT *nz, const double *x, const MKL_INT *indx, double *y);
void DSWAP(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
double DZASUM(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
double DZNRM2(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
MKL_INT IDAMAX(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
MKL_INT IDAMIN(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

void ZAXPY(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZAXPBY(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void ZAXPYI(const MKL_INT *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void ZCOPY(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZDOTC(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZDOTCI(MKL_Complex16 *pres,const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void ZDOTU(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZDOTUI(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void ZDROT(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const double *s) NOTHROW;
void ZDSCAL(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZGTHR(const MKL_INT *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void ZGTHRZ(const MKL_INT *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void ZROT(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const MKL_Complex16 *s) NOTHROW;
void ZROTG(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s) NOTHROW;
void ZSCAL(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZSCTR(const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void ZSWAP(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
MKL_INT IZAMAX(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
MKL_INT IZAMIN(const MKL_INT *n,const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

/* BLAS Level2 */

void SGBMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
           const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SGER(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
          const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda) NOTHROW;
void SSBMV(const char *uplo, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SSPMV(const char *uplo, const MKL_INT *n, const float *alpha, const float *ap,
           const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SSPR(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *ap) NOTHROW;
void SSPR2(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *ap) NOTHROW;
void SSYMV(const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void SSYR(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
          float *a, const MKL_INT *lda) NOTHROW;
void SSYR2(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda) NOTHROW;
void STBMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void STBSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void STPMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const float *ap, float *x, const MKL_INT *incx) NOTHROW;
void STPSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const float *ap, float *x, const MKL_INT *incx) NOTHROW;
void STRMV(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const float *a, const MKL_INT *lda, float *b, const MKL_INT *incx) NOTHROW;
void STRSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void SGEM2VU(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
             const float *x1, const MKL_INT *incx1, const float *x2, const MKL_INT *incx2,
             const float *beta, float *y1, const MKL_INT *incy1, float *y2, const MKL_INT *incy2);

void CGBMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CGERC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void CGERU(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void CHBMV(const char *uplo, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CHEMV(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CHER(const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
          MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void CHER2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void CHPMV(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void CHPR(const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
          MKL_Complex8 *ap) NOTHROW;
void CHPR2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *ap) NOTHROW;
void CTBMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CTBSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CTPMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CTPSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CTRMV(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *incx) NOTHROW;
void CTRSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void CGEM2VC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
             const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x1, const MKL_INT *incx1,
             const MKL_Complex8 *x2, const MKL_INT *incx2, const MKL_Complex8 *beta,
             MKL_Complex8 *y1, const MKL_INT *incy1, MKL_Complex8 *y2, const MKL_INT *incy2);
void SCGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
            const float *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
            const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);

void DGBMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const double *alpha, const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void DGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
           const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void DGER(const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
          const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda) NOTHROW;
void DSBMV(const char *uplo, const MKL_INT *n, const MKL_INT *k, const double *alpha,
           const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void DSPMV(const char *uplo, const MKL_INT *n, const double *alpha, const double *ap,
           const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void DSPR(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *ap) NOTHROW;
void DSPR2(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *ap) NOTHROW;
void DSYMV(const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void DSYR(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
          double *a, const MKL_INT *lda) NOTHROW;
void DSYR2(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda) NOTHROW;
void DTBMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void DTBSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void DTPMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *ap, double *x, const MKL_INT *incx) NOTHROW;
void DTPSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *ap, double *x, const MKL_INT *incx) NOTHROW;
void DTRMV(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const double *a, const MKL_INT *lda, double *b, const MKL_INT *incx) NOTHROW;
void DTRSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void DGEM2VU(const MKL_INT *m, const MKL_INT *n, const double *alpha,
             const double *a, const MKL_INT *lda, const double *x1, const MKL_INT *incx1,
             const double *x2, const MKL_INT *incx2, const double *beta,
             double *y1, const MKL_INT *incy1, double *y2, const MKL_INT *incy2);

void ZGBMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZGERC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void ZGERU(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void ZHBMV(const char *uplo, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZHEMV(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZHER(const char *uplo, const MKL_INT *n, const double *alpha,
          const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void ZHER2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void ZHPMV(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void ZHPR(const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x,
          const MKL_INT *incx, MKL_Complex16 *ap) NOTHROW;
void ZHPR2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *ap) NOTHROW;
void ZTBMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZTBSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZTPMV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZTPSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZTRMV(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *incx) NOTHROW;
void ZTRSV(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ZGEM2VC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
             const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x1, const MKL_INT *incx1,
             const MKL_Complex16 *x2, const MKL_INT *incx2, const MKL_Complex16 *beta,
             MKL_Complex16 *y1, const MKL_INT *incy1, MKL_Complex16 *y2, const MKL_INT *incy2);
void DZGEMV(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
            const double *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
            const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);

/* BLAS Level3 */

void SGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
size_t SGEMM_PACK_GET_SIZE(const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void SGEMM_PACK(const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                const float *alpha, const float *src, const MKL_INT *ld, float *dest);
void SGEMM_COMPUTE(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void SGEMM_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float **b_array, const MKL_INT *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void SGEMM_BATCH_STRIDED(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const float *beta, float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;
void SGEMMT(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void SSYMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void SSYR2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void SSYRK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void SSYRK_BATCH_STRIDED(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea, const float *beta,
           float *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void SSYRK_BATCH(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float *beta_array,
           float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void STRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;
void STRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;
void STRSM_BATCH(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float **a_array,
                 const MKL_INT *lda_array, float **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void STRSM_BATCH_STRIDED(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void CGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;

void CGEMM_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void CGEMM_BATCH_STRIDED(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;
void SCGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const float *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc);
void CGEMM3M(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
             const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
             const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
             MKL_Complex8 *c, const MKL_INT *ldc);
void CGEMM3M_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                   const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                   const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void CGEMMT(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb,
            const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CTRSM_BATCH(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                 const MKL_INT *lda_array, MKL_Complex8 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void CTRSM_BATCH_STRIDED(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void CHEMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CHER2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CHERK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const float *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CSYMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *b, const MKL_INT *ldb,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CSYR2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb,
            const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CSYRK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void CSYRK_BATCH(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void CSYRK_BATCH_STRIDED(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void CTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;
void CTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda,
           MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;

void DGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
size_t DGEMM_PACK_GET_SIZE(const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void DGEMM_PACK(const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                const double *alpha, const double *src, const MKL_INT *ld, double *dest);
void DGEMM_COMPUTE(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void DGEMM_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT *lda_array, const double **b_array, const MKL_INT *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void DGEMM_BATCH_STRIDED(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const double *beta, double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

void DGEMMT(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;

void DSYMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
void DSYR2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
void DSYRK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const double *beta,
           double *c, const MKL_INT *ldc) NOTHROW;
void DSYRK_BATCH(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const double *alpha_array, const double **a_array, const MKL_INT *lda_array,
           const double *beta_array, double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void DSYRK_BATCH_STRIDED(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea, const double *beta,
           double *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void DTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;
void DTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;
void DTRSM_BATCH(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double **a_array,
                 const MKL_INT *lda_array, double **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void DTRSM_BATCH_STRIDED(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void ZGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZGEMM_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ZGEMM_BATCH_STRIDED(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

void DZGEMM(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const double *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc);
void ZGEMM3M(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
             const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
             const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
             MKL_Complex16 *c, const MKL_INT *ldc);
void ZGEMM3M_BATCH(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                   const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                   const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void ZGEMMT(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZHEMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZHER2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZHERK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const double *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZSYMM(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZSYR2K(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZSYRK(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void ZSYRK_BATCH(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ZSYRK_BATCH_STRIDED(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void ZTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;
void ZTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;
void ZTRSM_BATCH(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                 const MKL_INT *lda_array, MKL_Complex16 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ZTRSM_BATCH_STRIDED(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void GEMM_S8U8S32(const char *transa, const char *transb, const char *offsetc,
                  const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                  const float *alpha, const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao,
                  const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo,
                  const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co) NOTHROW;
void GEMM_S16S16S32(const char *transa, const char *transb, const char *offsetc,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                    const float *alpha, const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao,
                    const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo,
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);

size_t GEMM_S8U8S32_PACK_GET_SIZE (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t GEMM_S16S16S32_PACK_GET_SIZE (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);

void GEMM_S8U8S32_PACK (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                        const void *src, const MKL_INT *ld, void *dest);
void GEMM_S16S16S32_PACK (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          const MKL_INT16 *src, const MKL_INT *ld, MKL_INT16 *dest);
void GEMM_S8U8S32_COMPUTE (const char *transa, const char *transb, const char *offsetc,
                           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                           const float *alpha,
                           const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao,
                           const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo,
                           const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void GEMM_S16S16S32_COMPUTE(const char *transa, const char *transb, const char *offsetc,
                            const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                            const float *alpha,
                            const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao,
                            const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo,
                            const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);

void HGEMM(const char *transa, const char *transb,
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_F16 *alpha, const MKL_F16 *a, const MKL_INT *lda,
           const MKL_F16 *b, const MKL_INT *ldb,
           const MKL_F16 *beta, MKL_F16 *c, const MKL_INT *ldc) NOTHROW;
size_t HGEMM_PACK_GET_SIZE (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void HGEMM_PACK (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_F16 *alpha, const MKL_F16 *src, const MKL_INT *ld, MKL_F16 *dest);
void HGEMM_COMPUTE(const char *transa, const char *transb,
                   const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const MKL_F16 *a, const MKL_INT *lda,
                   const MKL_F16 *b, const MKL_INT *ldb,
                   const MKL_F16 *beta, MKL_F16 *c, const MKL_INT *ldc);

/* Lower case declaration */

void xerbla(const char *srname, const int *info, const int lsrname);
int lsame(const char *ca, const char *cb, const MKL_INT lca, const MKL_INT lcb);

/* BLAS Level1 */
float scabs1(const MKL_Complex8 *c);
float sasum(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
void saxpy(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
void saxpby(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void saxpyi(const MKL_INT *nz, const float *a, const float *x, const MKL_INT *indx, float *y);
float scasum(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
float scnrm2(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void scopy(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
float sdot(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
float sdoti(const MKL_INT *nz, const float *x, const MKL_INT *indx, const float *y);
float sdsdot(const MKL_INT *n, const float *sb, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
void sgthr(const MKL_INT *nz, const float *y, float *x, const MKL_INT *indx);
void sgthrz(const MKL_INT *nz, float *y, float *x, const MKL_INT *indx);
float snrm2(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
void srot(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *c, const float *s) NOTHROW;
void srotg(float *a,float *b,float *c,float *s) NOTHROW;
void sroti(const MKL_INT *nz, float *x, const MKL_INT *indx, float *y, const float *c, const float *s);
void srotm(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *param) NOTHROW;
void srotmg(float *d1, float *d2, float *x1, const float *y1, float *param) NOTHROW;
void sscal(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx) NOTHROW;
void ssctr(const MKL_INT *nz, const float *x, const MKL_INT *indx, float *y);
void sswap(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy) NOTHROW;
MKL_INT isamax(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;
MKL_INT isamin(const MKL_INT *n, const float *x, const MKL_INT *incx) NOTHROW;

void caxpy(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void caxpby(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void caxpyi(const MKL_INT *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void ccopy(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cdotc(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cdotci(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void cdotu(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cdotui(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void cgthr(const MKL_INT *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void cgthrz(const MKL_INT *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void crot(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const MKL_Complex8 *s) NOTHROW;
void crotg(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s) NOTHROW;
void cscal(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void csctr(const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void csrot(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const float *s) NOTHROW;
void csscal(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void cswap(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
MKL_INT icamax(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
MKL_INT icamin(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;

double dcabs1(const MKL_Complex16 *z);
double dasum(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
void daxpy(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
void daxpby(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void daxpyi(const MKL_INT *nz, const double *a, const double *x, const MKL_INT *indx, double *y);
void dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
double ddot(const MKL_INT *n, const double *x, const MKL_INT *incx, const double *y, const MKL_INT *incy) NOTHROW;
double dsdot(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy) NOTHROW;
double ddoti(const MKL_INT *nz, const double *x, const MKL_INT *indx, const double *y);
void dgthr(const MKL_INT *nz, const double *y, double *x, const MKL_INT *indx);
void dgthrz(const MKL_INT *nz, double *y, double *x, const MKL_INT *indx);
double dnrm2(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
void drot(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *c, const double *s) NOTHROW;
void drotg(double *a, double *b, double *c, double *s) NOTHROW;
void droti(const MKL_INT *nz, double *x, const MKL_INT *indx, double *y, const double *c, const double *s);
void drotm(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *param) NOTHROW;
void drotmg(double *d1, double *d2, double *x1, const double *y1, double *param) NOTHROW;
void dscal(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx) NOTHROW;
void dsctr(const MKL_INT *nz, const double *x, const MKL_INT *indx, double *y);
void dswap(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy) NOTHROW;
double dzasum(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
double dznrm2(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
MKL_INT idamax(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;
MKL_INT idamin(const MKL_INT *n, const double *x, const MKL_INT *incx) NOTHROW;

void zaxpy(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zaxpby(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zaxpyi(const MKL_INT *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void zcopy(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zdotc(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zdotci(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void zdotu(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zdotui(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void zdrot(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const double *s) NOTHROW;
void zdscal(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void zgthr(const MKL_INT *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void zgthrz(const MKL_INT *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void zrot(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const MKL_Complex16 *s) NOTHROW;
void zrotg(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s) NOTHROW;
void zscal(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void zsctr(const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void zswap(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
MKL_INT izamax(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
MKL_INT izamin(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;

/* blas level2 */

void sgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
           const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void sger(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
          const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda) NOTHROW;
void ssbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, const float *alpha,
           const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
           const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void sspmv(const char *uplo, const MKL_INT *n, const float *alpha, const float *ap,
           const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void sspr(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
          float *ap) NOTHROW;
void sspr2(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *ap) NOTHROW;
void ssymv(const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy) NOTHROW;
void ssyr(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
          float *a, const MKL_INT *lda) NOTHROW;
void ssyr2(const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
           const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda) NOTHROW;
void stbmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void stbsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void stpmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *ap,
           float *x, const MKL_INT *incx) NOTHROW;
void stpsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *ap,
           float *x, const MKL_INT *incx) NOTHROW;
void strmv(const char *uplo, const char *transa, const char *diag, const MKL_INT *n, const float *a,
           const MKL_INT *lda, float *b, const MKL_INT *incx) NOTHROW;
void strsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx) NOTHROW;
void sgem2vu(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
             const float *x1, const MKL_INT *incx1, const float *x2, const MKL_INT *incx2,
             const float *beta, float *y1, const MKL_INT *incy1, float *y2, const MKL_INT *incy2);

void cgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void cgeru(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void chbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void chemv(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void cher(const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
          MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void cher2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
           MKL_Complex8 *a, const MKL_INT *lda) NOTHROW;
void chpmv(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
           const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
           MKL_Complex8 *y, const MKL_INT *incy) NOTHROW;
void chpr(const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
          MKL_Complex8 *ap) NOTHROW;
void chpr2(const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
           const MKL_Complex8 *y, const MKL_INT *incy, MKL_Complex8 *ap) NOTHROW;
void ctbmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void ctbsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void ctpmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void ctpsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void ctrmv(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *incx) NOTHROW;
void ctrsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx) NOTHROW;
void cgem2vc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
             const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x1, const MKL_INT *incx1,
             const MKL_Complex8 *x2, const MKL_INT *incx2, const MKL_Complex8 *beta,
             MKL_Complex8 *y1, const MKL_INT *incy1, MKL_Complex8 *y2, const MKL_INT *incy2);
void scgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
            const float *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
            const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);

void dgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const double *alpha, const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
           const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void dger(const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
          const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda) NOTHROW;
void dsbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, const double *alpha,
           const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
           const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void dspmv(const char *uplo, const MKL_INT *n, const double *alpha, const double *ap,
           const double *x, const MKL_INT *incx, const double *beta,
           double *y, const MKL_INT *incy) NOTHROW;
void dspr(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
          double *ap) NOTHROW;
void dspr2(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *ap) NOTHROW;
void dsymv(const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy) NOTHROW;
void dsyr(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
          double *a, const MKL_INT *lda) NOTHROW;
void dsyr2(const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
           const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda) NOTHROW;
void dtbmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void dtbsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void dtpmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *ap, double *x, const MKL_INT *incx) NOTHROW;
void dtpsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *ap, double *x, const MKL_INT *incx) NOTHROW;
void dtrmv(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const double *a, const MKL_INT *lda, double *b, const MKL_INT *incx) NOTHROW;
void dtrsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx) NOTHROW;
void dgem2vu(const MKL_INT *m, const MKL_INT *n, const double *alpha,
             const double *a, const MKL_INT *lda, const double *x1, const MKL_INT *incx1,
             const double *x2, const MKL_INT *incx2, const double *beta,
             double *y1, const MKL_INT *incy1, double *y2, const MKL_INT *incy2);

void zgbmv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void zgeru(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void zhbmv(const char *uplo, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zhemv(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zher(const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
          MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void zher2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
           MKL_Complex16 *a, const MKL_INT *lda) NOTHROW;
void zhpmv(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
           const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
           MKL_Complex16 *y, const MKL_INT *incy) NOTHROW;
void zhpr(const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
          MKL_Complex16 *ap) NOTHROW;
void zhpr2(const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
           const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *ap) NOTHROW;
void ztbmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ztbsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ztpmv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ztpsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void ztrmv(const char *uplo, const char *transa, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *incx) NOTHROW;
void ztrsv(const char *uplo, const char *trans, const char *diag, const MKL_INT *n,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx) NOTHROW;
void zgem2vc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
             const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x1, const MKL_INT *incx1,
             const MKL_Complex16 *x2, const MKL_INT *incx2, const MKL_Complex16 *beta,
             MKL_Complex16 *y1, const MKL_INT *incy1, MKL_Complex16 *y2, const MKL_INT *incy2);
void dzgemv(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
            const double *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
            const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);

/* blas level3 */

void sgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
size_t sgemm_pack_get_size(const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void sgemm_pack(const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                const float *alpha, const float *src, const MKL_INT *ld, float *dest);
void sgemm_compute(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void sgemm_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float **b_array, const MKL_INT *ldb_array,
                 const float *beta_array, float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void sgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const float *beta, float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

void sgemmt(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void ssymm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
           const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void ssyr2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
            const float *beta, float *c, const MKL_INT *ldc) NOTHROW;
void ssyrk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const float *beta,
           float *c, const MKL_INT *ldc) NOTHROW;
void ssyrk_batch(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float *beta_array,
           float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ssyrk_batch_strided(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea, const float *beta,
           float *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void strmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;
void strsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
           float *b, const MKL_INT *ldb) NOTHROW;
void strsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float **a_array,
                 const MKL_INT *lda_array, float **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void strsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const float *alpha, const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                 float *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void cgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void cgemm_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void cgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;

void scgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const float *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc);
void cgemm3m(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
             const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
             const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
             MKL_Complex8 *c, const MKL_INT *ldc);
void cgemm3m_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                   const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                   const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void cgemmt(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void chemm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void cher2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void cherk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const float *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void csymm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void csyr2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
            const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
            MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void csyrk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
           const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc) NOTHROW;
void csyrk_batch(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array,
           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void csyrk_batch_strided(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea, const MKL_Complex8 *beta,
           MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void ctrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;
void ctrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
           const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *ldb) NOTHROW;
void ctrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array,
                 const MKL_INT *lda_array, MKL_Complex8 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ctrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void dgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
size_t dgemm_pack_get_size(const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void dgemm_pack(const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                const double *alpha, const double *src, const MKL_INT *ld, double *dest);
void dgemm_compute(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void dgemm_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const double *alpha_array, const double **a_array, const MKL_INT *lda_array, const double **b_array, const MKL_INT *ldb_array,
                 const double *beta_array, double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void dgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const double *beta, double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;
void dgemmt(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
void dsymm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
           const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
void dsyr2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
            const double *beta, double *c, const MKL_INT *ldc) NOTHROW;
void dsyrk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const double *beta,
           double *c, const MKL_INT *ldc) NOTHROW;
void dsyrk_batch(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const double *alpha_array, const double **a_array, const MKL_INT *lda_array, const double *beta_array,
           double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void dsyrk_batch_strided(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea, const double *beta,
           double *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void dtrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;
void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
           double *b, const MKL_INT *ldb) NOTHROW;
void dtrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double **a_array,
                 const MKL_INT *lda_array, double **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void dtrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const double *alpha, const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                 double *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void zgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zgemm_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                 const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                 const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void zgemm_batch_strided(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                 const MKL_INT *batch_size) NOTHROW;
void dzgemm(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const double *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc);
void zgemm3m(const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
             const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
             const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
             MKL_Complex16 *c, const MKL_INT *ldc);
void zgemm3m_batch(const char *transa_array, const char *transb_array, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                   const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                   const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void zgemmt(const char *uplo, const char *transa, const char *transb, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zhemm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zher2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zherk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const double *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zsymm(const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zsyr2k(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
            const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
            const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
            MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zsyrk(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
           const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc) NOTHROW;
void zsyrk_batch(const char *uplo_array, const char *trans_array, const MKL_INT *n_array, const MKL_INT *k_array,
           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array,
           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void zsyrk_batch_strided(const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k,
           const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea, const MKL_Complex16 *beta,
           MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec, const MKL_INT *batch_size) NOTHROW;
void ztrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;
void ztrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
           const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb) NOTHROW;
void ztrsm_batch(const char *side_array, const char *uplo_array, const char *transa_array, const char *diag_array,
                 const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array,
                 const MKL_INT *lda_array, MKL_Complex16 **b_array, const MKL_INT *ldb, const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void ztrsm_batch_strided(const char *side, const char *uplo, const char *transa, const char *diag,
                 const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                 MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *strideb,
                 const MKL_INT *batch_size) NOTHROW;

void gemm_s16s16s32(const char *transa, const char *transb, const char *offsetc,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                    const float *alpha, const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao,
                    const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo,
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void gemm_s8u8s32(const char *transa, const char *transb, const char *offsetc,
                  const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                  const float *alpha, const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao,
                  const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo,
                  const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co) NOTHROW;

size_t gemm_s8u8s32_pack_get_size (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t gemm_s16s16s32_pack_get_size (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void gemm_s8u8s32_pack (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                        const void *src, const MKL_INT *ld, void *dest);
void gemm_s16s16s32_pack (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                          const MKL_INT16 *src, const MKL_INT *ld, MKL_INT16 *dest);
void gemm_s8u8s32_compute (const char *transa, const char *transb, const char *offsetc,
                           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                           const float *alpha,
                           const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao,
                           const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo,
                           const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void gemm_s16s16s32_compute(const char *transa, const char *transb, const char *offsetc,
                            const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                            const float *alpha,
                            const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao,
                            const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo,
                            const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);

void hgemm(const char *transa, const char *transb,
           const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const MKL_F16 *alpha, const MKL_F16 *a, const MKL_INT *lda,
           const MKL_F16 *b, const MKL_INT *ldb,
           const MKL_F16 *beta, MKL_F16 *c, const MKL_INT *ldc) NOTHROW;
size_t hgemm_pack_get_size (const char *identifier, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void hgemm_pack (const char *identifier, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                 const MKL_F16 *alpha, const MKL_F16 *src, const MKL_INT *ld, MKL_F16 *dest);
void hgemm_compute(const char *transa, const char *transb,
                   const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                   const MKL_F16 *a, const MKL_INT *lda,
                   const MKL_F16 *b, const MKL_INT *ldb,
                   const MKL_F16 *beta, MKL_F16 *c, const MKL_INT *ldc);

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

void SAXPY_BATCH(const MKL_INT *n, const float *alpha,
                 const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void saxpy_batch(const MKL_INT *n, const float *alpha,
                 const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void DAXPY_BATCH(const MKL_INT *n, const double *alpha,
                 const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void daxpy_batch(const MKL_INT *n, const double *alpha,
                 const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void CAXPY_BATCH(const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void caxpy_batch(const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void ZAXPY_BATCH(const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;
void zaxpy_batch(const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void SCOPY_BATCH(const MKL_INT *n, const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;
void scopy_batch(const MKL_INT *n, const float **x, const MKL_INT *incx,
                 float **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;

void DCOPY_BATCH(const MKL_INT *n, const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;
void dcopy_batch(const MKL_INT *n, const double **x, const MKL_INT *incx,
                 double **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;

void CCOPY_BATCH(const MKL_INT *n, const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;
void ccopy_batch(const MKL_INT *n, const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;

void ZCOPY_BATCH(const MKL_INT *n, const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;
void zcopy_batch(const MKL_INT *n, const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **y, const MKL_INT *incy, const MKL_INT *group_count,
                 const MKL_INT *group_size) NOTHROW;

void SAXPY_BATCH_STRIDED(const MKL_INT *n, const float *alpha,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;
void saxpy_batch_strided(const MKL_INT *n, const float *alpha,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void DAXPY_BATCH_STRIDED(const MKL_INT *n, const double *alpha,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;
void daxpy_batch_strided(const MKL_INT *n, const double *alpha,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void CAXPY_BATCH_STRIDED(const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;
void caxpy_batch_strided(const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void ZAXPY_BATCH_STRIDED(const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;
void zaxpy_batch_strided(const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void SCOPY_BATCH_STRIDED(const MKL_INT *n, const float* x, const MKL_INT *incx,
                         const MKL_INT* stridex, float*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;
void scopy_batch_strided(const MKL_INT *n, const float* x, const MKL_INT *incx,
                         const MKL_INT* stridex, float*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;

void DCOPY_BATCH_STRIDED(const MKL_INT *n, const double* x, const MKL_INT *incx,
                         const MKL_INT* stridex, double*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;
void dcopy_batch_strided(const MKL_INT *n, const double* x, const MKL_INT *incx,
                         const MKL_INT* stridex, double*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;

void CCOPY_BATCH_STRIDED(const MKL_INT *n, const MKL_Complex8* x, const MKL_INT *incx,
                         const MKL_INT* stridex, MKL_Complex8*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;
void ccopy_batch_strided(const MKL_INT *n, const MKL_Complex8* x, const MKL_INT *incx,
                         const MKL_INT* stridex, MKL_Complex8*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;

void ZCOPY_BATCH_STRIDED(const MKL_INT *n, const MKL_Complex16* x, const MKL_INT *incx,
                         const MKL_INT* stridex, MKL_Complex16*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;
void zcopy_batch_strided(const MKL_INT *n, const MKL_Complex16* x, const MKL_INT *incx,
                         const MKL_INT* stridex, MKL_Complex16*y, const MKL_INT* incy,
                         const MKL_INT* stridey, const MKL_INT *batch_size) NOTHROW;

/* Level2 BLAS batch API */
void sgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                 const float **a, const MKL_INT *lda, const float **x, const MKL_INT *incx,
                 const float *beta, float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void sgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const float *beta, float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void dgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                 const double **a, const MKL_INT *lda, const double **x, const MKL_INT *incx,
                 const double *beta, double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void dgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const double *beta, double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void cgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT *lda, const MKL_Complex8 **x, const MKL_INT *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void cgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void zgemv_batch(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT *lda, const MKL_Complex16 **x, const MKL_INT *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void zgemv_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;


void SGEMV_BATCH(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                 const float **a, const MKL_INT *lda, const float **x, const MKL_INT *incx,
                 const float *beta, float **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void SGEMV_BATCH_STRIDED(const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const float *beta, float *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void DGEMV_BATCH(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                 const double **a, const MKL_INT *lda, const double **x, const MKL_INT *incx,
                 const double *beta, double **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void DGEMV_BATCH_STRIDED(const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const double *beta, double *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void CGEMV_BATCH(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                 const MKL_Complex8 **a, const MKL_INT *lda, const MKL_Complex8 **x, const MKL_INT *incx,
                 const MKL_Complex8 *beta, MKL_Complex8 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void CGEMV_BATCH_STRIDED(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void ZGEMV_BATCH(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                 const MKL_Complex16 **a, const MKL_INT *lda, const MKL_Complex16 **x, const MKL_INT *incx,
                 const MKL_Complex16 *beta, MKL_Complex16 **y, const MKL_INT *incy,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void ZGEMV_BATCH_STRIDED(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy, const MKL_INT *stridey,
                         const MKL_INT *batch_size) NOTHROW;

void sdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const float **a, const MKL_INT *lda,
                 const float **x, const MKL_INT *incx,
                 float **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void sdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void ddgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const double **a, const MKL_INT *lda,
                 const double **x, const MKL_INT *incx,
                 double **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void ddgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void cdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 **a, const MKL_INT *lda,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void cdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void zdgmm_batch(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 **a, const MKL_INT *lda,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void zdgmm_batch_strided(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void SDGMM_BATCH(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const float **a, const MKL_INT *lda,
                 const float **x, const MKL_INT *incx,
                 float **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void SDGMM_BATCH_STRIDED(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const float *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const float *x, const MKL_INT *incx, const MKL_INT *stridex,
                         float *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void DDGMM_BATCH(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const double **a, const MKL_INT *lda,
                 const double **x, const MKL_INT *incx,
                 double **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void DDGMM_BATCH_STRIDED(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const double *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const double *x, const MKL_INT *incx, const MKL_INT *stridex,
                         double *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void CDGMM_BATCH(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex8 **a, const MKL_INT *lda,
                 const MKL_Complex8 **x, const MKL_INT *incx,
                 MKL_Complex8 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void CDGMM_BATCH_STRIDED(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex8 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex8 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

void ZDGMM_BATCH(const char *side, const MKL_INT *m, const MKL_INT *n,
                 const MKL_Complex16 **a, const MKL_INT *lda,
                 const MKL_Complex16 **x, const MKL_INT *incx,
                 MKL_Complex16 **c, const MKL_INT *ldc,
                 const MKL_INT *group_count, const MKL_INT *group_size) NOTHROW;

void ZDGMM_BATCH_STRIDED(const char *side, const MKL_INT *m, const MKL_INT *n,
                         const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stridea,
                         const MKL_Complex16 *x, const MKL_INT *incx, const MKL_INT *stridex,
                         MKL_Complex16 *c, const MKL_INT *ldc, const MKL_INT *stridec,
                         const MKL_INT *batch_size) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#include "mkl_blas_64.h"

#endif /* _MKL_BLAS_H_ */
