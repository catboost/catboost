/*******************************************************************************
* Copyright 2005-2022 Intel Corporation.
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
! Content:
!  Intel(R) oneAPI Math Kernel Library (oneMKL) interface for Sparse BLAS
!  level 2,3 routines
!
!******************************************************************************/

#ifndef _MKL_SPBLAS_H_
#define _MKL_SPBLAS_H_

#include "mkl_types.h"

#ifdef __GNUC__
#define MKL_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define MKL_DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: Intel oneMKL SpBLAS was declared deprecated. Use Intel oneMKL IE SpBLAS instead")
#define MKL_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Float */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void mkl_scsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_scsrsv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void mkl_scsrgemv(const char *transa, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scsrgemv(const char *transa, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_scsrsymv(const char *uplo, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scsrsymv(const char *uplo, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_scsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

MKL_DEPRECATED void mkl_scscmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_scscsv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);

MKL_DEPRECATED void mkl_scoomv(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_scoosv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_scoogemv(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scoogemv(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_scoosymv(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scoosymv(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_scootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_scootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);

MKL_DEPRECATED void mkl_sdiamv (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_sdiasv (const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void mkl_sdiagemv(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void mkl_sdiasymv(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void mkl_sdiatrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);

MKL_DEPRECATED void mkl_sskymv (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_sskysv(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *x, float *y);

MKL_DEPRECATED void mkl_sbsrmv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void mkl_sbsrsv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void mkl_sbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_sbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_sbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_sbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_sbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void mkl_cspblas_sbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void mkl_scsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_scsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_scscmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_scscsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_scoomm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_scoosm(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_sdiamm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_sdiasm (const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_sskysm (const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_sskymm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_sbsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_sbsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void MKL_SCSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SCSRSV(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void MKL_SCSRGEMV(const char *transa, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCSRGEMV(const char *transa, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_SCSRSYMV(const char *uplo, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCSRSYMV(const char *uplo, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_SCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

MKL_DEPRECATED void MKL_SCSCMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SCSCSV(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);

MKL_DEPRECATED void MKL_SCOOMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SCOOSV(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_SCOOGEMV(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCOOGEMV(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_SCOOSYMV(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCOOSYMV(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_SCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);

MKL_DEPRECATED void MKL_SDIAMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SDIASV (const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void MKL_SDIAGEMV(const char *transa, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void MKL_SDIASYMV(const char *uplo, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void MKL_SDIATRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);

MKL_DEPRECATED void MKL_SSKYMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SSKYSV(const char *transa, const MKL_INT *m, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *x, float *y);

MKL_DEPRECATED void MKL_SBSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void MKL_SBSRSV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void MKL_SBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_SBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_SBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void MKL_CSPBLAS_SBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void MKL_SCSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SCSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_SCSCMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SCSCSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_SCOOMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SCOOSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_SDIAMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SDIASM (const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_SSKYSM (const char *transa, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SSKYMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_SBSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_SBSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const float *alpha, const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

/* Double */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void mkl_dcsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_dcsrsv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void mkl_dcsrgemv(const char *transa, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcsrgemv(const char *transa, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_dcsrsymv(const char *uplo, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcsrsymv(const char *uplo, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_dcsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

MKL_DEPRECATED void mkl_dcscmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_dcscsv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);

MKL_DEPRECATED void mkl_dcoomv(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_dcoosv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_dcoogemv(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcoogemv(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_dcoosymv(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcoosymv(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_dcootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dcootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);

MKL_DEPRECATED void mkl_ddiamv (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_ddiasv (const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void mkl_ddiagemv(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void mkl_ddiasymv(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void mkl_ddiatrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);

MKL_DEPRECATED void mkl_dskymv (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_dskysv(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *x, double *y);

MKL_DEPRECATED void mkl_dbsrmv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void mkl_dbsrsv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void mkl_dbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_dbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_dbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void mkl_cspblas_dbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void mkl_dcsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_dcsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_dcscmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_dcscsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_dcoomm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_dcoosm(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_ddiamm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_ddiasm (const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_dskysm (const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_dskymm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_dbsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_dbsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void MKL_DCSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DCSRSV(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void MKL_DCSRGEMV(const char *transa, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCSRGEMV(const char *transa, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_DCSRSYMV(const char *uplo, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCSRSYMV(const char *uplo, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_DCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

MKL_DEPRECATED void MKL_DCSCMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DCSCSV(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);

MKL_DEPRECATED void MKL_DCOOMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DCOOSV(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_DCOOGEMV(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCOOGEMV(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_DCOOSYMV(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCOOSYMV(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_DCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);

MKL_DEPRECATED void MKL_DDIAMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DDIASV (const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void MKL_DDIAGEMV(const char *transa, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void MKL_DDIASYMV(const char *uplo, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void MKL_DDIATRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);

MKL_DEPRECATED void MKL_DSKYMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DSKYSV(const char *transa, const MKL_INT *m, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *x, double *y);

MKL_DEPRECATED void MKL_DBSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void MKL_DBSRSV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void MKL_DBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_DBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_DBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void MKL_CSPBLAS_DBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void MKL_DCSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DCSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_DCSCMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DCSCSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_DCOOMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DCOOSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_DDIAMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DDIASM (const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_DSKYSM (const char *transa, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DSKYMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_DBSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_DBSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const double *alpha, const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

/* MKL_Complex8 */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void mkl_ccsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccsrsv(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccsrgemv(const char *transa, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccsrgemv(const char *transa, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccsrsymv(const char *uplo, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccsrsymv(const char *uplo, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void mkl_ccscmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccscsv(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void mkl_ccoomv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccoosv(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccoogemv(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccoogemv(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccoosymv(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccoosymv(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_ccootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_ccootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void mkl_cdiamv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cdiasv (const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cdiagemv(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cdiasymv(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cdiatrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void mkl_cskymv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cskysv(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void mkl_cbsrmv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cbsrsv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_cbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_cbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void mkl_cspblas_cbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void mkl_ccsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_ccsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_ccscmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_ccscsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_ccoomm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_ccoosm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_cdiamm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_cdiasm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_cskysm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_cskymm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_cbsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_cbsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void MKL_CCSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCSRSV(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCSRGEMV(const char *transa, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCSRGEMV(const char *transa, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCSRSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCSRSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void MKL_CCSCMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCSCSV(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void MKL_CCOOMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCOOSV(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCOOGEMV(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCOOGEMV(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCOOSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCOOSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void MKL_CDIAMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CDIASV (const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CDIAGEMV(const char *transa, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CDIASYMV(const char *uplo, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CDIATRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void MKL_CSKYMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSKYSV(const char *transa, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void MKL_CBSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CBSRSV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void MKL_CSPBLAS_CBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void MKL_CCSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CCSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_CCSCMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CCSCSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_CCOOMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CCOOSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_CDIAMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CDIASM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_CSKYSM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CSKYMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_CBSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_CBSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

/* MKL_Complex16 */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void mkl_zcsrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcsrsv(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcsrgemv(const char *transa, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcsrgemv(const char *transa, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcsrsymv(const char *uplo, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcsrsymv(const char *uplo, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void mkl_zcscmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcscsv(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void mkl_zcoomv(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcoosv(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcoogemv(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcoogemv(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcoosymv(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcoosymv(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zcootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zcootrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void mkl_zdiamv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zdiasv (const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zdiagemv(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zdiasymv(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zdiatrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void mkl_zskymv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zskysv(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void mkl_zbsrmv (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zbsrsv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zbsrgemv(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zbsrsymv(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_zbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void mkl_cspblas_zbsrtrsv(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void mkl_zcsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zcsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_zcscmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zcscsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_zcoomm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zcoosm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_zdiamm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zdiasm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_zskysm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zskymm (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void mkl_zbsrmm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void mkl_zbsrsm(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void MKL_ZCSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCSRSV(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCSRGEMV(const char *transa, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCSRGEMV(const char *transa, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCSRSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCSRSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void MKL_ZCSCMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCSCSV(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void MKL_ZCOOMV(const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCOOSV(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCOOGEMV(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCOOGEMV(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCOOSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCOOSYMV(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZCOOTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void MKL_ZDIAMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZDIASV (const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZDIAGEMV(const char *transa, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZDIASYMV(const char *uplo, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZDIATRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void MKL_ZSKYMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZSKYSV(const char *transa, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void MKL_ZBSRMV (const char *transa, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZBSRSV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZBSRGEMV(const char *transa, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZBSRSYMV(const char *uplo, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_ZBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void MKL_CSPBLAS_ZBSRTRSV(const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void MKL_ZCSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZCSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_ZCSCMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZCSCSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_ZCOOMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZCOOSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_ZDIAMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZDIASM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_ZSKYSM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZSKYMM (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void MKL_ZBSRMM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZBSRSM(const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

/*Converters lower case*/
MKL_DEPRECATED void mkl_dcsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, double *Acsr, MKL_INT *AJ, MKL_INT *AI, double *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void mkl_dcsrcoo(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, double *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void mkl_dcsrcsc(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void mkl_dcsrdia(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, double *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void mkl_dcsrsky(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  double *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void mkl_scsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, float *Acsr, MKL_INT *AJ, MKL_INT *AI, float *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void mkl_scsrcoo(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, float *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void mkl_scsrcsc(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void mkl_scsrdia(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, float *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void mkl_scsrsky(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  float *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void mkl_ccsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex8 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void mkl_ccsrcoo(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex8 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void mkl_cdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex8 *Adns, const MKL_INT *lda, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void mkl_ccsrcsc(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void mkl_ccsrdia(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex8 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void mkl_ccsrsky(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex8 *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void mkl_zcsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex16 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void mkl_zcsrcoo(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex16 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void mkl_zdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex16 *Adns, const MKL_INT *lda, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void mkl_zcsrcsc(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void mkl_zcsrdia(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex16 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void mkl_zcsrsky(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex16 *Asky, MKL_INT *pointers, MKL_INT *info);

/*Converters upper case*/
MKL_DEPRECATED void MKL_DCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, double *Acsr, MKL_INT *AJ, MKL_INT *AI, double *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void MKL_DCSRCOO(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, double *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void MKL_DDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void MKL_DCSRCSC(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void MKL_DCSRDIA(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, double *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void MKL_DCSRSKY(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  double *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void MKL_SCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, float *Acsr, MKL_INT *AJ, MKL_INT *AI, float *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void MKL_SCSRCOO(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, float *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void MKL_SDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void MKL_SCSRCSC(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void MKL_SCSRDIA(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, float *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void MKL_SCSRSKY(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  float *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void MKL_CCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex8 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void MKL_CCSRCOO(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex8 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void MKL_CDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex8 *Adns, const MKL_INT *lda, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void MKL_CCSRCSC(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void MKL_CCSRDIA(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex8 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void MKL_CCSRSKY(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex8 *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void MKL_ZCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex16 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void MKL_ZCSRCOO(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex16 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void MKL_ZDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex16 *Adns, const MKL_INT *lda, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void MKL_ZCSRCSC(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void MKL_ZCSRDIA(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex16 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void MKL_ZCSRSKY(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex16 *Asky, MKL_INT *pointers, MKL_INT *info);


/*Sparse BLAS Level2 (CSR-CSR or CSR-DNS) lower case */
MKL_DEPRECATED void mkl_dcsrmultcsr(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a,  MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void mkl_dcsrmultd(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *ldc);
MKL_DEPRECATED void mkl_dcsradd(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, double *a, MKL_INT *ja, MKL_INT *ia,  const double *beta, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void mkl_scsrmultcsr(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void mkl_scsrmultd(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb,  MKL_INT *ib, float *c,  MKL_INT *ldc);
MKL_DEPRECATED void mkl_scsradd(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, float *a, MKL_INT *ja, MKL_INT *ia,  const float *beta, float *b, MKL_INT *jb, MKL_INT *ib, float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void mkl_ccsrmultcsr(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib, MKL_Complex8 *c, MKL_INT *jc, MKL_INT *ic, const MKL_INT *nnzmax, MKL_INT *ierr);
MKL_DEPRECATED void mkl_ccsrmultd(const char *transa,   const MKL_INT *m, const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib, MKL_Complex8 *c, MKL_INT *ldc);
MKL_DEPRECATED void mkl_ccsradd(const char *transa,  const MKL_INT *job, const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, const MKL_Complex8 *beta, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void mkl_zcsrmultcsr(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja,  MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void mkl_zcsrmultd(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *ldc);
MKL_DEPRECATED void mkl_zcsradd(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex16 *beta, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);


/*Sparse BLAS Level2 (CSR-CSR or CSR-DNS) upper case */
MKL_DEPRECATED void MKL_DCSRMULTCSR(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void MKL_DCSRMULTD(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *ldc);
MKL_DEPRECATED void MKL_DCSRADD(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, double *a, MKL_INT *ja, MKL_INT *ia, const double *beta, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void MKL_SCSRMULTCSR(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void MKL_SCSRMULTD(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *ldc);
MKL_DEPRECATED void MKL_SCSRADD(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, float *a, MKL_INT *ja, MKL_INT *ia,  const float *beta, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void MKL_CCSRMULTCSR(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void MKL_CCSRMULTD(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *ldc);
MKL_DEPRECATED void MKL_CCSRADD(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex8 *beta, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void MKL_ZCSRMULTCSR(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void MKL_ZCSRMULTD(const char *transa,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *ldc);
MKL_DEPRECATED void MKL_ZCSRADD(const char *transa,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex16 *beta, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);




/*****************************************************************************************/
/************** Basic types and constants for inspector-executor SpBLAS API **************/
/*****************************************************************************************/

    /* status of the routines */
    typedef enum
    {
        SPARSE_STATUS_SUCCESS           = 0,    /* the operation was successful */
        SPARSE_STATUS_NOT_INITIALIZED   = 1,    /* empty handle or matrix arrays */
        SPARSE_STATUS_ALLOC_FAILED      = 2,    /* internal error: memory allocation failed */
        SPARSE_STATUS_INVALID_VALUE     = 3,    /* invalid input value */
        SPARSE_STATUS_EXECUTION_FAILED  = 4,    /* e.g. 0-diagonal element for triangular solver, etc. */
        SPARSE_STATUS_INTERNAL_ERROR    = 5,    /* internal error */
        SPARSE_STATUS_NOT_SUPPORTED     = 6     /* e.g. operation for double precision doesn't support other types */
    } sparse_status_t;

    /* sparse matrix operations */
    typedef enum
    {
        SPARSE_OPERATION_NON_TRANSPOSE       = 10,
        SPARSE_OPERATION_TRANSPOSE           = 11,
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
    } sparse_operation_t;

    /* supported matrix types */
    typedef enum
    {
        SPARSE_MATRIX_TYPE_GENERAL            = 20,   /*    General case                    */
        SPARSE_MATRIX_TYPE_SYMMETRIC          = 21,   /*    Triangular part of              */
        SPARSE_MATRIX_TYPE_HERMITIAN          = 22,   /*    the matrix is to be processed   */
        SPARSE_MATRIX_TYPE_TRIANGULAR         = 23,
        SPARSE_MATRIX_TYPE_DIAGONAL           = 24,   /* diagonal matrix; only diagonal elements will be processed */
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25,
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26    /* block-diagonal matrix; only diagonal blocks will be processed */
    } sparse_matrix_type_t;

    /* sparse matrix indexing: C-style or Fortran-style */
    typedef enum
    {
        SPARSE_INDEX_BASE_ZERO  = 0,           /* C-style */
        SPARSE_INDEX_BASE_ONE   = 1            /* Fortran-style */
    } sparse_index_base_t;

    /* applies to triangular matrices only ( SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_MATRIX_TYPE_TRIANGULAR ) */
    typedef enum
    {
        SPARSE_FILL_MODE_LOWER  = 40,           /* lower triangular part of the matrix is stored */
        SPARSE_FILL_MODE_UPPER  = 41,            /* upper triangular part of the matrix is stored */
        SPARSE_FILL_MODE_FULL   = 42            /* upper triangular part of the matrix is stored */
    } sparse_fill_mode_t;

    /* applies to triangular matrices only ( SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_MATRIX_TYPE_TRIANGULAR ) */
    typedef enum
    {
        SPARSE_DIAG_NON_UNIT    = 50,           /* triangular matrix with non-unit diagonal */
        SPARSE_DIAG_UNIT        = 51            /* triangular matrix with unit diagonal */
    } sparse_diag_type_t;

    /* applicable for Level 3 operations with dense matrices; describes storage scheme for dense matrix (row major or column major) */
    typedef enum
    {
        SPARSE_LAYOUT_ROW_MAJOR    = 101,       /* C-style */
        SPARSE_LAYOUT_COLUMN_MAJOR = 102        /* Fortran-style */
    } sparse_layout_t;

    /* verbose mode; if verbose mode activated, handle should collect and report profiling / optimization info */
    typedef enum
    {
        SPARSE_VERBOSE_OFF      = 70,
        SPARSE_VERBOSE_BASIC    = 71,           /* output contains high-level information about optimization algorithms, issues, etc. */
        SPARSE_VERBOSE_EXTENDED = 72            /* provide detailed output information */
    } verbose_mode_t;

    /* memory optimization hints from user: describe how much memory could be used on optimization stage */
    typedef enum
    {
        SPARSE_MEMORY_NONE          = 80,       /* no memory should be allocated for matrix values and structures; auxiliary structures could be created only for workload balancing, parallelization, etc. */
        SPARSE_MEMORY_AGGRESSIVE    = 81        /* matrix could be converted to any internal format */
    } sparse_memory_usage_t;

    typedef enum
    {
        SPARSE_STAGE_FULL_MULT            = 90,
        SPARSE_STAGE_NNZ_COUNT            = 91,
        SPARSE_STAGE_FINALIZE_MULT        = 92,
        SPARSE_STAGE_FULL_MULT_NO_VAL     = 93,
        SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94
    } sparse_request_t;

    /* applies to SOR interface; define type of (S)SOR operation to perform */
    typedef enum
    {
        SPARSE_SOR_FORWARD   = 110, /* (omega∗L + D)∗x^1 = (D - omega*D - omega*U)∗alpha*x^0 + omega*b */
        SPARSE_SOR_BACKWARD  = 111, /* (omega∗U + D)∗x^1 = (D - omega*D - omega*L)∗alpha*x^0 + omega*b */
        SPARSE_SOR_SYMMETRIC = 112  /* SSOR, for e.g. with omega == 1 && alpha == 1, equal to solving a system:
                                       (L + D)∗x^1 = b - U*x; (U + D)∗x = b - L*x^1 */
    } sparse_sor_type_t;

/*************************************************************************************************/
/*** Opaque structure for sparse matrix in internal format, further D - means double precision ***/
/*************************************************************************************************/

    struct  sparse_matrix;
    typedef struct sparse_matrix *sparse_matrix_t;

    /* descriptor of main sparse matrix properties */
    struct matrix_descr {
        sparse_matrix_type_t  type;       /* matrix type: general, diagonal or triangular / symmetric / hermitian */
        sparse_fill_mode_t    mode;       /* upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case) */
        sparse_diag_type_t    diag;       /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
    };

/*****************************************************************************************/
/*************************************** Creation routines *******************************/
/*****************************************************************************************/

/*
    Matrix handle is used for storing information about the matrix and matrix values

    Create matrix from one of the existing sparse formats by creating the handle with matrix info and copy matrix values if requested.
    Collect high-level info about the matrix. Need to use this interface for the case with several calls in program for performance reasons,
    where optimizations are not required.

    coordinate format,
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call.
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t mkl_sparse_s_create_coo(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t mkl_sparse_d_create_coo(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t mkl_sparse_c_create_coo(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t mkl_sparse_z_create_coo(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );


/*
    compressed sparse row format (4-arrays version),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call.
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t mkl_sparse_s_create_csr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t mkl_sparse_d_create_csr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t mkl_sparse_c_create_csr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t mkl_sparse_z_create_csr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );

/*
    compressed sparse column format (4-arrays version),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call.
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t mkl_sparse_s_create_csc(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   float               *values );

    sparse_status_t mkl_sparse_d_create_csc(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   double              *values );

    sparse_status_t mkl_sparse_c_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t mkl_sparse_z_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   MKL_Complex16       *values );

/*
    compressed block sparse row format (4-arrays version, square blocks),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call.
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t mkl_sparse_s_create_bsr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t mkl_sparse_d_create_bsr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t mkl_sparse_c_create_bsr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t mkl_sparse_z_create_bsr(       sparse_matrix_t     *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );

/*
    Create copy of the existing handle; matrix properties could be changed.
    For example it could be used for extracting triangular or diagonal parts from existing matrix.
*/
    sparse_status_t mkl_sparse_copy( const sparse_matrix_t     source,
                                     const struct matrix_descr descr,        /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     sparse_matrix_t           *dest );

/*
    destroy matrix handle; if sparse matrix was stored inside the handle it also deallocates the matrix
    It is user's responsibility not to delete the handle with the matrix, if this matrix is shared with other handles
*/
    sparse_status_t mkl_sparse_destroy( sparse_matrix_t  A );
/*
    return extended error information from last operation;
    e.g. info about wrong input parameter, memory sizes that couldn't be allocated
*/
    sparse_status_t mkl_sparse_get_error_info( sparse_matrix_t  A, MKL_INT *info ); /* unsupported currently */


/*****************************************************************************************/
/************************ Converters of internal representation  *************************/
/*****************************************************************************************/

    /* converters from current format to another */
    sparse_status_t mkl_sparse_convert_csr ( const sparse_matrix_t    source,         /* convert original matrix to CSR representation */
                                             const sparse_operation_t operation,      /* as is, transposed or conjugate transposed */
                                             sparse_matrix_t          *dest );

    sparse_status_t mkl_sparse_convert_bsr ( const sparse_matrix_t    source,         /* convert original matrix to BSR representation */
                                             const MKL_INT            block_size,
                                             const sparse_layout_t    block_layout,   /* block storage: row-major or column-major */
                                             const sparse_operation_t operation,      /* as is, transposed or conjugate transposed */
                                             sparse_matrix_t          *dest );

    sparse_status_t mkl_sparse_s_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             float                  **values );

    sparse_status_t mkl_sparse_d_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             double                 **values );

    sparse_status_t mkl_sparse_c_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex8           **values );

    sparse_status_t mkl_sparse_z_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex16          **values );

    sparse_status_t mkl_sparse_s_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             float                  **values );

    sparse_status_t mkl_sparse_d_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             double                 **values );

    sparse_status_t mkl_sparse_c_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex8           **values );

    sparse_status_t mkl_sparse_z_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex16          **values );

    sparse_status_t mkl_sparse_s_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             float                  **values );

    sparse_status_t mkl_sparse_d_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             double                 **values );

    sparse_status_t mkl_sparse_c_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             MKL_Complex8           **values );

    sparse_status_t mkl_sparse_z_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             MKL_Complex16          **values );

/*****************************************************************************************/
/************************** Step-by-step modification routines ***************************/
/*****************************************************************************************/


    /* update existing value in the matrix ( for internal storage only, should not work with user-allocated matrices) */
    sparse_status_t mkl_sparse_s_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const float           value );

    sparse_status_t mkl_sparse_d_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const double          value );

    sparse_status_t mkl_sparse_c_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const MKL_Complex8    value );

    sparse_status_t mkl_sparse_z_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const MKL_Complex16   value );

    /* update existing values in the matrix for internal storage only
       can be used to either update all or selected values */
    sparse_status_t mkl_sparse_s_update_values( const sparse_matrix_t A,
                                                const MKL_INT         nvalues,
                                                const MKL_INT        *indx,
                                                const MKL_INT        *indy,
                                                      float          *values );

    sparse_status_t mkl_sparse_d_update_values( const sparse_matrix_t A,
                                                const MKL_INT         nvalues,
                                                const MKL_INT        *indx,
                                                const MKL_INT        *indy,
                                                      double         *values );

    sparse_status_t mkl_sparse_c_update_values( const sparse_matrix_t A,
                                                const MKL_INT         nvalues,
                                                const MKL_INT        *indx,
                                                const MKL_INT        *indy,
                                                      MKL_Complex8   *values );

    sparse_status_t mkl_sparse_z_update_values( const sparse_matrix_t A,
                                                const MKL_INT         nvalues,
                                                const MKL_INT        *indx,
                                                const MKL_INT        *indy,
                                                      MKL_Complex16  *values );

/*****************************************************************************************/
/****************************** Verbose mode routine *************************************/
/*****************************************************************************************/

    /* allow to switch on/off verbose mode */
    sparse_status_t mkl_sparse_set_verbose_mode ( verbose_mode_t verbose ); /* unsupported currently */

/*****************************************************************************************/
/****************************** Optimization routines ************************************/
/*****************************************************************************************/

    /* Describe expected operations with amount of iterations */
    sparse_status_t mkl_sparse_set_mv_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );

    sparse_status_t mkl_sparse_set_dotmv_hint ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation, /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,     /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expectedCalls );

    sparse_status_t mkl_sparse_set_mm_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                                const MKL_INT             dense_matrix_size, /* amount of columns in dense matrix */
                                                const MKL_INT             expected_calls );

    sparse_status_t mkl_sparse_set_sv_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );

    sparse_status_t mkl_sparse_set_sm_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                                const MKL_INT             dense_matrix_size, /* amount of columns in dense matrix */
                                                const MKL_INT             expected_calls );

    sparse_status_t mkl_sparse_set_symgs_hint ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );

    sparse_status_t mkl_sparse_set_lu_smoother_hint( const sparse_matrix_t     A,
                                                     const sparse_operation_t  operation,
                                                     const struct matrix_descr descr, /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                     const MKL_INT             expectedCalls );

    sparse_status_t mkl_sparse_set_sorv_hint( const sparse_sor_type_t   type,  /* choice of forward, backward sweep or SSOR operation */
                                              const sparse_matrix_t     A,
                                              const struct matrix_descr descr, /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                              const MKL_INT             expectedCalls );

    /* Describe memory usage model */
    sparse_status_t mkl_sparse_set_memory_hint ( const sparse_matrix_t       A,
                                                 const sparse_memory_usage_t policy );    /* SPARSE_MEMORY_AGGRESSIVE is default value */

/*
    Optimize matrix described by the handle. It uses hints (optimization and memory) that should be set up before this call.
    If hints were not explicitly defined, default vales are:
    SPARSE_OPERATION_NON_TRANSPOSE for matrix-vector multiply with infinite number of expected iterations.
*/
    sparse_status_t mkl_sparse_optimize ( sparse_matrix_t  A );

/*****************************************************************************************/
/****************************** Computational routines ***********************************/
/*****************************************************************************************/

    sparse_status_t mkl_sparse_order( const sparse_matrix_t A );

/*
    Perform computations based on created matrix handle

    Level 2
*/
    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t mkl_sparse_s_mv ( const sparse_operation_t  operation,
                                      const float               alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const float               *x,
                                      const float               beta,
                                      float                     *y );

    sparse_status_t mkl_sparse_d_mv ( const sparse_operation_t  operation,
                                      const double              alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const double              *x,
                                      const double              beta,
                                      double                    *y );

    sparse_status_t mkl_sparse_c_mv ( const sparse_operation_t  operation,
                                      const MKL_Complex8        alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const MKL_Complex8        *x,
                                      const MKL_Complex8        beta,
                                      MKL_Complex8              *y );

    sparse_status_t mkl_sparse_z_mv ( const sparse_operation_t  operation,
                                      const MKL_Complex16       alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const MKL_Complex16       *x,
                                      const MKL_Complex16       beta,
                                      MKL_Complex16             *y );

    /*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner product */
    sparse_status_t mkl_sparse_s_dotmv( const sparse_operation_t  transA,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const float               *x,
                                        const float               beta,
                                        float                     *y,
                                        float                     *d);

    sparse_status_t mkl_sparse_d_dotmv( const sparse_operation_t  transA,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const double              *x,
                                        const double              beta,
                                        double                    *y,
                                        double                    *d);

    sparse_status_t mkl_sparse_c_dotmv( const sparse_operation_t  transA,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex8        *x,
                                        const MKL_Complex8        beta,
                                        MKL_Complex8              *y,
                                        MKL_Complex8              *d);

    sparse_status_t mkl_sparse_z_dotmv( const sparse_operation_t  transA,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex16       *x,
                                        const MKL_Complex16       beta,
                                        MKL_Complex16             *y,
                                        MKL_Complex16             *d);


    /*   Solves triangular system y = alpha * A^{-1} * x   */
    sparse_status_t mkl_sparse_s_trsv ( const sparse_operation_t  operation,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const float               *x,
                                        float                     *y );

    sparse_status_t mkl_sparse_d_trsv ( const sparse_operation_t  operation,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const double              *x,
                                        double                    *y );

    sparse_status_t mkl_sparse_c_trsv ( const sparse_operation_t  operation,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t    A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex8        *x,
                                        MKL_Complex8              *y );

    sparse_status_t mkl_sparse_z_trsv ( const sparse_operation_t  operation,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex16       *x,
                                        MKL_Complex16             *y );

    /*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x = b, */
    /*   that is, it solves:                                                          */
    /*      x0       = alpha*x                                                        */
    /*      (L+D)*x1 = b - U*x0                                                       */
    /*      (D+U)*x  = b - L*x1                                                       */
    /*                                                                                */
    /*   SYMGS_MV also returns y = A*x                                                */
    sparse_status_t mkl_sparse_s_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const float               alpha,
                                         const float               *b,
                                         float                     *x);

    sparse_status_t mkl_sparse_d_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const double              alpha,
                                         const double              *b,
                                         double                    *x);

    sparse_status_t mkl_sparse_c_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const MKL_Complex8        alpha,
                                         const MKL_Complex8        *b,
                                         MKL_Complex8              *x);

    sparse_status_t mkl_sparse_z_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const MKL_Complex16       alpha,
                                         const MKL_Complex16       *b,
                                         MKL_Complex16             *x);

    sparse_status_t mkl_sparse_s_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const float               alpha,
                                            const float               *b,
                                            float                     *x,
                                            float                     *y);

    sparse_status_t mkl_sparse_d_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const double              alpha,
                                            const double              *b,
                                            double                    *x,
                                            double                    *y);

    sparse_status_t mkl_sparse_c_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const MKL_Complex8        alpha,
                                            const MKL_Complex8        *b,
                                            MKL_Complex8              *x,
                                            MKL_Complex8              *y);

    sparse_status_t mkl_sparse_z_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const MKL_Complex16       alpha,
                                            const MKL_Complex16       *b,
                                            MKL_Complex16             *x,
                                            MKL_Complex16             *y);

    /*   Computes an action of a preconditioner
         which corresponds to the approximate matrix decomposition A ≈ (L+D)*E*(U+D)
         for the system Ax = b.

         L is lower triangular part of A
         U is upper triangular part of A
         D is diagonal values of A
         E is approximate diagonal inverse

         That is, it solves:
             r = rhs - A*x0
             (L + D)*E*(U + D)*dx = r
             x1 = x0 + dx                                        */

    sparse_status_t mkl_sparse_s_lu_smoother ( const sparse_operation_t  op,
                                               const sparse_matrix_t     A,
                                               const struct matrix_descr descr,
                                               const float               *diag,
                                               const float               *approx_diag_inverse,
                                               float                     *x,
                                               const float               *rhs);

    sparse_status_t mkl_sparse_d_lu_smoother ( const sparse_operation_t  op,
                                               const sparse_matrix_t     A,
                                               const struct matrix_descr descr,
                                               const double              *diag,
                                               const double              *approx_diag_inverse,
                                               double                    *x,
                                               const double              *rhs);

    sparse_status_t mkl_sparse_c_lu_smoother ( const sparse_operation_t  op,
                                               const sparse_matrix_t     A,
                                               const struct matrix_descr descr,
                                               const MKL_Complex8        *diag,
                                               const MKL_Complex8        *approx_diag_inverse,
                                               MKL_Complex8              *x,
                                               const MKL_Complex8        *rhs);

    sparse_status_t mkl_sparse_z_lu_smoother ( const sparse_operation_t  op,
                                               const sparse_matrix_t     A,
                                               const struct matrix_descr descr,
                                               const MKL_Complex16       *diag,
                                               const MKL_Complex16       *approx_diag_inverse,
                                               MKL_Complex16             *x,
                                               const MKL_Complex16       *rhs);

    /* Level 3 */

    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t mkl_sparse_s_mm( const sparse_operation_t  operation,
                                     const float               alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const float               *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const float               beta,
                                     float                     *y,
                                     const MKL_INT             ldy );

    sparse_status_t mkl_sparse_d_mm( const sparse_operation_t  operation,
                                     const double              alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const double              *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const double              beta,
                                     double                    *y,
                                     const MKL_INT             ldy );

    sparse_status_t mkl_sparse_c_mm( const sparse_operation_t  operation,
                                     const MKL_Complex8        alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const MKL_Complex8        *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const MKL_Complex8        beta,
                                     MKL_Complex8              *y,
                                     const MKL_INT             ldy );

    sparse_status_t mkl_sparse_z_mm( const sparse_operation_t  operation,
                                     const MKL_Complex16       alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const MKL_Complex16       *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const MKL_Complex16       beta,
                                     MKL_Complex16             *y,
                                     const MKL_INT             ldy );

    /*   Solves triangular system y = alpha * A^{-1} * x   */
    sparse_status_t mkl_sparse_s_trsm ( const sparse_operation_t  operation,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const float               *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        float                     *y,
                                        const MKL_INT             ldy );

    sparse_status_t mkl_sparse_d_trsm ( const sparse_operation_t  operation,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const double              *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        double                    *y,
                                        const MKL_INT             ldy );

    sparse_status_t mkl_sparse_c_trsm ( const sparse_operation_t  operation,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const MKL_Complex8        *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        MKL_Complex8              *y,
                                        const MKL_INT             ldy );

    sparse_status_t mkl_sparse_z_trsm ( const sparse_operation_t  operation,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const MKL_Complex16       *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        MKL_Complex16             *y,
                                        const MKL_INT             ldy );

    /* Sparse-sparse functionality */


    /*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse   */
    sparse_status_t mkl_sparse_s_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const float              alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t mkl_sparse_d_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const double             alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t mkl_sparse_c_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const MKL_Complex8       alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t mkl_sparse_z_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const MKL_Complex16      alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    /*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
    sparse_status_t mkl_sparse_spmm ( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
    sparse_status_t mkl_sparse_sp2m ( const sparse_operation_t  transA,
                                      const struct matrix_descr descrA,
                                      const sparse_matrix_t     A,
                                      const sparse_operation_t  transB,
                                      const struct matrix_descr descrB,
                                      const sparse_matrix_t     B,
                                      const sparse_request_t    request,
                                      sparse_matrix_t           *C );

    /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is sparse   */
    sparse_status_t mkl_sparse_syrk ( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      sparse_matrix_t          *C );


    /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is sparse   */
    sparse_status_t mkl_sparse_sypr ( const sparse_operation_t  transA,
                                      const sparse_matrix_t     A,
                                      const sparse_matrix_t     B,
                                      const struct matrix_descr descrB,
                                      sparse_matrix_t           *C,
                                      const sparse_request_t    request );

    /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is dense */
    sparse_status_t mkl_sparse_s_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const float              *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const float              alpha,
                                         const float              beta,
                                         float                    *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t mkl_sparse_d_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const double             *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const double             alpha,
                                         const double             beta,
                                         double                   *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t mkl_sparse_c_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const MKL_Complex8       *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const MKL_Complex8       alpha,
                                         const MKL_Complex8       beta,
                                         MKL_Complex8             *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t mkl_sparse_z_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const MKL_Complex16      *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const MKL_Complex16      alpha,
                                         const MKL_Complex16      beta,
                                         MKL_Complex16            *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );


    /*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
    sparse_status_t mkl_sparse_s_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        float                    *C,
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_d_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        double                   *C,
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_c_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        MKL_Complex8             *C,
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_z_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        MKL_Complex16            *C,
                                        const MKL_INT            ldc );

    /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
    sparse_status_t mkl_sparse_s_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const float               alpha,
                                         const float               beta,
                                         float                     *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t mkl_sparse_d_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const double              alpha,
                                         const double              beta,
                                         double                    *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t mkl_sparse_c_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const MKL_Complex8        alpha,
                                         const MKL_Complex8        beta,
                                         MKL_Complex8              *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t mkl_sparse_z_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const MKL_Complex16       alpha,
                                         const MKL_Complex16       beta,
                                         MKL_Complex16             *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is dense */
    sparse_status_t mkl_sparse_s_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const float              alpha,
                                        const float              beta,
                                        float                    *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_d_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const double             alpha,
                                        const double             beta,
                                        double                   *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_c_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const MKL_Complex8       alpha,
                                        const MKL_Complex8       beta,
                                        MKL_Complex8             *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t mkl_sparse_z_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const MKL_Complex16      alpha,
                                        const MKL_Complex16      beta,
                                        MKL_Complex16            *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    /* Computes forward or backward sweep of successive over-relaxation (SOR),
       or Symmetric successive over-relaxation (SSOR) */
    sparse_status_t mkl_sparse_s_sorv ( const sparse_sor_type_t   type,   /* choice of forward, backward sweep or SSOR operation */
                                        const struct matrix_descr descrA,
                                        const sparse_matrix_t     A,
                                              float               omega,
                                              float               alpha,  /* alpha equals to 0 mean zero initial guess */
                                              float*              x,      /* solution vector and alpha * x is initial guess */
                                        const float*              b );    /* right-hand side */

    sparse_status_t mkl_sparse_d_sorv ( const sparse_sor_type_t   type,   /* choice of forward, backward sweep or SSOR operation */
                                        const struct matrix_descr descrA,
                                        const sparse_matrix_t     A,
                                              double              omega,
                                              double              alpha,  /* alpha equals to 0 mean zero initial guess */
                                              double*             x,      /* solution vector and alpha * x is initial guess */
                                        const double*             b );    /* right-hand side */

#ifdef __cplusplus
}
#endif /*__cplusplus */
#endif /*_MKL_SPBLAS_H_ */
