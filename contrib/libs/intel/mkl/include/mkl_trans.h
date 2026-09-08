/*******************************************************************************
* Copyright 2007-2022 Intel Corporation.
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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) include for transposition routines
!******************************************************************************/

#if !defined(_MKL_TRANS_H)
#define _MKL_TRANS_H

/* for size_t */
#include <stddef.h>
#include "mkl_types.h"
#include "mkl_trans_names.h"

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

/* In-place transposition routines */

void mkl_simatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const float alpha,
    float * AB, size_t lda, size_t ldb);

void mkl_dimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const double alpha,
    double * AB, size_t lda, size_t ldb);

void mkl_cimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    MKL_Complex8 * AB, size_t lda, size_t ldb);

void mkl_zimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    MKL_Complex16 * AB, size_t lda, size_t ldb);

/* Out-of-place transposition routines */

void mkl_somatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda,
    float * B, size_t ldb);

void mkl_domatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda,
    double * B, size_t ldb);

void mkl_comatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda,
    MKL_Complex8 * B, size_t ldb);

void mkl_zomatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda,
    MKL_Complex16 * B, size_t ldb);

/* Out-of-place transposition routines (all-strided case) */

void mkl_somatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda, size_t stridea,
    float * B, size_t ldb, size_t strideb);

void mkl_domatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda, size_t stridea,
    double * B, size_t ldb, size_t strideb);

void mkl_comatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda, size_t stridea,
    MKL_Complex8 * B, size_t ldb, size_t strideb);

void mkl_zomatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda, size_t stridea,
    MKL_Complex16 * B, size_t ldb, size_t strideb);

/* Out-of-place memory movement routines */

void mkl_somatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda,
    const float beta,
    const float * B, size_t ldb,
    float * C, size_t ldc);

void mkl_domatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda,
    const double beta,
    const double * B, size_t ldb,
    double * C, size_t ldc);

void mkl_comatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda,
    const MKL_Complex8 beta,
    const MKL_Complex8 * B, size_t ldb,
    MKL_Complex8 * C, size_t ldc);

void mkl_zomatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda,
    const MKL_Complex16 beta,
    const MKL_Complex16 * B, size_t ldb,
    MKL_Complex16 * C, size_t ldc);


/* Batch routines */

void mkl_simatcopy_batch_strided(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const float alpha,
    float * AB, size_t lda, size_t ldb,
    size_t stride, size_t batch_size) NOTHROW;

void mkl_dimatcopy_batch_strided(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const double alpha,
    double * AB, size_t lda, size_t ldb,
    size_t stride, size_t batch_size) NOTHROW;

void mkl_cimatcopy_batch_strided(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    MKL_Complex8 * AB, size_t lda, size_t ldb,
    size_t stride, size_t batch_size) NOTHROW;

void mkl_zimatcopy_batch_strided(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    MKL_Complex16 * AB, size_t lda, size_t ldb,
    size_t stride, size_t batch_size) NOTHROW;

void mkl_somatcopy_batch_strided(
    char ordering, char trans,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda, size_t stridea,
    float * B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

void mkl_domatcopy_batch_strided(
    char ordering, char trans,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda, size_t stridea,
    double * B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

void mkl_comatcopy_batch_strided(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda, size_t stridea,
    MKL_Complex8 * B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

void mkl_zomatcopy_batch_strided(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda, size_t stridea,
    MKL_Complex16 * B, size_t ldb, size_t strideb, size_t batch_size) NOTHROW;

void mkl_simatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array,
    float ** AB_array, const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

void mkl_dimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array,
    double ** AB_array, const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

void mkl_cimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array,
    MKL_Complex8 ** AB_array, const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

void mkl_zimatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array,
    MKL_Complex16 ** AB_array, const size_t * lda_array, const size_t * ldb_array,
    size_t group_count, const size_t * group_size) NOTHROW;

void mkl_somatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const float * alpha_array,
    const float ** A_array, const size_t * lda_array,
    float ** B, const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

void mkl_domatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const double * alpha_array,
    const double ** A_array, const size_t * lda_array,
    double ** B_array, const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

void mkl_comatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex8 * alpha_array,
    const MKL_Complex8 ** A_array, const size_t * lda_array,
    MKL_Complex8 ** B, const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

void mkl_zomatcopy_batch(
    char ordering, const char * trans_array,
    const size_t * rows_array, const size_t * cols_array,
    const MKL_Complex16 * alpha_array,
    const MKL_Complex16 ** A_array, const size_t * lda_array,
    MKL_Complex16 ** B, const size_t * ldb_array, size_t group_count,
    const size_t * group_size) NOTHROW;

void mkl_somatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const float alpha, const float * A, size_t lda, size_t stridea,
    const float beta, const float * B, size_t ldb, size_t strideb,
    float * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

void mkl_domatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const double alpha, const double * A, size_t lda, size_t stridea,
    const double beta, const double * B, size_t ldb, size_t strideb,
    double * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

void mkl_comatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha, const MKL_Complex8 * A, size_t lda, size_t stridea,
    const MKL_Complex8 beta, const MKL_Complex8 * B, size_t ldb, size_t strideb,
    MKL_Complex8 * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

void mkl_zomatadd_batch_strided(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha, const MKL_Complex16 * A, size_t lda, size_t stridea,
    const MKL_Complex16 beta, const MKL_Complex16 * B, size_t ldb, size_t strideb,
    MKL_Complex16 * C, size_t ldc, size_t stridec, size_t batch_size) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_TRANS_H */
