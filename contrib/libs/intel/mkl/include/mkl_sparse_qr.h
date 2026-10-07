/*******************************************************************************
* Copyright 2018-2022 Intel Corporation.
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
*   Content:
*           Intel(R) oneAPI Math Kernel Library (oneMKL) Sparse QR C header file
*
*           Contains interface to: MKL_SPARSE_X_QR
*                                  MKL_SPARSE_QR_REORDER
*                                  MKL_SPARSE_X_QR_FACTORIZE
*                                  MKL_SPARSE_X_QR_SOLVE
*                                  MKL_SPARSE_X_QR_QMULT
*                                  MKL_SPARSE_X_QR_RSOLVE
*                                  MKL_SPARSE_SET_QR_HINT
*
********************************************************************************
*/

#ifndef _MKL_SPARSE_QR_H_
#define _MKL_SPARSE_QR_H_

#include "mkl_types.h"
#include "mkl_spblas.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef enum { SPARSE_QR_WITH_PIVOTS } sparse_qr_hint_t;

sparse_status_t mkl_sparse_set_qr_hint    ( sparse_matrix_t A, sparse_qr_hint_t hint );

sparse_status_t mkl_sparse_d_qr           ( sparse_operation_t operation, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, MKL_INT columns, double *x, MKL_INT ldx, const double *b, MKL_INT ldb );
sparse_status_t mkl_sparse_s_qr           ( sparse_operation_t operation, sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, MKL_INT columns, float  *x, MKL_INT ldx, const float  *b, MKL_INT ldb );

sparse_status_t mkl_sparse_qr_reorder     ( sparse_matrix_t A, struct matrix_descr descr );

sparse_status_t mkl_sparse_d_qr_factorize ( sparse_matrix_t A, double *alt_values );
sparse_status_t mkl_sparse_s_qr_factorize ( sparse_matrix_t A, float  *alt_values );

sparse_status_t mkl_sparse_d_qr_solve     ( sparse_operation_t operation, sparse_matrix_t A, double *alt_values, sparse_layout_t layout, MKL_INT columns, double *x, MKL_INT ldx, const double *b, MKL_INT ldb );
sparse_status_t mkl_sparse_s_qr_solve     ( sparse_operation_t operation, sparse_matrix_t A, float  *alt_values, sparse_layout_t layout, MKL_INT columns, float  *x, MKL_INT ldx, const float  *b, MKL_INT ldb );

sparse_status_t mkl_sparse_d_qr_qmult     ( sparse_operation_t operation, sparse_matrix_t A, sparse_layout_t layout, MKL_INT columns, double *x, MKL_INT ldx, const double *b, MKL_INT ldb );
sparse_status_t mkl_sparse_s_qr_qmult     ( sparse_operation_t operation, sparse_matrix_t A, sparse_layout_t layout, MKL_INT columns, float  *x, MKL_INT ldx, const float  *b, MKL_INT ldb );

sparse_status_t mkl_sparse_d_qr_rsolve    ( sparse_operation_t operation, sparse_matrix_t A, sparse_layout_t layout, MKL_INT columns, double *x, MKL_INT ldx, const double *b, MKL_INT ldb );
sparse_status_t mkl_sparse_s_qr_rsolve    ( sparse_operation_t operation, sparse_matrix_t A, sparse_layout_t layout, MKL_INT columns, float  *x, MKL_INT ldx, const float  *b, MKL_INT ldb );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
