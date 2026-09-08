/*******************************************************************************
* Copyright 2020-2022 Intel Corporation.
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

#ifndef _MKL_GRAPHBLAS_H_
#define _MKL_GRAPHBLAS_H_

#include <stddef.h>
#include <stdint.h>
#include "mkl_types.h"

#define GRAPH_DEPRECATION_MESSAGE "WARNING: Intel oneMKL Graph APIs will be removed in the 2024.0 Beta and Gold releases."
#ifdef __GNUC__
#define MKL_GRAPH_DEPRECATED __attribute__((deprecated(GRAPH_DEPRECATION_MESSAGE)))
#elif defined(_MSC_VER)
#define MKL_GRAPH_DEPRECATED __declspec(deprecated(GRAPH_DEPRECATION_MESSAGE))
#else
#pragma message(GRAPH_DEPRECATION_MESSAGE)
#define MKL_GRAPH_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
*  Contents:
*  Objects, part 1: Opaque objects
*  Objects, part 2: Structures, enums and non-opaque objects
*  Methods, part 1: Creating amd destroying opaque objects
*  Methods, part 2: Manipulating objects and getting object properties
*  Methods, part 3: Main functionality
*/

/******************************************************************************
*  Objects, part 1: Opaque objects
******************************************************************************/

struct  mkl_graph_matrix;
typedef struct mkl_graph_matrix *mkl_graph_matrix_t;

struct  mkl_graph_vector;
typedef struct mkl_graph_vector *mkl_graph_vector_t;

struct  mkl_graph_descriptor;
typedef struct mkl_graph_descriptor *mkl_graph_descriptor_t;

/******************************************************************************
*  Objects, part 2: Structures, enums and non-opaque objects
******************************************************************************/

typedef enum
{
    MKL_GRAPH_STATUS_SUCCESS              = 0,    /* the operation was successful */
    MKL_GRAPH_STATUS_NOT_INITIALIZED      = 1,    /* empty object passed in */
    MKL_GRAPH_STATUS_ALLOC_FAILED         = 2,    /* internal error: memory allocation failed */
    MKL_GRAPH_STATUS_INVALID_VALUE        = 3,    /* invalid input value */
    MKL_GRAPH_STATUS_INTERNAL_ERROR       = 4,    /* internal error */
    MKL_GRAPH_STATUS_NOT_SUPPORTED        = 5     /* e.g. operation for double precision doesn't support other types */
} mkl_graph_status_t;

typedef enum
{
    MKL_GRAPH_SEMIRING_PLUS_TIMES_FP32    = 0,  /* standard +.x semiring for single precision */
    MKL_GRAPH_SEMIRING_PLUS_TIMES_FP64    = 1,  /* standard +.x semiring for double precision */
    MKL_GRAPH_SEMIRING_PLUS_TIMES_INT32   = 2,  /* standard +.x semiring for 32-bit integers  */
    MKL_GRAPH_SEMIRING_PLUS_TIMES_INT64   = 3,  /* standard +.x semiring for 64-bit integers  */
    MKL_GRAPH_SEMIRING_PLUS_FIRST_FP32    = 4,  /* standard +.x semiring ignoring the values of the second operand for single precision */
    MKL_GRAPH_SEMIRING_PLUS_SECOND_FP32   = 5,  /* standard +.x semiring ignoring the values of the left operand for single precision */
    MKL_GRAPH_SEMIRING_LOR_LAND_BOOL      = 6,  /* for simple BFS */
    MKL_GRAPH_SEMIRING_MIN_PLUS_INT32     = 7,  /*  */
    MKL_GRAPH_SEMIRING_MIN_PLUS_INT64     = 8,  /*  */
    MKL_GRAPH_SEMIRING_MIN_PLUS_FP32      = 9,  /*  */
    MKL_GRAPH_SEMIRING_MIN_PLUS_FP64      = 10, /*  */
    MKL_GRAPH_SEMIRING_MAX_FIRST_INT32    = 11, /*  */
    MKL_GRAPH_SEMIRING_MAX_FIRST_INT64    = 12, /*  */
    MKL_GRAPH_SEMIRING_MAX_FIRST_FP32     = 13, /*  */
    MKL_GRAPH_SEMIRING_MAX_FIRST_FP64     = 14, /*  */
    MKL_GRAPH_SEMIRING_ANY_FIRST_FP32     = 15, /* early-exit kernels without values of the second operand */
    MKL_GRAPH_SEMIRING_ANY_FIRST_INT32    = 16, /* early-exit kernels without values of the second operand */
    MKL_GRAPH_SEMIRING_ANY_SECOND_FP32    = 17, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_ANY_SECOND_FP64    = 18, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_ANY_SECOND_INT32   = 19, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_ANY_SECOND_INT64   = 20, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_ANY_PAIR_BOOL      = 21, /* early-exit kernels without any values read */
    MKL_GRAPH_SEMIRING_PLUS_PAIR_INT32    = 22, /*  */
    MKL_GRAPH_SEMIRING_PLUS_PAIR_INT64    = 23, /*  */
    MKL_GRAPH_SEMIRING_MIN_SECOND_INT32   = 24, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_MIN_SECOND_INT64   = 25, /* early-exit kernels without values of the first operand */
    MKL_GRAPH_SEMIRING_MIN_FIRST_INT32    = 26, /* early-exit kernels without values of the second operand */
    MKL_GRAPH_SEMIRING_MIN_FIRST_INT64    = 27  /* early-exit kernels without values of the second operand */
} mkl_graph_semiring_t;

typedef enum
{
    MKL_GRAPH_ACCUMULATOR_NONE                  = 0, /*  */
    MKL_GRAPH_ACCUMULATOR_PLUS                  = 1, /*  */
    MKL_GRAPH_ACCUMULATOR_LOR                   = 2, /*  */
    MKL_GRAPH_ACCUMULATOR_MIN                   = 3  /*  */
} mkl_graph_accumulator_t;

typedef enum
{
    MKL_GRAPH_TYPE_UNSET                  = -1, /* "no type was set" */
    MKL_GRAPH_TYPE_BOOL                   =  0, /*  */
    MKL_GRAPH_TYPE_INT32                  =  1, /*  */
    MKL_GRAPH_TYPE_INT64                  =  2, /*  */
    MKL_GRAPH_TYPE_FP32                   =  3, /*  */
    MKL_GRAPH_TYPE_FP64                   =  4  /*  */
} mkl_graph_type_t;

typedef enum
{
    MKL_GRAPH_PROPERTY_NROWS              = 0, /*  */
    MKL_GRAPH_PROPERTY_NCOLS              = 1, /*  */
    MKL_GRAPH_PROPERTY_NNZ                = 2, /*  */
    MKL_GRAPH_PROPERTY_MATRIX_HAS_CSR     = 3, /*  */
    MKL_GRAPH_PROPERTY_MATRIX_HAS_CSC     = 4, /*  */
    MKL_GRAPH_PROPERTY_VECTOR_HAS_DENSE   = 5, /*  */
    MKL_GRAPH_PROPERTY_VECTOR_HAS_SPARSE  = 6  /*  */
} mkl_graph_property_t;

typedef enum
{
    MKL_GRAPH_FIELD_OUTPUT                = 0, /* Field for the output modifiers */
    MKL_GRAPH_FIELD_FIRST_INPUT           = 1, /* Field for the first input modifiers */
    MKL_GRAPH_FIELD_SECOND_INPUT          = 2, /* Field for the second input modifiers */
    MKL_GRAPH_FIELD_MASK                  = 3  /* Field for mask modifiers */
} mkl_graph_descriptor_field_t;

typedef enum
{
    MKL_GRAPH_MOD_NONE                   = 0, /*  */
    MKL_GRAPH_MOD_COMPLEMENT             = 1, /*  */
    MKL_GRAPH_MOD_TRANSPOSE              = 2, /*  */
    MKL_GRAPH_MOD_REPLACE                = 3, /*  */
    MKL_GRAPH_MOD_ONLY_STRUCTURE         = 4, /*  */
    MKL_GRAPH_MOD_KEEP_MASK_STRUCTURE    = 5  /*  */
} mkl_graph_descriptor_field_value_t;

typedef enum
{
    MKL_GRAPH_METHOD_AUTO                 = 0, /* automatic choice */
    MKL_GRAPH_METHOD_DOT                  = 1, /* dot product */
    MKL_GRAPH_METHOD_GUSTAVSON            = 2, /* Gustavson */
    MKL_GRAPH_METHOD_HASH                 = 3  /* not yet supported */
} mkl_graph_method_t;

typedef enum
{
    MKL_GRAPH_REQUEST_COMPUTE_ALL  = 0, /* Compute all in a single stage */
    MKL_GRAPH_REQUEST_FILL_NNZ     = 1, /* First of two stages. Calculate nnz and fill related user-allocated buffer (rowStart or colStart for a matrix) */
    MKL_GRAPH_REQUEST_FILL_ENTRIES = 2  /* Second of two stages. Fill user-allocated buffers for indices and values of the entries */
} mkl_graph_request_t;

/******************************************************************************
*  Methods, part 1: Creating and destroying opaque graph objects
******************************************************************************/

/*
   [mkl_graph_<name>_create]  Allocates the memory for an internal
                              representation of a graph object and initializes
                              it to default values.
   [mkl_graph_<name>_destroy] Deallocates all internally allocated memory for an
                              object.
*/

/* For matrices */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_create (mkl_graph_matrix_t *A_pt);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_destroy(mkl_graph_matrix_t *A_pt);

/* For vectors */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_create (mkl_graph_vector_t *v_pt);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_destroy(mkl_graph_vector_t *v_pt);

/* For descriptors */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_descriptor_create (mkl_graph_descriptor_t *desc_pt);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_descriptor_destroy(mkl_graph_descriptor_t *desc_pt);


/******************************************************************************
*  Methods, part 2: Manipulating objects and getting object properties
******************************************************************************/

/* For matrices */

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_set_csr(mkl_graph_matrix_t A,
                                                                 int64_t nrows, int64_t ncols,
                                                                 void *rows_start, mkl_graph_type_t rows_start_type,
                                                                 void *col_indx,   mkl_graph_type_t col_indx_type,
                                                                 void  *values,    mkl_graph_type_t values_type);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_get_csr(mkl_graph_matrix_t A,
                                                                 int64_t *nrows_pt, int64_t *ncols_pt,
                                                                 void **rows_start_pt, mkl_graph_type_t *rows_start_type_pt,
                                                                 void **col_indx_pt,   mkl_graph_type_t *col_indx_type_pt,
                                                                 void **values_pt,     mkl_graph_type_t *values_type_pt);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_set_csc(mkl_graph_matrix_t A,
                                                                 int64_t nrows, int64_t ncols,
                                                                 void *cols_start, mkl_graph_type_t cols_start_type,
                                                                 void *row_indx,   mkl_graph_type_t row_indx_type,
                                                                 void *values,     mkl_graph_type_t values_type);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_get_csc(mkl_graph_matrix_t A, int64_t *nrows_pt, int64_t *ncols_pt,
                                                                 void **cols_start_pt, mkl_graph_type_t *cols_start_type_pt,
                                                                 void **row_indx_pt,   mkl_graph_type_t *row_indx_type_pt,
                                                                 void **values_pt,     mkl_graph_type_t *values_type_pt);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_matrix_get_property(mkl_graph_matrix_t A,
                                                                      mkl_graph_property_t property,
                                                                      void *value_pt);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_optimize_mxv(mkl_graph_vector_t mask, mkl_graph_semiring_t semiring, mkl_graph_matrix_t A,
                                                               mkl_graph_vector_t v, mkl_graph_descriptor_t desc, int64_t ncalls);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_optimize_mxm(mkl_graph_matrix_t Mask, mkl_graph_semiring_t semiring, mkl_graph_matrix_t A,
                                                               mkl_graph_matrix_t B, mkl_graph_descriptor_t desc, int64_t ncalls);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_transpose(mkl_graph_matrix_t C, mkl_graph_matrix_t Mask,
                                                            mkl_graph_accumulator_t accum, mkl_graph_matrix_t A,
                                                            mkl_graph_descriptor_t desc);

/* For vectors */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_set_dense(mkl_graph_vector_t v,
                                                                   int64_t  dim, void  *values,
                                                                   mkl_graph_type_t  values_type);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_get_dense(mkl_graph_vector_t v,
                                                                   int64_t *dim_pt, void **values_pt,
                                                                   mkl_graph_type_t *values_type_pt);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_set_sparse (mkl_graph_vector_t v,
                                                                     int64_t dim, int64_t nnz,
                                                                     void *indices, mkl_graph_type_t indices_type,
                                                                     void *values, mkl_graph_type_t  values_type);
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_get_sparse (mkl_graph_vector_t v, int64_t *dim_pt, int64_t *nnz_pt,
                                                                     void **indices_pt, mkl_graph_type_t* indices_type_pt,
                                                                     void **values_pt,  mkl_graph_type_t* values_type_pt);

MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vector_get_property(mkl_graph_vector_t v,
                                                                      mkl_graph_property_t property,
                                                                      void *value_pt);

/* For descriptors */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_descriptor_set_field(mkl_graph_descriptor_t desc,
                                                                       mkl_graph_descriptor_field_t field,
                                                                       mkl_graph_descriptor_field_value_t value);

/******************************************************************************
*  Methods, part 3: Main functionality
******************************************************************************/

/* mxm: computes C<M> = accum(C, A +.x B) with A, B and M possibly modified via
 * descriptor desc */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_mxm (mkl_graph_matrix_t C, mkl_graph_matrix_t M,
                                                       mkl_graph_accumulator_t accum,
                                                       mkl_graph_semiring_t semiring,
                                                       mkl_graph_matrix_t A, mkl_graph_matrix_t B,
                                                       mkl_graph_descriptor_t desc,
                                                       mkl_graph_request_t request,
                                                       mkl_graph_method_t method);

/* mxv: computes w<mask> = accum(w, A +.x u) */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_mxv (mkl_graph_vector_t w, mkl_graph_vector_t mask,
                                                       mkl_graph_accumulator_t accum,
                                                       mkl_graph_semiring_t semiring,
                                                       mkl_graph_matrix_t A, mkl_graph_vector_t u,
                                                       mkl_graph_descriptor_t desc,
                                                       mkl_graph_request_t request,
                                                       mkl_graph_method_t method);

/* vxm: computes w<mask> = accum(w, u +.x A) */
MKL_GRAPH_DEPRECATED mkl_graph_status_t mkl_graph_vxm (mkl_graph_vector_t w, mkl_graph_vector_t mask,
                                                       mkl_graph_accumulator_t accum,
                                                       mkl_graph_semiring_t semiring,
                                                       mkl_graph_vector_t u, mkl_graph_matrix_t A,
                                                       mkl_graph_descriptor_t desc,
                                                       mkl_graph_request_t request,
                                                       mkl_graph_method_t method);

#ifdef __cplusplus
}
#endif /*__cplusplus */

#endif /*_MKL_GRAPHBLAS_H_ */


