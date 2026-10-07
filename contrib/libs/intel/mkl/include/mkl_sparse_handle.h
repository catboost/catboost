/*******************************************************************************
* Copyright 2004-2022 Intel Corporation.
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
*           Intel(R) oneAPI Math Kernel Library (oneMKL) DSS C header file
*
*           Contains more detailed information on internal datatypes and
*           constants used by DSS interface to PARDISO.
*
********************************************************************************
*/
#ifndef __MKL_SPARSE_HANDLE_H
#define __MKL_SPARSE_HANDLE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MKL_CALL_CONV
#   if defined(_WIN32) & !defined(_WIN64)
#       define MKL_CALL_CONV __cdecl
#   else
#       define MKL_CALL_CONV
#   endif
#endif

typedef enum { MKL_ZERO_BASED, MKL_ONE_BASED } sparse_matrix_indexing;
typedef enum { MKL_C_STYLE, MKL_FORTRAN_STYLE } sparse_matrix_print_styles;
typedef enum { MKL_CSR } sparse_matrix_formats;
typedef enum { MKL_GENERAL_STRUCTURE, MKL_UPPER_TRIANGULAR, MKL_LOWER_TRIANGULAR, MKL_STRUCTURAL_SYMMETRIC } sparse_matrix_structures;
typedef enum { MKL_NO_PRINT, MKL_PRINT } sparse_matrix_message_levels;
typedef enum { MKL_SPARSE_CHECKER_SUCCESS = 0, MKL_SPARSE_CHECKER_NON_MONOTONIC = 21, MKL_SPARSE_CHECKER_OUT_OF_RANGE = 22, MKL_SPARSE_CHECKER_NONTRIANGULAR = 23, MKL_SPARSE_CHECKER_NONORDERED = 24} sparse_checker_error_values;

typedef struct _sparse_struct {
    MKL_INT                      n, *csr_ia, *csr_ja, check_result[3];
    sparse_matrix_indexing       indexing;
    sparse_matrix_structures     matrix_structure;
    sparse_matrix_formats        matrix_format;
    sparse_matrix_message_levels message_level;
    sparse_matrix_print_styles   print_style;
} sparse_struct;

extern void    MKL_CALL_CONV sparse_matrix_checker_init    (sparse_struct*);
extern sparse_checker_error_values MKL_CALL_CONV sparse_matrix_checker (sparse_struct*);

#ifdef __cplusplus
}
#endif

#endif
