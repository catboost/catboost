/*******************************************************************************
* Copyright 2004-2017 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
*   Content:
*           Intel(R) Math Kernel Library (Intel(R) MKL) DSS C header file
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
#       if defined(MKL_STDCALL)
#           define MKL_CALL_CONV __stdcall
#       else
#           define MKL_CALL_CONV __cdecl
#       endif
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
