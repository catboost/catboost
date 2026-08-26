/* file: mkl_df_types.h */
/*******************************************************************************
* Copyright 2006-2022 Intel Corporation.
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
//++
//  This file contains user-level type definitions.
//--
*/

#ifndef __MKL_DF_TYPES_H__
#define __MKL_DF_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "mkl_types.h"

/*
//++
//  TYPEDEFS
//--
*/

/*
//  POINTER TO DATA FITTING STRUCTURE
//  This is a void pointer to hide implementation details.
*/
typedef void* DFTaskPtr;

/*
//  DATA FITTING SEARCH CALLBACK INTERNAL PARAMETERS STRUCTURE
*/
typedef struct _dfSearchCallBackLibraryParams{
    int limit_type_flag;
} dfSearchCallBackLibraryParams;

/*
//  DATA FITTING INTERPOLATION CALLBACK INTERNAL PARAMETERS STRUCTURE
*/
typedef struct _dfInterpCallBackLibraryParams{
    int reserved1;
} dfInterpCallBackLibraryParams;

/*
//  DATA FITTING INTEGRATION CALLBACK INTERNAL PARAMETERS STRUCTURE
*/
typedef struct _dfIntegrCallBackLibraryParams{
    int reserved1;
} dfIntegrCallBackLibraryParams;

/*
//  DATA FITTING CALLBACK FOR SUPPORT OF USER-DEFINED INTERPOLATION AND
//  EXTRAPOLATION
*/
typedef int (*dfsInterpCallBack) ( MKL_INT64* n, MKL_INT64 cell[], float  site[], float  r[], void* user_param, dfInterpCallBackLibraryParams* library_params );
typedef int (*dfdInterpCallBack) ( MKL_INT64* n, MKL_INT64 cell[], double site[], double r[], void* user_param, dfInterpCallBackLibraryParams* library_params );

/*
//  DATA FITTING CALLBACK FOR SUPPORT OF USER-DEFINED INTEGRATION
*/
typedef int (*dfsIntegrCallBack) ( MKL_INT64* n, MKL_INT64 lcell[], float  llim[], MKL_INT64 rcell[], float  rlim[], float  r[], void* user_params, dfIntegrCallBackLibraryParams* library_params );
typedef int (*dfdIntegrCallBack) ( MKL_INT64* n, MKL_INT64 lcell[], double llim[], MKL_INT64 rcell[], double rlim[], double r[], void* user_params, dfIntegrCallBackLibraryParams* library_params );

/*
//  DATA FITTING CALLBACK FOR SUPPORT OF USER-DEFINED CELL SEARCH
*/
typedef int (*dfsSearchCellsCallBack) ( MKL_INT64* n, float  site[], MKL_INT64 cell[], int flag[], void* user_params, dfSearchCallBackLibraryParams* library_params );
typedef int (*dfdSearchCellsCallBack) ( MKL_INT64* n, double site[], MKL_INT64 cell[], int flag[], void* user_params, dfSearchCallBackLibraryParams* library_params );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_DF_TYPES_H__ */
