/* file: mkl_vml_types.h */
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
//  User-level type definitions.
//--
*/

#ifndef __MKL_VML_TYPES_H__
#define __MKL_VML_TYPES_H__

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
//  ERROR CALLBACK CONTEXT.
//  Error callback context structure is used in a user's error callback
//  function with the following interface:
//
//      int USER_CALLBACK_FUNC_NAME( DefVmlErrorContext par )
//
//  Error callback context fields:
//  iCode        - error status
//  iIndex       - index of bad argument
//  dbA1         - 1-st argument value, at which error occured
//  dbA2         - 2-nd argument value, at which error occured
//                 (2-argument functions only)
//  dbR1         - 1-st resulting value
//  dbR2         - 2-nd resulting value (2-result functions only)
//  cFuncName    - function name, for which error occured
//  iFuncNameLen - length of function name
*/
typedef struct _DefVmlErrorContext
{
    int     iCode;
    int     iIndex;
    double  dbA1;
    double  dbA2;
    double  dbR1;
    double  dbR2;
    char    cFuncName[64];
    int     iFuncNameLen;
    double  dbA1Im;
    double  dbA2Im;
    double  dbR1Im;
    double  dbR2Im;
} DefVmlErrorContext;

/*
// User error callback handler function type
*/
typedef int (*VMLErrorCallBack) (DefVmlErrorContext* pdefVmlErrorContext);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_TYPES_H__ */
