/* file: mkl_vml_types.h */
/*******************************************************************************
* Copyright 2006-2017 Intel Corporation All Rights Reserved.
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
