/* file: mkl_vml.h */
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
//  VML main header file. To use VML it is sufficient to include
//  mkl_vml.h only.
//--
*/

#ifndef __MKL_VML_H__
#define __MKL_VML_H__

/*
// Latest versions of the Microsoft (R) C/C++ Optimizing Compiler emit Spectre v1
// warning C5045 on vulnerable memory loads and suggests using /Qspectre to mitigate.
// Adding the switch still results in warnings being emitted.
// This disables the warning for oneMKL VML applications until this compiler issue
// is resolved by Microsoft.
*/
#if defined _MSC_VER && !(defined __INTEL_COMPILER || defined __INTEL_LLVM_COMPILER)
#pragma warning(disable :5045)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "mkl_vml_defines.h"
#include "mkl_vml_types.h"
#include "mkl_vml_functions.h"

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_H__ */
