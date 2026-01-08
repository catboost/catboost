/* file: mkl_vml_functions.h */
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
//  User-level VML function declarations
//--
*/

#ifndef __MKL_VML_FUNCTIONS_H__
#define __MKL_VML_FUNCTIONS_H__

#include "mkl_vml_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
//++
//  EXTERNAL API MACROS.
//  Used to construct VML function declaration. Change them if you are going to
//  provide different API for VML functions.
//--
*/

#if  !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)   extern rtype name    arg;
#endif

#if  !defined(_mkl_api)
#define _mkl_api(rtype,name,arg)   extern rtype name##_ arg;
#endif

#if  !defined(_MKL_API)
#define _MKL_API(rtype,name,arg)   extern rtype name##_ arg;
#endif

/*
//++
//  VML ELEMENTARY FUNCTION DECLARATIONS.
//--
*/
/* Absolute value: r[i] = |a[i]| */
_MKL_API(void,VSABS,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDABS,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsabs,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdabs,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAbs,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAbs,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSABS,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDABS,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsabs,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdabs,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAbs,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAbs,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex absolute value: r[i] = |a[i]| */
_MKL_API(void,VCABS,(const MKL_INT *n, const MKL_Complex8  a[], float  r[]))
_MKL_API(void,VZABS,(const MKL_INT *n, const MKL_Complex16 a[], double r[]))
_mkl_api(void,vcabs,(const MKL_INT *n, const MKL_Complex8  a[], float  r[]))
_mkl_api(void,vzabs,(const MKL_INT *n, const MKL_Complex16 a[], double r[]))
_Mkl_Api(void,vcAbs,(const MKL_INT n,  const MKL_Complex8  a[], float  r[]))
_Mkl_Api(void,vzAbs,(const MKL_INT n,  const MKL_Complex16 a[], double r[]))

_MKL_API(void,VMCABS,(const MKL_INT *n, const MKL_Complex8  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMZABS,(const MKL_INT *n, const MKL_Complex16 a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmcabs,(const MKL_INT *n, const MKL_Complex8  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmzabs,(const MKL_INT *n, const MKL_Complex16 a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAbs,(const MKL_INT n,  const MKL_Complex8  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAbs,(const MKL_INT n,  const MKL_Complex16 a[], double r[], MKL_INT64 mode))

/* Argument of complex value: r[i] = carg(a[i]) */
_MKL_API(void,VCARG,(const MKL_INT *n, const MKL_Complex8  a[], float  r[]))
_MKL_API(void,VZARG,(const MKL_INT *n, const MKL_Complex16 a[], double r[]))
_mkl_api(void,vcarg,(const MKL_INT *n, const MKL_Complex8  a[], float  r[]))
_mkl_api(void,vzarg,(const MKL_INT *n, const MKL_Complex16 a[], double r[]))
_Mkl_Api(void,vcArg,(const MKL_INT n,  const MKL_Complex8  a[], float  r[]))
_Mkl_Api(void,vzArg,(const MKL_INT n,  const MKL_Complex16 a[], double r[]))

_MKL_API(void,VMCARG,(const MKL_INT *n, const MKL_Complex8  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMZARG,(const MKL_INT *n, const MKL_Complex16 a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmcarg,(const MKL_INT *n, const MKL_Complex8  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmzarg,(const MKL_INT *n, const MKL_Complex16 a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcArg,(const MKL_INT n,  const MKL_Complex8  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzArg,(const MKL_INT n,  const MKL_Complex16 a[], double r[], MKL_INT64 mode))

/* Addition: r[i] = a[i] + b[i] */
_MKL_API(void,VSADD,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDADD,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vsadd,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdadd,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsAdd,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdAdd,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSADD,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDADD,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsadd,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdadd,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAdd,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAdd,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complex addition: r[i] = a[i] + b[i] */
_MKL_API(void,VCADD,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZADD,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcadd,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzadd,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAdd,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAdd,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCADD,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZADD,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcadd,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzadd,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAdd,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAdd,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Subtraction: r[i] = a[i] - b[i] */
_MKL_API(void,VSSUB,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDSUB,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vssub,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdsub,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsSub,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdSub,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSSUB,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDSUB,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmssub,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdsub,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSub,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdSub,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complex subtraction: r[i] = a[i] - b[i] */
_MKL_API(void,VCSUB,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZSUB,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcsub,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzsub,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcSub,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzSub,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCSUB,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZSUB,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcsub,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzsub,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcSub,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzSub,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Reciprocal: r[i] = 1.0 / a[i] */
_MKL_API(void,VSINV,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDINV,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsinv,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdinv,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsInv,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdInv,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSINV,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDINV,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsinv,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdinv,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsInv,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdInv,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Square root: r[i] = a[i]^0.5 */
_MKL_API(void,VSSQRT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDSQRT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vssqrt,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdsqrt,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsSqrt,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdSqrt,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSSQRT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDSQRT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmssqrt,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdsqrt,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSqrt,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdSqrt,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex square root: r[i] = a[i]^0.5 */
_MKL_API(void,VCSQRT,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZSQRT,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcsqrt,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzsqrt,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcSqrt,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzSqrt,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCSQRT,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZSQRT,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcsqrt,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzsqrt,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcSqrt,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzSqrt,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Reciprocal square root: r[i] = 1/a[i]^0.5 */
_MKL_API(void,VSINVSQRT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDINVSQRT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsinvsqrt,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdinvsqrt,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsInvSqrt,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdInvSqrt,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSINVSQRT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDINVSQRT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsinvsqrt,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdinvsqrt,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsInvSqrt,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdInvSqrt,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Cube root: r[i] = a[i]^(1/3) */
_MKL_API(void,VSCBRT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCBRT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vscbrt,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdcbrt,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCbrt,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCbrt,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCBRT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCBRT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmscbrt,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdcbrt,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCbrt,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCbrt,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Reciprocal cube root: r[i] = 1/a[i]^(1/3) */
_MKL_API(void,VSINVCBRT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDINVCBRT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsinvcbrt,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdinvcbrt,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsInvCbrt,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdInvCbrt,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSINVCBRT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDINVCBRT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsinvcbrt,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdinvcbrt,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsInvCbrt,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdInvCbrt,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Squaring: r[i] = a[i]^2 */
_MKL_API(void,VSSQR,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDSQR,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vssqr,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdsqr,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsSqr,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdSqr,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSSQR,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDSQR,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmssqr,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdsqr,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSqr,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdSqr,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Exponential function: r[i] = e^a[i] */
_MKL_API(void,VSEXP,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDEXP,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsexp,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdexp,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsExp,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdExp,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSEXP,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDEXP,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsexp,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdexp,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsExp,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdExp,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex exponential function: r[i] = e^a[i] */
_MKL_API(void, VCEXP, (const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void, VZEXP, (const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void, vcexp, (const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void, vzexp, (const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void, vcExp, (const MKL_INT n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void, vzExp, (const MKL_INT n, const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void, VMCEXP, (const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void, VMZEXP, (const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void, vmcexp, (const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void, vmzexp, (const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void, vmcExp, (const MKL_INT n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void, vmzExp, (const MKL_INT n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Exponential function (base 2): r[i] = 2^a[i] */
_MKL_API(void, VSEXP2, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDEXP2, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsexp2, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdexp2, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsExp2, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdExp2, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSEXP2, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDEXP2, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsexp2, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdexp2, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsExp2, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdExp2, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Exponential function (base 10): r[i] = 10^a[i] */
_MKL_API(void, VSEXP10, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDEXP10, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsexp10, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdexp10, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsExp10, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdExp10, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSEXP10, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDEXP10, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsexp10, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdexp10, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsExp10, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdExp10, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Exponential of arguments decreased by 1: r[i] = e^(a[i]-1) */
_MKL_API(void,VSEXPM1,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDEXPM1,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsexpm1,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdexpm1,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsExpm1,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdExpm1,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSEXPM1,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDEXPM1,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsexpm1,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdexpm1,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsExpm1,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdExpm1,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Logarithm (base e): r[i] = ln(a[i]) */
_MKL_API(void,VSLN,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDLN,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsln,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdln,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsLn,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdLn,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSLN,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDLN,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsln,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdln,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsLn,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdLn,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex logarithm (base e): r[i] = ln(a[i]) */
_MKL_API(void,VCLN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZLN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcln,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzln,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcLn,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzLn,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCLN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZLN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcln,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzln,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcLn,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzLn,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Logarithm (base 2): r[i] = lb(a[i]) */
_MKL_API(void, VSLOG2, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDLOG2, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vslog2, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdlog2, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsLog2, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdLog2, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSLOG2, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDLOG2, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmslog2, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdlog2, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsLog2, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdLog2, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Logarithm (base 10): r[i] = lg(a[i]) */
_MKL_API(void,VSLOG10,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDLOG10,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vslog10,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdlog10,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsLog10,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdLog10,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSLOG10,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDLOG10,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmslog10,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdlog10,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsLog10,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdLog10,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex logarithm (base 10): r[i] = lg(a[i]) */
_MKL_API(void,VCLOG10,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZLOG10,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vclog10,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzlog10,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcLog10,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzLog10,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCLOG10,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZLOG10,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmclog10,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzlog10,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcLog10,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzLog10,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Logarithm (base e) of arguments increased by 1: r[i] = log(1+a[i]) */
_MKL_API(void,VSLOG1P,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDLOG1P,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vslog1p,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdlog1p,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsLog1p,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdLog1p,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSLOG1P,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDLOG1P,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmslog1p,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdlog1p,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsLog1p,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdLog1p,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Computes the exponent: r[i] = logb(a[i]) */
_MKL_API(void, VSLOGB, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDLOGB, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vslogb, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdlogb, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsLogb, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdLogb, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSLOGB, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDLOGB, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmslogb, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdlogb, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsLogb, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdLogb, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Cosine: r[i] = cos(a[i]) */
_MKL_API(void,VSCOS,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCOS,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vscos,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdcos,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCos,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCos,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCOS,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCOS,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmscos,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdcos,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCos,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCos,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex cosine: r[i] = ccos(a[i]) */
_MKL_API(void,VCCOS,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZCOS,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vccos,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzcos,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcCos,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzCos,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCCOS,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZCOS,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmccos,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzcos,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcCos,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzCos,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Sine: r[i] = sin(a[i]) */
_MKL_API(void,VSSIN,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDSIN,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vssin,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdsin,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsSin,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdSin,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSSIN,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDSIN,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmssin,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdsin,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSin,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdSin,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex sine: r[i] = sin(a[i]) */
_MKL_API(void,VCSIN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZSIN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcsin,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzsin,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcSin,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzSin,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCSIN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZSIN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcsin,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzsin,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcSin,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzSin,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Tangent: r[i] = tan(a[i]) */
_MKL_API(void,VSTAN,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDTAN,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vstan,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdtan,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsTan,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdTan,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSTAN,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDTAN,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmstan,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdtan,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsTan,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdTan,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex tangent: r[i] = tan(a[i]) */
_MKL_API(void,VCTAN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZTAN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vctan,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vztan,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcTan,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzTan,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCTAN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZTAN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmctan,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmztan,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcTan,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzTan,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Cosine PI: r[i] = cos(a[i]*PI) */
_MKL_API(void, VSCOSPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDCOSPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vscospi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdcospi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsCospi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdCospi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSCOSPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDCospi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmscospi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdcospi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsCospi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdCospi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Sine PI: r[i] = sin(a[i]*PI) */
_MKL_API(void, VSSINPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDSINPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vssinpi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdsinpi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsSinpi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdSinpi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSSINPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDSinpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmssinpi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdsinpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsSinpi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdSinpi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Tangent PI: r[i] = tan(a[i]*PI) */
_MKL_API(void, VSTANPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDTANPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vstanpi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdtanpi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsTanpi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdTanpi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSTANPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDTanpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmstanpi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdtanpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsTanpi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdTanpi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Cosine degree: r[i] = cos(a[i]*PI/180) */
_MKL_API(void, VSCOSD, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDCOSD, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vscosd, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdcosd, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsCosd, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdCosd, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSCOSD, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDCosd, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmscosd, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdcosd, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsCosd, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdCosd, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Sine degree: r[i] = sin(a[i]*PI/180) */
_MKL_API(void, VSSIND, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDSIND, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vssind, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdsind, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsSind, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdSind, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSSIND, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDSind, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmssind, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdsind, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsSind, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdSind, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Tangent degree: r[i] = tan(a[i]*PI/180) */
_MKL_API(void, VSTAND, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDTAND, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vstand, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdtand, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsTand, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdTand, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSTAND, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDTand, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmstand, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdtand, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsTand, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdTand, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Hyperbolic cosine: r[i] = ch(a[i]) */
_MKL_API(void,VSCOSH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCOSH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vscosh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdcosh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCosh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCosh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCOSH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCOSH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmscosh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdcosh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCosh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCosh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic cosine: r[i] = ch(a[i]) */
_MKL_API(void,VCCOSH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZCOSH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vccosh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzcosh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcCosh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzCosh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCCOSH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZCOSH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmccosh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzcosh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcCosh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzCosh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Hyperbolic sine: r[i] = sh(a[i]) */
_MKL_API(void,VSSINH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDSINH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vssinh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdsinh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsSinh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdSinh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSSINH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDSINH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmssinh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdsinh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSinh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdSinh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic sine: r[i] = sh(a[i]) */
_MKL_API(void,VCSINH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZSINH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcsinh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzsinh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcSinh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzSinh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCSINH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZSINH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcsinh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzsinh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcSinh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzSinh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Hyperbolic tangent: r[i] = th(a[i]) */
_MKL_API(void,VSTANH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDTANH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vstanh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdtanh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsTanh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdTanh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSTANH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDTANH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmstanh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdtanh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsTanh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdTanh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic tangent: r[i] = th(a[i]) */
_MKL_API(void,VCTANH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZTANH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vctanh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vztanh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcTanh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzTanh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCTANH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZTANH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmctanh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmztanh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcTanh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzTanh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Arc cosine: r[i] = arccos(a[i]) */
_MKL_API(void,VSACOS,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDACOS,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsacos,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdacos,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAcos,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAcos,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSACOS,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDACOS,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsacos,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdacos,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAcos,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAcos,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex arc cosine: r[i] = arccos(a[i]) */
_MKL_API(void,VCACOS,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZACOS,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcacos,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzacos,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAcos,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAcos,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCACOS,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZACOS,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcacos,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzacos,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAcos,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAcos,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Arc sine: r[i] = arcsin(a[i]) */
_MKL_API(void,VSASIN,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDASIN,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsasin,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdasin,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAsin,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAsin,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSASIN,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDASIN,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsasin,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdasin,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAsin,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAsin,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex arc sine: r[i] = arcsin(a[i]) */
_MKL_API(void,VCASIN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZASIN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcasin,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzasin,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAsin,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAsin,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCASIN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZASIN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcasin,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzasin,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAsin,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAsin,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Arc tangent: r[i] = arctan(a[i]) */
_MKL_API(void,VSATAN,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDATAN,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsatan,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdatan,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAtan,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAtan,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSATAN,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDATAN,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsatan,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdatan,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAtan,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAtan,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex arc tangent: r[i] = arctan(a[i]) */
_MKL_API(void,VCATAN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZATAN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcatan,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzatan,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAtan,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAtan,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCATAN,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZATAN,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcatan,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzatan,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAtan,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAtan,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Arc cosine PI: r[i] = arccos(a[i])/PI */
_MKL_API(void, VSACOSPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDACOSPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsacospi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdacospi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsAcospi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdAcospi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSACOSPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDAcospi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsacospi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdacospi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsAcospi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdAcospi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Arc sine PI: r[i] = arcsin(a[i])/PI */
_MKL_API(void, VSASINPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDASINPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsasinpi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdasinpi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsAsinpi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdAsinpi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSASINPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDAsinpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsasinpi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdasinpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsAsinpi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdAsinpi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Arc tangent PI: r[i] = arctan(a[i])/PI */
_MKL_API(void, VSATANPI, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDATANPI, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsatanpi, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdatanpi, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsAtanpi, (const MKL_INT n, const float  a[], float  r[]))
_Mkl_Api(void, vdAtanpi, (const MKL_INT n, const double a[], double r[]))

_MKL_API(void, VMSATANPI, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDAtanpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsatanpi, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdatanpi, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsAtanpi, (const MKL_INT n, const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdAtanpi, (const MKL_INT n, const double a[], double r[], MKL_INT64 mode))

/* Hyperbolic arc cosine: r[i] = arcch(a[i]) */
_MKL_API(void,VSACOSH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDACOSH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsacosh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdacosh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAcosh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAcosh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSACOSH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDACOSH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsacosh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdacosh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAcosh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAcosh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic arc cosine: r[i] = arcch(a[i]) */
_MKL_API(void,VCACOSH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZACOSH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcacosh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzacosh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAcosh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAcosh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCACOSH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZACOSH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcacosh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzacosh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAcosh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAcosh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Hyperbolic arc sine: r[i] = arcsh(a[i]) */
_MKL_API(void,VSASINH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDASINH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsasinh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdasinh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAsinh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAsinh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSASINH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDASINH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsasinh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdasinh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAsinh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAsinh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic arc sine: r[i] = arcsh(a[i]) */
_MKL_API(void,VCASINH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZASINH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcasinh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzasinh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAsinh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAsinh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCASINH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZASINH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcasinh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzasinh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAsinh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAsinh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Hyperbolic arc tangent: r[i] = arcth(a[i]) */
_MKL_API(void,VSATANH,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDATANH,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsatanh,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdatanh,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsAtanh,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdAtanh,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSATANH,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDATANH,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsatanh,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdatanh,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAtanh,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAtanh,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Complex hyperbolic arc tangent: r[i] = arcth(a[i]) */
_MKL_API(void,VCATANH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZATANH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcatanh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzatanh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcAtanh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzAtanh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCATANH,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZATANH,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcatanh,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzatanh,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcAtanh,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzAtanh,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Error function: r[i] = erf(a[i]) */
_MKL_API(void,VSERF,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDERF,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vserf,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vderf,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsErf,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdErf,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSERF,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDERF,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmserf,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmderf,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsErf,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdErf,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Inverse error function: r[i] = erfinv(a[i]) */
_MKL_API(void,VSERFINV,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDERFINV,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vserfinv,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vderfinv,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsErfInv,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdErfInv,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSERFINV,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDERFINV,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmserfinv,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmderfinv,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsErfInv,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdErfInv,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Square root of the sum of the squares: r[i] = hypot(a[i],b[i]) */
_MKL_API(void,VSHYPOT,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDHYPOT,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vshypot,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdhypot,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsHypot,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdHypot,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSHYPOT,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDHYPOT,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmshypot,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdhypot,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsHypot,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdHypot,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complementary error function: r[i] = 1 - erf(a[i]) */
_MKL_API(void,VSERFC,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDERFC,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vserfc,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vderfc,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsErfc,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdErfc,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSERFC,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDERFC,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmserfc,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmderfc,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsErfc,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdErfc,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Inverse complementary error function: r[i] = erfcinv(a[i]) */
_MKL_API(void,VSERFCINV,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDERFCINV,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vserfcinv,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vderfcinv,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsErfcInv,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdErfcInv,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSERFCINV,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDERFCINV,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmserfcinv,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmderfcinv,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsErfcInv,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdErfcInv,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Cumulative normal distribution function: r[i] = cdfnorm(a[i]) */
_MKL_API(void,VSCDFNORM,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCDFNORM,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vscdfnorm,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdcdfnorm,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCdfNorm,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCdfNorm,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCDFNORM,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCDFNORM,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmscdfnorm,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdcdfnorm,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCdfNorm,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCdfNorm,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Inverse cumulative normal distribution function: r[i] = cdfnorminv(a[i]) */
_MKL_API(void,VSCDFNORMINV,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCDFNORMINV,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vscdfnorminv,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdcdfnorminv,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCdfNormInv,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCdfNormInv,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCDFNORMINV,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCDFNORMINV,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmscdfnorminv,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdcdfnorminv,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCdfNormInv,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCdfNormInv,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Logarithm (base e) of the absolute value of gamma function: r[i] = lgamma(a[i]) */
_MKL_API(void,VSLGAMMA,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDLGAMMA,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vslgamma,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdlgamma,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsLGamma,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdLGamma,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSLGAMMA,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDLGAMMA,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmslgamma,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdlgamma,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsLGamma,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdLGamma,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Gamma function: r[i] = tgamma(a[i]) */
_MKL_API(void,VSTGAMMA,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDTGAMMA,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vstgamma,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdtgamma,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsTGamma,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdTGamma,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSTGAMMA,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDTGAMMA,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmstgamma,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdtgamma,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsTGamma,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdTGamma,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Arc tangent of a/b: r[i] = arctan(a[i]/b[i]) */
_MKL_API(void,VSATAN2,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDATAN2,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vsatan2,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdatan2,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsAtan2,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdAtan2,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSATAN2,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDATAN2,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsatan2,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdatan2,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsAtan2,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdAtan2,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Arc tangent of a/b divided by PI: r[i] = arctan(a[i]/b[i])/PI */
_MKL_API(void, VSATAN2PI, (const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void, VDATAN2PI, (const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void, vsatan2pi, (const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void, vdatan2pi, (const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void, vsAtan2pi, (const MKL_INT n, const float  a[], const float  b[], float  r[]))
_Mkl_Api(void, vdAtan2pi, (const MKL_INT n, const double a[], const double b[], double r[]))

_MKL_API(void, VMSATAN2PI, (const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDATAN2PI, (const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsatan2pi, (const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdatan2pi, (const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsAtan2pi, (const MKL_INT n, const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdAtan2pi, (const MKL_INT n, const double a[], const double b[], double r[], MKL_INT64 mode))

/* Multiplicaton: r[i] = a[i] * b[i] */
_MKL_API(void,VSMUL,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDMUL,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vsmul,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdmul,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsMul,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdMul,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSMUL,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDMUL,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsmul,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdmul,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsMul,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdMul,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complex multiplication: r[i] = a[i] * b[i] */
_MKL_API(void,VCMUL,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZMUL,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcmul,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzmul,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcMul,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzMul,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCMUL,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZMUL,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcmul,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzmul,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcMul,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzMul,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Division: r[i] = a[i] / b[i] */
_MKL_API(void,VSDIV,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDDIV,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vsdiv,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vddiv,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsDiv,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdDiv,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSDIV,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDDIV,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsdiv,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmddiv,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsDiv,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdDiv,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complex division: r[i] = a[i] / b[i] */
_MKL_API(void,VCDIV,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZDIV,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcdiv,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzdiv,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcDiv,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzDiv,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCDIV,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZDIV,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcdiv,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzdiv,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcDiv,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzDiv,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Power function: r[i] = a[i]^b[i] */
_MKL_API(void,VSPOW,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void,VDPOW,(const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void,vspow,(const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void,vdpow,(const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void,vsPow,(const MKL_INT n,  const float  a[], const float  b[], float  r[]))
_Mkl_Api(void,vdPow,(const MKL_INT n,  const double a[], const double b[], double r[]))

_MKL_API(void,VMSPOW,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDPOW,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmspow,(const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdpow,(const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsPow,(const MKL_INT n,  const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdPow,(const MKL_INT n,  const double a[], const double b[], double r[], MKL_INT64 mode))

/* Complex power function: r[i] = a[i]^b[i] */
_MKL_API(void,VCPOW,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZPOW,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcpow,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzpow,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcPow,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzPow,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCPOW,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZPOW,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcpow,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzpow,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcPow,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzPow,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Power function: r[i] = a[i]^(3/2) */
_MKL_API(void,VSPOW3O2,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDPOW3O2,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vspow3o2,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdpow3o2,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsPow3o2,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdPow3o2,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSPOW3O2,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDPOW3O2,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmspow3o2,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdpow3o2,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsPow3o2,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdPow3o2,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Power function: r[i] = a[i]^(2/3) */
_MKL_API(void,VSPOW2O3,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDPOW2O3,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vspow2o3,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdpow2o3,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsPow2o3,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdPow2o3,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSPOW2O3,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDPOW2O3,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmspow2o3,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdpow2o3,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsPow2o3,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdPow2o3,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Power function with fixed degree: r[i] = a[i]^b */
_MKL_API(void,VSPOWX,(const MKL_INT *n, const float  a[], const float  *b, float  r[]))
_MKL_API(void,VDPOWX,(const MKL_INT *n, const double a[], const double *b, double r[]))
_mkl_api(void,vspowx,(const MKL_INT *n, const float  a[], const float  *b, float  r[]))
_mkl_api(void,vdpowx,(const MKL_INT *n, const double a[], const double *b, double r[]))
_Mkl_Api(void,vsPowx,(const MKL_INT n,  const float  a[], const float   b, float  r[]))
_Mkl_Api(void,vdPowx,(const MKL_INT n,  const double a[], const double  b, double r[]))

_MKL_API(void,VMSPOWX,(const MKL_INT *n, const float  a[], const float  *b, float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDPOWX,(const MKL_INT *n, const double a[], const double *b, double r[], MKL_INT64 *mode))
_mkl_api(void,vmspowx,(const MKL_INT *n, const float  a[], const float  *b, float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdpowx,(const MKL_INT *n, const double a[], const double *b, double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsPowx,(const MKL_INT n,  const float  a[], const float   b, float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdPowx,(const MKL_INT n,  const double a[], const double  b, double r[], MKL_INT64 mode))

/* Complex power function with fixed degree: r[i] = a[i]^b */
_MKL_API(void,VCPOWX,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  *b, MKL_Complex8  r[]))
_MKL_API(void,VZPOWX,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 *b, MKL_Complex16 r[]))
_mkl_api(void,vcpowx,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  *b, MKL_Complex8  r[]))
_mkl_api(void,vzpowx,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 *b, MKL_Complex16 r[]))
_Mkl_Api(void,vcPowx,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8   b, MKL_Complex8  r[]))
_Mkl_Api(void,vzPowx,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16  b, MKL_Complex16 r[]))

_MKL_API(void,VMCPOWX,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  *b, MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZPOWX,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 *b, MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcpowx,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  *b, MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzpowx,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 *b, MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcPowx,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8   b, MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzPowx,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16  b, MKL_Complex16 r[], MKL_INT64 mode))

/* Power function with a[i]>=0: r[i] = a[i]^b[i] */
_MKL_API(void, VSPOWR, (const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_MKL_API(void, VDPOWR, (const MKL_INT *n, const double a[], const double b[], double r[]))
_mkl_api(void, vspowr, (const MKL_INT *n, const float  a[], const float  b[], float  r[]))
_mkl_api(void, vdpowr, (const MKL_INT *n, const double a[], const double b[], double r[]))
_Mkl_Api(void, vsPowr, (const MKL_INT n, const float  a[], const float  b[], float  r[]))
_Mkl_Api(void, vdPowr, (const MKL_INT n, const double a[], const double b[], double r[]))

_MKL_API(void, VMSPOWR, (const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDPOWR, (const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmspowr, (const MKL_INT *n, const float  a[], const float  b[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdpowr, (const MKL_INT *n, const double a[], const double b[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsPowr, (const MKL_INT n, const float  a[], const float  b[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdPowr, (const MKL_INT n, const double a[], const double b[], double r[], MKL_INT64 mode))

/* Sine & cosine: r1[i] = sin(a[i]), r2[i]=cos(a[i]) */
_MKL_API(void,VSSINCOS,(const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void,VDSINCOS,(const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void,vssincos,(const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void,vdsincos,(const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void,vsSinCos,(const MKL_INT n,  const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void,vdSinCos,(const MKL_INT n,  const double a[], double r1[], double r2[]))

_MKL_API(void,VMSSINCOS,(const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void,VMDSINCOS,(const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void,vmssincos,(const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void,vmdsincos,(const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void,vmsSinCos,(const MKL_INT n,  const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void,vmdSinCos,(const MKL_INT n,  const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Linear fraction: r[i] = (a[i]*scalea + shifta)/(b[i]*scaleb + shiftb) */
_MKL_API(void,VSLINEARFRAC,(const MKL_INT *n, const float  a[], const float  b[], const float  *scalea, const float  *shifta, const float  *scaleb, const float  *shiftb, float  r[]))
_MKL_API(void,VDLINEARFRAC,(const MKL_INT *n, const double a[], const double b[], const double *scalea, const double *shifta, const double *scaleb, const double *shiftb, double r[]))
_mkl_api(void,vslinearfrac,(const MKL_INT *n, const float  a[], const float  b[], const float  *scalea, const float  *shifta, const float  *scaleb, const float  *shiftb, float  r[]))
_mkl_api(void,vdlinearfrac,(const MKL_INT *n, const double a[], const double b[], const double *scalea, const double *shifta, const double *scaleb, const double *shiftb, double r[]))
_Mkl_Api(void,vsLinearFrac,(const MKL_INT n,  const float  a[], const float  b[], const float  scalea, const float  shifta, const float  scaleb, const float  shiftb, float  r[]))
_Mkl_Api(void,vdLinearFrac,(const MKL_INT n,  const double a[], const double b[], const double scalea, const double shifta, const double scaleb, const double shiftb, double r[]))

_MKL_API(void,VMSLINEARFRAC,(const MKL_INT *n, const float  a[], const float  b[], const float  *scalea, const float  *shifta, const float  *scaleb, const float  *shiftb, float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDLINEARFRAC,(const MKL_INT *n, const double a[], const double b[], const double *scalea, const double *shifta, const double *scaleb, const double *shiftb, double r[], MKL_INT64 *mode))
_mkl_api(void,vmslinearfrac,(const MKL_INT *n, const float  a[], const float  b[], const float  *scalea, const float  *shifta, const float  *scaleb, const float  *shiftb, float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdlinearfrac,(const MKL_INT *n, const double a[], const double b[], const double *scalea, const double *shifta, const double *scaleb, const double *shiftb, double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsLinearFrac,(const MKL_INT n,  const float  a[], const float  b[], const float  scalea, const float  shifta, const float  scaleb, const float  shiftb, float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdLinearFrac,(const MKL_INT n,  const double a[], const double b[], const double scalea, const double shifta, const double scaleb, const double shiftb, double r[], MKL_INT64 mode))

/* Integer value rounded towards plus infinity: r[i] = ceil(a[i]) */
_MKL_API(void,VSCEIL,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDCEIL,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsceil,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdceil,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsCeil,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdCeil,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSCEIL,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDCEIL,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsceil,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdceil,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsCeil,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdCeil,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Integer value rounded towards minus infinity: r[i] = floor(a[i]) */
_MKL_API(void,VSFLOOR,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDFLOOR,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsfloor,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdfloor,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsFloor,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdFloor,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSFLOOR,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDFLOOR,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsfloor,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdfloor,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsFloor,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdFloor,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Signed fraction part */
_MKL_API(void,VSFRAC,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDFRAC,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsfrac,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdfrac,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsFrac,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdFrac,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSFRAC,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDFRAC,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsfrac,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdfrac,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsFrac,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdFrac,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Truncated integer value and the remaining fraction part */
_MKL_API(void,VSMODF,(const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void,VDMODF,(const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void,vsmodf,(const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void,vdmodf,(const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void,vsModf,(const MKL_INT n,  const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void,vdModf,(const MKL_INT n,  const double a[], double r1[], double r2[]))

_MKL_API(void,VMSMODF,(const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void,VMDMODF,(const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void,vmsmodf,(const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void,vmdmodf,(const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void,vmsModf,(const MKL_INT n,  const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void,vmdModf,(const MKL_INT n,  const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Modulus function: r[i] = fmod(a[i], b[i]) */
_MKL_API(void, VSFMOD, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDFMOD, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsfmod, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdfmod, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsFmod, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdFmod, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSFMOD, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDFMOD, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsfmod, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdfmod, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsFmod, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdFmod, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Remainder function: r[i] = remainder(a[i], b[i]) */
_MKL_API(void, VSREMAINDER, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDREMAINDER, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsremainder, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdremainder, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsRemainder, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdRemainder, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSREMAINDER, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDREMAINDER, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsremainder, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdremainder, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsRemainder, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdRemainder, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Next after function: r[i] = nextafter(a[i], b[i]) */
_MKL_API(void, VSNEXTAFTER, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDNEXTAFTER, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsnextafter, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdnextafter, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsNextAfter, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdNextAfter, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSNEXTAFTER, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDNEXTAFTER, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsnextafter, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdnextafter, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsNextAfter, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdNextAfter, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Copy sign function: r[i] = copysign(a[i], b[i]) */
_MKL_API(void, VSCOPYSIGN, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDCOPYSIGN, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vscopysign, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdcopysign, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsCopySign, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdCopySign, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSCOPYSIGN, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDCOPYSIGN, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmscopysign, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdcopysign, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsCopySign, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdCopySign, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Positive difference function: r[i] = fdim(a[i], b[i]) */
_MKL_API(void, VSFDIM, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDFDIM, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsfdim, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdfdim, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsFdim, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdFdim, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSFDIM, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDFDIM, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsfdim, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdfdim, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsFdim, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdFdim, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Maximum function: r[i] = fmax(a[i], b[i]) */
_MKL_API(void, VSFMAX, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDFMAX, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsfmax, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdfmax, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsFmax, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdFmax, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSFMAX, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDFMAX, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsfmax, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdfmax, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsFmax, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdFmax, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Minimum function: r[i] = fmin(a[i], b[i]) */
_MKL_API(void, VSFMIN, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDFMIN, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsfmin, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdfmin, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsFmin, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdFmin, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSFMIN, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDFMIN, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsfmin, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdfmin, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsFmin, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdFmin, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Maximum magnitude function: r[i] = maxmag(a[i], b[i]) */
_MKL_API(void, VSMAXMAG, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDMAXMAG, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsmaxmag, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdmaxmag, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsMaxMag, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdMaxMag, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSMAXMAG, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDMAXMAG, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsmaxmag, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdmaxmag, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsMaxMag, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdMaxMag, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Minimum magnitude function: r[i] = minmag(a[i], b[i]) */
_MKL_API(void, VSMINMAG, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_MKL_API(void, VDMINMAG, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_mkl_api(void, vsminmag, (const MKL_INT *n, const float  a[], float  r1[], float  r2[]))
_mkl_api(void, vdminmag, (const MKL_INT *n, const double a[], double r1[], double r2[]))
_Mkl_Api(void, vsMinMag, (const MKL_INT n, const float  a[], float  r1[], float  r2[]))
_Mkl_Api(void, vdMinMag, (const MKL_INT n, const double a[], double r1[], double r2[]))

_MKL_API(void, VMSMINMAG, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_MKL_API(void, VMDMINMAG, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_mkl_api(void, vmsminmag, (const MKL_INT *n, const float  a[], float  r1[], float  r2[], MKL_INT64 *mode))
_mkl_api(void, vmdminmag, (const MKL_INT *n, const double a[], double r1[], double r2[], MKL_INT64 *mode))
_Mkl_Api(void, vmsMinMag, (const MKL_INT n, const float  a[], float  r1[], float  r2[], MKL_INT64 mode))
_Mkl_Api(void, vmdMinMag, (const MKL_INT n, const double a[], double r1[], double r2[], MKL_INT64 mode))

/* Rounded integer value in the current rounding mode: r[i] = nearbyint(a[i]) */
_MKL_API(void,VSNEARBYINT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDNEARBYINT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsnearbyint,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdnearbyint,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsNearbyInt,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdNearbyInt,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSNEARBYINT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDNEARBYINT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsnearbyint,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdnearbyint,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsNearbyInt,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdNearbyInt,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Rounded integer value in the current rounding mode with inexact result exception raised for rach changed value: r[i] = rint(a[i]) */
_MKL_API(void,VSRINT,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDRINT,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsrint,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdrint,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsRint,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdRint,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSRINT,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDRINT,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsrint,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdrint,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsRint,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdRint,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Value rounded to the nearest integer: r[i] = round(a[i]) */
_MKL_API(void,VSROUND,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDROUND,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vsround,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdround,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsRound,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdRound,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSROUND,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDROUND,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmsround,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdround,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsRound,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdRound,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Integer value rounded towards zero: r[i] = trunc(a[i]) */
_MKL_API(void,VSTRUNC,(const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void,VDTRUNC,(const MKL_INT *n, const double a[], double r[]))
_mkl_api(void,vstrunc,(const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void,vdtrunc,(const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void,vsTrunc,(const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void,vdTrunc,(const MKL_INT n,  const double a[], double r[]))

_MKL_API(void,VMSTRUNC,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void,VMDTRUNC,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void,vmstrunc,(const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void,vmdtrunc,(const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void,vmsTrunc,(const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void,vmdTrunc,(const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/* Element by element conjugation: r[i] = conj(a[i]) */
_MKL_API(void,VCCONJ,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_MKL_API(void,VZCONJ,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_mkl_api(void,vcconj,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[]))
_mkl_api(void,vzconj,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcConj,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzConj,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[]))

_MKL_API(void,VMCCONJ,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZCONJ,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcconj,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzconj,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcConj,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzConj,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Element by element multiplication of vector A element and conjugated vector B element: r[i] = mulbyconj(a[i],b[i]) */
_MKL_API(void,VCMULBYCONJ,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_MKL_API(void,VZMULBYCONJ,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_mkl_api(void,vcmulbyconj,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_mkl_api(void,vzmulbyconj,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))
_Mkl_Api(void,vcMulByConj,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[]))
_Mkl_Api(void,vzMulByConj,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[]))

_MKL_API(void,VMCMULBYCONJ,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZMULBYCONJ,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmcmulbyconj,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzmulbyconj,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcMulByConj,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_Complex8  b[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzMulByConj,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_Complex16 b[], MKL_Complex16 r[], MKL_INT64 mode))

/* Complex exponent of real vector elements: r[i] = CIS(a[i]) */
_MKL_API(void,VCCIS,(const MKL_INT *n, const float  a[], MKL_Complex8  r[]))
_MKL_API(void,VZCIS,(const MKL_INT *n, const double a[], MKL_Complex16 r[]))
_mkl_api(void,vccis,(const MKL_INT *n, const float  a[], MKL_Complex8  r[]))
_mkl_api(void,vzcis,(const MKL_INT *n, const double a[], MKL_Complex16 r[]))
_Mkl_Api(void,vcCIS,(const MKL_INT n,  const float  a[], MKL_Complex8  r[]))
_Mkl_Api(void,vzCIS,(const MKL_INT n,  const double a[], MKL_Complex16 r[]))

_MKL_API(void,VMCCIS,(const MKL_INT *n, const float  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_MKL_API(void,VMZCIS,(const MKL_INT *n, const double a[], MKL_Complex16 r[], MKL_INT64 *mode))
_mkl_api(void,vmccis,(const MKL_INT *n, const float  a[], MKL_Complex8  r[], MKL_INT64 *mode))
_mkl_api(void,vmzcis,(const MKL_INT *n, const double a[], MKL_Complex16 r[], MKL_INT64 *mode))
_Mkl_Api(void,vmcCIS,(const MKL_INT n,  const float  a[], MKL_Complex8  r[], MKL_INT64 mode))
_Mkl_Api(void,vmzCIS,(const MKL_INT n,  const double a[], MKL_Complex16 r[], MKL_INT64 mode))

/* Exponential integral of real vector elements: r[i] = E1(a[i]) */
_MKL_API(void, VSEXPINT1, (const MKL_INT *n, const float  a[], float  r[]))
_MKL_API(void, VDEXPINT1, (const MKL_INT *n, const double a[], double r[]))
_mkl_api(void, vsexpint1, (const MKL_INT *n, const float  a[], float  r[]))
_mkl_api(void, vdexpint1, (const MKL_INT *n, const double a[], double r[]))
_Mkl_Api(void, vsExpInt1, (const MKL_INT n,  const float  a[], float  r[]))
_Mkl_Api(void, vdExpInt1, (const MKL_INT n,  const double a[], double r[]))

_MKL_API(void, VMSEXPINT1, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_MKL_API(void, VMDEXPINT1, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_mkl_api(void, vmsexpint1, (const MKL_INT *n, const float  a[], float  r[], MKL_INT64 *mode))
_mkl_api(void, vmdexpint1, (const MKL_INT *n, const double a[], double r[], MKL_INT64 *mode))
_Mkl_Api(void, vmsExpInt1, (const MKL_INT n,  const float  a[], float  r[], MKL_INT64 mode))
_Mkl_Api(void, vmdExpInt1, (const MKL_INT n,  const double a[], double r[], MKL_INT64 mode))

/*
//++
//  VML PACK FUNCTION DECLARATIONS.
//--
*/
/* Positive Increment Indexing */
_MKL_API(void,VSPACKI,(const MKL_INT *n, const float  a[], const MKL_INT * incra, float  y[]))
_MKL_API(void,VDPACKI,(const MKL_INT *n, const double a[], const MKL_INT * incra, double y[]))
_mkl_api(void,vspacki,(const MKL_INT *n, const float  a[], const MKL_INT * incra, float  y[]))
_mkl_api(void,vdpacki,(const MKL_INT *n, const double a[], const MKL_INT * incra, double y[]))
_Mkl_Api(void,vsPackI,(const MKL_INT n,  const float  a[], const MKL_INT   incra, float  y[]))
_Mkl_Api(void,vdPackI,(const MKL_INT n,  const double a[], const MKL_INT   incra, double y[]))

_MKL_API(void,VCPACKI,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT * incra, MKL_Complex8  y[]))
_MKL_API(void,VZPACKI,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT * incra, MKL_Complex16 y[]))
_mkl_api(void,vcpacki,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT * incra, MKL_Complex8  y[]))
_mkl_api(void,vzpacki,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT * incra, MKL_Complex16 y[]))
_Mkl_Api(void,vcPackI,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_INT   incra, MKL_Complex8  y[]))
_Mkl_Api(void,vzPackI,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_INT   incra, MKL_Complex16 y[]))

/* Index Vector Indexing */
_MKL_API(void,VSPACKV,(const MKL_INT *n, const float  a[], const MKL_INT ia[], float  y[]))
_MKL_API(void,VDPACKV,(const MKL_INT *n, const double a[], const MKL_INT ia[], double y[]))
_mkl_api(void,vspackv,(const MKL_INT *n, const float  a[], const MKL_INT ia[], float  y[]))
_mkl_api(void,vdpackv,(const MKL_INT *n, const double a[], const MKL_INT ia[], double y[]))
_Mkl_Api(void,vsPackV,(const MKL_INT n,  const float  a[], const MKL_INT ia[], float  y[]))
_Mkl_Api(void,vdPackV,(const MKL_INT n,  const double a[], const MKL_INT ia[], double y[]))

_MKL_API(void,VCPACKV,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT ia[], MKL_Complex8  y[]))
_MKL_API(void,VZPACKV,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT ia[], MKL_Complex16 y[]))
_mkl_api(void,vcpackv,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT ia[], MKL_Complex8  y[]))
_mkl_api(void,vzpackv,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT ia[], MKL_Complex16 y[]))
_Mkl_Api(void,vcPackV,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_INT ia[], MKL_Complex8  y[]))
_Mkl_Api(void,vzPackV,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_INT ia[], MKL_Complex16 y[]))

/* Mask Vector Indexing */
_MKL_API(void,VSPACKM,(const MKL_INT *n, const float  a[], const MKL_INT ma[], float  y[]))
_MKL_API(void,VDPACKM,(const MKL_INT *n, const double a[], const MKL_INT ma[], double y[]))
_mkl_api(void,vspackm,(const MKL_INT *n, const float  a[], const MKL_INT ma[], float  y[]))
_mkl_api(void,vdpackm,(const MKL_INT *n, const double a[], const MKL_INT ma[], double y[]))
_Mkl_Api(void,vsPackM,(const MKL_INT n,  const float  a[], const MKL_INT ma[], float  y[]))
_Mkl_Api(void,vdPackM,(const MKL_INT n,  const double a[], const MKL_INT ma[], double y[]))

_MKL_API(void,VCPACKM,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT ma[], MKL_Complex8  y[]))
_MKL_API(void,VZPACKM,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT ma[], MKL_Complex16 y[]))
_mkl_api(void,vcpackm,(const MKL_INT *n, const MKL_Complex8  a[], const MKL_INT ma[], MKL_Complex8  y[]))
_mkl_api(void,vzpackm,(const MKL_INT *n, const MKL_Complex16 a[], const MKL_INT ma[], MKL_Complex16 y[]))
_Mkl_Api(void,vcPackM,(const MKL_INT n,  const MKL_Complex8  a[], const MKL_INT ma[], MKL_Complex8  y[]))
_Mkl_Api(void,vzPackM,(const MKL_INT n,  const MKL_Complex16 a[], const MKL_INT ma[], MKL_Complex16 y[]))

/*
//++
//  VML UNPACK FUNCTION DECLARATIONS.
//--
*/
/* Positive Increment Indexing */
_MKL_API(void,VSUNPACKI,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT * incry))
_MKL_API(void,VDUNPACKI,(const MKL_INT *n, const double a[], double y[], const MKL_INT * incry))
_mkl_api(void,vsunpacki,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT * incry))
_mkl_api(void,vdunpacki,(const MKL_INT *n, const double a[], double y[], const MKL_INT * incry))
_Mkl_Api(void,vsUnpackI,(const MKL_INT n,  const float  a[], float  y[], const MKL_INT incry  ))
_Mkl_Api(void,vdUnpackI,(const MKL_INT n,  const double a[], double y[], const MKL_INT incry  ))

_MKL_API(void,VCUNPACKI,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT * incry))
_MKL_API(void,VZUNPACKI,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT * incry))
_mkl_api(void,vcunpacki,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT * incry))
_mkl_api(void,vzunpacki,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT * incry))
_Mkl_Api(void,vcUnpackI,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT incry  ))
_Mkl_Api(void,vzUnpackI,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT incry  ))

/* Index Vector Indexing */
_MKL_API(void,VSUNPACKV,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT iy[]   ))
_MKL_API(void,VDUNPACKV,(const MKL_INT *n, const double a[], double y[], const MKL_INT iy[]   ))
_mkl_api(void,vsunpackv,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT iy[]   ))
_mkl_api(void,vdunpackv,(const MKL_INT *n, const double a[], double y[], const MKL_INT iy[]   ))
_Mkl_Api(void,vsUnpackV,(const MKL_INT n,  const float  a[], float  y[], const MKL_INT iy[]   ))
_Mkl_Api(void,vdUnpackV,(const MKL_INT n,  const double a[], double y[], const MKL_INT iy[]   ))

_MKL_API(void,VCUNPACKV,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT iy[]))
_MKL_API(void,VZUNPACKV,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT iy[]))
_mkl_api(void,vcunpackv,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT iy[]))
_mkl_api(void,vzunpackv,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT iy[]))
_Mkl_Api(void,vcUnpackV,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT iy[]))
_Mkl_Api(void,vzUnpackV,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT iy[]))

/* Mask Vector Indexing */
_MKL_API(void,VSUNPACKM,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT my[]   ))
_MKL_API(void,VDUNPACKM,(const MKL_INT *n, const double a[], double y[], const MKL_INT my[]   ))
_mkl_api(void,vsunpackm,(const MKL_INT *n, const float  a[], float  y[], const MKL_INT my[]   ))
_mkl_api(void,vdunpackm,(const MKL_INT *n, const double a[], double y[], const MKL_INT my[]   ))
_Mkl_Api(void,vsUnpackM,(const MKL_INT n,  const float  a[], float  y[], const MKL_INT my[]   ))
_Mkl_Api(void,vdUnpackM,(const MKL_INT n,  const double a[], double y[], const MKL_INT my[]   ))

_MKL_API(void,VCUNPACKM,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT my[]))
_MKL_API(void,VZUNPACKM,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT my[]))
_mkl_api(void,vcunpackm,(const MKL_INT *n, const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT my[]))
_mkl_api(void,vzunpackm,(const MKL_INT *n, const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT my[]))
_Mkl_Api(void,vcUnpackM,(const MKL_INT n,  const MKL_Complex8  a[], MKL_Complex8  y[], const MKL_INT my[]))
_Mkl_Api(void,vzUnpackM,(const MKL_INT n,  const MKL_Complex16 a[], MKL_Complex16 y[], const MKL_INT my[]))


/*
//++
//  VML ERROR HANDLING FUNCTION DECLARATIONS.
//--
*/
/* Set VML Error Status */
_MKL_API(int,VMLSETERRSTATUS,(const MKL_INT * status))
_mkl_api(int,vmlseterrstatus,(const MKL_INT * status))
_Mkl_Api(int,vmlSetErrStatus,(const MKL_INT   status))

/* Get VML Error Status */
_MKL_API(int,VMLGETERRSTATUS,(void))
_mkl_api(int,vmlgeterrstatus,(void))
_Mkl_Api(int,vmlGetErrStatus,(void))

/* Clear VML Error Status */
_MKL_API(int,VMLCLEARERRSTATUS,(void))
_mkl_api(int,vmlclearerrstatus,(void))
_Mkl_Api(int,vmlClearErrStatus,(void))

/* Set VML Error Callback Function */
_MKL_API(VMLErrorCallBack,VMLSETERRORCALLBACK,(const VMLErrorCallBack func))
_mkl_api(VMLErrorCallBack,vmlseterrorcallback,(const VMLErrorCallBack func))
_Mkl_Api(VMLErrorCallBack,vmlSetErrorCallBack,(const VMLErrorCallBack func))

/* Get VML Error Callback Function */
_MKL_API(VMLErrorCallBack,VMLGETERRORCALLBACK,(void))
_mkl_api(VMLErrorCallBack,vmlgeterrorcallback,(void))
_Mkl_Api(VMLErrorCallBack,vmlGetErrorCallBack,(void))

/* Reset VML Error Callback Function */
_MKL_API(VMLErrorCallBack,VMLCLEARERRORCALLBACK,(void))
_mkl_api(VMLErrorCallBack,vmlclearerrorcallback,(void))
_Mkl_Api(VMLErrorCallBack,vmlClearErrorCallBack,(void))


/*
//++
//  VML MODE FUNCTION DECLARATIONS.
//--
*/
/* Set VML Mode */
_MKL_API(unsigned int,VMLSETMODE,(const MKL_UINT *newmode))
_mkl_api(unsigned int,vmlsetmode,(const MKL_UINT *newmode))
_Mkl_Api(unsigned int,vmlSetMode,(const MKL_UINT  newmode))

/* Get VML Mode */
_MKL_API(unsigned int,VMLGETMODE,(void))
_mkl_api(unsigned int,vmlgetmode,(void))
_Mkl_Api(unsigned int,vmlGetMode,(void))


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_FUNCTIONS_H__ */
