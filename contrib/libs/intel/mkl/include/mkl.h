/*******************************************************************************
* Copyright 1999-2017 Intel Corporation All Rights Reserved.
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
!  Content:
!      Intel(R) Math Kernel Library (Intel(R) MKL) interface
!******************************************************************************/

#ifndef _MKL_H_
#define _MKL_H_

#define _Mkl_Api(rtype,name,arg) extern rtype name    arg;
#define _mkl_api(rtype,name,arg) extern rtype name##_ arg;
#define _MKL_API(rtype,name,arg) extern rtype name##_ arg;

#include "mkl_version.h"
#include "mkl_types.h"
#include "mkl_blas.h"
#include "mkl_trans.h"
#include "mkl_cblas.h"
#include "mkl_spblas.h"
#include "mkl_lapack.h"
#include "mkl_lapacke.h"
#include "mkl_pardiso.h"
#include "mkl_sparse_handle.h"
#include "mkl_dss.h"
#include "mkl_rci.h"
#include "mkl_vml.h"
#include "mkl_vsl.h"
#include "mkl_df.h"
#include "mkl_service.h"
#include "mkl_dfti.h"
#include "mkl_trig_transforms.h"
#include "mkl_poisson.h"
#include "mkl_solvers_ee.h"
#include "mkl_direct_call.h"
#include "mkl_dnn.h"
#include "mkl_compact.h"

#endif /* _MKL_H_ */
