/*******************************************************************************
* Copyright 2012-2017 Intel Corporation All Rights Reserved.
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
*           Intel(R) Math Kernel Library (Intel(R) MKL) FEAST C header file
*
*           Contains interface to FEAST.
*
********************************************************************************
*/
#if !defined( __MKL_SOLVERS_EE_H )
#define __MKL_SOLVERS_EE_H
#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void feastinit (MKL_INT* fpm);
void FEASTINIT (MKL_INT* fpm);

void dfeast_scsrev (const char* uplo , const MKL_INT* n , const double* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SCSREV (const char* uplo , const MKL_INT* n , const double* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

void sfeast_sygv (const char* uplo , const MKL_INT* n , const float* a , const MKL_INT* lda , const float* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SYGV (const char* uplo , const MKL_INT* n , const float* a , const MKL_INT* lda , const float* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void sfeast_scsrev (const char* uplo , const MKL_INT* n , const float* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SCSREV (const char* uplo , const MKL_INT* n , const float* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void zfeast_hbgv (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex16* a , const MKL_INT* lda , const MKL_INT* klb , const MKL_Complex16* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HBGV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex16* a , const MKL_INT* lda , const MKL_INT* klb , const MKL_Complex16* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void zfeast_hcsrev (const char* uplo , const MKL_INT* n , const MKL_Complex16* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HCSREV (const char* uplo , const MKL_INT* n , const MKL_Complex16* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void cfeast_hbev (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex8* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HBEV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex8* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void zfeast_heev (const char* uplo , const MKL_INT* n , const MKL_Complex16* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HEEV (const char* uplo , const MKL_INT* n , const MKL_Complex16* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void zfeast_hcsrgv (const char* uplo , const MKL_INT* n , const MKL_Complex16* sa , const MKL_INT* isa , const MKL_INT* jsa , const MKL_Complex16* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HCSRGV (const char* uplo , const MKL_INT* n , const MKL_Complex16* sa , const MKL_INT* isa , const MKL_INT* jsa , const MKL_Complex16* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void sfeast_scsrgv (const char* uplo , const MKL_INT* n , const float* sa , const MKL_INT* isa , const MKL_INT* jsa , const float* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SCSRGV (const char* uplo , const MKL_INT* n , const float* sa , const MKL_INT* isa , const MKL_INT* jsa , const float* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void dfeast_srci (MKL_INT* ijob , const MKL_INT* n , MKL_Complex16* ze , double* work , MKL_Complex16* workc , double* aq , double* sq , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* lambda , double* q , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SRCI (MKL_INT* ijob , const MKL_INT* n , MKL_Complex16* ze , double* work , MKL_Complex16* workc , double* aq , double* sq , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* lambda , double* q , MKL_INT* m , double* res , MKL_INT* info);

void sfeast_sbev (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const float* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SBEV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const float* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void dfeast_sygv (const char* uplo , const MKL_INT* n , const double* a , const MKL_INT* lda , const double* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SYGV (const char* uplo , const MKL_INT* n , const double* a , const MKL_INT* lda , const double* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

void cfeast_hegv (const char* uplo , const MKL_INT* n , const MKL_Complex8* a , const MKL_INT* lda , const MKL_Complex8* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HEGV (const char* uplo , const MKL_INT* n , const MKL_Complex8* a , const MKL_INT* lda , const MKL_Complex8* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void zfeast_hrci (MKL_INT* ijob , const MKL_INT* n , MKL_Complex16* ze , MKL_Complex16* work , MKL_Complex16* workc , MKL_Complex16* zaq , MKL_Complex16* zsq , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* lambda , MKL_Complex16* q , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HRCI (MKL_INT* ijob , const MKL_INT* n , MKL_Complex16* ze , MKL_Complex16* work , MKL_Complex16* workc , MKL_Complex16* zaq , MKL_Complex16* zsq , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* lambda , MKL_Complex16* q , MKL_INT* m , double* res , MKL_INT* info);

void cfeast_hrci (MKL_INT* ijob , const MKL_INT* n , MKL_Complex8* ze , MKL_Complex8* work , MKL_Complex8* workc , MKL_Complex8* zaq , MKL_Complex8* zsq , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* lambda , MKL_Complex8* q , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HRCI (MKL_INT* ijob , const MKL_INT* n , MKL_Complex8* ze , MKL_Complex8* work , MKL_Complex8* workc , MKL_Complex8* zaq , MKL_Complex8* zsq , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* lambda , MKL_Complex8* q , MKL_INT* m , float* res , MKL_INT* info);

void zfeast_hegv (const char* uplo , const MKL_INT* n , const MKL_Complex16* a , const MKL_INT* lda , const MKL_Complex16* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HEGV (const char* uplo , const MKL_INT* n , const MKL_Complex16* a , const MKL_INT* lda , const MKL_Complex16* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void sfeast_srci (MKL_INT* ijob , const MKL_INT* n , MKL_Complex8* ze , float* work , MKL_Complex8* workc , float* aq , float* sq , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* lambda , float* q , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SRCI (MKL_INT* ijob , const MKL_INT* n , MKL_Complex8* ze , float* work , MKL_Complex8* workc , float* aq , float* sq , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* lambda , float* q , MKL_INT* m , float* res , MKL_INT* info);

void dfeast_scsrgv (const char* uplo , const MKL_INT* n , const double* sa , const MKL_INT* isa , const MKL_INT* jsa , const double* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SCSRGV (const char* uplo , const MKL_INT* n , const double* sa , const MKL_INT* isa , const MKL_INT* jsa , const double* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

void sfeast_syev (const char* uplo , const MKL_INT* n , const float* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);
void SFEAST_SYEV (const char* uplo , const MKL_INT* n , const float* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void cfeast_hcsrev (const char* uplo , const MKL_INT* n , const MKL_Complex8* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HCSREV (const char* uplo , const MKL_INT* n , const MKL_Complex8* sa , const MKL_INT* isa , const MKL_INT* jsa , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void zfeast_hbev (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex16* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);
void ZFEAST_HBEV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex16* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , MKL_Complex16* x , MKL_INT* m , double* res , MKL_INT* info);

void dfeast_sbgv (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const double* a , const MKL_INT* lda , const MKL_INT* klb , const double* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SBGV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const double* a , const MKL_INT* lda , const MKL_INT* klb , const double* b , const MKL_INT* ldb , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

void cfeast_hbgv (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex8* a , const MKL_INT* lda , const MKL_INT* klb , const MKL_Complex8* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HBGV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const MKL_Complex8* a , const MKL_INT* lda , const MKL_INT* klb , const MKL_Complex8* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void cfeast_heev (const char* uplo , const MKL_INT* n , const MKL_Complex8* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HEEV (const char* uplo , const MKL_INT* n , const MKL_Complex8* a , const MKL_INT* lda , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void sfeast_sbgv (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const float* a , const MKL_INT* lda , const MKL_INT* klb , const float* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* mode , float* res , MKL_INT* info);
void SFEAST_SBGV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const float* a , const MKL_INT* lda , const MKL_INT* klb , const float* b , const MKL_INT* ldb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , float* x , MKL_INT* m , float* res , MKL_INT* info);

void cfeast_hcsrgv (const char* uplo , const MKL_INT* n , const MKL_Complex8* sa , const MKL_INT* isa , const MKL_INT* jsa , const MKL_Complex8* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);
void CFEAST_HCSRGV (const char* uplo , const MKL_INT* n , const MKL_Complex8* sa , const MKL_INT* isa , const MKL_INT* jsa , const MKL_Complex8* sb , const MKL_INT* isb , const MKL_INT* jsb , MKL_INT* fpm , float* epsout , MKL_INT* loop , const float* emin , const float* emax , MKL_INT* m0 , float* e , MKL_Complex8* x , MKL_INT* m , float* res , MKL_INT* info);

void dfeast_syev (const char* uplo , const MKL_INT* n , const double* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SYEV (const char* uplo , const MKL_INT* n , const double* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

void dfeast_sbev (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const double* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);
void DFEAST_SBEV (const char* uplo , const MKL_INT* n , const MKL_INT* kla , const double* a , const MKL_INT* lda , MKL_INT* fpm , double* epsout , MKL_INT* loop , const double* emin , const double* emax , MKL_INT* m0 , double* e , double* x , MKL_INT* m , double* res , MKL_INT* info);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
