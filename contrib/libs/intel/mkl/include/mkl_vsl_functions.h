/* file: mkl_vsl_functions.h */
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
//  User-level VSL function declarations
//--
*/

#ifndef __MKL_VSL_FUNCTIONS_H__
#define __MKL_VSL_FUNCTIONS_H__

#include "mkl_vsl_types.h"

#ifdef __cplusplus
#if __cplusplus > 199711L
#define NOTHROW noexcept
#else
#define NOTHROW throw()
#endif
#else
#define NOTHROW
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
//++
//  EXTERNAL API MACROS.
//  Used to construct VSL function declaration. Change them if you are going to
//  provide different API for VSL functions.
//--
*/

#if  !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)   extern rtype name    arg
#endif

#if  !defined(_mkl_api)
#define _mkl_api(rtype,name,arg)   extern rtype name##_ arg
#endif

#if  !defined(_MKL_API)
#define _MKL_API(rtype,name,arg)   extern rtype name##_ arg
#endif

/*
//++
//  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Cauchy distribution */
_Mkl_Api(int,vdRngCauchy,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGCAUCHY,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrngcauchy,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngCauchy,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGCAUCHY,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrngcauchy,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);

/* Uniform distribution */
_Mkl_Api(int,vdRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);

/* Gaussian distribution */
_Mkl_Api(int,vdRngGaussian,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGGAUSSIAN,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrnggaussian,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngGaussian,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGGAUSSIAN,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnggaussian,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);

/* GaussianMV distribution */
_Mkl_Api(int,vdRngGaussianMV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const MKL_INT  ,  const MKL_INT  , const double *, const double *) NOTHROW);
_MKL_API(int,VDRNGGAUSSIANMV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const MKL_INT *,  const MKL_INT *, const double *, const double *) NOTHROW);
_mkl_api(int,vdrnggaussianmv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const MKL_INT *,  const MKL_INT *, const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngGaussianMV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const MKL_INT  ,  const MKL_INT  , const float *,  const float * ) NOTHROW);
_MKL_API(int,VSRNGGAUSSIANMV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const MKL_INT *,  const MKL_INT *, const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnggaussianmv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const MKL_INT *,  const MKL_INT *, const float *,  const float * ) NOTHROW);

/* Exponential distribution */
_Mkl_Api(int,vdRngExponential,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGEXPONENTIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrngexponential,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngExponential,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGEXPONENTIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrngexponential,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ) NOTHROW);

/* Laplace distribution */
_Mkl_Api(int,vdRngLaplace,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGLAPLACE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrnglaplace,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngLaplace,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGLAPLACE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnglaplace,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);

/* Weibull distribution */
_Mkl_Api(int,vdRngWeibull,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGWEIBULL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *) NOTHROW);
_mkl_api(int,vdrngweibull,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngWeibull,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGWEIBULL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrngweibull,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ) NOTHROW);

/* Rayleigh distribution */
_Mkl_Api(int,vdRngRayleigh,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGRAYLEIGH,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrngrayleigh,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngRayleigh,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGRAYLEIGH,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrngrayleigh,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ) NOTHROW);

/* Lognormal distribution */
_Mkl_Api(int,vdRngLognormal,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGLOGNORMAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *) NOTHROW);
_mkl_api(int,vdrnglognormal,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngLognormal,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGLOGNORMAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnglognormal,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ) NOTHROW);

/* Gumbel distribution */
_Mkl_Api(int,vdRngGumbel,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGGUMBEL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_mkl_api(int,vdrnggumbel,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngGumbel,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGGUMBEL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnggumbel,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ) NOTHROW);

/* Gamma distribution */
_Mkl_Api(int,vdRngGamma,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGGAMMA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *) NOTHROW);
_mkl_api(int,vdrnggamma,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngGamma,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGGAMMA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrnggamma,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ) NOTHROW);

/* Beta distribution */
_Mkl_Api(int,vdRngBeta,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  , const double  ) NOTHROW);
_MKL_API(int,VDRNGBETA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *) NOTHROW);
_mkl_api(int,vdrngbeta,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngBeta,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float  ,  const float   ) NOTHROW);
_MKL_API(int,VSRNGBETA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ) NOTHROW);
_mkl_api(int,vsrngbeta,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ) NOTHROW);

/* Chi-square distribution */
_Mkl_Api(int,vdRngChiSquare,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const int  ) NOTHROW);
_MKL_API(int,VDRNGCHISQUARE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const int *) NOTHROW);
_mkl_api(int,vdrngchisquare,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const int *) NOTHROW);
_Mkl_Api(int,vsRngChiSquare,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const int  ) NOTHROW);
_MKL_API(int,VSRNGCHISQUARE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const int *) NOTHROW);
_mkl_api(int,vsrngchisquare,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const int *) NOTHROW);

/*
//++
//  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Bernoulli distribution */
_Mkl_Api(int,viRngBernoulli,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ) NOTHROW);
_MKL_API(int,VIRNGBERNOULLI,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);
_mkl_api(int,virngbernoulli,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);

/* Uniform distribution */
_Mkl_Api(int,viRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const int  ) NOTHROW);
_MKL_API(int,VIRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *) NOTHROW);
_mkl_api(int,virnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *) NOTHROW);

/* UniformBits distribution */
_Mkl_Api(int,viRngUniformBits,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned int []) NOTHROW);
_MKL_API(int,VIRNGUNIFORMBITS,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []) NOTHROW);
_mkl_api(int,virnguniformbits,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []) NOTHROW);

/* UniformBits32 distribution */
_Mkl_Api(int,viRngUniformBits32,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned int []) NOTHROW);
_MKL_API(int,VIRNGUNIFORMBITS32,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []) NOTHROW);
_mkl_api(int,virnguniformbits32,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []) NOTHROW);

/* UniformBits64 distribution */
_Mkl_Api(int,viRngUniformBits64,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned MKL_INT64 []) NOTHROW);
_MKL_API(int,VIRNGUNIFORMBITS64,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned MKL_INT64 []) NOTHROW);
_mkl_api(int,virnguniformbits64,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned MKL_INT64 []) NOTHROW);

/* Geometric distribution */
_Mkl_Api(int,viRngGeometric,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ) NOTHROW);
_MKL_API(int,VIRNGGEOMETRIC,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);
_mkl_api(int,virnggeometric,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);

/* Binomial distribution */
_Mkl_Api(int,viRngBinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const double  ) NOTHROW);
_MKL_API(int,VIRNGBINOMIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const double *) NOTHROW);
_mkl_api(int,virngbinomial,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const double *) NOTHROW);

/* Multinomial distribution */
_Mkl_Api(int,viRngMultinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const int  , const double []) NOTHROW);
_MKL_API(int,VIRNGMULTINOMIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const double []) NOTHROW);
_mkl_api(int,virngmultinomial,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const double []) NOTHROW);

/* Hypergeometric distribution */
_Mkl_Api(int,viRngHypergeometric,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const int  , const int  ) NOTHROW);
_MKL_API(int,VIRNGHYPERGEOMETRIC,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const int *) NOTHROW);
_mkl_api(int,virnghypergeometric,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const int *) NOTHROW);

/* Negbinomial distribution */
_Mkl_Api(int,viRngNegbinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,viRngNegBinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  , const double  ) NOTHROW);
_MKL_API(int,VIRNGNEGBINOMIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *, const double *) NOTHROW);
_mkl_api(int,virngnegbinomial,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *, const double *) NOTHROW);

/* Poisson distribution */
_Mkl_Api(int,viRngPoisson,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ) NOTHROW);
_MKL_API(int,VIRNGPOISSON,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);
_mkl_api(int,virngpoisson,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *) NOTHROW);

/* PoissonV distribution */
_Mkl_Api(int,viRngPoissonV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double []) NOTHROW);
_MKL_API(int,VIRNGPOISSONV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double []) NOTHROW);
_mkl_api(int,virngpoissonv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double []) NOTHROW);


/*
//++
//  VSL SERVICE FUNCTION DECLARATIONS.
//--
*/
/* NewStream - stream creation/initialization */
_Mkl_Api(int,vslNewStream,(VSLStreamStatePtr* , const MKL_INT  , const MKL_UINT  ) NOTHROW);
_mkl_api(int,vslnewstream,(VSLStreamStatePtr* , const MKL_INT *, const MKL_UINT *) NOTHROW);
_MKL_API(int,VSLNEWSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const MKL_UINT *) NOTHROW);

/* NewStreamEx - advanced stream creation/initialization */
_Mkl_Api(int,vslNewStreamEx,(VSLStreamStatePtr* , const MKL_INT  , const MKL_INT  , const unsigned int[]) NOTHROW);
_mkl_api(int,vslnewstreamex,(VSLStreamStatePtr* , const MKL_INT *, const MKL_INT *, const unsigned int[]) NOTHROW);
_MKL_API(int,VSLNEWSTREAMEX,(VSLStreamStatePtr* , const MKL_INT *, const MKL_INT *, const unsigned int[]) NOTHROW);

_Mkl_Api(int,vsliNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const unsigned int[], const iUpdateFuncPtr) NOTHROW);
_mkl_api(int,vslinewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const unsigned int[], const iUpdateFuncPtr) NOTHROW);
_MKL_API(int,VSLINEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const unsigned int[], const iUpdateFuncPtr) NOTHROW);

_Mkl_Api(int,vsldNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const double[], const double  , const double  , const dUpdateFuncPtr) NOTHROW);
_mkl_api(int,vsldnewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const double[], const double *, const double *, const dUpdateFuncPtr) NOTHROW);
_MKL_API(int,VSLDNEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const double[], const double *, const double *, const dUpdateFuncPtr) NOTHROW);

_Mkl_Api(int,vslsNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const float[], const float  , const float  , const sUpdateFuncPtr) NOTHROW);
_mkl_api(int,vslsnewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const float[], const float *, const float *, const sUpdateFuncPtr) NOTHROW);
_MKL_API(int,VSLSNEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const float[], const float *, const float *, const sUpdateFuncPtr) NOTHROW);

/* DeleteStream - delete stream */
_Mkl_Api(int,vslDeleteStream,(VSLStreamStatePtr*) NOTHROW);
_mkl_api(int,vsldeletestream,(VSLStreamStatePtr*) NOTHROW);
_MKL_API(int,VSLDELETESTREAM,(VSLStreamStatePtr*) NOTHROW);

/* CopyStream - copy all stream information */
_Mkl_Api(int,vslCopyStream,(VSLStreamStatePtr*, const VSLStreamStatePtr) NOTHROW);
_mkl_api(int,vslcopystream,(VSLStreamStatePtr*, const VSLStreamStatePtr) NOTHROW);
_MKL_API(int,VSLCOPYSTREAM,(VSLStreamStatePtr*, const VSLStreamStatePtr) NOTHROW);

/* CopyStreamState - copy stream state only */
_Mkl_Api(int,vslCopyStreamState,(VSLStreamStatePtr  , const VSLStreamStatePtr  ) NOTHROW);
_mkl_api(int,vslcopystreamstate,(VSLStreamStatePtr *, const VSLStreamStatePtr *) NOTHROW);
_MKL_API(int,VSLCOPYSTREAMSTATE,(VSLStreamStatePtr *, const VSLStreamStatePtr *) NOTHROW);

/* LeapfrogStream - leapfrog method */
_Mkl_Api(int,vslLeapfrogStream,(VSLStreamStatePtr  , const MKL_INT  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslleapfrogstream,(VSLStreamStatePtr *, const MKL_INT *, const MKL_INT *) NOTHROW);
_MKL_API(int,VSLLEAPFROGSTREAM,(VSLStreamStatePtr *, const MKL_INT *, const MKL_INT *) NOTHROW);

/* SkipAheadStream - skip-ahead method */
_Mkl_Api(int,vslSkipAheadStream,(VSLStreamStatePtr  , const long long int  ) NOTHROW);
_mkl_api(int,vslskipaheadstream,(VSLStreamStatePtr *, const long long int *) NOTHROW);
_MKL_API(int,VSLSKIPAHEADSTREAM,(VSLStreamStatePtr *, const long long int *) NOTHROW);

/* SkipAheadStreamEx - skip-ahead extended method */
_Mkl_Api(int,vslSkipAheadStreamEx,(VSLStreamStatePtr  , const MKL_INT  , const MKL_UINT64[]) NOTHROW);
_mkl_api(int,vslskipaheadstreamex,(VSLStreamStatePtr *, const MKL_INT *, const MKL_UINT64[]) NOTHROW);
_MKL_API(int,VSLSKIPAHEADSTREAMEX,(VSLStreamStatePtr *, const MKL_INT *, const MKL_UINT64[]) NOTHROW);

/* GetStreamStateBrng - get BRNG associated with given stream */
_Mkl_Api(int,vslGetStreamStateBrng,(const VSLStreamStatePtr  ) NOTHROW);
_mkl_api(int,vslgetstreamstatebrng,(const VSLStreamStatePtr *) NOTHROW);
_MKL_API(int,VSLGETSTREAMSTATEBRNG,(const VSLStreamStatePtr *) NOTHROW);

/* GetNumRegBrngs - get number of registered BRNGs */
_Mkl_Api(int,vslGetNumRegBrngs,(void) NOTHROW);
_mkl_api(int,vslgetnumregbrngs,(void) NOTHROW);
_MKL_API(int,VSLGETNUMREGBRNGS,(void) NOTHROW);

/* RegisterBrng - register new BRNG */
_Mkl_Api(int,vslRegisterBrng,(const VSLBRngProperties* ) NOTHROW);
_mkl_api(int,vslregisterbrng,(const VSLBRngProperties* ) NOTHROW);
_MKL_API(int,VSLREGISTERBRNG,(const VSLBRngProperties* ) NOTHROW);

/* GetBrngProperties - get BRNG properties */
_Mkl_Api(int,vslGetBrngProperties,(const int  , VSLBRngProperties* ) NOTHROW);
_mkl_api(int,vslgetbrngproperties,(const int *, VSLBRngProperties* ) NOTHROW);
_MKL_API(int,VSLGETBRNGPROPERTIES,(const int *, VSLBRngProperties* ) NOTHROW);

/* SaveStreamF - save random stream descriptive data to file */
_Mkl_Api(int,vslSaveStreamF,(const VSLStreamStatePtr  , const char*             ) NOTHROW);
_mkl_api(int,vslsavestreamf,(const VSLStreamStatePtr *, const char* , const int ) NOTHROW);
_MKL_API(int,VSLSAVESTREAMF,(const VSLStreamStatePtr *, const char* , const int ) NOTHROW);

/* LoadStreamF - load random stream descriptive data from file */
_Mkl_Api(int,vslLoadStreamF,(VSLStreamStatePtr *, const char*             ) NOTHROW);
_mkl_api(int,vslloadstreamf,(VSLStreamStatePtr *, const char* , const int ) NOTHROW);
_MKL_API(int,VSLLOADSTREAMF,(VSLStreamStatePtr *, const char* , const int ) NOTHROW);

/* SaveStreamM - save random stream descriptive data to memory */
_Mkl_Api(int,vslSaveStreamM,(const VSLStreamStatePtr  , char* ) NOTHROW);
_mkl_api(int,vslsavestreamm,(const VSLStreamStatePtr *, char* ) NOTHROW);
_MKL_API(int,VSLSAVESTREAMM,(const VSLStreamStatePtr *, char* ) NOTHROW);

/* LoadStreamM - load random stream descriptive data from memory */
_Mkl_Api(int,vslLoadStreamM,(VSLStreamStatePtr *, const char* ) NOTHROW);
_mkl_api(int,vslloadstreamm,(VSLStreamStatePtr *, const char* ) NOTHROW);
_MKL_API(int,VSLLOADSTREAMM,(VSLStreamStatePtr *, const char* ) NOTHROW);

/* GetStreamSize - get size of random stream descriptive data */
_Mkl_Api(int,vslGetStreamSize,(const VSLStreamStatePtr) NOTHROW);
_mkl_api(int,vslgetstreamsize,(const VSLStreamStatePtr) NOTHROW);
_MKL_API(int,VSLGETSTREAMSIZE,(const VSLStreamStatePtr) NOTHROW);

/*
//++
//  VSL CONVOLUTION AND CORRELATION FUNCTION DECLARATIONS.
//--
*/

_Mkl_Api(int,vsldConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslsconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslcconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vsldCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldcorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslscorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzcorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslccorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []) NOTHROW);


_Mkl_Api(int,vsldConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslsconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vsldCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldcorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslscorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzcorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslccorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ) NOTHROW);


_Mkl_Api(int,vsldConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslsconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslcconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vsldCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldcorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslscorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzcorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslccorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []) NOTHROW);


_Mkl_Api(int,vsldConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslsconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vsldCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldcorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslscorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzcorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslccorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ) NOTHROW);


_Mkl_Api(int,vslConvDeleteTask,(VSLConvTaskPtr* ) NOTHROW);
_mkl_api(int,vslconvdeletetask,(VSLConvTaskPtr* ) NOTHROW);
_MKL_API(int,VSLCONVDeleteTask,(VSLConvTaskPtr* ) NOTHROW);

_Mkl_Api(int,vslCorrDeleteTask,(VSLCorrTaskPtr* ) NOTHROW);
_mkl_api(int,vslcorrdeletetask,(VSLCorrTaskPtr* ) NOTHROW);
_MKL_API(int,VSLCORRDeleteTask,(VSLCorrTaskPtr* ) NOTHROW);


_Mkl_Api(int,vslConvCopyTask,(VSLConvTaskPtr* , const VSLConvTaskPtr  ) NOTHROW);
_mkl_api(int,vslconvcopytask,(VSLConvTaskPtr* , const VSLConvTaskPtr* ) NOTHROW);
_MKL_API(int,VSLCONVCopyTask,(VSLConvTaskPtr* , const VSLConvTaskPtr* ) NOTHROW);

_Mkl_Api(int,vslCorrCopyTask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr  ) NOTHROW);
_mkl_api(int,vslcorrcopytask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr* ) NOTHROW);
_MKL_API(int,VSLCORRCopyTask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr* ) NOTHROW);


_Mkl_Api(int,vslConvSetMode,(VSLConvTaskPtr  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslconvsetmode,(VSLConvTaskPtr* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCONVSETMODE,(VSLConvTaskPtr* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslCorrSetMode,(VSLCorrTaskPtr  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcorrsetmode,(VSLCorrTaskPtr* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCORRSETMODE,(VSLCorrTaskPtr* , const MKL_INT* ) NOTHROW);


_Mkl_Api(int,vslConvSetInternalPrecision,(VSLConvTaskPtr  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslconvsetinternalprecision,(VSLConvTaskPtr* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCONVSETINTERNALPRECISION,(VSLConvTaskPtr* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslCorrSetInternalPrecision,(VSLCorrTaskPtr  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcorrsetinternalprecision,(VSLCorrTaskPtr* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCORRSETINTERNALPRECISION,(VSLCorrTaskPtr* , const MKL_INT* ) NOTHROW);


_Mkl_Api(int,vslConvSetStart,(VSLConvTaskPtr  , const MKL_INT []) NOTHROW);
_mkl_api(int,vslconvsetstart,(VSLConvTaskPtr* , const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCONVSETSTART,(VSLConvTaskPtr* , const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslCorrSetStart,(VSLCorrTaskPtr  , const MKL_INT []) NOTHROW);
_mkl_api(int,vslcorrsetstart,(VSLCorrTaskPtr* , const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCORRSETSTART,(VSLCorrTaskPtr* , const MKL_INT []) NOTHROW);


_Mkl_Api(int,vslConvSetDecimation,(VSLConvTaskPtr  , const MKL_INT []) NOTHROW);
_mkl_api(int,vslconvsetdecimation,(VSLConvTaskPtr* , const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCONVSETDECIMATION,(VSLConvTaskPtr* , const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslCorrSetDecimation,(VSLCorrTaskPtr  , const MKL_INT []) NOTHROW);
_mkl_api(int,vslcorrsetdecimation,(VSLCorrTaskPtr* , const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCORRSETDECIMATION,(VSLCorrTaskPtr* , const MKL_INT []) NOTHROW);


_Mkl_Api(int,vsldConvExec,(VSLConvTaskPtr  , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldconvexec,(VSLConvTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCONVEXEC,(VSLConvTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsConvExec,(VSLConvTaskPtr  , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslsconvexec,(VSLConvTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCONVEXEC,(VSLConvTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzConvExec,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzconvexec,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCONVEXEC,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcConvExec,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslcconvexec,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCONVEXEC,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vsldCorrExec,(VSLCorrTaskPtr  , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldcorrexec,(VSLCorrTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCORREXEC,(VSLCorrTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsCorrExec,(VSLCorrTaskPtr  , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslscorrexec,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCORREXEC,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzCorrExec,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzcorrexec,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCORREXEC,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcCorrExec,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslccorrexec,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCORREXEC,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);


_Mkl_Api(int,vsldConvExec1D,(VSLConvTaskPtr  , const double [], const MKL_INT  , const double [], const MKL_INT  , double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldconvexec1d,(VSLConvTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCONVEXEC1D,(VSLConvTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsConvExec1D,(VSLConvTaskPtr  , const float [],  const MKL_INT  , const float [],  const MKL_INT  , float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslsconvexec1d,(VSLConvTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCONVEXEC1D,(VSLConvTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzConvExec1D,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzconvexec1d,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCONVEXEC1D,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcConvExec1D,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcconvexec1d,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCONVEXEC1D,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vsldCorrExec1D,(VSLCorrTaskPtr  , const double [], const MKL_INT  , const double [], const MKL_INT  , double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldcorrexec1d,(VSLCorrTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCORREXEC1D,(VSLCorrTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsCorrExec1D,(VSLCorrTaskPtr  , const float [],  const MKL_INT  , const float [],  const MKL_INT  , float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslscorrexec1d,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCORREXEC1D,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzCorrExec1D,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzcorrexec1d,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCORREXEC1D,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcCorrExec1D,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslccorrexec1d,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCORREXEC1D,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);


_Mkl_Api(int,vsldConvExecX,(VSLConvTaskPtr  , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldconvexecx,(VSLConvTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCONVEXECX,(VSLConvTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsConvExecX,(VSLConvTaskPtr  , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslsconvexecx,(VSLConvTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCONVEXECX,(VSLConvTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzConvExecX,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzconvexecx,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCONVEXECX,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcConvExecX,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslcconvexecx,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCONVEXECX,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vsldCorrExecX,(VSLCorrTaskPtr  , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldcorrexecx,(VSLCorrTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDCORREXECX,(VSLCorrTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsCorrExecX,(VSLCorrTaskPtr  , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslscorrexecx,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSCORREXECX,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslzCorrExecX,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslzcorrexecx,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLZCORREXECX,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslcCorrExecX,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_mkl_api(int,vslccorrexecx,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);
_MKL_API(int,VSLCCORREXECX,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []) NOTHROW);


_Mkl_Api(int,vsldConvExecX1D,(VSLConvTaskPtr  , const double [], const MKL_INT  , double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldconvexecx1d,(VSLConvTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCONVEXECX1D,(VSLConvTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsConvExecX1D,(VSLConvTaskPtr  , const float [],  const MKL_INT  , float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslsconvexecx1d,(VSLConvTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCONVEXECX1D,(VSLConvTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzConvExecX1D,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzconvexecx1d,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCONVEXECX1D,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcConvExecX1D,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslcconvexecx1d,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCONVEXECX1D,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vsldCorrExecX1D,(VSLCorrTaskPtr  , const double [], const MKL_INT  , double [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldcorrexecx1d,(VSLCorrTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDCORREXECX1D,(VSLCorrTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsCorrExecX1D,(VSLCorrTaskPtr  , const float [],  const MKL_INT  , float [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslscorrexecx1d,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSCORREXECX1D,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslzCorrExecX1D,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ) NOTHROW);
_mkl_api(int,vslzcorrexecx1d,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLZCORREXECX1D,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslcCorrExecX1D,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ) NOTHROW);
_mkl_api(int,vslccorrexecx1d,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLCCORREXECX1D,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ) NOTHROW);


/*
//++
//  SUMMARARY STATTISTICS LIBRARY ROUTINES
//--
*/

/*
//  Task constructors
*/
_Mkl_Api(int,vsldSSNewTask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []) NOTHROW);
_mkl_api(int,vsldssnewtask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLDSSNEWTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []) NOTHROW);

_Mkl_Api(int,vslsSSNewTask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []) NOTHROW);
_mkl_api(int,vslsssnewtask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []) NOTHROW);
_MKL_API(int,VSLSSSNEWTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []) NOTHROW);


/*
// Task editors
*/

/*
// Editor to modify a task parameter
*/
_Mkl_Api(int,vsldSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const double* ) NOTHROW);
_mkl_api(int,vsldssedittask,(VSLSSTaskPtr* , const MKL_INT* , const double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const float* ) NOTHROW);
_mkl_api(int,vslsssedittask,(VSLSSTaskPtr* , const MKL_INT* , const float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const float* ) NOTHROW);

_Mkl_Api(int,vsliSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const MKL_INT* ) NOTHROW);
_mkl_api(int,vslissedittask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLISSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ) NOTHROW);

/*
// Task specific editors
*/

/*
// Editors to modify moments related parameters
*/
_Mkl_Api(int,vsldSSEditMoments,(VSLSSTaskPtr  , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);
_mkl_api(int,vsldsseditmoments,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITMOMENTS,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditMoments,(VSLSSTaskPtr  , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);
_mkl_api(int,vslssseditmoments,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITMOMENTS,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);


/*
// Editors to modify sums related parameters
*/
_Mkl_Api(int,vsldSSEditSums,(VSLSSTaskPtr  , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);
_mkl_api(int,vsldsseditsums,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITSUMS,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditSums,(VSLSSTaskPtr  , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);
_mkl_api(int,vslssseditsums,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITSUMS,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ) NOTHROW);


/*
// Editors to modify variance-covariance/correlation matrix related parameters
*/
_Mkl_Api(int,vsldSSEditCovCor,(VSLSSTaskPtr  , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vsldsseditcovcor,(VSLSSTaskPtr* , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSEDITCOVCOR,(VSLSSTaskPtr* , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSEditCovCor,(VSLSSTaskPtr  , float* , float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vslssseditcovcor,(VSLSSTaskPtr* , float* , float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSEDITCOVCOR,(VSLSSTaskPtr* , float* , float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);


/*
// Editors to modify cross-product matrix related parameters
*/
_Mkl_Api(int,vsldSSEditCP,(VSLSSTaskPtr  , double* , double* ,  double* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vsldsseditcp,(VSLSSTaskPtr* , double* , double* ,  double* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSEDITCP,(VSLSSTaskPtr* , double* , double* ,  double* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSEditCP,(VSLSSTaskPtr  , float* , float* , float* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vslssseditcp,(VSLSSTaskPtr* , float* , float* , float* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSEDITCP,(VSLSSTaskPtr* , float* , float* , float* , const MKL_INT* ) NOTHROW);


/*
// Editors to modify partial variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditPartialCovCor,(VSLSSTaskPtr  , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vsldsseditpartialcovcor,(VSLSSTaskPtr* , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSEDITPARTIALCOVCOR,(VSLSSTaskPtr* , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSEditPartialCovCor,(VSLSSTaskPtr  , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ) NOTHROW);
_mkl_api(int,vslssseditpartialcovcor,(VSLSSTaskPtr* , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSEDITPARTIALCOVCOR,(VSLSSTaskPtr* , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ) NOTHROW);


/*
// Editors to modify quantiles related parameters
*/
_Mkl_Api(int,vsldSSEditQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* , double* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vsldsseditquantiles,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , double* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSEDITQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , double* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSEditQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* , float* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vslssseditquantiles,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , float* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSEDITQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , float* , const MKL_INT* ) NOTHROW);


/*
// Editors to modify stream data quantiles related parameters
*/
_Mkl_Api(int,vsldSSEditStreamQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ) NOTHROW);
_mkl_api(int,vsldsseditstreamquantiles,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITSTREAMQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditStreamQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ) NOTHROW);
_mkl_api(int,vslssseditstreamquantiles,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITSTREAMQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ) NOTHROW);

/*
// Editors to modify pooled/group variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditPooledCovariance,(VSLSSTaskPtr  , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ) NOTHROW);
_mkl_api(int,vsldsseditpooledcovariance,(VSLSSTaskPtr* , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITPOOLEDCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditPooledCovariance,(VSLSSTaskPtr  , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ) NOTHROW);
_mkl_api(int,vslssseditpooledcovariance,(VSLSSTaskPtr* , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITPOOLEDCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ) NOTHROW);


/*
// Editors to modify robust variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditRobustCovariance,(VSLSSTaskPtr  , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ) NOTHROW);
_mkl_api(int,vsldsseditrobustcovariance,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITROBUSTCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditRobustCovariance,(VSLSSTaskPtr  , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ) NOTHROW);
_mkl_api(int,vslssseditrobustcovariance,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITROBUSTCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ) NOTHROW);


/*
// Editors to modify outliers detection parameters
*/
_Mkl_Api(int,vsldSSEditOutliersDetection,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* ) NOTHROW);
_mkl_api(int,vsldsseditoutliersdetection,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITOUTLIERSDETECTION,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditOutliersDetection,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* ) NOTHROW);
_mkl_api(int,vslssseditoutliersdetection,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITOUTLIERSDETECTION,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* ) NOTHROW);

/*
// Editors to modify missing values support parameters
*/
_Mkl_Api(int,vsldSSEditMissingValues,(VSLSSTaskPtr  , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ) NOTHROW);
_mkl_api(int,vsldsseditmissingvalues,(VSLSSTaskPtr* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ) NOTHROW);
_MKL_API(int,VSLDSSEDITMISSINGVALUES,(VSLSSTaskPtr* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ) NOTHROW);

_Mkl_Api(int,vslsSSEditMissingValues,(VSLSSTaskPtr  , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ) NOTHROW);
_mkl_api(int,vslssseditmissingvalues,(VSLSSTaskPtr* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ) NOTHROW);
_MKL_API(int,VSLSSSEDITMISSINGVALUES,(VSLSSTaskPtr* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ) NOTHROW);

/*
// Editors to modify matrixparametrization parameters
*/
_Mkl_Api(int,vsldSSEditCorParameterization,(VSLSSTaskPtr  , const double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vsldsseditcorparameterization,(VSLSSTaskPtr* , const double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSEDITCORPARAMETERIZATION,(VSLSSTaskPtr* , const double* , const MKL_INT* , double* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSEditCorParameterization,(VSLSSTaskPtr  , const float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);
_mkl_api(int,vslssseditcorparameterization,(VSLSSTaskPtr* , const float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSEDITCORPARAMETERIZATION,(VSLSSTaskPtr* , const float* , const MKL_INT* , float* , const MKL_INT* ) NOTHROW);


/*
// Compute routines
*/
_Mkl_Api(int,vsldSSCompute,(VSLSSTaskPtr  , const unsigned MKL_INT64  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vsldsscompute,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLDSSCOMPUTE,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ) NOTHROW);

_Mkl_Api(int,vslsSSCompute,(VSLSSTaskPtr  , const unsigned MKL_INT64  , const MKL_INT  ) NOTHROW);
_mkl_api(int,vslssscompute,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ) NOTHROW);
_MKL_API(int,VSLSSSCOMPUTE,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ) NOTHROW);


/*
// Task destructor
*/
_Mkl_Api(int,vslSSDeleteTask,(VSLSSTaskPtr* ) NOTHROW);
_mkl_api(int,vslssdeletetask,(VSLSSTaskPtr* ) NOTHROW);
_MKL_API(int,VSLSSDELETETASK,(VSLSSTaskPtr* ) NOTHROW);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#include "mkl_vsl_functions_64.h"

#endif /* __MKL_VSL_FUNCTIONS_H__ */
