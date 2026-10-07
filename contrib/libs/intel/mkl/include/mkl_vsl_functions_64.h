/* file: mkl_vsl_functions_64.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation.
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
//  User-level VSL function declarations _64
//--
*/

#ifndef __MKL_VSL_FUNCTIONS_64_H__
#define __MKL_VSL_FUNCTIONS_64_H__

#include "mkl_vsl_types.h"

#ifndef NOTHROW
  #ifdef __cplusplus
    #if __cplusplus > 199711L
      #define NOTHROW noexcept
    #else
      #define NOTHROW throw()
    #endif
  #else
   #define NOTHROW
  #endif
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

/* VSL routines with MKL_INT64 input parameters */
/* Note: ILP64 interfaces are not supported on IA-32 architecture */
#if defined(_WIN64) || defined(__MINGW64__) || defined(__x86_64__)

/*
//++
//  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Cauchy distribution */
_Mkl_Api(int,vdRngCauchy_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngCauchy_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float   ) NOTHROW);


/* Uniform distribution */
_Mkl_Api(int,vdRngUniform_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngUniform_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float   ) NOTHROW);


/* Gaussian distribution */
_Mkl_Api(int,vdRngGaussian_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngGaussian_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float   ) NOTHROW);


/* GaussianMV distribution */
_Mkl_Api(int,vdRngGaussianMV_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const MKL_INT64  ,  const MKL_INT64  , const double *, const double *) NOTHROW);
_Mkl_Api(int,vsRngGaussianMV_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const MKL_INT64  ,  const MKL_INT64  , const float *,  const float * ) NOTHROW);


/* Exponential distribution */
_Mkl_Api(int,vdRngExponential_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  ,  double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngExponential_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  ,  float [],  const float  ,  const float   ) NOTHROW);


/* Laplace distribution */
_Mkl_Api(int,vdRngLaplace_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngLaplace_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float   ) NOTHROW);


/* Weibull distribution */
_Mkl_Api(int,vdRngWeibull_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngWeibull_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float  ,  const float   ) NOTHROW);

/* Rayleigh distribution */
_Mkl_Api(int,vdRngRayleigh_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  ,  double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngRayleigh_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  ,  float [],  const float  ,  const float   ) NOTHROW);


/* Lognormal distribution */
_Mkl_Api(int,vdRngLognormal_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  , const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngLognormal_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float  ,  const float  ,  const float   ) NOTHROW);

/* Gumbel distribution */
_Mkl_Api(int,vdRngGumbel_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngGumbel_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float   ) NOTHROW);

/* Gamma distribution */
_Mkl_Api(int,vdRngGamma_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngGamma_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float  ,  const float   ) NOTHROW);

/* Beta distribution */
_Mkl_Api(int,vdRngBeta_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const double  , const double  , const double  , const double  ) NOTHROW);
_Mkl_Api(int,vsRngBeta_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const float  ,  const float  ,  const float  ,  const float   ) NOTHROW);

/* Chi-square distribution */
_Mkl_Api(int,vdRngChiSquare_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , double [], const int  ) NOTHROW);
_Mkl_Api(int,vsRngChiSquare_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , float [],  const int  ) NOTHROW);


/*
//++
//  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Bernoulli distribution */
_Mkl_Api(int,viRngBernoulli_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double  ) NOTHROW);

/* Uniform distribution */
_Mkl_Api(int,viRngUniform_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const int  , const int  ) NOTHROW);

/* UniformBits distribution */
_Mkl_Api(int,viRngUniformBits_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , unsigned int []) NOTHROW);

/* UniformBits32 distribution */
_Mkl_Api(int,viRngUniformBits32_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , unsigned int []) NOTHROW);

/* UniformBits64 distribution */
_Mkl_Api(int,viRngUniformBits64_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , unsigned MKL_INT64 []) NOTHROW);

/* Geometric distribution */
_Mkl_Api(int,viRngGeometric_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double  ) NOTHROW);

/* Binomial distribution */
_Mkl_Api(int,viRngBinomial_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const int  , const double  ) NOTHROW);

/* Multinomial distribution */
_Mkl_Api(int,viRngMultinomial_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const int  , const int  , const double []) NOTHROW);

/* Hypergeometric distribution */
_Mkl_Api(int,viRngHypergeometric_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const int  , const int  , const int  ) NOTHROW);

/* Negbinomial distribution */
_Mkl_Api(int,viRngNegbinomial_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double  , const double  ) NOTHROW);
_Mkl_Api(int,viRngNegBinomial_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double  , const double  ) NOTHROW);

/* Poisson distribution */
_Mkl_Api(int,viRngPoisson_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double  ) NOTHROW);

/* PoissonV distribution */
_Mkl_Api(int,viRngPoissonV_64,(const MKL_INT64  , VSLStreamStatePtr  , const MKL_INT64  , int [], const double []) NOTHROW);


/*
//++
//  VSL SERVICE FUNCTION DECLARATIONS.
//--
*/
/* NewStream - stream creation/initialization */
_Mkl_Api(int,vslNewStream_64,(VSLStreamStatePtr* , const MKL_INT64  , const MKL_UINT64  ) NOTHROW);

/* NewStreamEx - advanced stream creation/initialization */
_Mkl_Api(int,vslNewStreamEx_64,(VSLStreamStatePtr* , const MKL_INT64  , const MKL_INT64  , const unsigned int[]) NOTHROW);

/*
//++
//  SUMMARARY STATTISTICS LIBRARY ROUTINES
//--
*/

/*
//  Task constructors
*/
_Mkl_Api(int,vsldSSNewTask_64,(VSLSSTaskPtr* , const MKL_INT64* , const MKL_INT64* , const MKL_INT64* , const double [], const double [], const MKL_INT64 []) NOTHROW);

_Mkl_Api(int,vslsSSNewTask_64,(VSLSSTaskPtr* , const MKL_INT64* , const MKL_INT64* , const MKL_INT64* , const float  [], const float  [], const MKL_INT64 []) NOTHROW);

#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_FUNCTIONS_64_H__ */
