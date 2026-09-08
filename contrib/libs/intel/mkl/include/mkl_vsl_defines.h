/* file: mkl_vsl_defines.h */
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
//  User-level macro definitions
//--
*/

#ifndef __MKL_VSL_DEFINES_H__
#define __MKL_VSL_DEFINES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/*
// "No error" status
*/
#define VSL_STATUS_OK                      0
#define VSL_ERROR_OK                       0

/*
// Common errors (-1..-999)
*/
#define VSL_ERROR_FEATURE_NOT_IMPLEMENTED  -1
#define VSL_ERROR_UNKNOWN                  -2
#define VSL_ERROR_BADARGS                  -3
#define VSL_ERROR_MEM_FAILURE              -4
#define VSL_ERROR_NULL_PTR                 -5
#define VSL_ERROR_CPU_NOT_SUPPORTED        -6


/*
// RNG errors (-1000..-1999)
*/
/* brng errors */
#define VSL_RNG_ERROR_INVALID_BRNG_INDEX        -1000
#define VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED      -1002
#define VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED     -1003
#define VSL_RNG_ERROR_SKIPAHEADEX_UNSUPPORTED   -1004
#define VSL_RNG_ERROR_BRNGS_INCOMPATIBLE        -1005
#define VSL_RNG_ERROR_BAD_STREAM                -1006
#define VSL_RNG_ERROR_BRNG_TABLE_FULL           -1007
#define VSL_RNG_ERROR_BAD_STREAM_STATE_SIZE     -1008
#define VSL_RNG_ERROR_BAD_WORD_SIZE             -1009
#define VSL_RNG_ERROR_BAD_NSEEDS                -1010
#define VSL_RNG_ERROR_BAD_NBITS                 -1011
#define VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED       -1012
#define VSL_RNG_ERROR_LEAPFROG_NSTREAMS_TOO_BIG -1013
#define VSL_RNG_ERROR_BRNG_NOT_SUPPORTED        -1014

/* abstract stream related errors */
#define VSL_RNG_ERROR_BAD_UPDATE                -1120
#define VSL_RNG_ERROR_NO_NUMBERS                -1121
#define VSL_RNG_ERROR_INVALID_ABSTRACT_STREAM   -1122

/* non deterministic stream related errors */
#define VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED     -1130
#define VSL_RNG_ERROR_NONDETERM_NRETRIES_EXCEEDED -1131

/* ARS5 stream related errors */
#define VSL_RNG_ERROR_ARS5_NOT_SUPPORTED        -1140

/* Multinomial distribution probability array related errors */
#define VSL_DISTR_MULTINOMIAL_BAD_PROBABILITY_ARRAY    -1150

/* read/write stream to file errors */
#define VSL_RNG_ERROR_FILE_CLOSE                -1100
#define VSL_RNG_ERROR_FILE_OPEN                 -1101
#define VSL_RNG_ERROR_FILE_WRITE                -1102
#define VSL_RNG_ERROR_FILE_READ                 -1103

#define VSL_RNG_ERROR_BAD_FILE_FORMAT           -1110
#define VSL_RNG_ERROR_UNSUPPORTED_FILE_VER      -1111

#define VSL_RNG_ERROR_BAD_MEM_FORMAT            -1200

/* Convolution/correlation errors */
#define VSL_CC_ERROR_NOT_IMPLEMENTED        (-2000)
#define VSL_CC_ERROR_ALLOCATION_FAILURE     (-2001)
#define VSL_CC_ERROR_BAD_DESCRIPTOR         (-2200)
#define VSL_CC_ERROR_SERVICE_FAILURE        (-2210)
#define VSL_CC_ERROR_EDIT_FAILURE           (-2211)
#define VSL_CC_ERROR_EDIT_PROHIBITED        (-2212)
#define VSL_CC_ERROR_COMMIT_FAILURE         (-2220)
#define VSL_CC_ERROR_COPY_FAILURE           (-2230)
#define VSL_CC_ERROR_DELETE_FAILURE         (-2240)
#define VSL_CC_ERROR_BAD_ARGUMENT           (-2300)
#define VSL_CC_ERROR_DIMS                   (-2301)
#define VSL_CC_ERROR_START                  (-2302)
#define VSL_CC_ERROR_DECIMATION             (-2303)
#define VSL_CC_ERROR_XSHAPE                 (-2311)
#define VSL_CC_ERROR_YSHAPE                 (-2312)
#define VSL_CC_ERROR_ZSHAPE                 (-2313)
#define VSL_CC_ERROR_XSTRIDE                (-2321)
#define VSL_CC_ERROR_YSTRIDE                (-2322)
#define VSL_CC_ERROR_ZSTRIDE                (-2323)
#define VSL_CC_ERROR_X                      (-2331)
#define VSL_CC_ERROR_Y                      (-2332)
#define VSL_CC_ERROR_Z                      (-2333)
#define VSL_CC_ERROR_JOB                    (-2100)
#define VSL_CC_ERROR_KIND                   (-2110)
#define VSL_CC_ERROR_MODE                   (-2120)
#define VSL_CC_ERROR_TYPE                   (-2130)
#define VSL_CC_ERROR_PRECISION              (-2140)
#define VSL_CC_ERROR_EXTERNAL_PRECISION     (-2141)
#define VSL_CC_ERROR_INTERNAL_PRECISION     (-2142)
#define VSL_CC_ERROR_METHOD                 (-2400)
#define VSL_CC_ERROR_OTHER                  (-2800)

/*
//++
// SUMMARY STATTISTICS ERROR/WARNING CODES
//--
*/

/*
// Warnings
*/
#define VSL_SS_NOT_FULL_RANK_MATRIX                   4028
#define VSL_SS_SEMIDEFINITE_COR                       4029
/*
// Errors (-4000..-4999)
*/
#define VSL_SS_ERROR_ALLOCATION_FAILURE              -4000
#define VSL_SS_ERROR_BAD_DIMEN                       -4001
#define VSL_SS_ERROR_BAD_OBSERV_N                    -4002
#define VSL_SS_ERROR_STORAGE_NOT_SUPPORTED           -4003
#define VSL_SS_ERROR_BAD_INDC_ADDR                   -4004
#define VSL_SS_ERROR_BAD_WEIGHTS                     -4005
#define VSL_SS_ERROR_BAD_MEAN_ADDR                   -4006
#define VSL_SS_ERROR_BAD_2R_MOM_ADDR                 -4007
#define VSL_SS_ERROR_BAD_3R_MOM_ADDR                 -4008
#define VSL_SS_ERROR_BAD_4R_MOM_ADDR                 -4009
#define VSL_SS_ERROR_BAD_2C_MOM_ADDR                 -4010
#define VSL_SS_ERROR_BAD_3C_MOM_ADDR                 -4011
#define VSL_SS_ERROR_BAD_4C_MOM_ADDR                 -4012
#define VSL_SS_ERROR_BAD_KURTOSIS_ADDR               -4013
#define VSL_SS_ERROR_BAD_SKEWNESS_ADDR               -4014
#define VSL_SS_ERROR_BAD_MIN_ADDR                    -4015
#define VSL_SS_ERROR_BAD_MAX_ADDR                    -4016
#define VSL_SS_ERROR_BAD_VARIATION_ADDR              -4017
#define VSL_SS_ERROR_BAD_COV_ADDR                    -4018
#define VSL_SS_ERROR_BAD_COR_ADDR                    -4019
#define VSL_SS_ERROR_BAD_ACCUM_WEIGHT_ADDR           -4020
#define VSL_SS_ERROR_BAD_QUANT_ORDER_ADDR            -4021
#define VSL_SS_ERROR_BAD_QUANT_ORDER                 -4022
#define VSL_SS_ERROR_BAD_QUANT_ADDR                  -4023
#define VSL_SS_ERROR_BAD_ORDER_STATS_ADDR            -4024
#define VSL_SS_ERROR_MOMORDER_NOT_SUPPORTED          -4025
#define VSL_SS_ERROR_ALL_OBSERVS_OUTLIERS            -4026
#define VSL_SS_ERROR_BAD_ROBUST_COV_ADDR             -4027
#define VSL_SS_ERROR_BAD_ROBUST_MEAN_ADDR            -4028
#define VSL_SS_ERROR_METHOD_NOT_SUPPORTED            -4029
#define VSL_SS_ERROR_BAD_GROUP_INDC_ADDR             -4030
#define VSL_SS_ERROR_NULL_TASK_DESCRIPTOR            -4031
#define VSL_SS_ERROR_BAD_OBSERV_ADDR                 -4032
#define VSL_SS_ERROR_SINGULAR_COV                    -4033
#define VSL_SS_ERROR_BAD_POOLED_COV_ADDR             -4034
#define VSL_SS_ERROR_BAD_POOLED_MEAN_ADDR            -4035
#define VSL_SS_ERROR_BAD_GROUP_COV_ADDR              -4036
#define VSL_SS_ERROR_BAD_GROUP_MEAN_ADDR             -4037
#define VSL_SS_ERROR_BAD_GROUP_INDC                  -4038
#define VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_ADDR        -4039
#define VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_N_ADDR      -4040
#define VSL_SS_ERROR_BAD_OUTLIERS_WEIGHTS_ADDR       -4041
#define VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_ADDR      -4042
#define VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_N_ADDR    -4043
#define VSL_SS_ERROR_BAD_STORAGE_ADDR                -4044
#define VSL_SS_ERROR_BAD_PARTIAL_COV_IDX_ADDR        -4045
#define VSL_SS_ERROR_BAD_PARTIAL_COV_ADDR            -4046
#define VSL_SS_ERROR_BAD_PARTIAL_COR_ADDR            -4047
#define VSL_SS_ERROR_BAD_MI_PARAMS_ADDR              -4048
#define VSL_SS_ERROR_BAD_MI_PARAMS_N_ADDR            -4049
#define VSL_SS_ERROR_BAD_MI_BAD_PARAMS_N             -4050
#define VSL_SS_ERROR_BAD_MI_PARAMS                   -4051
#define VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_N_ADDR    -4052
#define VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_ADDR      -4053
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR          -4054
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N_ADDR        -4055
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_N_ADDR         -4056
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_ADDR           -4057
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N             -4058
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_N              -4059
#define VSL_SS_ERROR_BAD_MI_OUTPUT_PARAMS            -4060
#define VSL_SS_ERROR_BAD_MI_PRIOR_N_ADDR             -4061
#define VSL_SS_ERROR_BAD_MI_PRIOR_ADDR               -4062
#define VSL_SS_ERROR_BAD_MI_MISSING_VALS_N           -4063
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N_ADDR  -4064
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_ADDR    -4065
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N       -4066
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS         -4067
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER_ADDR     -4068
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER          -4069
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ADDR           -4070
#define VSL_SS_ERROR_BAD_PARAMTR_COR_ADDR            -4071
#define VSL_SS_ERROR_BAD_COR                         -4072
#define VSL_SS_ERROR_BAD_PARTIAL_COV_IDX             -4073
#define VSL_SS_ERROR_BAD_SUM_ADDR                    -4074
#define VSL_SS_ERROR_BAD_2R_SUM_ADDR                 -4075
#define VSL_SS_ERROR_BAD_3R_SUM_ADDR                 -4076
#define VSL_SS_ERROR_BAD_4R_SUM_ADDR                 -4077
#define VSL_SS_ERROR_BAD_2C_SUM_ADDR                 -4078
#define VSL_SS_ERROR_BAD_3C_SUM_ADDR                 -4079
#define VSL_SS_ERROR_BAD_4C_SUM_ADDR                 -4080
#define VSL_SS_ERROR_BAD_CP_ADDR                     -4081
#define VSL_SS_ERROR_BAD_MDAD_ADDR                   -4082
#define VSL_SS_ERROR_BAD_MNAD_ADDR                   -4083
#define VSL_SS_ERROR_BAD_SORTED_OBSERV_ADDR          -4084
#define VSL_SS_ERROR_INDICES_NOT_SUPPORTED           -4085


/*
// Internal errors caused by internal routines of the functions
*/
#define VSL_SS_ERROR_ROBCOV_INTERN_C1                -5000
#define VSL_SS_ERROR_PARTIALCOV_INTERN_C1            -5010
#define VSL_SS_ERROR_PARTIALCOV_INTERN_C2            -5011
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C1           -5021
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C2           -5022
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C3           -5023
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C4           -5024
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C5           -5025
#define VSL_SS_ERROR_PARAMTRCOR_INTERN_C1            -5030
#define VSL_SS_ERROR_COVRANK_INTERNAL_ERROR_C1       -5040
#define VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C1        -5041
#define VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C2        -5042


/*
// CONV/CORR RELATED MACRO DEFINITIONS
*/
#define VSL_CONV_MODE_AUTO        0
#define VSL_CORR_MODE_AUTO        0
#define VSL_CONV_MODE_DIRECT      1
#define VSL_CORR_MODE_DIRECT      1
#define VSL_CONV_MODE_FFT         2
#define VSL_CORR_MODE_FFT         2
#define VSL_CONV_PRECISION_SINGLE 1
#define VSL_CORR_PRECISION_SINGLE 1
#define VSL_CONV_PRECISION_DOUBLE 2
#define VSL_CORR_PRECISION_DOUBLE 2

/*
//++
//  BASIC RANDOM NUMBER GENERATOR (BRNG) RELATED MACRO DEFINITIONS
//--
*/

/*
//  MAX NUMBER OF BRNGS CAN BE REGISTERED IN VSL
//  No more than VSL_MAX_REG_BRNGS basic generators can be registered in VSL
//  (including predefined basic generators).
//
//  Change this number to increase/decrease number of BRNGs can be registered.
*/
#define VSL_MAX_REG_BRNGS           512

/*
//  PREDEFINED BRNG NAMES
*/
#define VSL_BRNG_SHIFT      20
#define VSL_BRNG_INC        (1<<VSL_BRNG_SHIFT)

#define VSL_BRNG_MCG31          (VSL_BRNG_INC)
#define VSL_BRNG_R250           (VSL_BRNG_MCG31    +VSL_BRNG_INC)
#define VSL_BRNG_MRG32K3A       (VSL_BRNG_R250     +VSL_BRNG_INC)
#define VSL_BRNG_MCG59          (VSL_BRNG_MRG32K3A +VSL_BRNG_INC)
#define VSL_BRNG_WH             (VSL_BRNG_MCG59    +VSL_BRNG_INC)
#define VSL_BRNG_SOBOL          (VSL_BRNG_WH       +VSL_BRNG_INC)
#define VSL_BRNG_NIEDERR        (VSL_BRNG_SOBOL    +VSL_BRNG_INC)
#define VSL_BRNG_MT19937        (VSL_BRNG_NIEDERR  +VSL_BRNG_INC)
#define VSL_BRNG_MT2203         (VSL_BRNG_MT19937  +VSL_BRNG_INC)
#define VSL_BRNG_IABSTRACT      (VSL_BRNG_MT2203   +VSL_BRNG_INC)
#define VSL_BRNG_DABSTRACT      (VSL_BRNG_IABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_SABSTRACT      (VSL_BRNG_DABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_SFMT19937      (VSL_BRNG_SABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_NONDETERM      (VSL_BRNG_SFMT19937+VSL_BRNG_INC)
#define VSL_BRNG_ARS5           (VSL_BRNG_NONDETERM+VSL_BRNG_INC)
#define VSL_BRNG_PHILOX4X32X10  (VSL_BRNG_ARS5     +VSL_BRNG_INC)


/*
// PREDEFINED PARAMETERS FOR NON-DETERMNINISTIC RANDOM NUMBER
// GENERATOR
// The library provides an abstraction to the source of non-deterministic
// random numbers supported in HW. Current version of the library provides
// interface to RDRAND-based only, available in latest Intel CPU.
*/
#define VSL_BRNG_RDRAND  0x0
#define VSL_BRNG_NONDETERM_NRETRIES 10

/*
//  LEAPFROG METHOD FOR GRAY-CODE BASED QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random number
//  basic generators. In contrast to pseudorandom number basic generators,
//  quasi-random ones take the dimension as initialization parameter.
//
//  Suppose that quasi-random number generator (QRNG) dimension is S. QRNG
//  sequence is a sequence of S-dimensional vectors:
//
//     x0=(x0[0],x0[1],...,x0[S-1]),x1=(x1[0],x1[1],...,x1[S-1]),...
//
//  VSL treats the output of any basic generator as 1-dimensional, however:
//
//     x0[0],x0[1],...,x0[S-1],x1[0],x1[1],...,x1[S-1],...
//
//  Because of nature of VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR QRNGs,
//  the only S-stride Leapfrog method is supported for them. In other words,
//  user can generate subsequences, which consist of fixed elements of
//  vectors x0,x1,... For example, if 0 element is fixed, the following
//  subsequence is generated:
//
//     x0[1],x1[1],x2[1],...
//
//  To use the s-stride Leapfrog method with given QRNG, user should call
//  vslLeapfrogStream function with parameter k equal to element to be fixed
//  (0<=k<S) and parameter nstreams equal to VSL_QRNG_LEAPFROG_COMPONENTS.
*/
#define VSL_QRNG_LEAPFROG_COMPONENTS    0x7fffffff

/*
//  USER-DEFINED PARAMETERS FOR QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random
//  number basic generators. Default parameters of the generators
//  support generation of quasi-random number vectors of dimensions
//  S<=40 for SOBOL and S<=318 for NIEDERRITER. The library provides
//  opportunity to register user-defined initial values for the
//  generators and generate quasi-random vectors of desirable dimension.
//  There is also opportunity to register user-defined parameters for
//  default dimensions and obtain another sequence of quasi-random vectors.
//  Service function vslNewStreamEx is used to pass the parameters to
//  the library. Data are packed into array params, parameter of the routine.
//  First element of the array is used for dimension S, second element
//  contains indicator, VSL_USER_QRNG_INITIAL_VALUES, of user-defined
//  parameters for quasi-random number generators.
//  Macros VSL_USER_PRIMITIVE_POLYMS and VSL_USER_INIT_DIRECTION_NUMBERS
//  are used to describe which data are passed to SOBOL QRNG and
//  VSL_USER_IRRED_POLYMS - which data are passed to NIEDERRITER QRNG.
//  For example, to demonstrate that both primitive polynomials and initial
//  direction numbers are passed in SOBOL one should set third element of the
//  array params to  VSL_USER_PRIMITIVE_POLYMS | VSL_USER_DIRECTION_NUMBERS.
//  Macro VSL_QRNG_OVERRIDE_1ST_DIM_INIT is used to override default
//  initialization for the first dimension. Macro VSL_USER_DIRECTION_NUMBERS
//  is used when direction numbers calculated on the user side are passed
//  into the generators. More detailed description of interface for
//  registration of user-defined QRNG initial parameters can be found
//  in VslNotes.pdf.
*/
#define VSL_USER_QRNG_INITIAL_VALUES     0x1
#define VSL_USER_PRIMITIVE_POLYMS        0x1
#define VSL_USER_INIT_DIRECTION_NUMBERS  0x2
#define VSL_USER_IRRED_POLYMS            0x1
#define VSL_USER_DIRECTION_NUMBERS       0x4
#define VSL_QRNG_OVERRIDE_1ST_DIM_INIT   0x8


/*
//  INITIALIZATION METHODS FOR USER-DESIGNED BASIC RANDOM NUMBER GENERATORS.
//  Each BRNG must support at least VSL_INIT_METHOD_STANDARD initialization
//  method. In addition, VSL_INIT_METHOD_LEAPFROG, VSL_INIT_METHOD_SKIPAHEAD and
//  VSL_INIT_METHOD_SKIPAHEADEX initialization methods can be supported.
//
//  If VSL_INIT_METHOD_LEAPFROG is not supported then initialization routine
//  must return VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED error code.
//
//  If VSL_INIT_METHOD_SKIPAHEAD is not supported then initialization routine
//  must return VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED error code.
//
//  If VSL_INIT_METHOD_SKIPAHEADEX is not supported then initialization routine
//  must return VSL_RNG_ERROR_SKIPAHEADEX_UNSUPPORTED error code.
//
//  If there is no error during initialization, the initialization routine must
//  return VSL_ERROR_OK code.
*/
#define VSL_INIT_METHOD_STANDARD    0
#define VSL_INIT_METHOD_LEAPFROG    1
#define VSL_INIT_METHOD_SKIPAHEAD   2
#define VSL_INIT_METHOD_SKIPAHEADEX 3

/*
//++
//  ACCURACY FLAG FOR DISTRIBUTION GENERATORS
//  This flag defines mode of random number generation.
//  If accuracy mode is set distribution generators will produce
//  numbers lying exactly within definitional domain for all values
//  of distribution parameters. In this case slight performance
//  degradation is expected. By default accuracy mode is switched off
//  admitting random numbers to be out of the definitional domain for
//  specific values of distribution parameters.
//  This macro is used to form names for accuracy versions of
//  distribution number generators
//--
*/
#define VSL_RNG_METHOD_ACCURACY_FLAG (1<<30)

/*
//++
//  TRANSFORMATION METHOD NAMES FOR DISTRIBUTION RANDOM NUMBER GENERATORS
//  VSL interface allows more than one generation method in a distribution
//  transformation subroutine. Following macro definitions are used to
//  specify generation method for given distribution generator.
//
//  Method name macro is constructed as
//
//     VSL_RNG_METHOD_<Distribution>_<Method>
//
//  where
//
//     <Distribution> - probability distribution
//     <Method> - method name
//
//  VSL_RNG_METHOD_<Distribution>_<Method> should be used with
//  vsl<precision>Rng<Distribution> function only, where
//
//     <precision> - s (single) or d (double)
//     <Distribution> - probability distribution
//--
*/

/*
// Uniform
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORM_STD 0 /* vsl{s,d,i}RngUniform */

#define VSL_RNG_METHOD_UNIFORM_STD_ACCURATE \
  VSL_RNG_METHOD_UNIFORM_STD | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngUniform */

/*
// Uniform Bits
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS_STD 0 /* vsliRngUniformBits */

/*
// Uniform Bits 32
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS32_STD 0 /* vsliRngUniformBits32 */

/*
// Uniform Bits 64
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS64_STD 0 /* vsliRngUniformBits64 */

/*
// Gaussian
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER   0 /* vsl{d,s}RngGaussian */
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2  1 /* vsl{d,s}RngGaussian */
#define VSL_RNG_METHOD_GAUSSIAN_ICDF        2 /* vsl{d,s}RngGaussian */

/*
// GaussianMV - multivariate (correlated) normal
// Multivariate (correlated) normal random number generator is based on
// uncorrelated Gaussian random number generator (see vslsRngGaussian and
// vsldRngGaussian functions):
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER   0 /* vsl{d,s}RngGaussianMV */
#define VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2  1 /* vsl{d,s}RngGaussianMV */
#define VSL_RNG_METHOD_GAUSSIANMV_ICDF        2 /* vsl{d,s}RngGaussianMV */

/*
// Exponential
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_EXPONENTIAL_ICDF 0 /* vsl{d,s}RngExponential */

#define VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE \
   VSL_RNG_METHOD_EXPONENTIAL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngExponential */

/*
// Laplace
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
//
// ICDF - inverse cumulative distribution function method:
//
//           x=+/-ln(u) with probability 1/2,
//
//        where
//
//           x - random number with Laplace distribution,
//           u - uniformly distributed random number
*/
#define VSL_RNG_METHOD_LAPLACE_ICDF 0 /* vsl{d,s}RngLaplace */

/*
// Weibull
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_WEIBULL_ICDF 0 /* vsl{d,s}RngWeibull */

#define VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE \
   VSL_RNG_METHOD_WEIBULL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngWeibull */


/*
// Cauchy
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_CAUCHY_ICDF 0 /* vsl{d,s}RngCauchy */

/*
// Rayleigh
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_RAYLEIGH_ICDF 0 /* vsl{d,s}RngRayleigh */

#define VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE \
   VSL_RNG_METHOD_RAYLEIGH_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngRayleigh */

/*
// Lognormal
//
// <Method>   <Short Description>
// BOXMULLER2       Box-Muller 2 algorithm based method
*/
#define VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 0 /* vsl{d,s}RngLognormal */
#define VSL_RNG_METHOD_LOGNORMAL_ICDF 1       /* vsl{d,s}RngLognormal */

#define VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE \
   VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngLognormal */

#define VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE \
   VSL_RNG_METHOD_LOGNORMAL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngLognormal */


/*
// Gumbel
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GUMBEL_ICDF 0 /* vsl{d,s}RngGumbel */

/*
// Gamma
//
// Comments:
// alpha>1             - algorithm of Marsaglia is used, nonlinear
//                       transformation of gaussian numbers based on
//                       acceptance/rejection method with squeezes;
// alpha>=0.6, alpha<1 - rejection from the Weibull distribution is used;
// alpha<0.6           - transformation of exponential power distribution
//                       (EPD) is used, EPD random numbers are generated
//                       by means of acceptance/rejection technique;
// alpha=1             - gamma distribution reduces to exponential
//                       distribution
*/
#define VSL_RNG_METHOD_GAMMA_GNORM 0 /* vsl{d,s}RngGamma */

#define VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE \
   VSL_RNG_METHOD_GAMMA_GNORM | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngGamma */


/*
// Beta
//
// Comments:
// CJA - stands for first letters of Cheng, Johnk, and Atkinson.
// Cheng    - for min(p,q) > 1 method of Cheng,
//            generation of beta random numbers of the second kind
//            based on acceptance/rejection technique and its
//            transformation to beta random numbers of the first kind;
// Johnk    - for max(p,q) < 1 methods of Johnk and Atkinson:
//            if q + K*p^2+C<=0, K=0.852..., C=-0.956...
//            algorithm of Johnk:
//            beta distributed random number is generated as
//            u1^(1/p) / (u1^(1/p)+u2^(1/q)), if u1^(1/p)+u2^(1/q)<=1;
//            otherwise switching algorithm of Atkinson: interval (0,1)
//            is divided into two domains (0,t) and (t,1), on each interval
//            acceptance/rejection technique with convenient majorizing
//            function is used;
// Atkinson - for min(p,q)<1, max(p,q)>1 switching algorithm of Atkinson
//            is used (with another point t, see short description above);
// ICDF     - inverse cumulative distribution function method according
//            to formulas x=1-u^(1/q) for p = 1, and x = u^(1/p) for q=1,
//            where x is beta distributed random number,
//            u - uniformly distributed random number.
//            for p=q=1 beta distribution reduces to uniform distribution.
//
*/
#define VSL_RNG_METHOD_BETA_CJA 0 /* vsl{d,s}RngBeta */

#define VSL_RNG_METHOD_BETA_CJA_ACCURATE \
   VSL_RNG_METHOD_BETA_CJA | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngBeta */


/*
// ChiSquare
//
// Comments:
// v = 1, v = 3               - chi-square distributed random number is
//                              generated as a sum of squares of v independent
//                              normal random numbers;
// v is even and v = 16       - chi-square distributed random number is
//                              generated using the following formula:
//                              x = -2*ln(u[0]*...*u[v/2-1]),
//                              where u[i] - random numbers uniformly
//                              distributed over the interval (0,1);
// v > 16, v is odd and v > 3 - chi-square distribution reduces to gamma
//                              distribution;
*/
#define VSL_RNG_METHOD_CHISQUARE_CHI2GAMMA 0 /* vsl{d,s}RngChiSquare */


/*
// Bernoulli
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_BERNOULLI_ICDF 0 /* vsliRngBernoulli */

/*
// Geometric
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GEOMETRIC_ICDF 0 /* vsliRngGeometric */

/*
// Binomial
//
// <Method>   <Short Description>
// BTPE       for ntrial*min(p,1-p)>30 acceptance/rejection method with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            otherwise table lookup method is used
*/
#define VSL_RNG_METHOD_BINOMIAL_BTPE 0 /* vsliRngBinomial */

/*
// Multinomial
//
// <Method>    <Short Description>
// MULTPOISSON Poisson Approximation of Multinomial Distribution method
*/
#define VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON 0 /* vsliRngMultinomial */

/*
// Hypergeometric
//
// <Method>   <Short Description>
// H2PE       if mode of distribution is large, acceptance/rejection method is
//            used with decomposition onto 3 regions:
//
//               * rectangular;
//               * left exponential tail;
//               * right exponential tail.
//
//            otherwise table lookup method is used
*/
#define VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE 0 /* vsliRngHypergeometric */

/*
// Poisson
//
// <Method>   <Short Description>
// PTPE       if lambda>=27, acceptance/rejection method is used with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            otherwise table lookup method is used
//
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            table lookup method is used.
*/
#define VSL_RNG_METHOD_POISSON_PTPE     0 /* vsliRngPoisson */
#define VSL_RNG_METHOD_POISSON_POISNORM 1 /* vsliRngPoisson */

/*
// Poisson
//
// <Method>   <Short Description>
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            ICDF method is used.
*/
#define VSL_RNG_METHOD_POISSONV_POISNORM 0 /* vsliRngPoissonV */

/*
// Negbinomial
//
// <Method>   <Short Description>
// NBAR       if (a-1)*(1-p)/p>=100, acceptance/rejection method is used with
//            decomposition onto 5 regions:
//
//               * rectangular;
//               * 2 trapezoid;
//               * left exponential tail;
//               * right exponential tail.
//
//            otherwise table lookup method is used.
*/
#define VSL_RNG_METHOD_NEGBINOMIAL_NBAR 0 /* vsliRngNegbinomial */

/*
//++
//  MATRIX STORAGE SCHEMES
//--
*/

/*
// Some multivariate random number generators, e.g. GaussianMV, operate
// with matrix parameters. To optimize matrix parameters usage VSL offers
// following matrix storage schemes. (See VSL documentation for more details).
//
// FULL     - whole matrix is stored
// PACKED   - lower/higher triangular matrix is packed in 1-dimensional array
// DIAGONAL - diagonal elements are packed in 1-dimensional array
*/
#define VSL_MATRIX_STORAGE_FULL     0
#define VSL_MATRIX_STORAGE_PACKED   1
#define VSL_MATRIX_STORAGE_DIAGONAL 2


/*
// SUMMARY STATISTICS (SS) RELATED MACRO DEFINITIONS
*/

/*
//++
//  MATRIX STORAGE SCHEMES
//--
*/
/*
// SS routines work with matrix parameters, e.g. matrix of observations,
// variance-covariance matrix. To optimize work with matrices the library
// provides the following storage matrix schemes.
*/
/*
// Matrix of observations:
// ROWS    - observations of the random vector are stored in raws, that
//           is, i-th row of the matrix of observations contains values
//           of i-th component of the random vector
// COLS    - observations of the random vector are stored in columns that
//           is, i-th column of the matrix of observations contains values
//           of i-th component of the random vector
*/
#define VSL_SS_MATRIX_STORAGE_ROWS     0x00010000
#define VSL_SS_MATRIX_STORAGE_COLS     0x00020000

/*
// Variance-covariance/correlation matrix:
// FULL     - whole matrix is stored
// L_PACKED - lower triangular matrix is stored as 1-dimensional array
// U_PACKED - upper triangular matrix is stored as 1-dimensional array
*/
#define VSL_SS_MATRIX_STORAGE_FULL            0x00000000
#define VSL_SS_MATRIX_STORAGE_L_PACKED        0x00000001
#define VSL_SS_MATRIX_STORAGE_U_PACKED        0x00000002


/*
//++
//  Summary Statistics METHODS
//--
*/
/*
// SS routines provide computation of basic statistical estimates
// (central/raw moments up to 4th order, variance-covariance,
//  minimum, maximum, skewness/kurtosis) using the following methods
//  - FAST  - estimates are computed for price of one or two passes over
//            observations using highly optimized oneMKL routines
//  - 1PASS - estimate is computed for price of one pass of the observations
//  - FAST_USER_MEAN - estimates are computed for price of one or two passes
//            over observations given user defined mean for central moments,
//            covariance and correlation
//  - CP_TO_COVCOR - convert cross-product matrix to variance-covariance/
//            correlation matrix
//  - SUM_TO_MOM - convert raw/central sums to raw/central moments
//
*/
#define VSL_SS_METHOD_FAST                    0x00000001
#define VSL_SS_METHOD_1PASS                   0x00000002
#define VSL_SS_METHOD_FAST_USER_MEAN          0x00000100
#define VSL_SS_METHOD_CP_TO_COVCOR            0x00000200
#define VSL_SS_METHOD_SUM_TO_MOM              0x00000400

/*
// SS provides routine for parametrization of correlation matrix using
// SPECTRAL DECOMPOSITION (SD) method
*/
#define VSL_SS_METHOD_SD                      0x00000004

/*
// SS routine for robust estimation of variance-covariance matrix
// and mean supports Rocke algorithm, TBS-estimator
*/
#define VSL_SS_METHOD_TBS                     0x00000008

/*
//  SS routine for estimation of missing values
//  supports Multiple Imputation (MI) method
*/
#define VSL_SS_METHOD_MI                      0x00000010

/*
// SS provides routine for detection of outliers, BACON method
*/
#define VSL_SS_METHOD_BACON                   0x00000020

/*
// SS supports routine for estimation of quantiles for streaming data
// using the following methods:
// - ZW      - intermediate estimates of quantiles during processing
//             the next block are computed
// - ZW_FAST - intermediate estimates of quantiles during processing
//             the next block are not computed
*/
#define VSL_SS_METHOD_SQUANTS_ZW              0x00000040
#define VSL_SS_METHOD_SQUANTS_ZW_FAST         0x00000080


/*
// Input of BACON algorithm is set of 3 parameters:
// - Initialization method of the algorithm
// - Parameter alfa such that 1-alfa is percentile of Chi2 distribution
// - Stopping criterion
*/
/*
// Number of BACON algorithm parameters
*/
#define VSL_SS_BACON_PARAMS_N         3

/*
// SS implementation of BACON algorithm supports two initialization methods:
// - Mahalanobis distance based method
// - Median based method
*/
#define VSL_SS_METHOD_BACON_MAHALANOBIS_INIT  0x00000001
#define VSL_SS_METHOD_BACON_MEDIAN_INIT       0x00000002

/*
// SS routine for sorting data, RADIX method
*/
#define VSL_SS_METHOD_RADIX                   0x00100000

/*
// Input of TBS algorithm is set of 4 parameters:
// - Breakdown point
// - Asymptotic rejection probability
// - Stopping criterion
// - Maximum number of iterations
*/
/*
// Number of TBS algorithm parameters
*/
#define VSL_SS_TBS_PARAMS_N           4

/*
// Input of MI algorithm is set of 5 parameters:
// - Maximal number of iterations for EM algorithm
// - Maximal number of iterations for DA algorithm
// - Stopping criterion
// - Number of sets to impute
// - Total number of missing values in dataset
*/
/*
// Number of MI algorithm parameters
*/
#define VSL_SS_MI_PARAMS_SIZE         5

/*
// SS MI algorithm expects that missing values are
// marked with NANs
*/
#define VSL_SS_DNAN                    0xFFF8000000000000
#define VSL_SS_SNAN                    0xFFC00000

/*
// Input of ZW algorithm is 1 parameter:
// - accuracy of quantile estimation
*/
/*
// Number of ZW algorithm parameters
*/
#define VSL_SS_SQUANTS_ZW_PARAMS_N   1


/*
//++
// MACROS USED SS EDIT AND COMPUTE ROUTINES
//--
*/

/*
// SS EditTask routine is way to edit input and output parameters of the task,
// e.g., pointers to arrays which hold observations, weights of observations,
// arrays of mean estimates or covariance estimates.
// Macros below define parameters available for modification
*/
#define VSL_SS_ED_DIMEN                                 1
#define VSL_SS_ED_OBSERV_N                              2
#define VSL_SS_ED_OBSERV                                3
#define VSL_SS_ED_OBSERV_STORAGE                        4
#define VSL_SS_ED_INDC                                  5
#define VSL_SS_ED_WEIGHTS                               6
#define VSL_SS_ED_MEAN                                  7
#define VSL_SS_ED_2R_MOM                                8
#define VSL_SS_ED_3R_MOM                                9
#define VSL_SS_ED_4R_MOM                               10
#define VSL_SS_ED_2C_MOM                               11
#define VSL_SS_ED_3C_MOM                               12
#define VSL_SS_ED_4C_MOM                               13
#define VSL_SS_ED_SUM                                  67
#define VSL_SS_ED_2R_SUM                               68
#define VSL_SS_ED_3R_SUM                               69
#define VSL_SS_ED_4R_SUM                               70
#define VSL_SS_ED_2C_SUM                               71
#define VSL_SS_ED_3C_SUM                               72
#define VSL_SS_ED_4C_SUM                               73
#define VSL_SS_ED_KURTOSIS                             14
#define VSL_SS_ED_SKEWNESS                             15
#define VSL_SS_ED_MIN                                  16
#define VSL_SS_ED_MAX                                  17
#define VSL_SS_ED_VARIATION                            18
#define VSL_SS_ED_COV                                  19
#define VSL_SS_ED_COV_STORAGE                          20
#define VSL_SS_ED_COR                                  21
#define VSL_SS_ED_COR_STORAGE                          22
#define VSL_SS_ED_CP                                   74
#define VSL_SS_ED_CP_STORAGE                           75
#define VSL_SS_ED_ACCUM_WEIGHT                         23
#define VSL_SS_ED_QUANT_ORDER_N                        24
#define VSL_SS_ED_QUANT_ORDER                          25
#define VSL_SS_ED_QUANT_QUANTILES                      26
#define VSL_SS_ED_ORDER_STATS                          27
#define VSL_SS_ED_GROUP_INDC                           28
#define VSL_SS_ED_POOLED_COV_STORAGE                   29
#define VSL_SS_ED_POOLED_MEAN                          30
#define VSL_SS_ED_POOLED_COV                           31
#define VSL_SS_ED_GROUP_COV_INDC                       32
#define VSL_SS_ED_REQ_GROUP_INDC                       32
#define VSL_SS_ED_GROUP_MEAN                           33
#define VSL_SS_ED_GROUP_COV_STORAGE                    34
#define VSL_SS_ED_GROUP_COV                            35
#define VSL_SS_ED_ROBUST_COV_STORAGE                   36
#define VSL_SS_ED_ROBUST_COV_PARAMS_N                  37
#define VSL_SS_ED_ROBUST_COV_PARAMS                    38
#define VSL_SS_ED_ROBUST_MEAN                          39
#define VSL_SS_ED_ROBUST_COV                           40
#define VSL_SS_ED_OUTLIERS_PARAMS_N                    41
#define VSL_SS_ED_OUTLIERS_PARAMS                      42
#define VSL_SS_ED_OUTLIERS_WEIGHT                      43
#define VSL_SS_ED_ORDER_STATS_STORAGE                  44
#define VSL_SS_ED_PARTIAL_COV_IDX                      45
#define VSL_SS_ED_PARTIAL_COV                          46
#define VSL_SS_ED_PARTIAL_COV_STORAGE                  47
#define VSL_SS_ED_PARTIAL_COR                          48
#define VSL_SS_ED_PARTIAL_COR_STORAGE                  49
#define VSL_SS_ED_MI_PARAMS_N                          50
#define VSL_SS_ED_MI_PARAMS                            51
#define VSL_SS_ED_MI_INIT_ESTIMATES_N                  52
#define VSL_SS_ED_MI_INIT_ESTIMATES                    53
#define VSL_SS_ED_MI_SIMUL_VALS_N                      54
#define VSL_SS_ED_MI_SIMUL_VALS                        55
#define VSL_SS_ED_MI_ESTIMATES_N                       56
#define VSL_SS_ED_MI_ESTIMATES                         57
#define VSL_SS_ED_MI_PRIOR_N                           58
#define VSL_SS_ED_MI_PRIOR                             59
#define VSL_SS_ED_PARAMTR_COR                          60
#define VSL_SS_ED_PARAMTR_COR_STORAGE                  61
#define VSL_SS_ED_STREAM_QUANT_PARAMS_N                62
#define VSL_SS_ED_STREAM_QUANT_PARAMS                  63
#define VSL_SS_ED_STREAM_QUANT_ORDER_N                 64
#define VSL_SS_ED_STREAM_QUANT_ORDER                   65
#define VSL_SS_ED_STREAM_QUANT_QUANTILES               66
#define VSL_SS_ED_MDAD                                 76
#define VSL_SS_ED_MNAD                                 77
#define VSL_SS_ED_SORTED_OBSERV                        78
#define VSL_SS_ED_SORTED_OBSERV_STORAGE                79


/*
// SS Compute routine calculates estimates supported by the library
// Macros below define estimates to compute
*/
#define VSL_SS_MEAN                       0x0000000000000001
#define VSL_SS_2R_MOM                     0x0000000000000002
#define VSL_SS_3R_MOM                     0x0000000000000004
#define VSL_SS_4R_MOM                     0x0000000000000008
#define VSL_SS_2C_MOM                     0x0000000000000010
#define VSL_SS_3C_MOM                     0x0000000000000020
#define VSL_SS_4C_MOM                     0x0000000000000040
#define VSL_SS_SUM                        0x0000000002000000
#define VSL_SS_2R_SUM                     0x0000000004000000
#define VSL_SS_3R_SUM                     0x0000000008000000
#define VSL_SS_4R_SUM                     0x0000000010000000
#define VSL_SS_2C_SUM                     0x0000000020000000
#define VSL_SS_3C_SUM                     0x0000000040000000
#define VSL_SS_4C_SUM                     0x0000000080000000
#define VSL_SS_KURTOSIS                   0x0000000000000080
#define VSL_SS_SKEWNESS                   0x0000000000000100
#define VSL_SS_VARIATION                  0x0000000000000200
#define VSL_SS_MIN                        0x0000000000000400
#define VSL_SS_MAX                        0x0000000000000800
#define VSL_SS_COV                        0x0000000000001000
#define VSL_SS_COR                        0x0000000000002000
#define VSL_SS_CP                         0x0000000100000000
#define VSL_SS_POOLED_COV                 0x0000000000004000
#define VSL_SS_GROUP_COV                  0x0000000000008000
#define VSL_SS_POOLED_MEAN                0x0000000800000000
#define VSL_SS_GROUP_MEAN                 0x0000001000000000
#define VSL_SS_QUANTS                     0x0000000000010000
#define VSL_SS_ORDER_STATS                0x0000000000020000
#define VSL_SS_SORTED_OBSERV              0x0000008000000000
#define VSL_SS_ROBUST_COV                 0x0000000000040000
#define VSL_SS_OUTLIERS                   0x0000000000080000
#define VSL_SS_PARTIAL_COV                0x0000000000100000
#define VSL_SS_PARTIAL_COR                0x0000000000200000
#define VSL_SS_MISSING_VALS               0x0000000000400000
#define VSL_SS_PARAMTR_COR                0x0000000000800000
#define VSL_SS_STREAM_QUANTS              0x0000000001000000
#define VSL_SS_MDAD                       0x0000000200000000
#define VSL_SS_MNAD                       0x0000000400000000

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_DEFINES_H__ */
