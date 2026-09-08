/*******************************************************************************
* Copyright 1999-2022 Intel Corporation.
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
! Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) types definition
!****************************************************************************/

#ifndef _MKL_TYPES_H_
#define _MKL_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* oneMKL Complex type for single precision */
#ifndef MKL_Complex8
typedef
struct _MKL_Complex8 {
    float real;
    float imag;
} MKL_Complex8;
#endif

/* oneMKL Complex type for double precision */
#ifndef MKL_Complex16
typedef
struct _MKL_Complex16 {
    double real;
    double imag;
} MKL_Complex16;
#endif

/* oneMKL Version type */
typedef
struct {
    int    MajorVersion;
    int    MinorVersion;
    int    UpdateVersion;
    char * ProductStatus;
    char * Build;
    char * Processor;
    char * Platform;
} MKLVersion;

/* oneMKL integer types for LP64 and ILP64 */
#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
    #define MKL_INT64 __int64
    #define MKL_UINT64 unsigned __int64
#else
    #define MKL_INT64 long long int
    #define MKL_UINT64 unsigned long long int
#endif

#ifdef MKL_ILP64

/* oneMKL ILP64 integer types */
#ifndef MKL_INT
    #define MKL_INT MKL_INT64
#endif
#ifndef MKL_UINT
    #define MKL_UINT MKL_UINT64
#endif
#define MKL_LONG MKL_INT64

#else

/* oneMKL LP64 integer types */
#ifndef MKL_INT
    #define MKL_INT int
#endif
#ifndef MKL_UINT
    #define MKL_UINT unsigned int
#endif
#define MKL_LONG long int

#endif

/* oneMKL integer types */
#ifndef MKL_UINT8
    #define MKL_UINT8 unsigned char
#endif
#ifndef MKL_INT8
    #define MKL_INT8 char
#endif
#ifndef MKL_INT16
    #define MKL_INT16 short
#endif
#ifndef MKL_BF16
    #define MKL_BF16 unsigned short
#endif
#ifndef MKL_INT32
    #define MKL_INT32 int
#endif
#ifndef MKL_F16
    #define MKL_F16 unsigned short
#endif

/* oneMKL domain names */
#define MKL_DOMAIN_ALL      0
#define MKL_DOMAIN_BLAS     1
#define MKL_DOMAIN_FFT      2
#define MKL_DOMAIN_VML      3
#define MKL_DOMAIN_PARDISO  4
#define MKL_DOMAIN_LAPACK   5

/* oneMKL CBWR */

/* mkl_cbwr_get options */
#define MKL_CBWR_BRANCH 1
#define MKL_CBWR_ALL   ~0

/* flag specific values */
#define MKL_CBWR_STRICT    0x10000

/* branch specific values */
#define MKL_CBWR_UNSET_ALL      MKL_CBWR_OFF
#define MKL_CBWR_OFF            0
#define MKL_CBWR_BRANCH_OFF     1
#define MKL_CBWR_AUTO           2
#define MKL_CBWR_COMPATIBLE     3
#define MKL_CBWR_SSE2           4
#define MKL_CBWR_SSSE3          6
#define MKL_CBWR_SSE4_1         7
#define MKL_CBWR_SSE4_2         8
#define MKL_CBWR_AVX            9
#define MKL_CBWR_AVX2          10
#define MKL_CBWR_AVX512_MIC    11
#define MKL_CBWR_AVX512        12
#define MKL_CBWR_AVX512_MIC_E1 13
#define MKL_CBWR_AVX512_E1     14

/* error codes */
#define MKL_CBWR_SUCCESS                   0
#define MKL_CBWR_ERR_INVALID_SETTINGS     -1
#define MKL_CBWR_ERR_INVALID_INPUT        -2
#define MKL_CBWR_ERR_UNSUPPORTED_BRANCH   -3
#define MKL_CBWR_ERR_UNKNOWN_BRANCH       -4
#define MKL_CBWR_ERR_MODE_CHANGE_FAILURE  -8

/* obsolete */
#define MKL_CBWR_SSE3           5

typedef enum {
    MKL_ROW_MAJOR = 101,
    MKL_COL_MAJOR = 102
} MKL_LAYOUT;

typedef enum {
    MKL_NOTRANS = 111,
    MKL_TRANS = 112,
    MKL_CONJTRANS = 113,
    MKL_CONJ = 114
} MKL_TRANSPOSE;

typedef enum {
    MKL_UPPER = 121,
    MKL_LOWER = 122
} MKL_UPLO;

typedef enum {
    MKL_NONUNIT = 131,
    MKL_UNIT = 132
} MKL_DIAG;

typedef enum {
    MKL_LEFT = 141,
    MKL_RIGHT = 142
} MKL_SIDE;

typedef enum {
    MKL_COMPACT_SSE = 181,
    MKL_COMPACT_AVX = 182,
    MKL_COMPACT_AVX512 = 183
} MKL_COMPACT_PACK;

typedef void (*sgemm_jit_kernel_t)(void*, float*,         float*,         float*);
typedef void (*dgemm_jit_kernel_t)(void*, double*,        double*,        double*);
typedef void (*cgemm_jit_kernel_t)(void*, MKL_Complex8*,  MKL_Complex8*,  MKL_Complex8*);
typedef void (*zgemm_jit_kernel_t)(void*, MKL_Complex16*, MKL_Complex16*, MKL_Complex16*);

typedef enum {
    MKL_JIT_SUCCESS = 0,
    MKL_NO_JIT = 1,
    MKL_JIT_ERROR = 2
} mkl_jit_status_t;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_TYPES_H_ */
