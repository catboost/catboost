/* file: mkl_vsl_types.h */
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
//  This file contains user-level type definitions.
//--
*/

#ifndef __MKL_VSL_TYPES_H__
#define __MKL_VSL_TYPES_H__

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
//  POINTER TO STREAM STATE STRUCTURE
//  This is a void pointer to hide implementation details.
*/
typedef void* VSLStreamStatePtr;
typedef void* VSLConvTaskPtr;
typedef void* VSLCorrTaskPtr;
typedef void* VSLSSTaskPtr;

/*
//  POINTERS TO BASIC RANDOM NUMBER GENERATOR FUNCTIONS
//  Each BRNG must have following implementations:
//
//  * Stream initialization (InitStreamPtr)
//  * Integer-value recurrence implementation (iBRngPtr)
//  * Single precision implementation (sBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
//  * Double precision implementation (dBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
*/
typedef int (*InitStreamPtr)( int method, VSLStreamStatePtr stream, \
        int n, const unsigned int params[] );
typedef int (*sBRngPtr)( VSLStreamStatePtr stream, int n, float r[], \
        float a, float b );
typedef int (*dBRngPtr)( VSLStreamStatePtr stream, int n, double r[], \
        double a, double b );
typedef int (*iBRngPtr)( VSLStreamStatePtr stream, int n, unsigned int r[] );

/*********** Pointers to callback functions for abstract streams *************/
typedef int (*iUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     unsigned int ibuf[], int* nmin, int* nmax, int* idx );
typedef int (*dUpdateFuncPtr)( VSLStreamStatePtr stream, int* n,
     double dbuf[], int* nmin, int* nmax, int* idx );
typedef int (*sUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     float sbuf[], int* nmin, int* nmax, int* idx );


/*
//  BASIC RANDOM NUMBER GENERATOR PROPERTIES STRUCTURE
//  The structure describes the properties of given basic generator, e.g. size
//  of the stream state structure, pointers to function implementations, etc.
//
//  BRNG properties structure fields:
//  StreamStateSize - size of the stream state structure (in bytes)
//  WordSize        - size of base word (in bytes). Typically this is 4 bytes.
//  NSeeds          - number of words necessary to describe generator's state
//  NBits           - number of bits actually used in base word. For example,
//                    only 31 least significant bits are actually used in
//                    basic random number generator MCG31m1 with 4-byte base
//                    word. NBits field is useful while interpreting random
//                    words as a sequence of random bits.
//  IncludesZero    - FALSE if 0 cannot be generated in integer-valued
//                    implementation; TRUE if 0 can be potentially generated in
//                    integer-valued implementation.
//  InitStream      - pointer to stream state initialization function
//  sBRng           - pointer to single precision implementation
//  dBRng           - pointer to double precision implementation
//  iBRng           - pointer to integer-value implementation
*/
typedef struct _VSLBRngProperties {
    int StreamStateSize;       /* Stream state size (in bytes) */
    int NSeeds;                /* Number of seeds */
    int IncludesZero;          /* Zero flag */
    int WordSize;              /* Size (in bytes) of base word */
    int NBits;                 /* Number of actually used bits */
    InitStreamPtr InitStream;  /* Pointer to InitStream func */
    sBRngPtr sBRng;            /* Pointer to S func */
    dBRngPtr dBRng;            /* Pointer to D func */
    iBRngPtr iBRng;            /* Pointer to I func */
} VSLBRngProperties;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_TYPES_H__ */
