/*******************************************************************************
* Copyright 2015-2017 Intel Corporation All Rights Reserved.
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

#ifndef _MKL_DNN_TYPES_H
#define _MKL_DNN_TYPES_H

#include <stdlib.h>

#if defined(__cplusplus_cli)
struct _uniPrimitive_s {};
struct _dnnLayout_s {};
#endif

typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;
typedef void* dnnPrimitiveAttributes_t;

#define DNN_MAX_DIMENSION       32
#define DNN_QUERY_MAX_LENGTH    128

typedef enum {
    E_SUCCESS                   =  0,
    E_INCORRECT_INPUT_PARAMETER = -1,
    E_UNEXPECTED_NULL_POINTER   = -2,
    E_MEMORY_ERROR              = -3,
    E_UNSUPPORTED_DIMENSION     = -4,
    E_UNIMPLEMENTED             = -127
} dnnError_t;

typedef enum {
    /** GEMM base convolution (unimplemented) */
    dnnAlgorithmConvolutionGemm,
    /** Direct convolution */
    dnnAlgorithmConvolutionDirect,
    /** FFT based convolution (unimplemented) */
    dnnAlgorithmConvolutionFFT,
    /** Maximum pooling */
    dnnAlgorithmPoolingMax,
    /** Minimum pooling */
    dnnAlgorithmPoolingMin,
    /** Average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvgExcludePadding,
    /** Alias for average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvg = dnnAlgorithmPoolingAvgExcludePadding,
    /** Average pooling (padded values are taken into account) */
    dnnAlgorithmPoolingAvgIncludePadding
} dnnAlgorithm_t;

typedef enum {
    dnnResourceSrc            = 0,
    dnnResourceFrom           = 0,
    dnnResourceDst            = 1,
    dnnResourceTo             = 1,
    dnnResourceFilter         = 2,
    dnnResourceScaleShift     = 2,
    dnnResourceBias           = 3,
    dnnResourceMean           = 3,
    dnnResourceDiffSrc        = 4,
    dnnResourceDiffFilter     = 5,
    dnnResourceDiffScaleShift = 5,
    dnnResourceDiffBias       = 6,
    dnnResourceVariance       = 6,
    dnnResourceDiffDst        = 7,
    dnnResourceWorkspace      = 8,
    dnnResourceMultipleSrc    = 16,
    dnnResourceMultipleDst    = 24,
    dnnResourceNumber         = 32
} dnnResourceType_t;

typedef enum {
    dnnBorderZeros          = 0x0,
    dnnBorderZerosAsymm     = 0x100,
    dnnBorderExtrapolation  = 0x3
} dnnBorder_t;

typedef enum {
    dnnUseInputMeanVariance = 0x1U,
    dnnUseScaleShift        = 0x2U
} dnnBatchNormalizationFlag_t;

#endif
