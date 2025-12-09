/* file: mkl_df_defines.h */
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
//  User-level macro definitions
//--
*/



#ifndef __MKL_DF_DEFINES_H__
#define __MKL_DF_DEFINES_H__



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#define DF_STATUS_OK                       0

/*
// Common errors (-1..-999)
*/
#define DF_ERROR_CPU_NOT_SUPPORTED         -1


/*
//++
// DATA FITTING ERROR/WARNING CODES
//--
*/
/*
// Errors (-1000..-1999)
*/
#define DF_ERROR_NULL_TASK_DESCRIPTOR     -1000
#define DF_ERROR_MEM_FAILURE              -1001
#define DF_ERROR_METHOD_NOT_SUPPORTED     -1002
#define DF_ERROR_COMP_TYPE_NOT_SUPPORTED  -1003
#define DF_ERROR_NULL_PTR                 -1037

#define DF_ERROR_BAD_NX                   -1004
#define DF_ERROR_BAD_X                    -1005
#define DF_ERROR_BAD_X_HINT               -1006
#define DF_ERROR_BAD_NY                   -1007
#define DF_ERROR_BAD_Y                    -1008
#define DF_ERROR_BAD_Y_HINT               -1009
#define DF_ERROR_BAD_SPLINE_ORDER         -1010
#define DF_ERROR_BAD_SPLINE_TYPE          -1011
#define DF_ERROR_BAD_IC_TYPE              -1012
#define DF_ERROR_BAD_IC                   -1013
#define DF_ERROR_BAD_BC_TYPE              -1014
#define DF_ERROR_BAD_BC                   -1015
#define DF_ERROR_BAD_PP_COEFF             -1016
#define DF_ERROR_BAD_PP_COEFF_HINT        -1017
#define DF_ERROR_BAD_PERIODIC_VAL         -1018
#define DF_ERROR_BAD_DATA_ATTR            -1019
#define DF_ERROR_BAD_DATA_IDX             -1020


#define DF_ERROR_BAD_NSITE                -1021
#define DF_ERROR_BAD_SITE                 -1022
#define DF_ERROR_BAD_SITE_HINT            -1023
#define DF_ERROR_BAD_NDORDER              -1024
#define DF_ERROR_BAD_DORDER               -1025
#define DF_ERROR_BAD_DATA_HINT            -1026
#define DF_ERROR_BAD_INTERP               -1027
#define DF_ERROR_BAD_INTERP_HINT          -1028
#define DF_ERROR_BAD_CELL_IDX             -1029
#define DF_ERROR_BAD_NLIM                 -1030
#define DF_ERROR_BAD_LLIM                 -1031
#define DF_ERROR_BAD_RLIM                 -1032
#define DF_ERROR_BAD_INTEGR               -1033
#define DF_ERROR_BAD_INTEGR_HINT          -1034
#define DF_ERROR_BAD_LOOKUP_INTERP_SITE   -1035
#define DF_ERROR_BAD_CHECK_FLAG           -1036



/*
// Internal errors caused by internal routines of the functions
*/
#define VSL_DF_ERROR_INTERNAL_C1          -1500
#define VSL_DF_ERROR_INTERNAL_C2          -1501

/*
// User-defined callback status
*/
#define DF_STATUS_EXACT_RESULT             1000

/*
//++
// MACROS USED IN DATAFITTING EDITORS AND COMPUTE ROUTINES
//--
*/

/*
// Attributes of parameters that can be modified in Data Fitting task
*/
#define DF_X                                    1
#define DF_Y                                    2
#define DF_IC                                   3
#define DF_BC                                   4
#define DF_PP_SCOEFF                            5

#define DF_NX                                  14
#define DF_XHINT                               15
#define DF_NY                                  16
#define DF_YHINT                               17
#define DF_SPLINE_ORDER                        18
#define DF_SPLINE_TYPE                         19
#define DF_IC_TYPE                             20
#define DF_BC_TYPE                             21
#define DF_PP_COEFF_HINT                       22
#define DF_CHECK_FLAG                          23

/*
//++
// SPLINE ORDERS SUPPORTED IN DATA FITTING ROUTINES
//--
*/
#define DF_PP_STD                        0
#define DF_PP_LINEAR                     2
#define DF_PP_QUADRATIC                  3
#define DF_PP_CUBIC                      4

/*
//++
// SPLINE TYPES SUPPORTED IN DATA FITTING ROUTINES
//--
*/

#define DF_PP_DEFAULT                       0
#define DF_PP_SUBBOTIN                      1
#define DF_PP_NATURAL                       2
#define DF_PP_HERMITE                       3
#define DF_PP_BESSEL                        4
#define DF_PP_AKIMA                         5
#define DF_LOOKUP_INTERPOLANT               6
#define DF_CR_STEPWISE_CONST_INTERPOLANT    7
#define DF_CL_STEPWISE_CONST_INTERPOLANT    8
#define DF_PP_HYMAN                         9

/*
//++
// TYPES OF BOUNDARY CONDITIONS USED IN SPLINE CONSTRUCTION
//--
*/
#define DF_NO_BC                           0
#define DF_BC_NOT_A_KNOT                   1
#define DF_BC_FREE_END                     2
#define DF_BC_1ST_LEFT_DER                 4
#define DF_BC_1ST_RIGHT_DER                8
#define DF_BC_2ND_LEFT_DER                16
#define DF_BC_2ND_RIGHT_DER               32
#define DF_BC_PERIODIC                    64
#define DF_BC_Q_VAL                      128

/*
//++
// TYPES OF INTERNAL CONDITIONS USED IN SPLINE CONSTRUCTION
//--
*/
#define DF_NO_IC                           0
#define DF_IC_1ST_DER                      1
#define DF_IC_2ND_DER                      2
#define DF_IC_Q_KNOT                       8



/*
//++
// TYPES OF SUPPORTED HINTS
//--
*/
#define DF_NO_HINT                    0x00000000
#define DF_NON_UNIFORM_PARTITION      0x00000001
#define DF_QUASI_UNIFORM_PARTITION    0x00000002
#define DF_UNIFORM_PARTITION          0x00000004

#define DF_MATRIX_STORAGE_ROWS        0x00000010
#define DF_MATRIX_STORAGE_COLS        0x00000020

#define DF_SORTED_DATA                0x00000040
#define DF_1ST_COORDINATE             0x00000080

#define DF_MATRIX_STORAGE_FUNCS_SITES_DERS    DF_MATRIX_STORAGE_ROWS
#define DF_MATRIX_STORAGE_FUNCS_DERS_SITES    DF_MATRIX_STORAGE_COLS
#define DF_MATRIX_STORAGE_SITES_FUNCS_DERS    0x00000100
#define DF_MATRIX_STORAGE_SITES_DERS_FUNCS    0x00000200

/*
//++
// TYPES OF APRIORI INFORMATION
// ABOUT DATA STRUCTURE
//--
*/
#define DF_NO_APRIORI_INFO             0x00000000
#define DF_APRIORI_MOST_LIKELY_CELL    0x00000001



/*
//++
// ESTIMATES TO BE COMPUTED WITH DATA FITTING COMPUTE ROUTINE
//--
*/
#define DF_INTERP           0x00000001
#define DF_CELL             0x00000002
#define DF_INTERP_USER_CELL 0x00000004


/*
//++
// METHODS TO BE USED FOR EVALUATION OF THE SPLINE RELATED ESTIMATES
//--
*/
#define DF_METHOD_STD                             0
#define DF_METHOD_PP                              1

/*
//++
// POSSIBLE VALUES FOR DF_CHECK_FLAG
//--
*/
#define DF_ENABLE_CHECK_FLAG     0x00000000
#define DF_DISABLE_CHECK_FLAG    0x00000001


/*
//++
// SPLINE FORMATS SUPPORTED IN SPLINE CONSTRUCTION ROUTINE
//--
*/

#define DF_PP_SPLINE                              0

/*
//++
// VALUES OF FLAG INDICATING WHICH, LEFT OR RIGHT, INTEGRATION LIMITS
// ARE PASSED BY INTEGRATION ROUTINE INTO SEARCH CALLBACK
//--
*/

#define DF_INTEGR_SEARCH_CB_LLIM_FLAG                    0
#define DF_INTEGR_SEARCH_CB_RLIM_FLAG                    1

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_DF_DEFINES_H__ */
