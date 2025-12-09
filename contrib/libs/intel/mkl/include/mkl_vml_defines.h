/* file: mkl_vml_defines.h */
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
//  Macro definitions visible on user level.
//--
*/

#ifndef __MKL_VML_DEFINES_H__
#define __MKL_VML_DEFINES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
//++
//  MACRO DEFINITIONS
//  Macro definitions for VML mode and VML error status.
//
//  VML mode controls VML function accuracy, floating-point settings (rounding
//  mode and precision) and VML error handling options. Default VML mode is
//  VML_HA | VML_ERRMODE_DEFAULT, i.e. VML high accuracy functions are
//  called, and current floating-point precision and the rounding mode is used.
//
//  Error status macros are used for error classification.
//--
*/

/*
//  VML FUNCTION ACCURACY CONTROL
//  VML_HA - when VML_HA is set, high accuracy VML functions are called
//  VML_LA - when VML_LA is set, low accuracy VML functions are called
//  VML_EP - when VML_EP is set, enhanced performance VML functions are called
//
//  NOTE: VML_HA, VML_LA and VML_EP must not be used in combination
*/
#define VML_LA 0x00000001
#define VML_HA 0x00000002
#define VML_EP 0x00000003


/*
//  SETTING OPTIMAL FLOATING-POINT PRECISION AND ROUNDING MODE
//  Definitions below are to set optimal floating-point control word
//  (precision and rounding mode).
//
//  For their correct work, VML functions change floating-point precision and
//  rounding mode (if necessary). Since control word changing is typically
//  expensive operation, it is recommended to set precision and rounding mode
//  to optimal values before VML function calls.
//
//  VML_FLOAT_CONSISTENT  - use this value if the calls are typically to single
//                          precision VML functions
//  VML_DOUBLE_CONSISTENT - use this value if the calls are typically to double
//                          precision VML functions
//  VML_RESTORE           - restore original floating-point precision and
//                          rounding mode
//  VML_DEFAULT_PRECISION - use default (current) floating-point precision and
//                          rounding mode
//  NOTE: VML_FLOAT_CONSISTENT, VML_DOUBLE_CONSISTENT, VML_RESTORE and
//        VML_DEFAULT_PRECISION must not be used in combination
*/
#define VML_DEFAULT_PRECISION 0x00000000
#define VML_FLOAT_CONSISTENT  0x00000010
#define VML_DOUBLE_CONSISTENT 0x00000020
#define VML_RESTORE           0x00000030

/*
//  VML ERROR HANDLING CONTROL
//  Macros below are used to control VML error handler.
//
//  VML_ERRMODE_IGNORE   - ignore errors
//  VML_ERRMODE_ERRNO    - errno variable is set on error
//  VML_ERRMODE_STDERR   - error description text is written to stderr on error
//  VML_ERRMODE_EXCEPT   - exception is raised on error
//  VML_ERRMODE_CALLBACK - user's error handler function is called on error
//  VML_ERRMODE_NOERR    - ignore errors and do not update status
//  VML_ERRMODE_DEFAULT  - errno variable is set, exceptions are raised and
//                         user's error handler is called on error
//  NOTE: VML_ERRMODE_IGNORE must not be used in combination with
//        VML_ERRMODE_ERRNO, VML_ERRMODE_STDERR, VML_ERRMODE_EXCEPT,
//        VML_ERRMODE_CALLBACK and VML_ERRMODE_DEFAULT.
//  NOTE: VML_ERRMODE_NOERR must not be used in combination with any
//        other VML_ERRMODE setting.
*/
#define VML_ERRMODE_IGNORE   0x00000100
#define VML_ERRMODE_ERRNO    0x00000200
#define VML_ERRMODE_STDERR   0x00000400
#define VML_ERRMODE_EXCEPT   0x00000800
#define VML_ERRMODE_CALLBACK 0x00001000
#define VML_ERRMODE_NOERR    0x00002000
#define VML_ERRMODE_DEFAULT  \
VML_ERRMODE_ERRNO | VML_ERRMODE_CALLBACK | VML_ERRMODE_EXCEPT

/*
//  OpenMP(R) number of threads mode macros
//  VML_NUM_THREADS_OMP_AUTO   - Maximum number of threads is determined by
//                               environmental variable OPM_NUM_THREADS or
//                               omp_set_num_threads() function
//  VML_NUM_THREADS_OMP_FIXED  - Number of threads is determined by
//                               environmental variable OPM_NUM_THREADS
//                               omp_set_num_threads() functions
*/
#define VML_NUM_THREADS_OMP_AUTO   0x00000000
#define VML_NUM_THREADS_OMP_FIXED  0x00010000

/*
//  FTZ & DAZ mode macros
//  VML_FTZDAZ_ON   - FTZ & DAZ MXCSR mode enabled
//                    for faster (sub)denormal values processing
//  VML_FTZDAZ_OFF  - FTZ & DAZ MXCSR mode disabled
//                    for accurate (sub)denormal values processing
*/
#define VML_FTZDAZ_ON   0x00280000
#define VML_FTZDAZ_OFF  0x00140000

/*
//  Exception trap macros
//  VML_TRAP_INVALID             Trap invalid arithmetic operand exception
//  VML_TRAP_DIVBYZERO           Trap divide-by-zero exception
//  VML_TRAP_OVERFLOW            Trap numeric overflow exception
//  VML_TRAP_UNDERFLOW           Trap numeric underflow exception
*/

#define VML_TRAP_INVALID    0x01000000
#define VML_TRAP_DIVBYZERO  0x02000000
#define VML_TRAP_OVERFLOW   0x04000000
#define VML_TRAP_UNDERFLOW  0x08000000

/*
//  ACCURACY, FLOATING-POINT CONTROL, FTZDAZ AND ERROR HANDLING MASKS
//  Accuracy, floating-point and error handling control are packed in
//  the VML mode variable. Macros below are useful to extract accuracy and/or
//  floating-point control and/or error handling control settings.
//
//  VML_ACCURACY_MASK           - extract accuracy bits
//  VML_FPUMODE_MASK            - extract floating-point control bits
//  VML_ERRMODE_MASK            - extract error handling control bits
//                                (including error callback bits)
//  VML_ERRMODE_STDHANDLER_MASK - extract error handling control bits
//                                (not including error callback bits)
//  VML_ERRMODE_CALLBACK_MASK   - extract error callback bits
//  VML_NUM_THREADS_OMP_MASK    - extract OpenMP(R) number of threads mode bits
//  VML_FTZDAZ_MASK             - extract FTZ & DAZ bits
*/
#define VML_ACCURACY_MASK           0x0000000F
#define VML_FPUMODE_MASK            0x000000F0
#define VML_ERRMODE_MASK            0x0000FF00
#define VML_ERRMODE_STDHANDLER_MASK 0x00002F00
#define VML_ERRMODE_CALLBACK_MASK   0x00001000
#define VML_NUM_THREADS_OMP_MASK    0x00030000
#define VML_FTZDAZ_MASK             0x003C0000
#define VML_TRAP_EXCEPTIONS_MASK    0x0F000000

/*
//  ERROR STATUS MACROS
//  VML_STATUS_OK        - no errors
//  VML_STATUS_BADSIZE   - array dimension is not positive
//  VML_STATUS_BADMEM    - invalid pointer passed
//  VML_STATUS_ERRDOM    - at least one of arguments is out of function domain
//  VML_STATUS_SING      - at least one of arguments caused singularity
//  VML_STATUS_OVERFLOW  - at least one of arguments caused overflow
//  VML_STATUS_UNDERFLOW - at least one of arguments caused underflow
//  VML_STATUS_ACCURACYWARNING - function doesn't support set accuracy mode,
//                               lower accuracy mode was used instead
*/
#define VML_STATUS_OK                    0
#define VML_STATUS_BADSIZE              -1
#define VML_STATUS_BADMEM               -2
#define VML_STATUS_ERRDOM                1
#define VML_STATUS_SING                  2
#define VML_STATUS_OVERFLOW              3
#define VML_STATUS_UNDERFLOW             4
#define VML_STATUS_ACCURACYWARNING       1000

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_DEFINES_H__ */
