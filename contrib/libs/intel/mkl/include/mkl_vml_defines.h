/* file: mkl_vml_defines.h */
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
//  VML_HA | VML_ERRMODE_DEFAULT | VML_FTZDAZ_CURRENT,
//  i.e. VML high accuracy functions are called,
//  current floating-point precision and the rounding mode is used,
//  current MXCSR settings are left unchanged.
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

#define VML_LA_64 0x0000000000000001
#define VML_HA_64 0x0000000000000002
#define VML_EP_64 0x0000000000000003

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

#define VML_DEFAULT_PRECISION_64 0x0000000000000000
#define VML_FLOAT_CONSISTENT_64  0x0000000000000010
#define VML_DOUBLE_CONSISTENT_64 0x0000000000000020
#define VML_RESTORE_64           0x0000000000000030

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

#define VML_ERRMODE_IGNORE_64   0x0000000000000100
#define VML_ERRMODE_ERRNO_64    0x0000000000000200
#define VML_ERRMODE_STDERR_64   0x0000000000000400
#define VML_ERRMODE_EXCEPT_64   0x0000000000000800
#define VML_ERRMODE_CALLBACK_64 0x0000000000001000
#define VML_ERRMODE_NOERR_64    0x0000000000002000
#define VML_ERRMODE_DEFAULT_64  \
VML_ERRMODE_ERRNO_64 | VML_ERRMODE_CALLBACK_64 | VML_ERRMODE_EXCEPT_64

/*
//  OpenMP(R) number of threads mode macros
//  VML_NUM_THREADS_OMP_AUTO   - Maximum number of threads is determined by
//                               environmental variable OMP_NUM_THREADS or
//                               omp_set_num_threads() function
//  VML_NUM_THREADS_OMP_FIXED  - Number of threads is determined by
//                               environmental variable OMP_NUM_THREADS
//                               omp_set_num_threads() functions
*/
#define VML_NUM_THREADS_OMP_AUTO   0x00000000
#define VML_NUM_THREADS_OMP_FIXED  0x00010000

#define VML_NUM_THREADS_OMP_AUTO_64   0x0000000000000000
#define VML_NUM_THREADS_OMP_FIXED_64  0x0000000000010000

/*
//  TBB partitioner control macros
//  VML_TBB_PARTITIONER_AUTO   - Automatic TBB partitioner tbb::auto_partitioner().
//                               Performs sufficient splitting to balance load.
//  VML_TBB_PARTITIONER_STATIC - Static TBB partitioner tbb::static_partitioner().
//                               Distributes range iterations among worker threads as uniformly as possible,
//                               without a possibility for further load balancing.
//  VML_TBB_PARTITIONER_SIMPLE - Simple TBB partitioner tbb::simple_partitioner().
//                               Recursively splits a range until it is no longer divisible.
//
*/
#define VML_TBB_PARTITIONER_AUTO   0x00000000
#define VML_TBB_PARTITIONER_STATIC 0x00010000
#define VML_TBB_PARTITIONER_SIMPLE 0x00020000

#define VML_TBB_PARTITIONER_AUTO_64   0x0000000000000000
#define VML_TBB_PARTITIONER_STATIC_64 0x0000000000010000
#define VML_TBB_PARTITIONER_SIMPLE_64 0x0000000000020000

/*
//  FTZ & DAZ mode macros
//  VML_FTZDAZ_ON      - FTZ & DAZ MXCSR mode enabled
//                       for faster (sub)denormal values processing
//  VML_FTZDAZ_OFF     - FTZ & DAZ MXCSR mode disabled
//                       for accurate (sub)denormal values processing
//  VML_FTZDAZ_CURRENT - FTZ & DAZ MXCSR mode is kept as currently on CPU
//                       (no slow MXCSR access during entry/exit)
//                       for increased performance
*/
#define VML_FTZDAZ_ON      0x00280000
#define VML_FTZDAZ_OFF     0x00140000
#define VML_FTZDAZ_CURRENT 0x00000000

#define VML_FTZDAZ_ON_64      0x0000000000280000
#define VML_FTZDAZ_OFF_64     0x0000000000140000
#define VML_FTZDAZ_CURRENT_64 0x0000000000000000

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

#define VML_TRAP_INVALID_64    0x0000000001000000
#define VML_TRAP_DIVBYZERO_64  0x0000000002000000
#define VML_TRAP_OVERFLOW_64   0x0000000004000000
#define VML_TRAP_UNDERFLOW_64  0x0000000008000000

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
//  VML_TBB_PARTITIONER_MASK    - extract TBB partitioner control bits
//  VML_FTZDAZ_MASK             - extract FTZ & DAZ bits
//  VML_TRAP_EXCEPTIONS_MASK    - extract exception trap bits
*/
#define VML_ACCURACY_MASK           0x0000000F
#define VML_FPUMODE_MASK            0x000000F0
#define VML_ERRMODE_MASK            0x0000FF00
#define VML_ERRMODE_STDHANDLER_MASK 0x00002F00
#define VML_ERRMODE_CALLBACK_MASK   0x00001000
#define VML_NUM_THREADS_OMP_MASK    0x00030000
#define VML_TBB_PARTITIONER_MASK    0x00030000
#define VML_FTZDAZ_MASK             0x003C0000
#define VML_TRAP_EXCEPTIONS_MASK    0x0F000000

#define VML_ACCURACY_MASK_64           0x000000000000000F
#define VML_FPUMODE_MASK_64            0x00000000000000F0
#define VML_ERRMODE_MASK_64            0x000000000000FF00
#define VML_ERRMODE_STDHANDLER_MASK_64 0x0000000000002F00
#define VML_ERRMODE_CALLBACK_MASK_64   0x0000000000001000
#define VML_NUM_THREADS_OMP_MASK_64    0x0000000000030000
#define VML_TBB_PARTITIONER_MASK_64    0x0000000000030000
#define VML_FTZDAZ_MASK_64             0x00000000003C0000
#define VML_TRAP_EXCEPTIONS_MASK_64    0x000000000F000000

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

#define VML_STATUS_OK_64                 0
#define VML_STATUS_BADSIZE_64           -1
#define VML_STATUS_BADMEM_64            -2
#define VML_STATUS_ERRDOM_64             1
#define VML_STATUS_SING_64               2
#define VML_STATUS_OVERFLOW_64           3
#define VML_STATUS_UNDERFLOW_64          4
#define VML_STATUS_ACCURACYWARNING_64    1000

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_DEFINES_H__ */
