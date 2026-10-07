/*******************************************************************************
* Copyright 2004-2022 Intel Corporation.
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
*   Content:
*           Intel(R) oneAPI Math Kernel Library (oneMKL) DSS C header file
*
*           Contains main datatypes, prototypes and constants definition
*
********************************************************************************
*/
#if !defined( __MKL_DSS_H )

#define __MKL_DSS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "mkl_types.h"
/*
** Basic data types
*/

typedef void *             _MKL_DSS_HANDLE_t;
typedef char               _CHARACTER_t;
typedef char               _CHARACTER_STR_t;
/*
** _INTEGER_t will be deprecated
*/
#define _INTEGER_t MKL_INT

#if !defined( _WIN64 )
typedef long               _LONG_t;
#else
typedef long long          _LONG_t;
#endif
typedef float              _REAL_t;
typedef double             _DOUBLE_PRECISION_t;
#define _DoubleComplexType struct { double r, i; }
typedef _DoubleComplexType _DOUBLE_COMPLEX_t;

/*
** MKL_DSS_DEFAULTS
*/

#define MKL_DSS_DEFAULTS   0



/*
** Out-of-core level option definitions
*/

#define MKL_DSS_OOC_VARIABLE 1024
#define MKL_DSS_OOC_STRONG   2048

/*
** Refinement steps on / off
*/

#define MKL_DSS_REFINEMENT_OFF 4096
#define MKL_DSS_REFINEMENT_ON  8192

/*
** Solver step's substitution
*/

#define MKL_DSS_FORWARD_SOLVE		16384
#define MKL_DSS_DIAGONAL_SOLVE		32768
#define MKL_DSS_BACKWARD_SOLVE		49152
#define MKL_DSS_TRANSPOSE_SOLVE		262144
#define MKL_DSS_CONJUGATE_SOLVE		524288

/*
** Single precision
*/

#define MKL_DSS_SINGLE_PRECISION		65536

/*
** Zero-based indexing
*/

#define MKL_DSS_ZERO_BASED_INDEXING		131072

/*
** Message level option definitions
*/

#define MKL_DSS_MSG_LVL_SUCCESS	-2147483647
#define MKL_DSS_MSG_LVL_DEBUG  	-2147483646
#define MKL_DSS_MSG_LVL_INFO   	-2147483645
#define MKL_DSS_MSG_LVL_WARNING	-2147483644
#define MKL_DSS_MSG_LVL_ERROR  	-2147483643
#define MKL_DSS_MSG_LVL_FATAL  	-2147483642

/*
** Termination level option definitions
*/

#define MKL_DSS_TERM_LVL_SUCCESS	1073741832
#define MKL_DSS_TERM_LVL_DEBUG  	1073741840
#define MKL_DSS_TERM_LVL_INFO   	1073741848
#define MKL_DSS_TERM_LVL_WARNING	1073741856
#define MKL_DSS_TERM_LVL_ERROR  	1073741864
#define MKL_DSS_TERM_LVL_FATAL  	1073741872

/*
** Structure option definitions
*/

#define MKL_DSS_SYMMETRIC                     536870976
#define MKL_DSS_SYMMETRIC_STRUCTURE           536871040
#define MKL_DSS_NON_SYMMETRIC                 536871104
#define MKL_DSS_SYMMETRIC_COMPLEX             536871168
#define MKL_DSS_SYMMETRIC_STRUCTURE_COMPLEX   536871232
#define MKL_DSS_NON_SYMMETRIC_COMPLEX         536871296

/*
** Reordering option definitions
*/

#define MKL_DSS_AUTO_ORDER   	        268435520
#define MKL_DSS_MY_ORDER     	        268435584
#define MKL_DSS_OPTION1_ORDER	        268435648
#define MKL_DSS_GET_ORDER     	        268435712
#define MKL_DSS_METIS_ORDER	        268435776
#define MKL_DSS_METIS_OPENMP_ORDER	268435840

/*
** Factorization option definitions
*/

#define MKL_DSS_POSITIVE_DEFINITE          	134217792
#define MKL_DSS_INDEFINITE                 	134217856
#define MKL_DSS_HERMITIAN_POSITIVE_DEFINITE	134217920
#define MKL_DSS_HERMITIAN_INDEFINITE       	134217984

/*
** Return status values
*/

#define MKL_DSS_SUCCESS        	0
#define MKL_DSS_ZERO_PIVOT     	-1
#define MKL_DSS_OUT_OF_MEMORY  	-2
#define MKL_DSS_FAILURE        	-3
#define MKL_DSS_ROW_ERR        	-4
#define MKL_DSS_COL_ERR        	-5
#define MKL_DSS_TOO_FEW_VALUES 	-6
#define MKL_DSS_TOO_MANY_VALUES	-7
#define MKL_DSS_NOT_SQUARE     	-8
#define MKL_DSS_STATE_ERR      	-9
#define MKL_DSS_INVALID_OPTION 	-10
#define MKL_DSS_OPTION_CONFLICT	-11
#define MKL_DSS_MSG_LVL_ERR    	-12
#define MKL_DSS_TERM_LVL_ERR   	-13
#define MKL_DSS_STRUCTURE_ERR  	-14
#define MKL_DSS_REORDER_ERR    	-15
#define MKL_DSS_VALUES_ERR     	-16
#define MKL_DSS_STATISTICS_INVALID_MATRIX    	-17
#define MKL_DSS_STATISTICS_INVALID_STATE    	-18
#define MKL_DSS_STATISTICS_INVALID_STRING    	-19
#define MKL_DSS_REORDER1_ERR    -20
#define MKL_DSS_PREORDER_ERR    -21
#define MKL_DSS_DIAG_ERR        -22
#define MKL_DSS_I32BIT_ERR      -23
#define MKL_DSS_OOC_MEM_ERR     -24
#define MKL_DSS_OOC_OC_ERR      -25
#define MKL_DSS_OOC_RW_ERR      -26

/*
** Function prototypes for DSS routines
*/

extern MKL_INT dss_create_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *);
extern MKL_INT dss_define_structure_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *, MKL_INT const *, MKL_INT const *,
	 MKL_INT const *, MKL_INT const *, MKL_INT const *);
extern MKL_INT dss_reorder_(_MKL_DSS_HANDLE_t *, MKL_INT const *,
	 MKL_INT const *);
extern MKL_INT dss_factor_real_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *, void const *);
extern MKL_INT dss_factor_complex_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *, void const *);
extern MKL_INT dss_solve_real_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *, void const *, MKL_INT const *,
	 void *);
extern MKL_INT dss_solve_complex_(_MKL_DSS_HANDLE_t *,
	 MKL_INT const *, void const *, MKL_INT const *,
	 void *);
extern MKL_INT dss_statistics_( _MKL_DSS_HANDLE_t *, MKL_INT const *,
	 _CHARACTER_STR_t const *, _DOUBLE_PRECISION_t *);
extern MKL_INT dss_delete_(_MKL_DSS_HANDLE_t const *,
	 MKL_INT const *);

/*
** In order to promote portability and to avoid having most users deal with these issues, the C header
** file mkl_dss.h provides a set of macros and type definitions that are intended to hide the
** inter-language calling conventions and provide an interface to the DSS that appears natural for
** C/C++.
*/

#define dss_create(handle, opt) dss_create_(&(handle), &(opt))
#define dss_define_structure(handle, opt, rowIndex, nRows, rCols,\
	 columns, nNonZeros) dss_define_structure_(&(handle), &(opt),\
	 (rowIndex), &(nRows), &(rCols), (columns), &(nNonZeros))
#define dss_reorder(handle, opt, perm) dss_reorder_(&(handle), &(opt),\
	 (perm))
#define dss_factor_real(handle, opt,\
	 rValues) dss_factor_real_(&(handle), &(opt), (rValues))
#define dss_factor_complex(handle, opt,\
	 rValues) dss_factor_complex_(&(handle), &(opt), (rValues))
#define dss_solve_real(handle, opt, rRhsValues, nRhs,\
	 rSolValues) dss_solve_real_(&(handle), &(opt), (rRhsValues), &(nRhs),\
	 (rSolValues))
#define dss_solve_complex(handle, opt, rRhsValues, nRhs,\
	 rSolValues) dss_solve_complex_(&(handle), &(opt), (rRhsValues),\
	 &(nRhs), (rSolValues))
#define dss_statistics(handle, opt, stat, ret) \
	 dss_statistics_(&(handle), &(opt), (stat), (ret))
#define dss_delete(handle, opt) dss_delete_(&(handle), &(opt))


#define DSS_CREATE(handle, opt) dss_create_(&(handle), &(opt))
#define DSS_DEFINE_STRUCTURE(handle, opt, rowIndex, nRows, rCols,\
	 columns, nNonZeros) dss_define_structure_(&(handle), &(opt),\
	 (rowIndex), &(nRows), &(rCols), (columns), &(nNonZeros))
#define DSS_REORDER(handle, opt, perm) dss_reorder_(&(handle), &(opt),\
	 (perm))
#define DSS_FACTOR_REAL(handle, opt,\
	 rValues) dss_factor_real_(&(handle), &(opt), (rValues))
#define DSS_FACTOR_COMPLEX(handle, opt,\
	 rValues) dss_factor_complex_(&(handle), &(opt), (rValues))
#define DSS_SOLVE_REAL(handle, opt, rRhsValues, nRhs,\
	 rSolValues) dss_solve_real_(&(handle), &(opt), (rRhsValues), &(nRhs),\
	 (rSolValues))
#define DSS_SOLVE_COMPLEX(handle, opt, rRhsValues, nRhs,\
	 rSolValues) dss_solve_complex_(&(handle), &(opt), (rRhsValues),\
	 &(nRhs), (rSolValues))
#define DSS_STATISTICS(handle, opt, stat, ret) \
	 dss_statistics_(&(handle), &(opt), (stat), (ret))
#define DSS_DELETE(handle, opt) dss_delete_(&(handle), &(opt))


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
