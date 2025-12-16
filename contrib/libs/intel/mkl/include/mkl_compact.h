/*******************************************************************************
* Copyright 2017 Intel Corporation All Rights Reserved.
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
!  Content:
!      Intel(R) Math Kernel Library (Intel(R) MKL) interfaces for Compact format
!******************************************************************************/
#ifndef _MKL_COMPACT_H
#define _MKL_COMPACT_H

#include "mkl_types.h"

#ifndef mkl_compact_complex_float
#define mkl_compact_complex_float MKL_Complex8
#endif

#ifndef mkl_compact_complex_double
#define mkl_compact_complex_double MKL_Complex16
#endif

#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */

MKL_COMPACT_PACK mkl_get_format_compact( void );

MKL_INT mkl_sget_size_compact( MKL_INT ld, MKL_INT sd,
                               MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_sgepack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                          const float * const *a, MKL_INT lda, float *ap, MKL_INT ldap,
                          MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_sgeunpack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                            float * const *a, MKL_INT lda, const float *ap, MKL_INT ldap,
                            MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_sgemm_compact( MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                        MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n,
                        MKL_INT k, float alpha, const float *ap,
                        MKL_INT ldap, const float *bp, MKL_INT ldbp,
                        float beta, float *cp, MKL_INT ldcp,
                        MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_strsm_compact( MKL_LAYOUT layout, MKL_SIDE side,
                        MKL_UPLO uplo, MKL_TRANSPOSE transa,
                        MKL_DIAG diag, MKL_INT m, MKL_INT n,
                        float alpha, const float *ap, MKL_INT ldap,
                        float *bp, MKL_INT ldbp,
                        MKL_COMPACT_PACK format, MKL_INT nm );

MKL_INT mkl_dget_size_compact( MKL_INT ld, MKL_INT sd,
                               MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_dgepack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                          const double * const *a, MKL_INT lda, double *ap, MKL_INT ldap,
                          MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_dgeunpack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                            double * const *a, MKL_INT lda, const double *ap, MKL_INT ldap,
                            MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_dgemm_compact( MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                        MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n,
                        MKL_INT k, double alpha, const double *a,
                        MKL_INT ldap, const double *b, MKL_INT ldbp,
                        double beta, double *c, MKL_INT ldcp,
                        MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_dtrsm_compact( MKL_LAYOUT layout, MKL_SIDE side,
                        MKL_UPLO uplo, MKL_TRANSPOSE transa,
                        MKL_DIAG diag, MKL_INT m, MKL_INT n,
                        double alpha, const double *a, MKL_INT ldap,
                        double *b, MKL_INT ldbp,
                        MKL_COMPACT_PACK format, MKL_INT nm );

MKL_INT mkl_cget_size_compact( MKL_INT ld, MKL_INT sd,
                               MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_cgepack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                          const mkl_compact_complex_float * const *a, MKL_INT lda, float *ap, MKL_INT ldap,
                          MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_cgeunpack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                            mkl_compact_complex_float * const *a, MKL_INT lda, const float *ap, MKL_INT ldap,
                            MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_cgemm_compact( MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                        MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n,
                        MKL_INT k, const mkl_compact_complex_float *alpha, const float *a,
                        MKL_INT ldap, const float *b, MKL_INT ldbp,
                        const mkl_compact_complex_float *beta, float *c, MKL_INT ldcp,
                        MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_ctrsm_compact( MKL_LAYOUT layout, MKL_SIDE side,
                        MKL_UPLO uplo, MKL_TRANSPOSE transa,
                        MKL_DIAG diag, MKL_INT m, MKL_INT n,
                        const mkl_compact_complex_float *alpha, const float *a, MKL_INT ldap,
                        float *b, MKL_INT ldbp,
                        MKL_COMPACT_PACK format, MKL_INT nm );

MKL_INT mkl_zget_size_compact( MKL_INT ld, MKL_INT sd,
                               MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_zgepack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                          const mkl_compact_complex_double * const *a, MKL_INT lda, double *ap, MKL_INT ldap,
                          MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_zgeunpack_compact( MKL_LAYOUT layout, MKL_INT rows, MKL_INT columns,
                            mkl_compact_complex_double * const *a, MKL_INT lda, const double *ap, MKL_INT ldap,
                            MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_zgemm_compact( MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                        MKL_TRANSPOSE transb, MKL_INT m, MKL_INT n,
                        MKL_INT k, const mkl_compact_complex_double *alpha, const double *a,
                        MKL_INT ldap, const double *b, MKL_INT ldbp,
                        const mkl_compact_complex_double *beta, double *c, MKL_INT ldcp,
                        MKL_COMPACT_PACK format, MKL_INT nm );
void mkl_ztrsm_compact( MKL_LAYOUT layout, MKL_SIDE side,
                        MKL_UPLO uplo, MKL_TRANSPOSE transa,
                        MKL_DIAG diag, MKL_INT m, MKL_INT n,
                        const mkl_compact_complex_double *alpha, const double *a, MKL_INT ldap,
                        double *b, MKL_INT ldbp,
                        MKL_COMPACT_PACK format, MKL_INT nm );

/* LAPACK compact routines */

void mkl_cgetrinp_compact( MKL_LAYOUT layout, MKL_INT n, float* ap,
                           MKL_INT ldap, float* work, MKL_INT lwork,
                           MKL_INT* info, MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_dgetrinp_compact( MKL_LAYOUT layout, MKL_INT n, double* ap, MKL_INT ldap,
                           double* work, MKL_INT lwork, MKL_INT* info,
                           MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_sgetrinp_compact( MKL_LAYOUT layout, MKL_INT n, float* ap, MKL_INT ldap,
                           float* work, MKL_INT lwork, MKL_INT* info,
                           MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_zgetrinp_compact( MKL_LAYOUT layout, MKL_INT n, double* ap,
                           MKL_INT ldap, double* work, MKL_INT lwork,
                           MKL_INT* info, MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_cgetrfnp_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
                           float* ap, MKL_INT ldap, MKL_INT* info,
                           MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_dgetrfnp_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double* ap,
                           MKL_INT ldap, MKL_INT* info, MKL_COMPACT_PACK format,
                           MKL_INT nm );

void mkl_sgetrfnp_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float* ap,
                           MKL_INT ldap, MKL_INT* info, MKL_COMPACT_PACK format,
                           MKL_INT nm );

void mkl_zgetrfnp_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
                           double* ap, MKL_INT ldap, MKL_INT* info,
                           MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_cpotrf_compact( MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                         float* ap, MKL_INT ldap, MKL_INT* info,
                         MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_dpotrf_compact( MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double* ap,
                         MKL_INT ldap, MKL_INT* info, MKL_COMPACT_PACK format,
                         MKL_INT nm );

void mkl_spotrf_compact( MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float* ap,
                         MKL_INT ldap, MKL_INT* info, MKL_COMPACT_PACK format,
                         MKL_INT nm );

void mkl_zpotrf_compact( MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                         double* ap, MKL_INT ldap, MKL_INT* info,
                         MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_cgeqrf_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float* ap,
                         MKL_INT ldap, float* taup, float* work, MKL_INT lwork,
                         MKL_INT* info, MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_dgeqrf_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double* ap,
                         MKL_INT ldap, double* taup, double* work,
                         MKL_INT lwork, MKL_INT* info, MKL_COMPACT_PACK format,
                         MKL_INT nm );

void mkl_sgeqrf_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float* ap,
                         MKL_INT ldap, float* taup, float* work, MKL_INT lwork,
                         MKL_INT* info, MKL_COMPACT_PACK format, MKL_INT nm );

void mkl_zgeqrf_compact( MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double* ap,
                         MKL_INT ldap, double* taup, double* work,
                         MKL_INT lwork, MKL_INT* info, MKL_COMPACT_PACK format,
                         MKL_INT nm );

#ifdef __cplusplus
}
#endif    /* __cplusplus */

#endif /* _MKL_COMPACT_H */
