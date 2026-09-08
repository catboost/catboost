/*******************************************************************************
* Copyright 2017-2022 Intel Corporation.
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
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) interfaces for Compact format
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


MKL_INT64 mkl_sget_size_compact_64( MKL_INT64 ld, MKL_INT64 sd,
                                    MKL_COMPACT_PACK format, MKL_INT64 nm );

MKL_INT64 mkl_dget_size_compact_64( MKL_INT64 ld, MKL_INT64 sd,
                                    MKL_COMPACT_PACK format, MKL_INT64 nm );

MKL_INT64 mkl_cget_size_compact_64( MKL_INT64 ld, MKL_INT64 sd,
                                    MKL_COMPACT_PACK format, MKL_INT64 nm );

MKL_INT64 mkl_zget_size_compact_64( MKL_INT64 ld, MKL_INT64 sd,
                                    MKL_COMPACT_PACK format, MKL_INT64 nm );

#ifdef __cplusplus
}
#endif    /* __cplusplus */

#endif /* _MKL_COMPACT_H */
