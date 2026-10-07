/*******************************************************************************
* Copyright 2014-2022 Intel Corporation.
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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) C functions that can be inlined
!******************************************************************************/
#include "mkl_types.h"
#include "immintrin.h"

#undef mkl_dc_gemm
#undef mkl_dc_syrk
#undef mkl_dc_trsm

#if defined(MKL_DOUBLE)
#define mkl_dc_gemm mkl_dc_dgemm
#define mkl_dc_syrk mkl_dc_dsyrk
#define mkl_dc_trsm mkl_dc_dtrsm
#define mkl_dc_axpy mkl_dc_daxpy
#define mkl_dc_dot mkl_dc_ddot
#define MKL_DC_DOT_CONVERT mkl_dc_ddot_convert

#elif defined(MKL_SINGLE)
#define mkl_dc_gemm mkl_dc_sgemm
#define mkl_dc_syrk mkl_dc_ssyrk
#define mkl_dc_trsm mkl_dc_strsm
#define mkl_dc_axpy mkl_dc_saxpy
#define mkl_dc_dot mkl_dc_sdot
#define MKL_DC_DOT_CONVERT mkl_dc_sdot_convert

#elif defined(MKL_COMPLEX)
#define mkl_dc_gemm mkl_dc_cgemm
#define mkl_dc_syrk mkl_dc_csyrk
#define mkl_dc_trsm mkl_dc_ctrsm
#define mkl_dc_axpy mkl_dc_caxpy

#elif defined(MKL_COMPLEX16)
#define mkl_dc_gemm mkl_dc_zgemm
#define mkl_dc_syrk mkl_dc_zsyrk
#define mkl_dc_trsm mkl_dc_ztrsm
#define mkl_dc_axpy mkl_dc_zaxpy

#endif

#define mkl_dc_gemm_xx_mnk_pst_beta(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, \
		a_op, b_op, c_op, a_access, b_access) \
do { \
	MKL_INT i, j, l; \
	MKL_DC_PRAGMA_VECTOR \
	for (i = 0; i < m; i++) \
	for (j = 0; j < n; j++) { \
		mkl_dc_type c, temp; MKL_DC_SET_ZERO(temp); \
		for (l = 0; l < k; l++) { \
			mkl_dc_type a, b, temp1; \
			a = a_access(A, lda, i, l); a_op(a, a); \
			b = b_access(B, ldb, l, j); b_op(b, b); \
			MKL_DC_MUL(temp1, a, b); \
			MKL_DC_ADD(temp, temp, temp1); \
		} \
		MKL_DC_MUL(temp, alpha, temp); \
		c = C[i + j * ldc]; \
		c_op(c, beta); \
		MKL_DC_ADD(C[i + j * ldc], c, temp); \
	} \
} while (0)

#define mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, \
		a_op, b_op, a_access, b_access) \
do { \
	if (MKL_DC_IS_ZERO(beta)) \
		mkl_dc_gemm_xx_mnk_pst_beta(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, \
				a_op, b_op, MKL_DC_ZERO_C, a_access, b_access); \
	else \
		mkl_dc_gemm_xx_mnk_pst_beta(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, \
				a_op, b_op, MKL_DC_MUL_C, a_access, b_access); \
} while (0)

#ifdef __AVX2__
#ifdef MKL_DOUBLE

#define MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_c, ymm_tmp)  \
    ymm_c = _mm256_fmadd_pd(ymm_a, ymm_b, ymm_c); 

#define MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_c, xmm_tmp)  \
    xmm_c = _mm_fmadd_pd(xmm_a, xmm_b, xmm_c); 

#define MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_c, xmm_tmp)  \
    xmm_c = _mm_fmadd_sd(xmm_a, xmm_b, xmm_c); 

#define MKL_DC_XOR_YMM _mm256_xor_pd
#define MKL_DC_SETZERO_YMM _mm256_setzero_pd
#define MKL_DC_BCAST_YMM _mm256_broadcast_sd
#define MKL_DC_LOAD_YMM _mm256_loadu_pd
#define MKL_DC_STORE_YMM _mm256_storeu_pd
#define MKL_DC_ADD_YMM _mm256_add_pd
#define MKL_DC_MUL_YMM _mm256_mul_pd
#define MKL_DC_MASKLOAD_YMM _mm256_maskload_pd
#define MKL_DC_MASKSTORE_YMM _mm256_maskstore_pd

#define MKL_DC_HADD_YMM _mm256_hadd_pd
#define MKL_DC_PERM2F128_YMM _mm256_permute2f128_pd
#define MKL_DC_UNPACKHI_YMM _mm256_unpackhi_pd
#define MKL_DC_UNPACKLO_YMM _mm256_unpacklo_pd

#define MKL_DC_CAST_YMM_TO_XMM _mm256_castpd256_pd128
#define MKL_DC_CAST_XMM_TO_YMM _mm256_castpd128_pd256

#define MKL_DC_XOR_XMM _mm_xor_pd
#define MKL_DC_SETZERO_XMM _mm_setzero_pd
#define MKL_DC_LOAD_XMM _mm_loadu_pd
#define MKL_DC_STORE_XMM _mm_storeu_pd
#define MKL_DC_ADD_XMM _mm_add_pd
#define MKL_DC_MUL_XMM _mm_mul_pd

#define MKL_DC_LOAD_XMM_S _mm_load_sd
#define MKL_DC_STORE_XMM_S _mm_store_sd
#define MKL_DC_LOADDUP_XMM _mm_loaddup_pd
#define MKL_DC_ADD_XMM_S _mm_add_sd
#define MKL_DC_MUL_XMM_S _mm_mul_sd

#define MKL_DC_UNPACKHI_XMM _mm_unpackhi_pd
#define MKL_DC_UNPACKLO_XMM _mm_unpacklo_pd

#define MKL_DC_VEC_TRANSPOSE_YMM(x1, x2, x3, x4, tmp1, tmp2, tmp3, tmp4) \
    do { \
        tmp1 = _mm256_unpacklo_pd(x1, x2);             \
        tmp2 = _mm256_unpackhi_pd(x1, x2);             \
        tmp3 = _mm256_unpacklo_pd(x3, x4);             \
        tmp4 = _mm256_unpackhi_pd(x3, x4);             \
        x1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20); \
        x2 = _mm256_permute2f128_pd(tmp2, tmp4, 0x20); \
        x3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31); \
        x4 = _mm256_permute2f128_pd(tmp2, tmp4, 0x31); \
    } while (0)  

#define MKL_DC_VEC_TRANSPOSE_XMM(out1, out2, in1, in2) \
    do { \
        out1 = _mm_unpacklo_pd(in1, in2);       \
        out2 = _mm_unpackhi_pd(in1, in2);       \
    } while (0)  


typedef __m128d MKL_DC_XMMTYPE;
typedef __m256d MKL_DC_YMMTYPE;

#endif
#endif

#define MKL_DC_DGEMM_KERNELS(kernel_type, arch) \
    do { \
        if (AisN && BisN) { \
            mkl_dc_dgemm_nn_mnk_ ## kernel_type ## _ ## arch ## _pst(m, n, k, ALPHA, A, lda, B, ldb, BETA, C, ldc); \
        } else if (AisN && !BisN) { \
            mkl_dc_dgemm_nt_mnk_ ## kernel_type ## _ ## arch ## _pst(m, n, k, ALPHA, A, lda, B, ldb, BETA, C, ldc); \
        } else if (!AisN && BisN) { \
            mkl_dc_dgemm_tn_mnk_ ## kernel_type ## _ ## arch ## _pst(m, n, k, ALPHA, A, lda, B, ldb, BETA, C, ldc); \
        } else { \
            mkl_dc_dgemm_tt_mnk_ ## kernel_type ## _ ## arch ## _pst(m, n, k, ALPHA, A, lda, B, ldb, BETA, C, ldc); \
        } \
    } while (0) 


#include "mkl_direct_blas_kernels.h"

#define MKL_DC_ALPHA_ONE
#include "mkl_direct_blas_kernels.h"
#undef MKL_DC_ALPHA_ONE

#define MKL_DC_BETA_ZERO
#include "mkl_direct_blas_kernels.h"
#undef MKL_DC_BETA_ZERO

#define MKL_DC_BETA_ONE
#include "mkl_direct_blas_kernels.h"
#undef MKL_DC_BETA_ONE

#define MKL_DC_BETA_ONE
#define MKL_DC_ALPHA_ONE
#include "mkl_direct_blas_kernels.h"
#undef MKL_DC_BETA_ONE
#undef MKL_DC_ALPHA_ONE

#define MKL_DC_BETA_ZERO
#define MKL_DC_ALPHA_ONE
#include "mkl_direct_blas_kernels.h"
#undef MKL_DC_BETA_ZERO
#undef MKL_DC_ALPHA_ONE




static __inline void mkl_dc_gemm(const char * TRANSA, const char * TRANSB,
            const MKL_INT * M, const MKL_INT * N, const MKL_INT * K,
            const mkl_dc_type * ALPHA,
			const mkl_dc_type * A, const MKL_INT * LDA,
            const mkl_dc_type * B, const MKL_INT * LDB,
            const mkl_dc_type * BETA,
            mkl_dc_type * C, const MKL_INT * LDC)
{
	int AisN, BisN;
#ifndef MKL_REAL_DATA_TYPE
	int AisT, AisC;
	int BisT, BisC;
#endif
	mkl_dc_type alpha = *ALPHA, beta = *BETA;
	MKL_INT m = *M, n = *N, k = *K;
	MKL_INT lda = *LDA, ldb = *LDB, ldc = *LDC;

	if (m <= 0 || n <= 0 || ((MKL_DC_IS_ZERO(alpha) || k <= 0) && MKL_DC_IS_ONE(beta)))
		return;

	AisN = MKL_DC_MisN(*TRANSA);
	BisN = MKL_DC_MisN(*TRANSB);
#ifndef MKL_REAL_DATA_TYPE
	AisT = MKL_DC_MisT(*TRANSA);
	BisT = MKL_DC_MisT(*TRANSB);
	AisC = !(AisN || AisT);
	BisC = !(BisN || BisT);
#endif

	if (MKL_DC_IS_ZERO(alpha)) {
		MKL_INT i, j;
		if (MKL_DC_IS_ZERO(beta))
			for (j = 0; j < n; j++)
#pragma vector
				for (i = 0; i < m; i++)
					MKL_DC_SET_ZERO(C[i + ldc * j]);
		else
			for (j = 0; j < n; j++)
#pragma vector
				for (i = 0; i < m; i++)
					MKL_DC_MUL(C[i + ldc * j], beta, C[i + ldc * j]);
		return;
	}

#if defined(MKL_DOUBLE) && defined(__AVX2__)
    if (MKL_DC_IS_ONE(alpha) && MKL_DC_IS_ONE(beta)) {
        MKL_DC_DGEMM_KERNELS(a1b1, avx2);
    } else if (MKL_DC_IS_ONE(alpha) && MKL_DC_IS_ZERO(beta)) {
        MKL_DC_DGEMM_KERNELS(a1b0, avx2);
    } else if (MKL_DC_IS_ZERO(beta)) {
        MKL_DC_DGEMM_KERNELS(axb0, avx2);
    } else if (MKL_DC_IS_ONE(beta)) {
        MKL_DC_DGEMM_KERNELS(axb1, avx2);
    } else if (MKL_DC_IS_ONE(alpha)) {
        MKL_DC_DGEMM_KERNELS(a1bx, avx2);
    } else {
        MKL_DC_DGEMM_KERNELS(axbx, avx2);
    }
#else
	if (AisN && BisN)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_MOV, MKL_DC_MN, MKL_DC_MN);
#ifndef MKL_REAL_DATA_TYPE
	else if (AisC && BisN)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_CONJ, MKL_DC_MOV, MKL_DC_MT, MKL_DC_MN);
	else if (AisC && BisT)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_CONJ, MKL_DC_MOV, MKL_DC_MT, MKL_DC_MT);
	else if (AisN && BisC)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_CONJ, MKL_DC_MN, MKL_DC_MT);
	else if (AisT && BisC)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_CONJ, MKL_DC_MT, MKL_DC_MT);
	else if (AisC && BisC)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_CONJ, MKL_DC_CONJ, MKL_DC_MT, MKL_DC_MT);
#endif
	else if (AisN && !BisN)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_MOV, MKL_DC_MN, MKL_DC_MT);
	else if (!AisN && BisN)
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_MOV, MKL_DC_MT, MKL_DC_MN);
	else
		mkl_dc_gemm_xx_mnk_pst(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, MKL_DC_MOV, MKL_DC_MOV, MKL_DC_MT, MKL_DC_MT);
#endif

}

/* ?TRSM */
#define mkl_dc_trsm_lxnx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, diag_op ) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (j = 0; j < n; j++) { \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = m-1; k >= 0; k--) { \
            diag_op( B[k + j * ldb], A[k + k * lda] ); \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = 0; i <= k-1; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[i + k * lda]; \
                MKL_DC_MUL(temp1, B[k + j * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
	} \
    } else { \
	for (j = 0; j < n; j++) { \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = 0; k < m; k++) { \
            diag_op( B[k + j * ldb], A[k + k * lda] ); \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = k+1; i < m; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[i + k * lda]; \
                MKL_DC_MUL(temp1, B[k + j * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_lxtx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, diag_op ) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (j = 0; j < n; j++) { \
        mkl_dc_type temp; \
        for(i = 0; i < m; i++) { \
            MKL_DC_MUL(temp, alpha, B[i+j*ldb]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (k = 0; k <= i-1; k++) { \
                mkl_dc_type a, temp1; \
                a = A[k + i * lda]; \
                MKL_DC_MUL(temp1, B[k + j * ldb], a); \
                MKL_DC_SUB(temp, temp, temp1); \
            } \
            diag_op( temp, A[i + i * lda] ); \
            MKL_DC_MOV(B[i+j*ldb], temp); \
	    } \
	} \
    } else { \
	for (j = 0; j < n; j++) { \
        mkl_dc_type temp; \
        for(i = m-1; i >= 0; i--) { \
            MKL_DC_MUL(temp, alpha, B[i+j*ldb]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (k = i+1; k < m; k++) { \
                mkl_dc_type a, temp1; \
                a = A[k + i * lda]; \
                MKL_DC_MUL(temp1, B[k + j * ldb], a); \
                MKL_DC_SUB(temp, temp, temp1); \
            } \
            diag_op( temp, A[i + i * lda] ); \
            MKL_DC_MOV(B[i+j*ldb], temp); \
	    } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_lxcx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, diag_op ) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (j = 0; j < n; j++) { \
        mkl_dc_type temp; \
        for(i = 0; i < m; i++) { \
            mkl_dc_type a; \
            MKL_DC_MUL(temp, alpha, B[i+j*ldb]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (k = 0; k <= i-1; k++) { \
                mkl_dc_type a1, temp1; \
                a1 = A[k + i * lda]; \
                MKL_DC_CONJ(a1, a1); \
                MKL_DC_MUL(temp1, B[k + j * ldb], a1); \
                MKL_DC_SUB(temp, temp, temp1); \
            } \
            a = A[i + i * lda]; \
            MKL_DC_CONJ(a, a); \
            diag_op( temp, a ); \
            MKL_DC_MOV(B[i+j*ldb], temp); \
	    } \
	} \
    } else { \
	for (j = 0; j < n; j++) { \
        mkl_dc_type temp; \
        for(i = m-1; i >= 0; i--) { \
            mkl_dc_type a; \
            MKL_DC_MUL(temp, alpha, B[i+j*ldb]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (k = i+1; k < m; k++) { \
                mkl_dc_type a1, temp1; \
                a1 = A[k + i * lda]; \
                MKL_DC_CONJ(a1, a1); \
                MKL_DC_MUL(temp1, B[k + j * ldb], a1); \
                MKL_DC_SUB(temp, temp, temp1); \
            } \
            a = A[i + i * lda]; \
            MKL_DC_CONJ(a, a); \
            diag_op( temp, a ); \
            MKL_DC_MOV(B[i+j*ldb], temp); \
	    } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxnn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb ) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (j = 0; j < n; j++) { \
        mkl_dc_type temp, one; \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = 0; k <= j-1; k++) { \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = 0; i < m; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[k + j * lda]; \
                MKL_DC_MUL(temp1, B[i + k * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
        MKL_DC_SET_ONE(one); \
        MKL_DC_DIV(temp, one, A[j + j * lda]); \
        for ( i = 0; i < m; i++ ) { \
            MKL_DC_MUL(B[i + j * ldb], temp, B[i + j * ldb]); \
        } \
	} \
    } else { \
	for (j = n-1; j >= 0; j--) { \
        mkl_dc_type temp, one; \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = j+1; k < n; k++) { \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = 0; i < m; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[k + j * lda]; \
                MKL_DC_MUL(temp1, B[i + k * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
        MKL_DC_SET_ONE(one); \
        MKL_DC_DIV(temp, one, A[j + j * lda]); \
        for ( i = 0; i < m; i++ ) { \
            MKL_DC_MUL(B[i + j * ldb], temp, B[i + j * ldb]); \
        } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxnu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb ) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (j = 0; j < n; j++) { \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = 0; k <= j-1; k++) { \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = 0; i < m; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[k + j * lda]; \
                MKL_DC_MUL(temp1, B[i + k * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
	} \
    } else { \
	for (j = n-1; j >= 0; j--) { \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for(i = 0; i < m; i++) { \
                MKL_DC_MUL_C( B[i+j*ldb], alpha ); \
            } \
        } \
	    for (k = j+1; k < n; k++) { \
            MKL_DC_PRAGMA_VECTOR \
            for ( i = 0; i < m; i++ ) { \
                mkl_dc_type a, temp1; \
                a = A[k + j * lda]; \
                MKL_DC_MUL(temp1, B[i + k * ldb], a); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
	        } \
	    } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxtn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (k = n-1; k >= 0; k--) { \
        mkl_dc_type temp, one; \
        MKL_DC_SET_ONE(one); \
        MKL_DC_DIV(temp, one, A[k + k * lda]); \
        for (i = 0; i < m; i++) { \
            MKL_DC_MUL(B[i + k*ldb], temp, B[i + k*ldb]); \
        } \
        for(j = 0; j <= k-1; j++) { \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } else { \
	for (k = 0; k < n; k++) { \
        mkl_dc_type temp, one; \
        MKL_DC_SET_ONE(one); \
        MKL_DC_DIV(temp, one, A[k + k * lda]); \
        for (i = 0; i < m; i++) { \
            MKL_DC_MUL(B[i + k*ldb], temp, B[i + k*ldb]); \
        } \
        for(j = k+1; j < n; j++) { \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxcn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (k = n-1; k >= 0; k--) { \
        mkl_dc_type temp, one, a; \
        MKL_DC_SET_ONE(one); \
        a = A[k + k * lda]; \
        MKL_DC_CONJ(a, a); \
        MKL_DC_DIV(temp, one, a); \
        for (i = 0; i < m; i++) { \
            MKL_DC_MUL(B[i + k*ldb], temp, B[i + k*ldb]); \
        } \
        for(j = 0; j <= k-1; j++) { \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_CONJ(temp, temp); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } else { \
	for (k = 0; k < n; k++) { \
        mkl_dc_type temp, one, a; \
        MKL_DC_SET_ONE(one); \
        a = A[k + k * lda]; \
        MKL_DC_CONJ(a, a); \
        MKL_DC_DIV(temp, one, a); \
        for (i = 0; i < m; i++) { \
            MKL_DC_MUL(B[i + k*ldb], temp, B[i + k*ldb]); \
        } \
        for(j = k+1; j < n; j++) { \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_CONJ(temp, temp); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxtu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (k = n-1; k >= 0; k--) { \
        for(j = 0; j <= k-1; j++) { \
            mkl_dc_type temp; \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } else { \
	for (k = 0; k < n; k++) { \
        for(j = k+1; j < n; j++) { \
            mkl_dc_type temp; \
            MKL_DC_MOV(temp, A[j+k*lda]); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } \
} while (0)

#define mkl_dc_trsm_rxcu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb) \
do { \
	MKL_INT i, j, k; \
    if ( MKL_DC_MisU(uplo) ) { \
	for (k = n-1; k >= 0; k--) { \
        for(j = 0; j <= k-1; j++) { \
            mkl_dc_type temp; \
            temp = A[j+k*lda]; \
            MKL_DC_CONJ(temp, temp); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } else { \
	for (k = 0; k < n; k++) { \
        for(j = k+1; j < n; j++) { \
            mkl_dc_type temp; \
            temp = A[j+k*lda]; \
            MKL_DC_CONJ(temp, temp); \
            MKL_DC_PRAGMA_VECTOR \
	        for (i = 0; i < m; i++) { \
                mkl_dc_type b, temp1; \
                b = B[i + k * ldb]; \
                MKL_DC_MUL(temp1, temp, b); \
                MKL_DC_SUB(B[i + j * ldb], B[i + j * ldb], temp1); \
            } \
	    } \
        if ( !(MKL_DC_IS_ONE(alpha)) ) { \
            for ( i = 0; i < m; i++ ) { \
                MKL_DC_MUL(B[i + k * ldb], alpha, B[i + k * ldb]); \
            } \
        } \
	} \
    } \
} while (0)

static __inline void mkl_dc_trsm(const char * SIDE, const char * UPLO,
            const char * TRANSA, const char * DIAG,
            const MKL_INT * M, const MKL_INT * N,
            const mkl_dc_type * ALPHA,
			const mkl_dc_type * A, const MKL_INT * LDA,
            mkl_dc_type * B, const MKL_INT * LDB)
{
	int AisN;
	int lside, unit;
#ifndef MKL_REAL_DATA_TYPE
    int noconj;
#endif
	mkl_dc_type alpha = *ALPHA;
	MKL_INT m = *M, n = *N;
	MKL_INT lda = *LDA, ldb = *LDB;
    char uplo = *UPLO;

	if (m <= 0 || n <= 0 )
		return;

	AisN = MKL_DC_MisN(*TRANSA);
	lside = MKL_DC_MisL(*SIDE);
    unit = MKL_DC_MisU(*DIAG);
#ifndef MKL_REAL_DATA_TYPE
    noconj = MKL_DC_MisT(*TRANSA);
#endif

	if (MKL_DC_IS_ZERO(alpha)) {
		MKL_INT i, j;
			for (j = 0; j < n; j++)
#pragma vector
				for (i = 0; i < m; i++)
					MKL_DC_SET_ZERO(B[i + ldb * j]);
		return;
	}

    if (lside)
        if (AisN)
            if (unit)
                mkl_dc_trsm_lxnx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_NOOP);
            else
                mkl_dc_trsm_lxnx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_DIV_B);
        else
#ifndef MKL_REAL_DATA_TYPE
            if (!noconj)
                if (unit)
                    mkl_dc_trsm_lxcx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_NOOP);
                else
                    mkl_dc_trsm_lxcx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_DIV_B);
            else
#endif
            if (unit)
                mkl_dc_trsm_lxtx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_NOOP);
            else
                mkl_dc_trsm_lxtx_mn_pst(uplo, m, n, alpha, A, lda, B, ldb, MKL_DC_DIV_B);
    else
        if (AisN)
            if (unit)
                mkl_dc_trsm_rxnu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
            else
                mkl_dc_trsm_rxnn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
        else
#ifndef MKL_REAL_DATA_TYPE
            if (!noconj)
                if (unit)
                    mkl_dc_trsm_rxcu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
                else
                    mkl_dc_trsm_rxcn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
            else
#endif
            if (unit)
                mkl_dc_trsm_rxtu_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
            else
                mkl_dc_trsm_rxtn_mn_pst(uplo, m, n, alpha, A, lda, B, ldb);
}

/* ?SYRK */
#define mkl_dc_syrk_xx_nk_pst_beta(uplo, n, k, alpha, A, lda, beta, C, ldc, \
		c_op, a_access) \
do { \
	MKL_INT i, j, l; \
    if (MKL_DC_MisU(uplo)) { \
	    for (j = 0; j < n; j++) \
		MKL_DC_PRAGMA_VECTOR \
	    for (i = 0; i <= j; i++) { \
		    mkl_dc_type c, temp; MKL_DC_SET_ZERO(temp); \
		    for (l = 0; l < k; l++) { \
			    mkl_dc_type a, b, temp1; \
			    a = a_access(A, lda, j, l); \
			    b = a_access(A, lda, i, l); \
			    MKL_DC_MUL(temp1, a, b); \
			    MKL_DC_ADD(temp, temp, temp1); \
		    } \
		    MKL_DC_MUL(temp, alpha, temp); \
		    c = C[i + j * ldc]; \
		    c_op(c, beta); \
		    MKL_DC_ADD(C[i + j * ldc], c, temp); \
	    } \
    } else { \
	    for (j = 0; j < n; j++) \
		MKL_DC_PRAGMA_VECTOR \
	    for (i = j; i < n; i++) { \
		    mkl_dc_type c, temp; MKL_DC_SET_ZERO(temp); \
		    for (l = 0; l < k; l++) { \
			    mkl_dc_type a, b, temp1; \
			    a = a_access(A, lda, j, l); \
			    b = a_access(A, lda, i, l); \
			    MKL_DC_MUL(temp1, a, b); \
			    MKL_DC_ADD(temp, temp, temp1); \
		    } \
		    MKL_DC_MUL(temp, alpha, temp); \
		    c = C[i + j * ldc]; \
		    c_op(c, beta); \
		    MKL_DC_ADD(C[i + j * ldc], c, temp); \
	    } \
    } \
} while (0)


#define mkl_dc_syrk_xx_nk_pst(uplo, n, k, alpha, A, lda, beta, C, ldc, \
		a_access) \
do { \
	if (MKL_DC_IS_ZERO(beta)) \
		mkl_dc_syrk_xx_nk_pst_beta(uplo, n, k, alpha, A, lda, beta, C, ldc, \
				MKL_DC_ZERO_C, a_access); \
	else \
		mkl_dc_syrk_xx_nk_pst_beta(uplo, n, k, alpha, A, lda, beta, C, ldc, \
				MKL_DC_MUL_C, a_access); \
} while (0)

static __inline void mkl_dc_syrk(const char * UPLO, const char * TRANS,
            const MKL_INT * N, const MKL_INT * K,
            const mkl_dc_type * ALPHA,
			const mkl_dc_type * A, const MKL_INT * LDA,
            const mkl_dc_type * BETA,
            mkl_dc_type * C, const MKL_INT * LDC)
{
	int AisN, CisU;
	mkl_dc_type alpha = *ALPHA, beta = *BETA;
	MKL_INT n = *N, k = *K;
	MKL_INT lda = *LDA, ldc = *LDC;
    char uplo = *UPLO;

	if (n <= 0 || ((MKL_DC_IS_ZERO(alpha) || k <= 0) && MKL_DC_IS_ONE(beta)))
		return;

	AisN = MKL_DC_MisN(*TRANS);
    CisU = MKL_DC_MisU(*UPLO);

	if (MKL_DC_IS_ZERO(alpha)) {
		MKL_INT i, j;
		if (MKL_DC_IS_ZERO(beta))
            if (CisU)
			    for (j = 0; j < n; j++)
#pragma vector
				    for (i = 0; i <= j; i++)
					    MKL_DC_SET_ZERO(C[i + ldc * j]);
            else
			    for (j = 0; j < n; j++)
#pragma vector
				    for (i = j; i < n; i++)
					    MKL_DC_SET_ZERO(C[i + ldc * j]);
		else
            if (CisU)
			    for (j = 0; j < n; j++)
#pragma vector
				    for (i = 0; i <= j; i++)
					    MKL_DC_MUL(C[i + ldc * j], beta, C[i + ldc * j]);
            else
			    for (j = 0; j < n; j++)
#pragma vector
				    for (i = j; i < n; i++)
					    MKL_DC_MUL(C[i + ldc * j], beta, C[i + ldc * j]);
		return;
	}

	if (AisN)
		mkl_dc_syrk_xx_nk_pst(uplo, n, k, alpha, A, lda, beta, C, ldc, MKL_DC_MN);
    else
		mkl_dc_syrk_xx_nk_pst(uplo, n, k, alpha, A, lda, beta, C, ldc, MKL_DC_MT);

}

static __inline void mkl_dc_axpy (const MKL_INT *N, const mkl_dc_type *ALPHA, const mkl_dc_type *x,  const MKL_INT *INCX, mkl_dc_type *y, const MKL_INT *INCY)
{
    MKL_INT i;
    MKL_INT n = *N, ix = 0, iy = 0;
    if (*INCX == 1 && *INCY == 1) {
#pragma vector vecremainder 
        for(i = 0; i < n; i++) MKL_DC_MUL_ADD(y[i], (*ALPHA), x[i], y[i]);
    } else {
        if (*INCX < 0) ix = (-(n) + 1) * *INCX;
        if (*INCY < 0) iy = (-(n) + 1) * *INCY;
        for(i = 0; i < n; i++, ix += *INCX, iy += *INCY) MKL_DC_MUL_ADD(y[iy], (*ALPHA), x[ix], y[iy]);
    }
}

#if defined(MKL_DOUBLE) || defined(MKL_SINGLE)

static __inline mkl_dc_type mkl_dc_dot (const MKL_INT* N, const mkl_dc_type *x,
        const MKL_INT* INCX, const mkl_dc_type *y, const MKL_INT* INCY)
{
    MKL_INT n = *N, ix = 0, iy = 0;
    mkl_dc_type ret = 0.0;
    MKL_INT i;

    if (*INCX == 1 && *INCY == 1)
#pragma vector vecremainder 
        for(i = 0; i < n; i++) ret += y[i] * x[i];
    else  {
        if (*INCX < 0) ix = (-(n) + 1) * *INCX;
        if (*INCY < 0) iy = (-(n) + 1) * *INCY;
        for(i = 0; i < n; i++, ix += *INCX, iy += *INCY) ret += y[iy] * x[ix];
    }
    return ret;
}

#endif

#undef MKL_DC_DOT_CONVERT
#undef mkl_dc_gemm_xx_mnk_pst_beta
#undef mkl_dc_gemm_xx_mnk_pst
#undef mkl_dc_syrk_xx_nk_pst_beta
#undef mkl_dc_syrk_xx_nk_pst
#undef mkl_dc_trsm_lxnx_mn_pst
#undef mkl_dc_trsm_lxtx_mn_pst
#undef mkl_dc_trsm_lxcx_mn_pst
#undef mkl_dc_trsm_rxnn_mn_pst
#undef mkl_dc_trsm_rxnu_mn_pst
#undef mkl_dc_trsm_rxtn_mn_pst
#undef mkl_dc_trsm_rxtu_mn_pst
#undef mkl_dc_trsm_rxcn_mn_pst
#undef mkl_dc_trsm_rxcu_mn_pst
#undef mkl_dc_axpy
#undef mkl_dc_dot
