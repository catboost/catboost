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
!      Intel(R) oneAPI Math Kernel Library (oneMKL) intrinsics code
!******************************************************************************/

#ifdef __AVX2__
#if defined(MKL_DC_ALPHA_ONE) && defined(MKL_DC_BETA_ONE)
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _a1b1_avx2_pst
#elif defined(MKL_DC_ALPHA_ONE) && defined(MKL_DC_BETA_ZERO)
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _a1b0_avx2_pst
#elif defined(MKL_DC_ALPHA_ONE)
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _a1bx_avx2_pst
#elif defined(MKL_DC_BETA_ONE)
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _axb1_avx2_pst
#elif defined(MKL_DC_BETA_ZERO)
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _axb0_avx2_pst
#else
#define MKL_DC_FNAME_GEMM_KERNEL(fname) mkl_dc_ ## fname ## _axbx_avx2_pst
#endif
#endif

#ifdef __AVX2__
#ifdef MKL_DOUBLE

static __inline void MKL_DC_FNAME_GEMM_KERNEL(dgemm_nn_mnk)
(MKL_INT m, MKL_INT n, MKL_INT kK,
 const mkl_dc_type * ALPHA,
 const mkl_dc_type * A, MKL_INT lda,
 const mkl_dc_type * B, MKL_INT ldb,
 const mkl_dc_type * BETA,
 mkl_dc_type * C, MKL_INT ldc) 
{
#undef MKL_DC_AA
#undef MKL_DC_BB
#undef MKL_DC_CC
#define MKL_DC_AA(i,j) ((A)[(i)+lda*(j)])
#define MKL_DC_BB(i,j) ((B)[(i)+ldb*(j)])
#define MKL_DC_CC(i,j) ((C)[(i)+ldc*(j)])
    const MKL_INT m_in_ker = 8;
    const MKL_INT n_in_ker = 4;
    const MKL_INT k_in_ker = 4;

    const MKL_INT MKER1    = 4;
    const MKL_INT MKER2    = 2;
    const MKL_INT MKER3    = 1;
    const MKL_INT MKER4    = 0;

    MKL_INT m0 = (m/m_in_ker)*m_in_ker;
    MKL_INT n0 = (n/n_in_ker)*n_in_ker;
    MKL_INT k0 = (kK/k_in_ker)*k_in_ker;

    MKL_INT krem = kK - k0;

    MKL_DC_YMMTYPE ymm_temp;
    MKL_DC_YMMTYPE ymm_temp0, ymm_temp1;
    MKL_DC_YMMTYPE ymm_temp2, ymm_temp3;
    MKL_DC_YMMTYPE ymm_temp4, ymm_temp5;
    MKL_DC_YMMTYPE ymm_temp6, ymm_temp7;

    MKL_DC_YMMTYPE ymm_c0, ymm_c1;
    MKL_DC_YMMTYPE ymm_c2, ymm_c3;
    MKL_DC_YMMTYPE ymm_c4, ymm_c5;
    MKL_DC_YMMTYPE ymm_c6, ymm_c7;

    MKL_DC_YMMTYPE ymm_a, ymm_a1, ymm_b;
    MKL_DC_YMMTYPE ymm_alpha;

    MKL_DC_XMMTYPE xmm_a, xmm_b;
    MKL_DC_XMMTYPE xmm_temp0, xmm_temp3, xmm_temp5, xmm_temp7;
    MKL_DC_XMMTYPE xmm_temp;
    MKL_DC_XMMTYPE xmm_c, xmm_c3, xmm_c5, xmm_c7;
    MKL_DC_XMMTYPE xmm_alpha;

#if !defined(MKL_DC_ALPHA_ZERO) && !defined(MKL_DC_ALPHA_ONE)
    ymm_alpha = MKL_DC_BCAST_YMM(ALPHA);
    xmm_alpha = MKL_DC_CAST_YMM_TO_XMM(ymm_alpha);
#endif

#if !defined(MKL_DC_BETA_ZERO) && !defined(MKL_DC_BETA_ONE)
    MKL_DC_YMMTYPE ymm_beta = MKL_DC_BCAST_YMM(BETA);
    MKL_DC_XMMTYPE xmm_beta = MKL_DC_CAST_YMM_TO_XMM(ymm_beta);
#endif

    MKL_INT j;
    for (j=0; j<n0; j+=n_in_ker) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_c0, ymm_temp0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_c4, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 =  ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+2), ymm_c5);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c6 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_c6, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_ADD_YMM(ymm_temp6, ymm_c6);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp6, ymm_c6, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp6);
#else
            ymm_c6 =  ymm_temp6;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c6);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c7 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_c7, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_ADD_YMM(ymm_temp7, ymm_c7);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp7, ymm_c7, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp7);
#else
            ymm_c7 =  ymm_temp7;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+3), ymm_c7);
        }

        if ((m-i) & MKER1) {

            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c7 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_c7, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_ADD_YMM(ymm_temp7, ymm_c7);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp7, ymm_c7, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp7);
#else
            ymm_c7 =  ymm_temp7;
#endif
#endif

            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c5);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c7);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c7 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_c7, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_ADD_XMM(xmm_temp7, xmm_c7);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp7, xmm_c7, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp7);
#else
            xmm_c7 = xmm_temp7;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2), xmm_c5);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+3), xmm_c7);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c  = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c  = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM_S(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c7 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_c7, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_ADD_XMM_S(xmm_temp7, xmm_c7);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp7, xmm_c7, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp7);
#else
            xmm_c7 = xmm_temp7;
#endif
#endif

            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2), xmm_c5);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+3), xmm_c7);

            i += MKER3;
        }
    }

    if ((n-j) == 3) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_c0, ymm_temp0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_c4, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 =  ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+2), ymm_c5);

        }

        if ((m-i) & MKER1) {

            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif

            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c5);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2), xmm_c5);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c  = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c  = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM_S(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2), xmm_c5);

            i += MKER3;
        }
        
    } else if ((n-j) == 2) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp4);
                ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_temp5);
                ymm_temp2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_temp6);
                ymm_temp3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_temp7);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 =  ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 =  ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

        }

        if ((m-i) & MKER1) {

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#endif
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp4);
                ymm_temp3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_temp7);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 = ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#endif
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp5);
                xmm_temp3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_temp7);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }


#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);

            i += MKER2;
        }

        if ((m - i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#endif

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);
            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);

            i += MKER3;
        }


    } else if ((n-j) == 1) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp2);
                ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_temp3);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 =  ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

        }

        if ((m-i) & MKER1) {

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#endif
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp1);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#endif
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp3);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;
            }


#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#endif

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_temp3);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);

            i += MKER3;
        }
    }
}

static __inline void MKL_DC_FNAME_GEMM_KERNEL(dgemm_nt_mnk)
(MKL_INT m, MKL_INT n, MKL_INT kK,
 const mkl_dc_type * ALPHA,
 const mkl_dc_type * A, MKL_INT lda,
 const mkl_dc_type * B, MKL_INT ldb,
 const mkl_dc_type * BETA,
 mkl_dc_type * C, MKL_INT ldc) 
{
#undef MKL_DC_AA
#undef MKL_DC_BB
#undef MKL_DC_CC
#define MKL_DC_AA(i,j) ((A)[(i)+lda*(j)])
#define MKL_DC_BB(i,j) ((B)[(j)+ldb*(i)])
#define MKL_DC_CC(i,j) ((C)[(i)+ldc*(j)])
    const MKL_INT m_in_ker = 8;
    const MKL_INT n_in_ker = 4;
    const MKL_INT k_in_ker = 4;

    const MKL_INT MKER1    = 4;
    const MKL_INT MKER2    = 2;
    const MKL_INT MKER3    = 1;
    const MKL_INT MKER4    = 0;

    MKL_INT m0 = (m/m_in_ker)*m_in_ker;
    MKL_INT n0 = (n/n_in_ker)*n_in_ker;
    MKL_INT k0 = (kK/k_in_ker)*k_in_ker;

    MKL_INT krem = kK - k0;

    MKL_DC_YMMTYPE ymm_temp;
    MKL_DC_YMMTYPE ymm_temp0, ymm_temp1;
    MKL_DC_YMMTYPE ymm_temp2, ymm_temp3;
    MKL_DC_YMMTYPE ymm_temp4, ymm_temp5;
    MKL_DC_YMMTYPE ymm_temp6, ymm_temp7;

    MKL_DC_YMMTYPE ymm_c0, ymm_c1;
    MKL_DC_YMMTYPE ymm_c2, ymm_c3;
    MKL_DC_YMMTYPE ymm_c4, ymm_c5;
    MKL_DC_YMMTYPE ymm_c6, ymm_c7;

    MKL_DC_YMMTYPE ymm_a, ymm_a1, ymm_b;
    MKL_DC_YMMTYPE ymm_alpha;

    MKL_DC_XMMTYPE xmm_a, xmm_b;
    MKL_DC_XMMTYPE xmm_temp0, xmm_temp3, xmm_temp5, xmm_temp7;
    MKL_DC_XMMTYPE xmm_temp;
    MKL_DC_XMMTYPE xmm_c, xmm_c3, xmm_c5, xmm_c7;
    MKL_DC_XMMTYPE xmm_alpha;

#if !defined(MKL_DC_ALPHA_ZERO) && !defined(MKL_DC_ALPHA_ONE)
    ymm_alpha = MKL_DC_BCAST_YMM(ALPHA);
    xmm_alpha = MKL_DC_CAST_YMM_TO_XMM(ymm_alpha);
#endif

#if !defined(MKL_DC_BETA_ZERO) && !defined(MKL_DC_BETA_ONE)
    MKL_DC_YMMTYPE ymm_beta = MKL_DC_BCAST_YMM(BETA);
    MKL_DC_XMMTYPE xmm_beta = MKL_DC_CAST_YMM_TO_XMM(ymm_beta);
#endif

    MKL_INT j;
    for (j=0; j<n0; j+=n_in_ker) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_c0, ymm_temp0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_c4, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 =  ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+2), ymm_c5);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c6 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_c6, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_ADD_YMM(ymm_temp6, ymm_c6);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp6, ymm_c6, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp6);
#else
            ymm_c6 =  ymm_temp6;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c6);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c7 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_c7, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_ADD_YMM(ymm_temp7, ymm_c7);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp7, ymm_c7, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp7);
#else
            ymm_c7 =  ymm_temp7;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+3), ymm_c7);
        }

        if ((m-i) & MKER1) {

            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c7 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_c7, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_ADD_YMM(ymm_temp7, ymm_c7);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp7, ymm_c7, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp7);
#else
            ymm_c7 =  ymm_temp7;
#endif
#endif

            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c5);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c7);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+3));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c7 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_c7, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_ADD_XMM(xmm_temp7, xmm_c7);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp7, xmm_c7, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp7);
#else
            xmm_c7 = xmm_temp7;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2), xmm_c5);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+3), xmm_c7);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c  = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c  = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM_S(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c7 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM(xmm_c7, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_ADD_XMM_S(xmm_temp7, xmm_c7);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp7, xmm_c7, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c7 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp7);
#else
            xmm_c7 = xmm_temp7;
#endif
#endif

            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2), xmm_c5);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+3), xmm_c7);

            i += MKER3;
        }
    }

    if ((n-j) == 3) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_c0, ymm_temp0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_c4, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 =  ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+2), ymm_c5);

        }

        if ((m-i) & MKER1) {

            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);


                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp5, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_c5, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 =  ymm_temp5;
#endif
#endif

            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c5);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+2));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2), xmm_c5);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);

            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp5, xmm_b);


                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c  = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c  = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c5 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM(xmm_c5, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_ADD_XMM_S(xmm_temp5, xmm_c5);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp5, xmm_c5, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c5 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp5);
#else
            xmm_c5 = xmm_temp5;
#endif
#endif

            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2), xmm_c5);

            i += MKER3;
        }
        
    } else if ((n-j) == 2) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);
                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp5, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp4);
                ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_temp5);
                ymm_temp2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_temp6);
                ymm_temp3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_temp7);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 =  ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_c2, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 =  ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 =  ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j+1), ymm_c3);

        }

        if ((m-i) & MKER1) {

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#endif
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp7, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp4);
                ymm_temp3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_temp7);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_c3, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 = ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c3);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
            xmm_c3 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#endif
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp5, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp7, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp5);
                xmm_temp3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_temp7);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j+1));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }


#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c3);

            i += MKER2;
        }

        if ((m - i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
            xmm_c3 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#endif

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);
            }

            if (krem & 2) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM(xmm_c3, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_ADD_XMM_S(xmm_temp3, xmm_c3);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp3, xmm_c3, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c3 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp3);
#else
            xmm_c3 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c3);

            i += MKER3;
        }


    } else if ((n-j) == 1) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 1));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 2));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 3));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp2, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp3, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp2);
                ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_temp3);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a1       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+4,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a1, ymm_b, ymm_temp1, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 =  ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i+4,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_c1, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 =  ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i+4,j), ymm_c1);

        }

        if ((m-i) & MKER1) {

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#endif
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 1));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 1,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 2));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 2,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 3));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 3,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);
            }

            if (krem & 2) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                k++;

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp1, ymm_temp);

                k++;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp1);
            }

            if (krem & 1) {
                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i,k + 0));
                ymm_b       = MKL_DC_BCAST_YMM(&MKL_DC_BB(k + 0,j));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b, ymm_temp0, ymm_temp);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_c0, ymm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

            i += MKER1;
        }

        if ((m-i) & MKER2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j));
#endif
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+1));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+1,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+2));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+2,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k+3));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k+3,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);
            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp3);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM(&MKL_DC_AA(i,k));
                ymm_b      = MKL_DC_BCAST_YMM(&MKL_DC_BB(k,j));
                xmm_b      = MKL_DC_CAST_YMM_TO_XMM(ymm_b);
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;
            }


#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j), xmm_c);

            i += MKER2;
        }

        if ((m-i) & MKER3) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j));
#endif

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+1));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+2));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);


                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k+3));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

            }

            if (krem & 2) {

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;

                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp3, xmm_b);

                k++;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_temp3);
            }

            if (krem & 1) {
                xmm_a      = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i,k));
                xmm_b      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k,j));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b, xmm_temp0, xmm_b);

                k++;
            }

#if !defined(MKL_DC_BETA_ZERO)
#if !defined(MKL_DC_BETA_ONE)
            xmm_c = MKL_DC_MUL_XMM(xmm_c, xmm_beta);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j), xmm_c);

            i += MKER3;
        }
    }
}

static __inline void MKL_DC_FNAME_GEMM_KERNEL(dgemm_tn_mnk)
(MKL_INT m, MKL_INT n, MKL_INT kK,
 const mkl_dc_type * ALPHA,
 const mkl_dc_type * A, MKL_INT lda,
 const mkl_dc_type * B, MKL_INT ldb,
 const mkl_dc_type * BETA,
 mkl_dc_type * C, MKL_INT ldc) 
{
#undef MKL_DC_AA
#undef MKL_DC_BB
#undef MKL_DC_CC
#define MKL_DC_AA(i,j) ((A)[(j)+lda*(i)])
#define MKL_DC_BB(i,j) ((B)[(i)+ldb*(j)])
#define MKL_DC_CC(i,j) ((C)[(i)+ldc*(j)])

    const MKL_INT m_in_ker = 4;
    const MKL_INT n_in_ker = 2;
    const MKL_INT k_in_ker = 4;
    const MKL_INT64 MASKBIT =  ((MKL_INT64) 1<<63);

    MKL_INT m0   = (m/m_in_ker)*m_in_ker;
    MKL_INT n0   = (n/n_in_ker)*n_in_ker;
    MKL_INT k0   = (kK/k_in_ker)*k_in_ker;

    MKL_INT krem = kK - k0;

    MKL_DC_YMMTYPE ymm_temp;
    MKL_DC_YMMTYPE ymm_temp0, ymm_temp1;
    MKL_DC_YMMTYPE ymm_temp2, ymm_temp3;
    MKL_DC_YMMTYPE ymm_temp4, ymm_temp5;
    MKL_DC_YMMTYPE ymm_temp6, ymm_temp7;
    MKL_DC_YMMTYPE ymm_temp8, ymm_temp9;
    MKL_DC_YMMTYPE ymm_temp10, ymm_temp11;

    MKL_DC_YMMTYPE ymm_temp20, ymm_temp21;
    MKL_DC_YMMTYPE ymm_temp22, ymm_temp23;

    MKL_DC_YMMTYPE ymm_c0, ymm_c1;
    MKL_DC_YMMTYPE ymm_c2, ymm_c3;
    MKL_DC_YMMTYPE ymm_c4, ymm_c5;
    MKL_DC_YMMTYPE ymm_c6, ymm_c7;

    MKL_DC_YMMTYPE ymm_b0, ymm_b1, ymm_b2;
    MKL_DC_YMMTYPE ymm_a;
    MKL_DC_YMMTYPE ymm_alpha;

    MKL_DC_XMMTYPE xmm_a, xmm_b0, xmm_b1;
    MKL_DC_XMMTYPE xmm_temp0, xmm_temp1, xmm_temp2, xmm_temp3;
    MKL_DC_XMMTYPE xmm_temp4, xmm_temp5, xmm_temp6, xmm_temp7;
    MKL_DC_XMMTYPE xmm_temp8, xmm_temp9, xmm_temp10, xmm_temp11;
    MKL_DC_XMMTYPE xmm_temp;
    MKL_DC_XMMTYPE xmm_c0, xmm_c1, xmm_c2, xmm_c3;
    MKL_DC_XMMTYPE xmm_c4, xmm_c5, xmm_c6, xmm_c7;
    MKL_DC_XMMTYPE xmm_alpha;

#if !defined(MKL_DC_ALPHA_ZERO) && !defined(MKL_DC_ALPHA_ONE)
    ymm_alpha = MKL_DC_BCAST_YMM(ALPHA);
    xmm_alpha = MKL_DC_CAST_YMM_TO_XMM(ymm_alpha);
#endif

#if !defined(MKL_DC_BETA_ZERO) && !defined(MKL_DC_BETA_ONE)
    MKL_DC_YMMTYPE ymm_beta = MKL_DC_BCAST_YMM(BETA);
    MKL_DC_XMMTYPE xmm_beta = MKL_DC_CAST_YMM_TO_XMM(ymm_beta);
#endif

    __m256i k_mask;
    if (krem == 1) {
        k_mask = _mm256_set_epi64x(0, 0, 0, MASKBIT);
    } else if (krem == 2) {
        k_mask = _mm256_set_epi64x(0, 0, MASKBIT, MASKBIT);
    } else if (krem == 3) {
        k_mask = _mm256_set_epi64x(0, MASKBIT, MASKBIT, MASKBIT);
    }


    MKL_INT j;
    for (j=0; j<n0; j+=n_in_ker) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp1  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();
            ymm_temp3  = MKL_DC_SETZERO_YMM();
            ymm_temp4  = MKL_DC_SETZERO_YMM();
            ymm_temp5  = MKL_DC_SETZERO_YMM();
            ymm_temp6  = MKL_DC_SETZERO_YMM();
            ymm_temp7  = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i + 0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                ymm_b1       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+2,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+3,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp7, ymm_temp);
            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                ymm_b1       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+1), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+2,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+3,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp6, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp7, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp4, ymm_temp6);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp1, ymm_temp3);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp5, ymm_temp7);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_beta, ymm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c1);
        }

        if ((m-i) == 3) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp1  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();
            ymm_temp3  = MKL_DC_SETZERO_YMM();
            ymm_temp4  = MKL_DC_SETZERO_YMM();
            ymm_temp5  = MKL_DC_SETZERO_YMM();
            ymm_temp6  = MKL_DC_SETZERO_YMM();
            ymm_temp7  = MKL_DC_SETZERO_YMM();
            __m256i m_mask = _mm256_set_epi64x(0, MASKBIT, MASKBIT, MASKBIT);

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i + 0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                ymm_b1       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+2,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                ymm_b1       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+1), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+2,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp4, ymm_temp6);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_MASKLOAD_YMM(&MKL_DC_CC(i,j), m_mask);
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_MASKSTORE_YMM(&MKL_DC_CC(i,j), m_mask, ymm_c0);

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp1, ymm_temp3);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp5, ymm_temp7);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_MASKLOAD_YMM(&MKL_DC_CC(i,j+1), m_mask);
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_beta, ymm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_MASKSTORE_YMM(&MKL_DC_CC(i,j+1), m_mask, ymm_c1);

        } else if ((m-i) == 2) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp1  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();
            ymm_temp3  = MKL_DC_SETZERO_YMM();
            ymm_temp4  = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                ymm_b1       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                ymm_b1       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+1), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c1 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM(xmm_beta, xmm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c1);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c1, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c1 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0), xmm_c1);

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp1, ymm_temp3);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp1 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp1);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp1, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp1, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp1);
#else
            xmm_c2 = xmm_temp1;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c2);

        } else if ((m-i) == 1) {

            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp1  = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                ymm_b1       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);
            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                ymm_b1       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+1), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp0);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c1 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c1);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c1, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c1 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0), xmm_c1);

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp1, ymm_temp1);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp1 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp1 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp1);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM_S(xmm_temp1, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp1, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp1);
#else
            xmm_c2 = xmm_temp1;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1), xmm_c2);
        }
    }

    if (n-j) {

        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();
            ymm_temp4  = MKL_DC_SETZERO_YMM();
            ymm_temp6  = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+2,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+3,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp6, ymm_temp);
            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+2,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+3,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp6, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp4, ymm_temp6);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j), ymm_c0);
        }

        if ((m-i) == 3) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();
            ymm_temp4  = MKL_DC_SETZERO_YMM();
            ymm_temp6  = MKL_DC_SETZERO_YMM();
            __m256i m_mask = _mm256_set_epi64x(0, MASKBIT, MASKBIT, MASKBIT);

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+2,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);

            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+2,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp4, ymm_temp);

            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp21 = MKL_DC_HADD_YMM(ymm_temp4, ymm_temp6);

            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x21);
            ymm_temp23 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp21, 0x30);

            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp22, ymm_temp23);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_MASKLOAD_YMM(&MKL_DC_CC(i,j), m_mask);
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_MASKSTORE_YMM(&MKL_DC_CC(i,j), m_mask, ymm_c0);


        } else if ((m-i) == 2) {

            ymm_temp0  = MKL_DC_SETZERO_YMM();
            ymm_temp2  = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+1,k + 0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);

            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+1,k + 0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp2, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp2);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c1 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM(xmm_beta, xmm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c1);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c1, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c1 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0), xmm_c1);

        } else if ((m-i) == 1) {
            ymm_temp0  = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_a        = MKL_DC_LOAD_YMM(&MKL_DC_AA(i+0, k + 0));
                ymm_b0       = MKL_DC_LOAD_YMM(&MKL_DC_BB(k + 0, j+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
            }

            if (krem) {
                ymm_a        = MKL_DC_MASKLOAD_YMM(&MKL_DC_AA(i+0, k + 0), k_mask);
                ymm_b0       = MKL_DC_MASKLOAD_YMM(&MKL_DC_BB(k + 0, j+0), k_mask);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b0, ymm_temp0, ymm_temp);
            }

            ymm_temp20 = MKL_DC_HADD_YMM(ymm_temp0, ymm_temp0);
            ymm_temp22 = MKL_DC_PERM2F128_YMM(ymm_temp20, ymm_temp20, 0x01);
            ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp20, ymm_temp22);
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c1 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM(xmm_beta, xmm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c1);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c1, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c1 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c1 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0), xmm_c1);
        }
    }
}

static __inline void MKL_DC_FNAME_GEMM_KERNEL(dgemm_tt_mnk)
(MKL_INT m, MKL_INT n, MKL_INT kK,
 const mkl_dc_type * ALPHA,
 const mkl_dc_type * A, MKL_INT lda,
 const mkl_dc_type * B, MKL_INT ldb,
 const mkl_dc_type * BETA,
 mkl_dc_type * C, MKL_INT ldc) 
{
#undef MKL_DC_AA
#undef MKL_DC_BB
#undef MKL_DC_CC
#define MKL_DC_AA(i,j) ((A)[(j)+lda*(i)])
#define MKL_DC_BB(i,j) ((B)[(j)+ldb*(i)])
#define MKL_DC_CC(i,j) ((C)[(i)+ldc*(j)])

    const MKL_INT m_in_ker = 4;
    const MKL_INT n_in_ker = 8;
    const MKL_INT k_in_ker = 4;

    MKL_INT m0   = (m/m_in_ker)*m_in_ker;
    MKL_INT n0   = (n/n_in_ker)*n_in_ker;
    MKL_INT k0   = (kK/k_in_ker)*k_in_ker;

    MKL_INT krem = kK - k0;

    MKL_DC_YMMTYPE ymm_temp;
    MKL_DC_YMMTYPE ymm_temp0, ymm_temp1;
    MKL_DC_YMMTYPE ymm_temp2, ymm_temp3;
    MKL_DC_YMMTYPE ymm_temp4, ymm_temp5;
    MKL_DC_YMMTYPE ymm_temp6, ymm_temp7;

    MKL_DC_YMMTYPE ymm_c0, ymm_c1;
    MKL_DC_YMMTYPE ymm_c2, ymm_c3;
    MKL_DC_YMMTYPE ymm_c4, ymm_c5;
    MKL_DC_YMMTYPE ymm_c6, ymm_c7;

    MKL_DC_YMMTYPE ymm_a, ymm_b1, ymm_b2;
    MKL_DC_YMMTYPE ymm_alpha;

    MKL_DC_XMMTYPE xmm_a, xmm_b1;
    MKL_DC_XMMTYPE xmm_temp0, xmm_temp1, xmm_temp2, xmm_temp3;
    MKL_DC_XMMTYPE xmm_temp4, xmm_temp5, xmm_temp6, xmm_temp7;
    MKL_DC_XMMTYPE xmm_temp;

    MKL_DC_XMMTYPE xmm_c0, xmm_c1, xmm_c2, xmm_c3;
    MKL_DC_XMMTYPE xmm_c4, xmm_c5, xmm_c6, xmm_c7;
    MKL_DC_XMMTYPE xmm_alpha;

#if !defined(MKL_DC_ALPHA_ZERO) && !defined(MKL_DC_ALPHA_ONE)
    ymm_alpha = MKL_DC_BCAST_YMM(ALPHA);
    xmm_alpha = MKL_DC_CAST_YMM_TO_XMM(ymm_alpha);
#endif

#if !defined(MKL_DC_BETA_ZERO) && !defined(MKL_DC_BETA_ONE)
    MKL_DC_YMMTYPE ymm_beta = MKL_DC_BCAST_YMM(BETA);
    MKL_DC_XMMTYPE xmm_beta = MKL_DC_CAST_YMM_TO_XMM(ymm_beta);
#endif

    MKL_INT j;
    for (j=0; j<n0; j+=n_in_ker) {
        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);


                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);


                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);
            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);
                k+=2;
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp7, ymm_temp);
            }

            MKL_DC_VEC_TRANSPOSE_YMM(ymm_temp0, ymm_temp2, ymm_temp4, ymm_temp6, ymm_c0, ymm_c2, ymm_c4, ymm_c6);
#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_beta, ymm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_beta, ymm_c4);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 = ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c6 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_beta, ymm_c6);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_ADD_YMM(ymm_temp6, ymm_c6);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp6, ymm_c6, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp6);
#else
            ymm_c6 = ymm_temp6;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c6);

            MKL_DC_VEC_TRANSPOSE_YMM(ymm_temp1, ymm_temp3, ymm_temp5, ymm_temp7, ymm_c1, ymm_c3, ymm_c5, ymm_c7);
#if !defined(MKL_DC_BETA_ZERO)
            ymm_c1 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+4));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_beta, ymm_c1);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_ADD_YMM(ymm_temp1, ymm_c1);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp1, ymm_c1, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c1 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp1);
#else
            ymm_c1 = ymm_temp1;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+4), ymm_c1);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c3 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+5));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_beta, ymm_c3);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_ADD_YMM(ymm_temp3, ymm_c3);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp3, ymm_c3, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c3 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp3);
#else
            ymm_c3 = ymm_temp3;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+5), ymm_c3);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c5 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+6));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_beta, ymm_c5);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_ADD_YMM(ymm_temp5, ymm_c5);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp5, ymm_c5, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c5 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp5);
#else
            ymm_c5 = ymm_temp5;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+6), ymm_c5);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c7 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+7));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_beta, ymm_c7);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_ADD_YMM(ymm_temp7, ymm_c7);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp7, ymm_c7, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c7 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp7);
#else
            ymm_c7 = ymm_temp7;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+7), ymm_c7);
        }

        if ((m-i) & 2) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);

                k+=2;
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp3, ymm_temp);
            }

            ymm_temp4 = MKL_DC_UNPACKLO_YMM(ymm_temp0, ymm_temp2);             
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0),xmm_c0);

            ymm_temp5 = MKL_DC_UNPACKHI_YMM(ymm_temp0, ymm_temp2);             
            xmm_temp5 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp5);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c2 = xmm_temp5;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1),xmm_c2);

            ymm_temp0 = MKL_DC_PERM2F128_YMM(ymm_temp4, ymm_temp4, 0x11); 
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2),xmm_c0);

            ymm_temp0 = MKL_DC_PERM2F128_YMM(ymm_temp5, ymm_temp5, 0x11); 
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c2 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+3),xmm_c2);

            ymm_temp4 = MKL_DC_UNPACKLO_YMM(ymm_temp1, ymm_temp3);             
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+4));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+4),xmm_c0);

            ymm_temp5 = MKL_DC_UNPACKHI_YMM(ymm_temp1, ymm_temp3);             
            xmm_temp5 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp5);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+5));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c2 = xmm_temp5;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+5),xmm_c2);

            ymm_temp0 = MKL_DC_PERM2F128_YMM(ymm_temp4, ymm_temp4, 0x11); 
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+6));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+6),xmm_c0);

            ymm_temp1 = MKL_DC_PERM2F128_YMM(ymm_temp5, ymm_temp5, 0x11); 
            xmm_temp1 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp1);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+7));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp1, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp1, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp1);
#else
            xmm_c2 = xmm_temp1;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+7),xmm_c2);

            i+=2;
        }

        if ((m-i) & 1) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

                k+=2;
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b2      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j+4));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b2, ymm_temp1, ymm_temp);

            }

            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1),xmm_c0);

            ymm_temp4 = MKL_DC_PERM2F128_YMM(ymm_temp0, ymm_temp0, 0x11); 
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+3),xmm_c0);

            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp1);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+4));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+4),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+5));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+5),xmm_c0);

            ymm_temp4 = MKL_DC_PERM2F128_YMM(ymm_temp1, ymm_temp1, 0x11); 
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+6));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+6),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+7));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+7),xmm_c0);
        }
    }

    if ((n-j) & 4) {
        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            ymm_temp4 = MKL_DC_SETZERO_YMM();
            ymm_temp5 = MKL_DC_SETZERO_YMM();
            ymm_temp6 = MKL_DC_SETZERO_YMM();
            ymm_temp7 = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp7, ymm_temp);


                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);


                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp7, ymm_temp);
            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp5, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp7, ymm_temp);

                k+=2;
            }

            if (kK>=2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp1);
                ymm_temp2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_temp3);
                ymm_temp4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_temp5);
                ymm_temp6 = MKL_DC_ADD_YMM(ymm_temp6, ymm_temp7);
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp4, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp6, ymm_temp);
            }

            MKL_DC_VEC_TRANSPOSE_YMM(ymm_temp0, ymm_temp2, ymm_temp4, ymm_temp6, ymm_c0, ymm_c2, ymm_c4, ymm_c6);
#if !defined(MKL_DC_BETA_ZERO)
            ymm_c0 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_beta, ymm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_c0);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp0, ymm_c0, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c0 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp0);
#else
            ymm_c0 = ymm_temp0;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+0), ymm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c2 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_beta, ymm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_c2);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp2, ymm_c2, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c2 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp2);
#else
            ymm_c2 = ymm_temp2;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+1), ymm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c4 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_beta, ymm_c4);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_ADD_YMM(ymm_temp4, ymm_c4);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp4, ymm_c4, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c4 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp4);
#else
            ymm_c4 = ymm_temp4;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+2), ymm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            ymm_c6 = MKL_DC_LOAD_YMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_beta, ymm_c6);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_ADD_YMM(ymm_temp6, ymm_c6);
#else
            MKL_DC_MUL_ADD_YMM(ymm_alpha, ymm_temp6, ymm_c6, ymm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            ymm_c6 = MKL_DC_MUL_YMM(ymm_alpha, ymm_temp6);
#else
            ymm_c6 = ymm_temp6;
#endif
#endif
            MKL_DC_STORE_YMM(&MKL_DC_CC(i,j+3), ymm_c6);
        }

        if ((m-i) & 2) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            ymm_temp2 = MKL_DC_SETZERO_YMM();
            ymm_temp3 = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);
            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);


                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);


                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp3, ymm_temp);

                k+=2;
            }

            if (kK >= 2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp1);
                ymm_temp2 = MKL_DC_ADD_YMM(ymm_temp2, ymm_temp3);
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp2, ymm_temp);
            }

            ymm_temp4 = MKL_DC_UNPACKLO_YMM(ymm_temp0, ymm_temp2);             
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0),xmm_c0);

            ymm_temp5 = MKL_DC_UNPACKHI_YMM(ymm_temp0, ymm_temp2);             
            xmm_temp5 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp5);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c2 = xmm_temp5;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1),xmm_c2);

            ymm_temp0 = MKL_DC_PERM2F128_YMM(ymm_temp4, ymm_temp4, 0x11); 
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+2),xmm_c0);

            ymm_temp0 = MKL_DC_PERM2F128_YMM(ymm_temp5, ymm_temp5, 0x11); 
            xmm_temp0 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp0, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp0, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp0);
#else
            xmm_c2 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+3),xmm_c2);

            i+=2;
        }

        if ((m-i) & 1) {
            ymm_temp0 = MKL_DC_SETZERO_YMM();
            ymm_temp1 = MKL_DC_SETZERO_YMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+2, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+3, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);
            }

            if ((kK-k) & 2) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);

                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+1, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp1, ymm_temp);

                k+=2;
            }

            if (kK >= 2) {
                ymm_temp0 = MKL_DC_ADD_YMM(ymm_temp0, ymm_temp1);
            }

            if (kK-k) {
                ymm_b1      = MKL_DC_LOAD_YMM(&MKL_DC_BB(k+0, j));
                ymm_a       = MKL_DC_BCAST_YMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_YMM(ymm_a, ymm_b1, ymm_temp0, ymm_temp);
            }

            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp0);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1),xmm_c0);

            ymm_temp4 = MKL_DC_PERM2F128_YMM(ymm_temp0, ymm_temp0, 0x11); 
            xmm_temp4 = MKL_DC_CAST_YMM_TO_XMM(ymm_temp4);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+2));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+2),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp4, xmm_temp4);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+3));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+3),xmm_c0);
        }
        j+=4;
    }

    if ((n-j) & 2) {
        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            xmm_temp2 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp4 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp6 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();

            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp6, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp7, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp6, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp7, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp6, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp7, xmm_temp);
                k+=2;
            }

            if (kK >= 2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp1);
                xmm_temp2 = MKL_DC_ADD_XMM(xmm_temp2, xmm_temp3);
                xmm_temp4 = MKL_DC_ADD_XMM(xmm_temp4, xmm_temp5);
                xmm_temp6 = MKL_DC_ADD_XMM(xmm_temp6, xmm_temp7);
            }

            if (kK-k) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp6, xmm_temp);
            }

            MKL_DC_VEC_TRANSPOSE_XMM(xmm_temp1, xmm_temp3, xmm_temp0, xmm_temp2);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp1, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp1, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp1);
#else
            xmm_c0 = xmm_temp1;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0), xmm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c2 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1), xmm_c2);

            MKL_DC_VEC_TRANSPOSE_XMM(xmm_temp1, xmm_temp3, xmm_temp4, xmm_temp6);
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c4 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i+2,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c4 = MKL_DC_MUL_XMM(xmm_beta, xmm_c4);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c4 = MKL_DC_ADD_XMM(xmm_temp1, xmm_c4);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp1, xmm_c4, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c4 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp1);
#else
            xmm_c4 = xmm_temp1;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i+2,j+0), xmm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c6 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i+2,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c6 = MKL_DC_MUL_XMM(xmm_beta, xmm_c6);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c6 = MKL_DC_ADD_XMM(xmm_temp3, xmm_c6);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp3, xmm_c6, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c6 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp3);
#else
            xmm_c6 = xmm_temp3;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i+2,j+1), xmm_c6);
        }

        if ((m-i) & 2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            xmm_temp2 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp); 
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                k+=2;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp1);
                xmm_temp2 = MKL_DC_ADD_XMM(xmm_temp2, xmm_temp3);
            }

            if ((kK-k)) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp2, xmm_temp);
            }

            xmm_temp4 = MKL_DC_UNPACKLO_XMM(xmm_temp0, xmm_temp2);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+0),xmm_c0);

            xmm_temp5 = MKL_DC_UNPACKHI_XMM(xmm_temp0, xmm_temp2);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM(xmm_temp5, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM(xmm_alpha, xmm_temp5, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM(xmm_alpha, xmm_temp5);
#else
            xmm_c2 = xmm_temp5;
#endif
#endif
            MKL_DC_STORE_XMM(&MKL_DC_CC(i,j+1),xmm_c2);

            i+=2;
        }

        if ((m-i) & 1) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                k+=2;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM(xmm_temp0, xmm_temp1);
            }

            if ((kK-k)) {
                xmm_b1      = MKL_DC_LOAD_XMM(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOADDUP_XMM(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM(xmm_a, xmm_b1, xmm_temp0, xmm_temp);
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0),xmm_c0);

            xmm_temp4 = MKL_DC_UNPACKHI_XMM(xmm_temp0, xmm_temp0);             
#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+1));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c0 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+1),xmm_c0);
        }
        j+=2;
    }

    if ((n-j)) {
        MKL_INT i;
        for (i=0; i<m0; i+=m_in_ker) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            xmm_temp2 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            xmm_temp4 = MKL_DC_SETZERO_XMM();
            xmm_temp5 = MKL_DC_SETZERO_XMM();
            xmm_temp6 = MKL_DC_SETZERO_XMM();
            xmm_temp7 = MKL_DC_SETZERO_XMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp6, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp7, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp6, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp7, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp6, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp5, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp7, xmm_temp);

                k+=2;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_temp1);
                xmm_temp2 = MKL_DC_ADD_XMM_S(xmm_temp2, xmm_temp3);
                xmm_temp4 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_temp5);
                xmm_temp6 = MKL_DC_ADD_XMM_S(xmm_temp6, xmm_temp7);
            }

            if ((kK-k)) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+2, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp4, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+3, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp6, xmm_temp);
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0), xmm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i+1,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM_S(xmm_temp2, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp2, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp2);
#else
            xmm_c2 = xmm_temp2;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i+1,j+0), xmm_c2);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c4 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i+2,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c4 = MKL_DC_MUL_XMM(xmm_beta, xmm_c4);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c4 = MKL_DC_ADD_XMM_S(xmm_temp4, xmm_c4);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp4, xmm_c4, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c4 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp4);
#else
            xmm_c4 = xmm_temp4;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i+2,j+0), xmm_c4);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c6 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i+3,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c6 = MKL_DC_MUL_XMM(xmm_beta, xmm_c6);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c6 = MKL_DC_ADD_XMM_S(xmm_temp6, xmm_c6);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp6, xmm_c6, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c6 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp6);
#else
            xmm_c6 = xmm_temp6;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i+3,j+0), xmm_c6);
        }

        if ((m-i) & 2) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            xmm_temp2 = MKL_DC_SETZERO_XMM();
            xmm_temp3 = MKL_DC_SETZERO_XMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);


                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp3, xmm_temp);

                k+=2;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_temp1);
                xmm_temp2 = MKL_DC_ADD_XMM_S(xmm_temp2, xmm_temp3);
            }

            if ((kK-k)) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i+1, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp2, xmm_temp);
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0),xmm_c0);

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c2 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i+1,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c2);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_ADD_XMM_S(xmm_temp2, xmm_c2);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp2, xmm_c2, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c2 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp2);
#else
            xmm_c2 = xmm_temp2;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i+1,j+0),xmm_c2);

            i+=2;
        }

        if ((m-i) & 1) {
            xmm_temp0 = MKL_DC_SETZERO_XMM();
            xmm_temp1 = MKL_DC_SETZERO_XMM();
            MKL_INT k;
            for (k=0; k<k0; k+=k_in_ker) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+2, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+2));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+3, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+3));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);
            }

            if ((kK-k) & 2) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);

                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+1, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+1));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp1, xmm_temp);
                k+=2;
            }

            if (kK>=2) {
                xmm_temp0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_temp1);
            }

            if (kK-k) {
                xmm_b1      = MKL_DC_LOAD_XMM_S(&MKL_DC_BB(k+0, j));
                xmm_a       = MKL_DC_LOAD_XMM_S(&MKL_DC_AA(i, k+0));
                MKL_DC_MUL_ADD_XMM_S(xmm_a, xmm_b1, xmm_temp0, xmm_temp);
            }

#if !defined(MKL_DC_BETA_ZERO)
            xmm_c0 = MKL_DC_LOAD_XMM_S(&MKL_DC_CC(i,j+0));
#if !defined(MKL_DC_BETA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_beta, xmm_c0);
#endif
#if defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_ADD_XMM_S(xmm_temp0, xmm_c0);
#else
            MKL_DC_MUL_ADD_XMM_S(xmm_alpha, xmm_temp0, xmm_c0, xmm_temp);
#endif
#else
#if !defined(MKL_DC_ALPHA_ONE)
            xmm_c0 = MKL_DC_MUL_XMM_S(xmm_alpha, xmm_temp0);
#else
            xmm_c0 = xmm_temp0;
#endif
#endif
            MKL_DC_STORE_XMM_S(&MKL_DC_CC(i,j+0),xmm_c0);
        }
    }
}

#endif
#endif

#undef MKL_DC_AA
#undef MKL_DC_BB
#undef MKL_DC_CC
#undef MKL_DC_FNAME_GEMM_KERNEL
