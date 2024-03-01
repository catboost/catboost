/*******************************************************************************
* INTEL CONFIDENTIAL
* Copyright 1996-2023 Intel Corporation.
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
 * ALGORITHM DESCRIPTION:
 *  *
 *  *   Compute 0.5*[log(1+x)-log(1-x)], using small table
 *  *   lookups that map to AVX3 permute instructions
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_atanhf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_atanhf16_ha

__svml_atanhf16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_satanh_ha_data_internal_avx512(%rip), %zmm7

/* round reciprocals to 1+5b mantissas */
        vmovups   384+__svml_satanh_ha_data_internal_avx512(%rip), %zmm2
        vmovups   448+__svml_satanh_ha_data_internal_avx512(%rip), %zmm3
        vmovaps   %zmm0, %zmm13
        vandps    320+__svml_satanh_ha_data_internal_avx512(%rip), %zmm13, %zmm9

/* 1+y */
        vaddps    {rn-sae}, %zmm7, %zmm9, %zmm12

/* 1-y */
        vsubps    {rn-sae}, %zmm9, %zmm7, %zmm11
        vxorps    %zmm9, %zmm13, %zmm14

/* Yp_high */
        vsubps    {rn-sae}, %zmm7, %zmm12, %zmm5

/* -Ym_high */
        vsubps    {rn-sae}, %zmm7, %zmm11, %zmm8

/* RcpP ~ 1/Yp */
        vrcp14ps  %zmm12, %zmm15

/* RcpM ~ 1/Ym */
        vrcp14ps  %zmm11, %zmm0

/* input outside (-1, 1) ? */
        vcmpps    $21, {sae}, %zmm7, %zmm9, %k0
        vpaddd    %zmm2, %zmm15, %zmm1
        vpaddd    %zmm2, %zmm0, %zmm4

/* Yp_low */
        vsubps    {rn-sae}, %zmm5, %zmm9, %zmm6
        vandps    %zmm3, %zmm1, %zmm10
        vandps    %zmm3, %zmm4, %zmm15

/* Ym_low */
        vaddps    {rn-sae}, %zmm8, %zmm9, %zmm8

/* Reduced argument: Rp = (RcpP*Yp - 1)+RcpP*Yp_low */
        vfmsub213ps {rn-sae}, %zmm7, %zmm10, %zmm12

/* exponents */
        vgetexpps {sae}, %zmm10, %zmm1

/* Table lookups */
        vmovups   __svml_satanh_ha_data_internal_avx512(%rip), %zmm9

/* Reduced argument: Rm = (RcpM*Ym - 1)+RcpM*Ym_low */
        vfmsub231ps {rn-sae}, %zmm15, %zmm11, %zmm7
        vmovups   128+__svml_satanh_ha_data_internal_avx512(%rip), %zmm11
        vfmadd231ps {rn-sae}, %zmm10, %zmm6, %zmm12
        vmovups   192+__svml_satanh_ha_data_internal_avx512(%rip), %zmm0
        vgetexpps {sae}, %zmm15, %zmm2
        vfnmadd231ps {rn-sae}, %zmm15, %zmm8, %zmm7

/* Prepare table index */
        vpsrld    $18, %zmm10, %zmm6
        vpsrld    $18, %zmm15, %zmm5
        vmovups   64+__svml_satanh_ha_data_internal_avx512(%rip), %zmm10
        vmovups   640+__svml_satanh_ha_data_internal_avx512(%rip), %zmm15

/* Km-Kp */
        vsubps    {rn-sae}, %zmm1, %zmm2, %zmm4
        vmovups   576+__svml_satanh_ha_data_internal_avx512(%rip), %zmm1
        kmovw     %k0, %edx
        vmovaps   %zmm6, %zmm3
        vpermi2ps %zmm10, %zmm9, %zmm3
        vpermt2ps %zmm10, %zmm5, %zmm9
        vpermi2ps %zmm0, %zmm11, %zmm6
        vpermt2ps %zmm0, %zmm5, %zmm11

/* table values */
        vsubps    {rn-sae}, %zmm3, %zmm9, %zmm3

/* K*L2H + Th */
        vmovups   704+__svml_satanh_ha_data_internal_avx512(%rip), %zmm5

/* polynomials */
        vmovups   512+__svml_satanh_ha_data_internal_avx512(%rip), %zmm10
        vsubps    {rn-sae}, %zmm6, %zmm11, %zmm8

/* K*L2L + Tl */
        vmovups   768+__svml_satanh_ha_data_internal_avx512(%rip), %zmm6
        vfmadd231ps {rn-sae}, %zmm4, %zmm5, %zmm3

/* Rp^2 */
        vmulps    {rn-sae}, %zmm12, %zmm12, %zmm5
        vfmadd213ps {rn-sae}, %zmm8, %zmm6, %zmm4

/* Rm^2 */
        vmulps    {rn-sae}, %zmm7, %zmm7, %zmm8

/* (K*L2H+Th) + Rph */
        vaddps    {rn-sae}, %zmm3, %zmm12, %zmm6
        vmovaps   %zmm1, %zmm2
        vfmadd231ps {rn-sae}, %zmm12, %zmm10, %zmm2
        vfmadd231ps {rn-sae}, %zmm7, %zmm10, %zmm1

/* -Rp_high */
        vsubps    {rn-sae}, %zmm6, %zmm3, %zmm3

/* (K*L2H+Th) + (Rph - Rmh) */
        vsubps    {rn-sae}, %zmm7, %zmm6, %zmm0
        vfmadd213ps {rn-sae}, %zmm15, %zmm12, %zmm2
        vfmadd213ps {rn-sae}, %zmm15, %zmm7, %zmm1

/* Rpl */
        vaddps    {rn-sae}, %zmm3, %zmm12, %zmm12

/* -Rm_high */
        vsubps    {rn-sae}, %zmm6, %zmm0, %zmm3

/* (K*L2L + Tl) + Rp^2*PolyP */
        vfmadd213ps {rn-sae}, %zmm4, %zmm5, %zmm2

/* Rpl - Rm^2*PolyM */
        vfnmadd213ps {rn-sae}, %zmm12, %zmm8, %zmm1

/* Rml */
        vaddps    {rn-sae}, %zmm3, %zmm7, %zmm3

/* low part */
        vaddps    {rn-sae}, %zmm1, %zmm2, %zmm7
        vorps     832+__svml_satanh_ha_data_internal_avx512(%rip), %zmm14, %zmm1
        vsubps    {rn-sae}, %zmm3, %zmm7, %zmm14
        vaddps    {rn-sae}, %zmm14, %zmm0, %zmm0
        vmulps    {rn-sae}, %zmm1, %zmm0, %zmm0
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm13, 64(%rsp)
        vmovups   %zmm0, 128(%rsp)
        je        .LBL_1_2


        xorl      %eax, %eax


        vzeroupper
        kmovw     %k4, 24(%rsp)
        kmovw     %k5, 16(%rsp)
        kmovw     %k6, 8(%rsp)
        kmovw     %k7, (%rsp)
        movq      %rsi, 40(%rsp)
        movq      %rdi, 32(%rsp)
        movq      %r12, 56(%rsp)
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 48(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_7:

        btl       %r12d, %r13d
        jc        .LBL_1_10

.LBL_1_8:

        incl      %r12d
        cmpl      $16, %r12d
        jl        .LBL_1_7


        kmovw     24(%rsp), %k4
	.cfi_restore 122
        kmovw     16(%rsp), %k5
	.cfi_restore 123
        kmovw     8(%rsp), %k6
	.cfi_restore 124
        kmovw     (%rsp), %k7
	.cfi_restore 125
        vmovups   128(%rsp), %zmm0
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_2
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_10:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi

        call      __svml_satanh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanhf16_ha,@function
	.size	__svml_atanhf16_ha,.-__svml_atanhf16_ha
..LN__svml_atanhf16_ha.0:

.L_2__routine_start___svml_satanh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_satanh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    2(%rdi), %edx
        movss     (%rdi), %xmm1
        andl      $32640, %edx
        movb      3(%rdi), %al
        andb      $127, %al
        movss     %xmm1, -8(%rsp)
        movb      %al, -5(%rsp)
        cmpl      $32640, %edx
        je        .LBL_2_6


        cmpl      $1065353216, -8(%rsp)
        jne       .LBL_2_4


        divss     4+__satanh_ha__imlsAtanhTab(%rip), %xmm1
        movss     %xmm1, (%rsi)
        movl      $2, %eax
        ret

.LBL_2_4:

        movss     8+__satanh_ha__imlsAtanhTab(%rip), %xmm0
        movl      $1, %eax
        mulss     4+__satanh_ha__imlsAtanhTab(%rip), %xmm0
        movss     %xmm0, (%rsi)


        ret

.LBL_2_6:

        cmpl      $2139095040, -8(%rsp)
        jne       .LBL_2_8


        movss     4+__satanh_ha__imlsAtanhTab(%rip), %xmm0
        movl      $1, %eax
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_8:

        mulss     (%rdi), %xmm1
        xorl      %eax, %eax
        movss     %xmm1, (%rsi)


        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_satanh_ha_cout_rare_internal,@function
	.size	__svml_satanh_ha_cout_rare_internal,.-__svml_satanh_ha_cout_rare_internal
..LN__svml_satanh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_satanh_ha_data_internal_avx512:
	.long	0
	.long	1023148032
	.long	1031307264
	.long	1035436032
	.long	1039220736
	.long	1041539072
	.long	1043333120
	.long	1045078016
	.long	1046773760
	.long	1048428544
	.long	1049313280
	.long	1050099712
	.long	1050873856
	.long	1051627520
	.long	1052364800
	.long	1053085696
	.long	1053794304
	.long	1054482432
	.long	1055162368
	.long	1055825920
	.long	1056477184
	.long	1057040384
	.long	1057353728
	.long	1057662976
	.long	1057964032
	.long	1058260992
	.long	1058553856
	.long	1058840576
	.long	1059123200
	.long	1059399680
	.long	1059672064
	.long	1059940352
	.long	0
	.long	925287326
	.long	3090802686
	.long	928156389
	.long	3078132181
	.long	942242832
	.long	3083833176
	.long	3092427142
	.long	3045295702
	.long	940324527
	.long	3089323092
	.long	945994465
	.long	3085466567
	.long	3078914384
	.long	3072337169
	.long	927865605
	.long	3093041984
	.long	947354573
	.long	3053684310
	.long	936642948
	.long	940531631
	.long	941968204
	.long	946194506
	.long	3086005293
	.long	943635681
	.long	943465747
	.long	3080925892
	.long	3078186319
	.long	3093311347
	.long	3061074424
	.long	934582639
	.long	939563115
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.type	__svml_satanh_ha_data_internal_avx512,@object
	.size	__svml_satanh_ha_data_internal_avx512,896
	.align 4
__satanh_ha__imlsAtanhTab:
	.long	1065353216
	.long	0
	.long	2139095040
	.type	__satanh_ha__imlsAtanhTab,@object
	.size	__satanh_ha__imlsAtanhTab,12
      	.section        .note.GNU-stack,"",@progbits
