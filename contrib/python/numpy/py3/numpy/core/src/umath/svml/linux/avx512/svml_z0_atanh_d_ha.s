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
.L_2__routine_start___svml_atanh8_ha_z0_0:

	.align    16,0x90
	.globl __svml_atanh8_ha

__svml_atanh8_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_datanh_ha_data_internal_avx512(%rip), %zmm3

/* round reciprocals to 1+4b mantissas */
        vmovups   384+__svml_datanh_ha_data_internal_avx512(%rip), %zmm4
        vmovups   448+__svml_datanh_ha_data_internal_avx512(%rip), %zmm10
        vandpd    320+__svml_datanh_ha_data_internal_avx512(%rip), %zmm0, %zmm14

/* 1+y */
        vaddpd    {rn-sae}, %zmm3, %zmm14, %zmm2

/* 1-y */
        vsubpd    {rn-sae}, %zmm14, %zmm3, %zmm7
        vxorpd    %zmm14, %zmm0, %zmm1

/* Yp_high */
        vsubpd    {rn-sae}, %zmm3, %zmm2, %zmm8

/* -Ym_high */
        vsubpd    {rn-sae}, %zmm3, %zmm7, %zmm13

/* RcpP ~ 1/Yp */
        vrcp14pd  %zmm2, %zmm5

/* RcpM ~ 1/Ym */
        vrcp14pd  %zmm7, %zmm6

/* input outside (-1, 1) ? */
        vcmppd    $21, {sae}, %zmm3, %zmm14, %k0
        vpaddq    %zmm4, %zmm5, %zmm12
        vpaddq    %zmm4, %zmm6, %zmm11

/* Yp_low */
        vsubpd    {rn-sae}, %zmm8, %zmm14, %zmm9
        vandpd    %zmm10, %zmm12, %zmm15
        vandpd    %zmm10, %zmm11, %zmm5

/* Ym_low */
        vaddpd    {rn-sae}, %zmm13, %zmm14, %zmm14

/* Reduced argument: Rp = (RcpP*Yp - 1)+RcpP*Yp_low */
        vfmsub213pd {rn-sae}, %zmm3, %zmm15, %zmm2

/* exponents */
        vgetexppd {sae}, %zmm15, %zmm6
        vmovups   128+__svml_datanh_ha_data_internal_avx512(%rip), %zmm8

/* Table lookups */
        vmovups   __svml_datanh_ha_data_internal_avx512(%rip), %zmm10

/* polynomials */
        vmovups   512+__svml_datanh_ha_data_internal_avx512(%rip), %zmm4
        vfmadd231pd {rn-sae}, %zmm15, %zmm9, %zmm2
        vpsrlq    $48, %zmm5, %zmm9

/* Prepare table index */
        vpsrlq    $48, %zmm15, %zmm12
        vmovups   64+__svml_datanh_ha_data_internal_avx512(%rip), %zmm15
        kmovw     %k0, %edx

/* Reduced argument: Rm = (RcpM*Ym - 1)+RcpM*Ym_low */
        vmovaps   %zmm3, %zmm13
        vfmsub231pd {rn-sae}, %zmm5, %zmm7, %zmm13
        vmovups   192+__svml_datanh_ha_data_internal_avx512(%rip), %zmm7
        vfnmadd231pd {rn-sae}, %zmm5, %zmm14, %zmm13
        vgetexppd {sae}, %zmm5, %zmm5
        vmovups   704+__svml_datanh_ha_data_internal_avx512(%rip), %zmm14

/* Km-Kp */
        vsubpd    {rn-sae}, %zmm6, %zmm5, %zmm6
        vmovups   576+__svml_datanh_ha_data_internal_avx512(%rip), %zmm5
        vmovaps   %zmm12, %zmm11
        vpermi2pd %zmm7, %zmm8, %zmm12
        vpermt2pd %zmm7, %zmm9, %zmm8
        vpermi2pd %zmm15, %zmm10, %zmm11
        vpermt2pd %zmm15, %zmm9, %zmm10
        vsubpd    {rn-sae}, %zmm12, %zmm8, %zmm8
        vmovups   640+__svml_datanh_ha_data_internal_avx512(%rip), %zmm9

/* K*L2H + Th */
        vmovups   1088+__svml_datanh_ha_data_internal_avx512(%rip), %zmm12

/* K*L2L + Tl */
        vmovups   1152+__svml_datanh_ha_data_internal_avx512(%rip), %zmm15
        vmovaps   %zmm5, %zmm7
        vfmadd231pd {rn-sae}, %zmm2, %zmm4, %zmm7
        vfmadd231pd {rn-sae}, %zmm13, %zmm4, %zmm5

/* table values */
        vsubpd    {rn-sae}, %zmm11, %zmm10, %zmm4
        vfmadd213pd {rn-sae}, %zmm9, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm9, %zmm13, %zmm5
        vmovups   768+__svml_datanh_ha_data_internal_avx512(%rip), %zmm10
        vmovups   832+__svml_datanh_ha_data_internal_avx512(%rip), %zmm11
        vfmadd231pd {rn-sae}, %zmm6, %zmm12, %zmm4
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm14, %zmm13, %zmm5
        vfmadd213pd {rn-sae}, %zmm8, %zmm15, %zmm6
        vmovups   896+__svml_datanh_ha_data_internal_avx512(%rip), %zmm8

/* Rp^2 */
        vmulpd    {rn-sae}, %zmm2, %zmm2, %zmm9
        vfmadd213pd {rn-sae}, %zmm10, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm10, %zmm13, %zmm5

/* (K*L2H+Th) + Rph */
        vaddpd    {rn-sae}, %zmm4, %zmm2, %zmm10
        vfmadd213pd {rn-sae}, %zmm11, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm11, %zmm13, %zmm5

/* Rm^2 */
        vmulpd    {rn-sae}, %zmm13, %zmm13, %zmm11
        vfmadd213pd {rn-sae}, %zmm8, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm5
        vmovups   960+__svml_datanh_ha_data_internal_avx512(%rip), %zmm8

/* -Rp_high */
        vsubpd    {rn-sae}, %zmm10, %zmm4, %zmm4

/* (K*L2H+Th) + (Rph - Rmh) */
        vsubpd    {rn-sae}, %zmm13, %zmm10, %zmm3
        vfmadd213pd {rn-sae}, %zmm8, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm5
        vmovups   1024+__svml_datanh_ha_data_internal_avx512(%rip), %zmm8
        vfmadd213pd {rn-sae}, %zmm8, %zmm2, %zmm7
        vfmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm5

/* Rpl */
        vaddpd    {rn-sae}, %zmm4, %zmm2, %zmm2

/* -Rm_high */
        vsubpd    {rn-sae}, %zmm10, %zmm3, %zmm4

/* (K*L2L + Tl) + Rp^2*PolyP */
        vfmadd213pd {rn-sae}, %zmm6, %zmm9, %zmm7

/* Rpl - Rm^2*PolyM */
        vfnmadd213pd {rn-sae}, %zmm2, %zmm11, %zmm5

/* Rml */
        vaddpd    {rn-sae}, %zmm4, %zmm13, %zmm2
        vorpd     1216+__svml_datanh_ha_data_internal_avx512(%rip), %zmm1, %zmm4

/* low part */
        vaddpd    {rn-sae}, %zmm5, %zmm7, %zmm5
        vsubpd    {rn-sae}, %zmm2, %zmm5, %zmm1
        vaddpd    {rn-sae}, %zmm1, %zmm3, %zmm1
        vmulpd    {rn-sae}, %zmm4, %zmm1, %zmm1
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm1, 128(%rsp)
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
        cmpl      $8, %r12d
        jl        .LBL_1_7


        kmovw     24(%rsp), %k4
	.cfi_restore 122
        kmovw     16(%rsp), %k5
	.cfi_restore 123
        kmovw     8(%rsp), %k6
	.cfi_restore 124
        kmovw     (%rsp), %k7
	.cfi_restore 125
        vmovups   128(%rsp), %zmm1
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

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi

        call      __svml_datanh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanh8_ha,@function
	.size	__svml_atanh8_ha,.-__svml_atanh8_ha
..LN__svml_atanh8_ha.0:

.L_2__routine_start___svml_datanh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_datanh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    6(%rdi), %eax
        andl      $32752, %eax
        movsd     (%rdi), %xmm0
        movb      7(%rdi), %dl
        andb      $127, %dl
        movsd     %xmm0, -8(%rsp)
        cmpl      $32752, %eax
        je        .LBL_2_6

.LBL_2_2:

        cmpl      $0, -8(%rsp)
        jne       .LBL_2_5


        movb      %dl, -1(%rsp)
        cmpl      $1072693248, -4(%rsp)
        jne       .LBL_2_5


        divsd     8+__datanh_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        movl      $2, %eax
        ret

.LBL_2_5:

        movsd     8+__datanh_ha_CoutTab(%rip), %xmm0
        movl      $1, %eax
        mulsd     16+__datanh_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_6:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_8


        cmpl      $0, (%rdi)
        je        .LBL_2_2

.LBL_2_8:

        mulsd     %xmm0, %xmm0
        xorl      %eax, %eax
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_datanh_ha_cout_rare_internal,@function
	.size	__svml_datanh_ha_cout_rare_internal,.-__svml_datanh_ha_cout_rare_internal
..LN__svml_datanh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_datanh_ha_data_internal_avx512:
	.long	0
	.long	0
	.long	3222405120
	.long	1068436016
	.long	1848311808
	.long	1069426439
	.long	1890025472
	.long	1069940528
	.long	3348791296
	.long	1070370807
	.long	2880159744
	.long	1070688092
	.long	3256631296
	.long	1070883211
	.long	4139499520
	.long	1071069655
	.long	3971973120
	.long	1071248163
	.long	3348791296
	.long	1071419383
	.long	1605304320
	.long	1071583887
	.long	3827638272
	.long	1071693426
	.long	1584414720
	.long	1071769695
	.long	860823552
	.long	1071843287
	.long	3896934400
	.long	1071914383
	.long	643547136
	.long	1071983149
	.long	0
	.long	0
	.long	3496399314
	.long	3176377139
	.long	720371772
	.long	3173659692
	.long	1944193543
	.long	1027855304
	.long	634920691
	.long	1028268460
	.long	1664625295
	.long	3176788476
	.long	192624563
	.long	1029620349
	.long	3796653051
	.long	1028654748
	.long	3062724207
	.long	1029196786
	.long	634920691
	.long	1029317036
	.long	1913570380
	.long	1027322573
	.long	825194088
	.long	1028982125
	.long	2335489660
	.long	1025116093
	.long	2497625109
	.long	3177087936
	.long	914782743
	.long	3176833847
	.long	3743595607
	.long	1028041657
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.type	__svml_datanh_ha_data_internal_avx512,@object
	.size	__svml_datanh_ha_data_internal_avx512,1280
	.align 8
__datanh_ha_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720
	.type	__datanh_ha_CoutTab,@object
	.size	__datanh_ha_CoutTab,32
      	.section        .note.GNU-stack,"",@progbits
