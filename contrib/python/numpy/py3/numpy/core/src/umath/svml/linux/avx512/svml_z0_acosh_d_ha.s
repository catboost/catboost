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
 *  *   Compute log(x+sqrt(x*x-1)) using RSQRT14/RSQRT28 for starting the
 *  *   square root approximation, and small table lookups for log (mapping to
 *  *   AVX3 permute instructions).
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_acosh8_ha_z0_0:

	.align    16,0x90
	.globl __svml_acosh8_ha

__svml_acosh8_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp

/* x^2 */
        vmulpd    {rn-sae}, %zmm0, %zmm0, %zmm1
        vmovups   256+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm7

/* polynomial computation for small inputs */
        vmovups   576+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm8
        vmovups   640+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm11
        vmovups   704+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm9
        vcmppd    $18, {sae}, %zmm7, %zmm0, %k1

/* x^2 - 1 */
        vsubpd    {rn-sae}, %zmm7, %zmm1, %zmm5
        kmovw     %k1, %eax
        vsubpd    {rn-sae}, %zmm5, %zmm1, %zmm2

/* (x^2)_low */
        vmovaps   %zmm0, %zmm12
        vfmsub213pd {rn-sae}, %zmm1, %zmm0, %zmm12

/* -One_l */
        vsubpd    {rn-sae}, %zmm7, %zmm2, %zmm3

/* not a very small input ? */
        vmovups   384+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm2

/* Yl0 */
        vaddpd    {rn-sae}, %zmm3, %zmm12, %zmm6

/* very large inputs ? */
        vmovups   448+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm3

/* Yh */
        vaddpd    {rn-sae}, %zmm6, %zmm5, %zmm1
        vcmppd    $21, {sae}, %zmm3, %zmm0, %k2
        vrsqrt14pd %zmm1, %zmm15
        vcmppd    $21, {sae}, %zmm2, %zmm1, %k3
        vmovups   896+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm3
        vfmadd231pd {rn-sae}, %zmm1, %zmm8, %zmm11

/* Sh ~sqrt(-1+x^2) */
        vmulpd    {rn-sae}, %zmm15, %zmm1, %zmm13
        vfmadd213pd {rn-sae}, %zmm9, %zmm1, %zmm11

/* Ylh */
        vsubpd    {rn-sae}, %zmm5, %zmm1, %zmm4

/* Sh+x */
        vaddpd    {rn-sae}, %zmm0, %zmm13, %zmm8

/* out of range inputs? */
        vmovups   512+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm5
        vmulpd    {rn-sae}, %zmm1, %zmm11, %zmm12
        vcmppd    $22, {sae}, %zmm5, %zmm0, %k0
        vmovups   960+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm5

/* Yl */
        vsubpd    {rn-sae}, %zmm4, %zmm6, %zmm14
        vmovups   832+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm4
        kmovw     %k0, %edx

/* (Yh*R0)_low */
        vmovaps   %zmm1, %zmm10
        orl       %eax, %edx
        vfmsub213pd {rn-sae}, %zmm13, %zmm15, %zmm10

/* rel. error term: Eh=1-Sh*R0 */
        vmovaps   %zmm7, %zmm6
        vfnmadd231pd {rn-sae}, %zmm13, %zmm15, %zmm6

/* Sl = (Yh*R0)_low+(R0*Yl) */
        vfmadd213pd {rn-sae}, %zmm10, %zmm15, %zmm14

/* rel. error term: Eh=(1-Sh*R0)-Sl*R0 */
        vfnmadd231pd {rn-sae}, %zmm14, %zmm15, %zmm6

/*
 * sqrt(1+x^2) ~ Sh + Sl + Sh*Eh*poly_s
 * poly_s = c1+c2*Eh+c3*Eh^2
 */
        vmovups   768+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm15

/* Sh*Eh */
        vmulpd    {rn-sae}, %zmm6, %zmm13, %zmm10
        vfmadd231pd {rn-sae}, %zmm6, %zmm15, %zmm4
        vmovups   1600+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm15
        vfmadd213pd {rn-sae}, %zmm3, %zmm6, %zmm4

/* Shh */
        vsubpd    {rn-sae}, %zmm0, %zmm8, %zmm3
        vfmadd213pd {rn-sae}, %zmm5, %zmm6, %zmm4

/* Shl */
        vsubpd    {rn-sae}, %zmm3, %zmm13, %zmm6

/* Sl + Sh*Eh*poly_s */
        vfmadd213pd {rn-sae}, %zmm14, %zmm4, %zmm10

/* polynomial computation for small inputs */
        vaddpd    {rn-sae}, %zmm10, %zmm13, %zmm14

/* Xin0+Sl+Sh*Eh*poly_s ~ x+sqrt(1+x^2) */
        vaddpd    {rn-sae}, %zmm10, %zmm8, %zmm11
        vfmadd213pd {rn-sae}, %zmm10, %zmm12, %zmm14

/* Sl_high */
        vsubpd    {rn-sae}, %zmm8, %zmm11, %zmm3
        vaddpd    {rn-sae}, %zmm13, %zmm14, %zmm12

/* Sl_l */
        vsubpd    {rn-sae}, %zmm3, %zmm10, %zmm5

/* fixup for very large inputs */
        vmovups   1280+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm13
        vmovups   128+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm14
        vmovups   1472+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm3
        vmovups   1344+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm10
        vmulpd    {rn-sae}, %zmm13, %zmm0, %zmm11{%k2}
        vmovups   1536+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm13

/* Xin_low */
        vaddpd    {rn-sae}, %zmm6, %zmm5, %zmm9
        vrcp14pd  %zmm11, %zmm4

/* fixup for very large inputs */
        vxorpd    %zmm9, %zmm9, %zmm9{%k2}

/* round reciprocal to 1+4b mantissas */
        vpaddq    1152+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm4, %zmm8

/* Table lookups */
        vmovups   __svml_dacosh_ha_data_internal_avx512(%rip), %zmm4
        vandpd    1216+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm8, %zmm8

/* reduced argument for log(): (Rcp*Xin-1)+Rcp*Xin_low */
        vfmsub231pd {rn-sae}, %zmm8, %zmm11, %zmm7

/* exponents */
        vgetexppd {sae}, %zmm8, %zmm5

/* Prepare table index */
        vpsrlq    $48, %zmm8, %zmm11
        vsubpd    {rn-sae}, %zmm10, %zmm5, %zmm5{%k2}
        vpermt2pd 192+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm11, %zmm14
        vpermt2pd 64+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm11, %zmm4
        vmovups   1728+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm10

/* -K*L2H + Th */
        vmovups   1984+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm11
        vmovaps   %zmm8, %zmm6
        vfmadd213pd {rn-sae}, %zmm7, %zmm9, %zmm6

/* Tl+Rcp*Xin_low */
        vfmadd213pd {rn-sae}, %zmm14, %zmm9, %zmm8

/* polynomials */
        vmovups   1408+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm9
        vfnmadd231pd {rn-sae}, %zmm5, %zmm11, %zmm4
        vmovups   1856+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm14
        vfmadd231pd {rn-sae}, %zmm6, %zmm9, %zmm3
        vmovups   1664+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm9
        vfmadd213pd {rn-sae}, %zmm13, %zmm6, %zmm3

/* -K*L2L + Tl + Rcp*Xin_low */
        vmovups   2048+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm13
        vfmadd213pd {rn-sae}, %zmm15, %zmm6, %zmm3
        vfnmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm5
        vmovups   1792+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm8
        vmovups   1920+__svml_dacosh_ha_data_internal_avx512(%rip), %zmm15
        vfmadd213pd {rn-sae}, %zmm9, %zmm6, %zmm3

/* (K*L2H+Th) + Rh0_h */
        vaddpd    {rn-sae}, %zmm4, %zmm7, %zmm9
        vfmadd213pd {rn-sae}, %zmm10, %zmm6, %zmm3

/* -R_high */
        vsubpd    {rn-sae}, %zmm9, %zmm4, %zmm4
        vfmadd213pd {rn-sae}, %zmm8, %zmm6, %zmm3

/* R^2 */
        vmulpd    {rn-sae}, %zmm6, %zmm6, %zmm8
        vfmadd213pd {rn-sae}, %zmm14, %zmm6, %zmm3

/* Rl */
        vaddpd    {rn-sae}, %zmm4, %zmm7, %zmm7
        vfmadd213pd {rn-sae}, %zmm15, %zmm6, %zmm3

/* (K*L2L + Tl) + R^2*Poly */
        vfmadd213pd {rn-sae}, %zmm5, %zmm8, %zmm3
        vaddpd    {rn-sae}, %zmm7, %zmm3, %zmm3
        vaddpd    {rn-sae}, %zmm3, %zmm9, %zmm12{%k3}
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm12, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm12, 128(%rsp)
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
        vmovups   128(%rsp), %zmm12
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

        call      __svml_dacosh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_acosh8_ha,@function
	.size	__svml_acosh8_ha,.-__svml_acosh8_ha
..LN__svml_acosh8_ha.0:

.L_2__routine_start___svml_dacosh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_dacosh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    6(%rdi), %eax
        andl      $32752, %eax
        movsd     (%rdi), %xmm0
        cmpl      $32752, %eax
        je        .LBL_2_9

.LBL_2_2:

        movsd     %xmm0, -8(%rsp)


        cmpl      $0, -8(%rsp)
        jne       .LBL_2_8


        cmpl      $1072693248, -4(%rsp)
        je        .LBL_2_7


        cmpl      $2146435072, -4(%rsp)
        jne       .LBL_2_8


        movsd     %xmm0, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_7:

        movq      8+__dacosh_ha_CoutTab(%rip), %rax
        movq      %rax, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_8:

        movsd     16+__dacosh_ha_CoutTab(%rip), %xmm0
        movl      $1, %eax
        mulsd     8+__dacosh_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_9:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_12


        cmpl      $0, (%rdi)
        je        .LBL_2_2

.LBL_2_12:

        mulsd     %xmm0, %xmm0
        xorl      %eax, %eax
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dacosh_ha_cout_rare_internal,@function
	.size	__svml_dacosh_ha_cout_rare_internal,.-__svml_dacosh_ha_cout_rare_internal
..LN__svml_dacosh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dacosh_ha_data_internal_avx512:
	.long	0
	.long	0
	.long	3222405120
	.long	3215919664
	.long	1848311808
	.long	3216910087
	.long	1890025472
	.long	3217424176
	.long	3348791296
	.long	3217854455
	.long	2880159744
	.long	3218171740
	.long	3256631296
	.long	3218366859
	.long	4139499520
	.long	3218553303
	.long	3971973120
	.long	3218731811
	.long	3348791296
	.long	3218903031
	.long	1605304320
	.long	3219067535
	.long	3827638272
	.long	3219177074
	.long	1584414720
	.long	3219253343
	.long	860823552
	.long	3219326935
	.long	3896934400
	.long	3219398031
	.long	643547136
	.long	3219466797
	.long	0
	.long	0
	.long	3496399314
	.long	1028893491
	.long	720371772
	.long	1026176044
	.long	1944193543
	.long	3175338952
	.long	634920691
	.long	3175752108
	.long	1664625295
	.long	1029304828
	.long	192624563
	.long	3177103997
	.long	3796653051
	.long	3176138396
	.long	3062724207
	.long	3176680434
	.long	634920691
	.long	3176800684
	.long	1913570380
	.long	3174806221
	.long	825194088
	.long	3176465773
	.long	2335489660
	.long	3172599741
	.long	2497625109
	.long	1029604288
	.long	914782743
	.long	1029350199
	.long	3743595607
	.long	3175525305
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
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1061158912
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	2890702885
	.long	3215382410
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	207003370
	.long	1068708659
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	1431623933
	.long	3217380693
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
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
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
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
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
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
	.type	__svml_dacosh_ha_data_internal_avx512,@object
	.size	__svml_dacosh_ha_data_internal_avx512,2176
	.align 8
__dacosh_ha_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720
	.type	__dacosh_ha_CoutTab,@object
	.size	__dacosh_ha_CoutTab,32
      	.section        .note.GNU-stack,"",@progbits
