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
 *  *  After computing exp(x) in high-low parts, an accurate computation is performed to obtain exp(x)-1
 *  *  Typical exp() implementation, except that:
 *  *   - tables are small (32 elements), allowing for fast gathers
 *  *   - all arguments processed in the main path
 *  *       - final VSCALEF assists branch-free design (correct overflow/underflow and special case responses)
 *  *       - a VAND is used to ensure the reduced argument |R|<2, even for large inputs
 *  *       - RZ mode used to avoid oveflow to +/-Inf for x*log2(e); helps with special case handling
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_expm1f16_ha_z0_0:

	.align    16,0x90
	.globl __svml_expm1f16_ha

__svml_expm1f16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   128+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm9
        vmovups   192+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm8
        vmovups   384+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm10
        vmovups   448+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm11

/* set Z0=max(Z0, -64.0) */
        vmovups   512+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm13
        vmovups   256+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm7
        vmovaps   %zmm0, %zmm6

/* 2^(23-4)*1.5 + x * log2(e) */
        vfmadd213ps {rn-sae}, %zmm8, %zmm6, %zmm9

/* ensure |R|<2 even for special cases */
        vmovups   576+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm0
        vcmpps    $21, {sae}, %zmm7, %zmm6, %k0

/* polynomial */
        vmovups   640+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm7

/* Z0 ~ x*log2(e), rounded to 4 fractional bits */
        vsubps    {rn-sae}, %zmm8, %zmm9, %zmm12
        vpermps   __svml_sexpm1_ha_data_internal_avx512(%rip), %zmm9, %zmm14
        vpermps   64+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm9, %zmm4
        vandps    320+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm6, %zmm5

/* R = x - Z0*log(2) */
        vfnmadd213ps {rn-sae}, %zmm6, %zmm12, %zmm10
        vmovups   768+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm8
        vmaxps    {sae}, %zmm13, %zmm12, %zmm15
        kmovw     %k0, %edx

/* scale Th */
        vscalefps {rn-sae}, %zmm15, %zmm14, %zmm3
        vmovaps   %zmm11, %zmm2
        vfnmadd213ps {rn-sae}, %zmm10, %zmm12, %zmm2

/* (Z0*L2L)_high */
        vsubps    {rn-sae}, %zmm2, %zmm10, %zmm1
        vandps    %zmm0, %zmm2, %zmm2

/* Rl = -(Z0*L2L)_low */
        vfnmadd231ps {rn-sae}, %zmm12, %zmm11, %zmm1

/* Th - 1 */
        vmovups   832+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm11
        vmovups   896+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm12
        vandps    %zmm0, %zmm1, %zmm9
        vmovups   704+__svml_sexpm1_ha_data_internal_avx512(%rip), %zmm1
        vmulps    {rn-sae}, %zmm2, %zmm2, %zmm0
        vcmpps    $17, {sae}, %zmm11, %zmm3, %k1
        vfmadd231ps {rn-sae}, %zmm2, %zmm7, %zmm1

/* Rl+Tlr */
        vaddps    {rn-sae}, %zmm4, %zmm9, %zmm10
        vaddps    {rn-sae}, %zmm12, %zmm3, %zmm7
        vfmadd213ps {rn-sae}, %zmm8, %zmm2, %zmm1

/* Rl+Tlr + Tlr*R */
        vfmadd213ps {rn-sae}, %zmm10, %zmm2, %zmm4
        vblendmps %zmm12, %zmm3, %zmm13{%k1}
        vblendmps %zmm3, %zmm12, %zmm15{%k1}

/* polynomial + Tlr*(1+R) */
        vfmadd213ps {rn-sae}, %zmm4, %zmm0, %zmm1
        vsubps    {rn-sae}, %zmm7, %zmm13, %zmm14

/* Th-1 + (Th*R)_high */
        vmovaps   %zmm7, %zmm0
        vfmadd231ps {rn-sae}, %zmm3, %zmm2, %zmm0
        vaddps    {rn-sae}, %zmm15, %zmm14, %zmm8

/* (Th*R)_low */
        vsubps    {rn-sae}, %zmm7, %zmm0, %zmm4
        vfmsub231ps {rn-sae}, %zmm3, %zmm2, %zmm4

/* Tl1 + Th*(polynomial + Tlr*(1+R)) */
        vfmadd213ps {rn-sae}, %zmm8, %zmm1, %zmm3

/* (Th*R)_low + Tl1 + Th*(polynomial + Tlr*(1+R)) */
        vaddps    {rn-sae}, %zmm4, %zmm3, %zmm3
        vaddps    {rn-sae}, %zmm3, %zmm0, %zmm0
        vorps     %zmm5, %zmm0, %zmm0
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

        vmovups   %zmm6, 64(%rsp)
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

        call      __svml_sexpm1_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_expm1f16_ha,@function
	.size	__svml_expm1f16_ha,.-__svml_expm1f16_ha
..LN__svml_expm1f16_ha.0:

.L_2__routine_start___svml_sexpm1_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_sexpm1_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movss     (%rdi), %xmm7
        pxor      %xmm0, %xmm0
        comiss    %xmm7, %xmm0
        ja        .LBL_2_18


        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        shrl      $7, %edx
        cmpl      $255, %edx
        je        .LBL_2_19


        cvtss2sd  %xmm7, %xmm7
        cmpl      $74, %edx
        jle       .LBL_2_16


        movsd     1080+_vmldExpHATab(%rip), %xmm0
        comisd    %xmm7, %xmm0
        jb        .LBL_2_15


        comisd    1096+_vmldExpHATab(%rip), %xmm7
        jb        .LBL_2_14


        movsd     1024+_vmldExpHATab(%rip), %xmm1
        movaps    %xmm7, %xmm6
        mulsd     %xmm7, %xmm1
        lea       _vmldExpHATab(%rip), %r10
        movsd     %xmm1, -16(%rsp)
        movsd     -16(%rsp), %xmm2
        movsd     1072+_vmldExpHATab(%rip), %xmm1
        movsd     1136+_vmldExpHATab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        addsd     1032+_vmldExpHATab(%rip), %xmm2
        movsd     %xmm2, -8(%rsp)
        movsd     -8(%rsp), %xmm3
        movl      -8(%rsp), %r8d
        movl      %r8d, %ecx
        andl      $63, %r8d
        subsd     1032+_vmldExpHATab(%rip), %xmm3
        movsd     %xmm3, -16(%rsp)
        lea       1(%r8,%r8), %r9d
        movsd     -16(%rsp), %xmm5
        lea       (%r8,%r8), %edi
        movsd     -16(%rsp), %xmm4
        mulsd     1112+_vmldExpHATab(%rip), %xmm4
        mulsd     1104+_vmldExpHATab(%rip), %xmm5
        subsd     %xmm4, %xmm6
        shrl      $6, %ecx
        subsd     %xmm5, %xmm6
        comisd    1088+_vmldExpHATab(%rip), %xmm7
        mulsd     %xmm6, %xmm1
        movsd     (%r10,%rdi,8), %xmm0
        lea       1023(%rcx), %edx
        addsd     1064+_vmldExpHATab(%rip), %xmm1
        mulsd     %xmm6, %xmm1
        addsd     1056+_vmldExpHATab(%rip), %xmm1
        mulsd     %xmm6, %xmm1
        addsd     1048+_vmldExpHATab(%rip), %xmm1
        mulsd     %xmm6, %xmm1
        addsd     1040+_vmldExpHATab(%rip), %xmm1
        mulsd     %xmm6, %xmm1
        mulsd     %xmm6, %xmm1
        addsd     %xmm6, %xmm1
        addsd     (%r10,%r9,8), %xmm1
        mulsd     %xmm0, %xmm1
        jb        .LBL_2_10


        andl      $2047, %edx
        addsd     %xmm0, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_9


        movzwl    1142+_vmldExpHATab(%rip), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -18(%rsp)
        movsd     -24(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_9:

        decl      %edx
        andl      $2047, %edx
        movzwl    -18(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -18(%rsp)
        movsd     -24(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        mulsd     1152+_vmldExpHATab(%rip), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_10:

        addl      $1083, %ecx
        andl      $2047, %ecx
        movl      %ecx, %eax
        movzwl    -18(%rsp), %edx
        shll      $4, %eax
        andl      $-32753, %edx
        orl       %eax, %edx
        movw      %dx, -18(%rsp)
        movsd     -24(%rsp), %xmm3
        mulsd     %xmm3, %xmm1
        mulsd     %xmm0, %xmm3
        movaps    %xmm1, %xmm2
        addsd     %xmm3, %xmm2
        cmpl      $50, %ecx
        ja        .LBL_2_12


        movsd     1160+_vmldExpHATab(%rip), %xmm0
        mulsd     %xmm2, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        jmp       .LBL_2_13

.LBL_2_12:

        movsd     %xmm2, -72(%rsp)
        movsd     -72(%rsp), %xmm0
        subsd     %xmm0, %xmm3
        movsd     %xmm3, -64(%rsp)
        movsd     -64(%rsp), %xmm2
        addsd     %xmm1, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm1
        mulsd     1168+_vmldExpHATab(%rip), %xmm1
        movsd     %xmm1, -56(%rsp)
        movsd     -72(%rsp), %xmm4
        movsd     -56(%rsp), %xmm3
        addsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)
        movsd     -48(%rsp), %xmm6
        movsd     -56(%rsp), %xmm5
        subsd     %xmm5, %xmm6
        movsd     %xmm6, -40(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     -40(%rsp), %xmm7
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -32(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -32(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -32(%rsp)
        movsd     -40(%rsp), %xmm11
        mulsd     1160+_vmldExpHATab(%rip), %xmm11
        movsd     %xmm11, -40(%rsp)
        movsd     -32(%rsp), %xmm12
        mulsd     1160+_vmldExpHATab(%rip), %xmm12
        movsd     %xmm12, -32(%rsp)
        movsd     -40(%rsp), %xmm14
        movsd     -32(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        cvtsd2ss  %xmm14, %xmm14
        movss     %xmm14, (%rsi)

.LBL_2_13:

        movl      $4, %eax
        ret

.LBL_2_14:

        movsd     1120+_vmldExpHATab(%rip), %xmm0
        movl      $4, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_15:

        movsd     1128+_vmldExpHATab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_16:

        movsd     1144+_vmldExpHATab(%rip), %xmm0
        addsd     %xmm7, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)


        ret

.LBL_2_18:

        movl      $-1082130432, (%rsi)
        ret

.LBL_2_19:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_21

.LBL_2_20:

        mulss     %xmm7, %xmm7
        movss     %xmm7, (%rsi)
        ret

.LBL_2_21:

        testl     $8388607, (%rdi)
        jne       .LBL_2_20


        movsd     1136+_vmldExpHATab(%rip), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sexpm1_ha_cout_rare_internal,@function
	.size	__svml_sexpm1_ha_cout_rare_internal,.-__svml_sexpm1_ha_cout_rare_internal
..LN__svml_sexpm1_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sexpm1_ha_data_internal_avx512:
	.long	1065353216
	.long	1065724611
	.long	1066112450
	.long	1066517459
	.long	1066940400
	.long	1067382066
	.long	1067843287
	.long	1068324927
	.long	1068827891
	.long	1069353124
	.long	1069901610
	.long	1070474380
	.long	1071072509
	.long	1071697119
	.long	1072349383
	.long	1073030525
	.long	0
	.long	860277610
	.long	2991457809
	.long	860562562
	.long	856238081
	.long	857441778
	.long	3003456168
	.long	2986372182
	.long	848495278
	.long	3003311522
	.long	833868005
	.long	835836658
	.long	2994528642
	.long	2981408986
	.long	2983366846
	.long	833659207
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3263168512
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1026207428
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
	.long	1042984644
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
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.type	__svml_sexpm1_ha_data_internal_avx512,@object
	.size	__svml_sexpm1_ha_data_internal_avx512,960
	.align 32
_vmldExpHATab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	2631457885
	.long	3161546771
	.long	3541402996
	.long	1072716208
	.long	896005651
	.long	1015861842
	.long	410360776
	.long	1072727877
	.long	1642514529
	.long	1012987726
	.long	1828292879
	.long	1072739672
	.long	1568897901
	.long	1016568486
	.long	852742562
	.long	1072751596
	.long	1882168529
	.long	1010744893
	.long	3490863953
	.long	1072763649
	.long	707771662
	.long	3163903570
	.long	2930322912
	.long	1072775834
	.long	3117806614
	.long	3163670819
	.long	1014845819
	.long	1072788152
	.long	3936719688
	.long	3162512149
	.long	3949972341
	.long	1072800603
	.long	1058231231
	.long	1015777676
	.long	828946858
	.long	1072813191
	.long	1044000608
	.long	1016786167
	.long	2288159958
	.long	1072825915
	.long	1151779725
	.long	1015705409
	.long	1853186616
	.long	1072838778
	.long	3819481236
	.long	1016499965
	.long	1709341917
	.long	1072851781
	.long	2552227826
	.long	1015039787
	.long	4112506593
	.long	1072864925
	.long	1829350193
	.long	1015216097
	.long	2799960843
	.long	1072878213
	.long	1913391796
	.long	1015756674
	.long	171030293
	.long	1072891646
	.long	1303423926
	.long	1015238005
	.long	2992903935
	.long	1072905224
	.long	1574172746
	.long	1016061241
	.long	926591435
	.long	1072918951
	.long	3427487848
	.long	3163704045
	.long	887463927
	.long	1072932827
	.long	1049900754
	.long	3161575912
	.long	1276261410
	.long	1072946854
	.long	2804567149
	.long	1015390024
	.long	569847338
	.long	1072961034
	.long	1209502043
	.long	3159926671
	.long	1617004845
	.long	1072975368
	.long	1623370769
	.long	1011049453
	.long	3049340112
	.long	1072989858
	.long	3667985273
	.long	1013894369
	.long	3577096743
	.long	1073004506
	.long	3145379760
	.long	1014403278
	.long	1990012071
	.long	1073019314
	.long	7447438
	.long	3163526196
	.long	1453150082
	.long	1073034283
	.long	3171891295
	.long	3162037958
	.long	917841882
	.long	1073049415
	.long	419288974
	.long	1016280325
	.long	3712504873
	.long	1073064711
	.long	3793507337
	.long	1016095713
	.long	363667784
	.long	1073080175
	.long	728023093
	.long	1016345318
	.long	2956612997
	.long	1073095806
	.long	1005538728
	.long	3163304901
	.long	2186617381
	.long	1073111608
	.long	2018924632
	.long	3163803357
	.long	1719614413
	.long	1073127582
	.long	3210617384
	.long	3163796463
	.long	1013258799
	.long	1073143730
	.long	3094194670
	.long	3160631279
	.long	3907805044
	.long	1073160053
	.long	2119843535
	.long	3161988964
	.long	1447192521
	.long	1073176555
	.long	508946058
	.long	3162904882
	.long	1944781191
	.long	1073193236
	.long	3108873501
	.long	3162190556
	.long	919555682
	.long	1073210099
	.long	2882956373
	.long	1013312481
	.long	2571947539
	.long	1073227145
	.long	4047189812
	.long	3163777462
	.long	2604962541
	.long	1073244377
	.long	3631372142
	.long	3163870288
	.long	1110089947
	.long	1073261797
	.long	3253791412
	.long	1015920431
	.long	2568320822
	.long	1073279406
	.long	1509121860
	.long	1014756995
	.long	2966275557
	.long	1073297207
	.long	2339118633
	.long	3160254904
	.long	2682146384
	.long	1073315202
	.long	586480042
	.long	3163702083
	.long	2191782032
	.long	1073333393
	.long	730975783
	.long	1014083580
	.long	2069751141
	.long	1073351782
	.long	576856675
	.long	3163014404
	.long	2990417245
	.long	1073370371
	.long	3552361237
	.long	3163667409
	.long	1434058175
	.long	1073389163
	.long	1853053619
	.long	1015310724
	.long	2572866477
	.long	1073408159
	.long	2462790535
	.long	1015814775
	.long	3092190715
	.long	1073427362
	.long	1457303226
	.long	3159737305
	.long	4076559943
	.long	1073446774
	.long	950899508
	.long	3160987380
	.long	2420883922
	.long	1073466398
	.long	174054861
	.long	1014300631
	.long	3716502172
	.long	1073486235
	.long	816778419
	.long	1014197934
	.long	777507147
	.long	1073506289
	.long	3507050924
	.long	1015341199
	.long	3706687593
	.long	1073526560
	.long	1821514088
	.long	1013410604
	.long	1242007932
	.long	1073547053
	.long	1073740399
	.long	3163532637
	.long	3707479175
	.long	1073567768
	.long	2789017511
	.long	1014276997
	.long	64696965
	.long	1073588710
	.long	3586233004
	.long	1015962192
	.long	863738719
	.long	1073609879
	.long	129252895
	.long	3162690849
	.long	3884662774
	.long	1073631278
	.long	1614448851
	.long	1014281732
	.long	2728693978
	.long	1073652911
	.long	2413007344
	.long	3163551506
	.long	3999357479
	.long	1073674779
	.long	1101668360
	.long	1015989180
	.long	1533953344
	.long	1073696886
	.long	835814894
	.long	1015702697
	.long	2174652632
	.long	1073719233
	.long	1301400989
	.long	1014466875
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
	.long	0
	.long	1071644672
	.long	1431652600
	.long	1069897045
	.long	1431670732
	.long	1067799893
	.long	984555731
	.long	1065423122
	.long	472530941
	.long	1062650218
	.long	3758096384
	.long	1079389762
	.long	3758096384
	.long	3226850697
	.long	2147483648
	.long	3227123254
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	1
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	1073741824
	.long	0
	.long	1009778688
	.long	0
	.long	1106771968
	.type	_vmldExpHATab,@object
	.size	_vmldExpHATab,1176
	.align 4
.L_2il0floatpacket.76:
	.long	0xbf800000
	.type	.L_2il0floatpacket.76,@object
	.size	.L_2il0floatpacket.76,4
      	.section        .note.GNU-stack,"",@progbits
