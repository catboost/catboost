/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Absolute arguments: xa = |x|
 *                       ya = |y|
 *   Argument signs: sign(x)
 *                   sign(x)
 *   High = (sgn_x)? Pi: 0
 *   Mask for difference: diff_msk = (xa<ya) ? 1 : 0;
 *   Maximum and minimum: xa1=max(xa,ya)
 *   ya1=min(ya,xa)
 *
 *   if(diff_msk) High = Pi2;
 *
 *   Get result sign: sgn_r = -sgn_x for diff_msk=1
 *   Divide: xa = ya1/xa1
 *   set branch mask if xa1 is Inf/NaN/zero
 *
 *   Result: High + xa*Poly
 *
 */

        .text

        .align    16,0x90
        .globl __svml_atan2s32

__svml_atan2s32:

        .cfi_startproc

        pushq     %rbp
        .cfi_def_cfa_offset 16
        movq      %rsp, %rbp
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp
        vmovdqu16 __svml_hatan2_data_internal(%rip), %zmm5
        vmovdqu16 384+__svml_hatan2_data_internal(%rip), %zmm13
        vmovdqu16 448+__svml_hatan2_data_internal(%rip), %zmm14
        vmovdqu16 512+__svml_hatan2_data_internal(%rip), %zmm15
        vmovaps   %zmm1, %zmm4
        vmovdqu16 64+__svml_hatan2_data_internal(%rip), %zmm1

/* xa = |x| */
        vpandd    %zmm5, %zmm4, %zmm8

/* ya = |y| */
        vpandd    %zmm5, %zmm0, %zmm7

/* diff_msk = (xa<ya) ? 1 : 0; */
        vcmpph    $17, {sae}, %zmm7, %zmm8, %k1

/* xa1=max(xa,ya) */
        vmaxph    {sae}, %zmm7, %zmm8, %zmm12

/* ya1=min(ya,xa) */
        vminph    {sae}, %zmm8, %zmm7, %zmm11

/* set branch mask if xa1 is Inf/NaN/zero */
        vfpclassph $159, %zmm12, %k0
        kmovd     %k0, %edx

/* sign(x) */
        vpxord    %zmm4, %zmm8, %zmm2

/* sgn_r = -sgn_x for diff_msk=1 */
        vpxord    256+__svml_hatan2_data_internal(%rip), %zmm2, %zmm10

/* High = (sgn_x)? Pi: 0 */
        vpsraw    $15, %zmm2, %zmm6
        vmovdqu16 %zmm10, %zmm2{%k1}

/* xa = ya1/xa1 */
        vdivph    {rn-sae}, %zmm12, %zmm11, %zmm10

/* polynomial */
        vmovdqu16 320+__svml_hatan2_data_internal(%rip), %zmm12

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm13, %zmm10, %zmm12
        vfmadd213ph {rn-sae}, %zmm14, %zmm10, %zmm12
        vfmadd213ph {rn-sae}, %zmm15, %zmm10, %zmm12
        vfmadd213ph {rn-sae}, %zmm1, %zmm10, %zmm12
        vpandd    128+__svml_hatan2_data_internal(%rip), %zmm6, %zmm9

/* if(diff_msk) High = Pi2; */
        vpblendmw 192+__svml_hatan2_data_internal(%rip), %zmm9, %zmm5{%k1}

/* set polynomial sign */
        vpxord    %zmm2, %zmm12, %zmm1

/* High + xa*Poly */
        vfmadd213ph {rn-sae}, %zmm5, %zmm10, %zmm1

/* sign(x) */
        vpxord    %zmm0, %zmm7, %zmm3

/* set sign */
        vpxord    %zmm3, %zmm1, %zmm1
        testl     %edx, %edx

/* Go to special inputs processing branch */
        jne       .LBL_SPECIAL_VALUES_BRANCH
                                # LOE rbx r12 r13 r14 r15 edx zmm0 zmm1 zmm4

/* Restore registers
 * and exit the function
 */

.LBL_EXIT:
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
        .cfi_def_cfa 7, 8
        .cfi_restore 6
        ret
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16

/* Branch to process
 * special inputs
 */

.LBL_SPECIAL_VALUES_BRANCH:
        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm4, 128(%rsp)
        vmovups   %zmm1, 192(%rsp)
                                # LOE rbx r12 r13 r14 r15 edx

        xorl      %eax, %eax
        movq      %r12, 16(%rsp)
        /*  DW_CFA_expression: r12 (r12) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -240; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 8(%rsp)
        /*  DW_CFA_expression: r13 (r13) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -248; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
        movq      %r14, (%rsp)
        /*  DW_CFA_expression: r14 (r14) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -256; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22
                                # LOE rbx r15 r12d r13d

/* Range mask
 * bits check
 */

.LBL_RANGEMASK_CHECK:
        btl       %r12d, %r13d

/* Call scalar math function */
        jc        .LBL_SCALAR_MATH_CALL
                                # LOE rbx r15 r12d r13d

/* Special inputs
 * processing loop
 */

.LBL_SPECIAL_VALUES_LOOP:
        incl      %r12d
        cmpl      $32, %r12d

/* Check bits in range mask */
        jl        .LBL_RANGEMASK_CHECK
                                # LOE rbx r15 r12d r13d

        movq      16(%rsp), %r12
        .cfi_restore 12
        movq      8(%rsp), %r13
        .cfi_restore 13
        movq      (%rsp), %r14
        .cfi_restore 14
        vmovups   192(%rsp), %zmm1

/* Go to exit */
        jmp       .LBL_EXIT
        /*  DW_CFA_expression: r12 (r12) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -240; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
        /*  DW_CFA_expression: r13 (r13) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -248; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
        /*  DW_CFA_expression: r14 (r14) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -256; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22
                                # LOE rbx r12 r13 r14 r15 zmm1

/* Scalar math fucntion call
 * to process special input
 */

.LBL_SCALAR_MATH_CALL:
        movl      %r12d, %r14d
        vmovsh    64(%rsp,%r14,2), %xmm1
        vmovsh    128(%rsp,%r14,2), %xmm2
        vcvtsh2ss %xmm1, %xmm1, %xmm0
        vcvtsh2ss %xmm2, %xmm2, %xmm1
        vzeroupper
        call      atan2f@PLT
                                # LOE rbx r14 r15 r12d r13d xmm0

        vmovaps   %xmm0, %xmm1
        vxorps    %xmm0, %xmm0, %xmm0
        vmovss    %xmm1, %xmm0, %xmm2
        vcvtss2sh %xmm2, %xmm2, %xmm3
        vmovsh    %xmm3, 192(%rsp,%r14,2)

/* Process special inputs in loop */
        jmp       .LBL_SPECIAL_VALUES_LOOP
                                # LOE rbx r15 r12d r13d
        .cfi_endproc

        .type	__svml_atan2s32,@function
        .size	__svml_atan2s32,.-__svml_atan2s32

        .section .rodata, "a"
        .align 64

__svml_hatan2_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x4248
	.endr
	.rept	32
        .word	0x3e48
	.endr
	.rept	32
        .word	0x8000
	.endr
	.rept	32
        .word	0xa528
	.endr
	.rept	32
        .word	0x3248
	.endr
	.rept	32
        .word	0xb65d
	.endr
	.rept	32
        .word	0x1f7a
	.endr
        .type	__svml_hatan2_data_internal,@object
        .size	__svml_hatan2_data_internal,576
	 .section        .note.GNU-stack,"",@progbits
