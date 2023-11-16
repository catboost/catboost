/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *    For pow computation sequences for log2() and exp2() are done with small tables
 *    The log2() part uses VGETEXP/VGETMANT (which treat denormals correctly)
 *    Branches are not needed for overflow/underflow:
 *    RZ mode used to prevent overflow to +/-Inf in intermediate computations
 *    Final VSCALEF properly handles overflow and underflow cases
 *    Callout is used for Inf/NaNs or x<=0
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_pows32

__svml_pows32:

        .cfi_startproc

        pushq     %rbp
        .cfi_def_cfa_offset 16
        movq      %rsp, %rbp
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp

/* x<=0 or Inf/NaN? */
        vfpclassph $223, %zmm0, %k0

/* y is Inf/NaN? */
        vfpclassph $153, %zmm1, %k1

/* mx - 1 */
        vmovups   __svml_hpow_data_internal(%rip), %zmm8

/* log polynomial */
        vmovups   64+__svml_hpow_data_internal(%rip), %zmm4
        vmovups   320+__svml_hpow_data_internal(%rip), %zmm7
        kmovd     %k0, %edx
        kmovd     %k1, %eax

/* set range mask */
        orl       %eax, %edx
        vextractf32x8 $1, %zmm0, %ymm14
        vcvtph2psx %ymm0, %zmm12
        vcvtph2psx %ymm14, %zmm10

/* GetMant(x), range [0.75, 1.5) */
        vgetmantps $11, {sae}, %zmm12, %zmm5
        vgetmantps $11, {sae}, %zmm10, %zmm6

/* GetExp(x) */
        vgetexpps {sae}, %zmm12, %zmm13
        vgetexpps {sae}, %zmm10, %zmm11

/* exponent correction */
        vgetexpps {sae}, %zmm5, %zmm9
        vgetexpps {sae}, %zmm6, %zmm3
        vsubps    {rn-sae}, %zmm8, %zmm5, %zmm12
        vsubps    {rn-sae}, %zmm8, %zmm6, %zmm10

/* apply exponent correction */
        vsubps    {rn-sae}, %zmm3, %zmm11, %zmm3
        vmovups   192+__svml_hpow_data_internal(%rip), %zmm5
        vmovups   256+__svml_hpow_data_internal(%rip), %zmm6
        vextractf32x8 $1, %zmm1, %ymm2
        vcvtph2psx %ymm2, %zmm14
        vmovups   128+__svml_hpow_data_internal(%rip), %zmm2
        vmovaps   %zmm2, %zmm8
        vfmadd231ps {rn-sae}, %zmm12, %zmm4, %zmm8
        vfmadd231ps {rn-sae}, %zmm10, %zmm4, %zmm2
        vsubps    {rn-sae}, %zmm9, %zmm13, %zmm4

/* poly */
        vfmadd213ps {rn-sae}, %zmm5, %zmm12, %zmm8
        vfmadd213ps {rn-sae}, %zmm5, %zmm10, %zmm2
        vmovdqu16 512+__svml_hpow_data_internal(%rip), %zmm5
        vfmadd213ps {rn-sae}, %zmm6, %zmm12, %zmm8
        vfmadd213ps {rn-sae}, %zmm6, %zmm10, %zmm2
        vmovdqu16 576+__svml_hpow_data_internal(%rip), %zmm6
        vfmadd213ps {rn-sae}, %zmm7, %zmm12, %zmm8
        vfmadd213ps {rn-sae}, %zmm7, %zmm10, %zmm2
        vmovups   384+__svml_hpow_data_internal(%rip), %zmm7
        vfmadd213ps {rn-sae}, %zmm7, %zmm12, %zmm8
        vfmadd213ps {rn-sae}, %zmm7, %zmm10, %zmm2

/* log2(x) */
        vfmadd213ps {rn-sae}, %zmm4, %zmm12, %zmm8
        vfmadd213ps {rn-sae}, %zmm3, %zmm10, %zmm2

/* polynomial */
        vmovdqu16 448+__svml_hpow_data_internal(%rip), %zmm10

/* y*log2(x) */
        vmulps    {rn-sae}, %zmm2, %zmm14, %zmm13
        vcvtph2psx %ymm1, %zmm15
        vmulps    {rn-sae}, %zmm8, %zmm15, %zmm9

/* reduced argument for exp2() computation */
        vreduceps $9, {sae}, %zmm13, %zmm3
        vmovdqu16 640+__svml_hpow_data_internal(%rip), %zmm8
        vreduceps $9, {sae}, %zmm9, %zmm2

/* y*log2(x) in FP16 */
        vcvtps2phx {rd-sae}, %zmm9, %ymm11

/* reduced argument in FP16 format */
        vcvtps2phx %zmm2, %ymm9
        vcvtps2phx %zmm3, %ymm4
        vcvtps2phx {rd-sae}, %zmm13, %ymm15
        vinsertf32x8 $1, %ymm4, %zmm9, %zmm7

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm5, %zmm7, %zmm10
        vfmadd213ph {rn-sae}, %zmm6, %zmm7, %zmm10
        vfmadd213ph {rn-sae}, %zmm8, %zmm7, %zmm10
        vinsertf32x8 $1, %ymm15, %zmm11, %zmm11

/* Poly*2^floor(xe) */
        vscalefph {rn-sae}, %zmm11, %zmm10, %zmm2

/* Go to special inputs processing branch */
        jne       .LBL_SPECIAL_VALUES_BRANCH
                                # LOE rbx r12 r13 r14 r15 edx zmm0 zmm1 zmm2

/* Restore registers
 * and exit the function
 */

.LBL_EXIT:
        vmovaps   %zmm2, %zmm0
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
        vmovups   %zmm1, 128(%rsp)
        vmovups   %zmm2, 192(%rsp)
                                # LOE rbx r12 r13 r14 r15 edx zmm2

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
        vmovups   192(%rsp), %zmm2

/* Go to exit */
        jmp       .LBL_EXIT
        /*  DW_CFA_expression: r12 (r12) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -240; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
        /*  DW_CFA_expression: r13 (r13) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -248; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
        /*  DW_CFA_expression: r14 (r14) (DW_OP_lit8; DW_OP_minus; DW_OP_const4s: -64; DW_OP_and; DW_OP_const4s: -256; DW_OP_plus)  */
        .cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22
                                # LOE rbx r12 r13 r14 r15 zmm2

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
        call      powf@PLT
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

        .type	__svml_pows32,@function
        .size	__svml_pows32,.-__svml_pows32

        .section .rodata, "a"
        .align 64

__svml_hpow_data_internal:
	.rept	16
        .long	0x3f800000
	.endr
	.rept	16
        .long	0xbe1ffcac
	.endr
	.rept	16
        .long	0x3e9864bc
	.endr
	.rept	16
        .long	0xbebd2851
	.endr
	.rept	16
        .long	0x3ef64dc7
	.endr
	.rept	16
        .long	0xbf389e6d
	.endr
	.rept	16
        .long	0x3fb8aa2c
	.endr
	.rept	32
        .word	0x2d12
	.endr
	.rept	32
        .word	0x332e
	.endr
	.rept	32
        .word	0x3992
	.endr
	.rept	32
        .word	0x3c00
	.endr
        .type	__svml_hpow_data_internal,@object
        .size	__svml_hpow_data_internal,704
	 .section        .note.GNU-stack,"",@progbits
