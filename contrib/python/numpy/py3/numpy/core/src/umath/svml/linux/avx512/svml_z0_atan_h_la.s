/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Absolute argument and sign: xa = |x|, sign(x)
 *   Selection mask: SelMask=1 for |x|>1.0
 *   Reciprocal:  y=RCP(xa)
 *   xa=y for |x|>1
 *   High = Pi/2 for |x|>1, 0 otherwise
 *   Result: High + xa*Poly
 *
 */

        .text

        .align    16,0x90
        .globl __svml_atans32

__svml_atans32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hatan_data_internal(%rip), %zmm30
        vmovdqu16 64+__svml_hatan_data_internal(%rip), %zmm5
        vmovdqu16 128+__svml_hatan_data_internal(%rip), %zmm31
        vmovdqu16 192+__svml_hatan_data_internal(%rip), %zmm1
        vmovdqu16 256+__svml_hatan_data_internal(%rip), %zmm28
        vmovdqu16 320+__svml_hatan_data_internal(%rip), %zmm29
        vmovdqu16 384+__svml_hatan_data_internal(%rip), %zmm2
        vmovdqu16 448+__svml_hatan_data_internal(%rip), %zmm3
        vmovdqu16 512+__svml_hatan_data_internal(%rip), %zmm4

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_atan_h:
        cmpq    $31, %rdx
        ja .loaddata_atanh
/* set up mask %k7 for masked load instruction */
        movl    $1, %eax
        movl    %edx, %ecx
        sall    %cl, %eax
        subl    $1, %eax
        kmovd   %eax, %k7
/* Constant required for masked load */
        movl    $0, %eax
        vpbroadcastw    %eax, %zmm0
        vmovdqu16 (%rdi), %zmm0{%k7}
        jmp .funcbegin_atan_h
.loaddata_atanh:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_atan_h:

        vmovdqu16 %zmm29, %zmm9

/*
 * No callout
 * xa = |x|
 */
        vpandd %zmm30, %zmm0, %zmm7

/* SelMask=1 for |x|>1.0 */
        vcmpph    $22, {sae}, %zmm5, %zmm7, %k1

/* High = Pi/2 for |x|>1, 0 otherwise */
        vpblendmw %zmm31, %zmm1, %zmm8{%k1}

/* sign(x) */
        vpxord    %zmm0, %zmm7, %zmm10

/* xa=y for |x|>1 */
        vrcpph    %zmm7, %zmm7{%k1}

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm2, %zmm7, %zmm9
        vfmadd213ph {rn-sae}, %zmm3, %zmm7, %zmm9
        vfmadd213ph {rn-sae}, %zmm4, %zmm7, %zmm9
        vfmadd213ph {rn-sae}, %zmm5, %zmm7, %zmm9

/* change sign of xa, for |x|>1 */
        vpxord %zmm28, %zmm7, %zmm6
        vmovdqu16 %zmm6, %zmm7{%k1}

/* High + xa*Poly */
        vfmadd213ph {rn-sae}, %zmm8, %zmm7, %zmm9

/* set sign */
        vpxord    %zmm10, %zmm9, %zmm0
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_atan_h
        ret

        .cfi_endproc

        .type	__svml_atans32,@function
        .size	__svml_atans32,.-__svml_atans32

        .section .rodata, "a"
        .align 64

__svml_hatan_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x3e48
	.endr
	.rept	32
        .word	0x0000
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
        .type	__svml_hatan_data_internal,@object
        .size	__svml_hatan_data_internal,576
	 .section        .note.GNU-stack,"",@progbits
