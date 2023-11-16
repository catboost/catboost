/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *      SelMask = (|x| >= 0.5) ? 1 : 0;
 *      R = SelMask ? sqrt(0.5 - 0.5*|x|) : |x|
 *      acos(|x|) = SelMask ? 2*Poly(R) : (Pi/2 - Poly(R))
 *      acos(x) = sign(x) ? (Pi - acos(|x|)) : acos(|x|)
 *
 */

        .text

        .align    16,0x90
        .globl __svml_acoss32

__svml_acoss32:

        .cfi_startproc

        vmovdqu16 __svml_hacos_data_internal(%rip), %zmm30
        vmovdqu16 64+__svml_hacos_data_internal(%rip), %zmm1
        vmovdqu16 128+__svml_hacos_data_internal(%rip), %zmm31
        vmovdqu16 192+__svml_hacos_data_internal(%rip), %zmm5
        vmovdqu16 256+__svml_hacos_data_internal(%rip), %zmm29
        vmovdqu16 320+__svml_hacos_data_internal(%rip), %zmm7
        vmovdqu16 384+__svml_hacos_data_internal(%rip), %zmm9
        vmovdqu16 448+__svml_hacos_data_internal(%rip), %zmm11

        kxnord  %k7, %k7, %k7

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_acos_h:
        cmpq    $31, %rdx
        ja .loaddata
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
        jmp .funcbegin_acos_h
.loaddata:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_acos_h:

/* restore over-written constants */
        vmovdqu16 %zmm31, %zmm3

/* x^2 */
        vmulph    {rn-sae}, %zmm0, %zmm0, %zmm2

/* y = 0.5 -0.5*|x| */
        vmovaps   %zmm1, %zmm10

/*
 * No callout
 * xa = |x|
 */
        vpandd  %zmm30, %zmm0, %zmm12
        vfnmadd231ph {rn-sae}, %zmm12, %zmm1, %zmm10

/* SelMask=1 for |x|>=0.5 */
        vcmpph    $21, {sae}, %zmm1, %zmm12, %k2

/* set y = y*rsqrt(y) ~ sqrt(y) */
        vrsqrtph  %zmm10, %zmm6

/* SqrtMask=0 if y==+/-0 */
        vcmpph    $4, {sae}, %zmm5, %zmm10, %k1

/* set x2=y for |x|>=0.5 */
        vminph    {sae}, %zmm10, %zmm2, %zmm8
        vmulph    {rn-sae}, %zmm6, %zmm10, %zmm10{%k1}

/* sign(x) */
        vpxord    %zmm0, %zmm12, %zmm13

/* polynomial */
        vmovdqu16 %zmm29, %zmm0
        vmulph    {rn-sae}, %zmm11, %zmm10, %zmm12{%k2}

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm7, %zmm8, %zmm0
        vfmadd213ph {rn-sae}, %zmm9, %zmm8, %zmm0
        vpxord    %zmm3, %zmm13, %zmm4
        vsubph    {rn-sae}, %zmm4, %zmm3, %zmm3{%k2}
        vpxord    %zmm13, %zmm12, %zmm14

/* result */
        vfnmadd213ph {rn-sae}, %zmm3, %zmm14, %zmm0

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_acos_h
        ret

        .cfi_endproc

        .type	__svml_acoss32,@function
        .size	__svml_acoss32,.-__svml_acoss32

        .section .rodata, "a"
        .align 64

__svml_hacos_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x3800
	.endr
	.rept	32
        .word	0x3e48
	.endr
	.rept	32
        .word	0x0000
	.endr
	.rept	32
        .word	0x2e26
	.endr
	.rept	32
        .word	0x3144
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0xc000
	.endr
        .type	__svml_hacos_data_internal,@object
        .size	__svml_hacos_data_internal,512
	 .section        .note.GNU-stack,"",@progbits
