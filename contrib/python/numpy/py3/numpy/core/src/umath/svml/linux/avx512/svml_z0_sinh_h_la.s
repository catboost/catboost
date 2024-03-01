/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Compute sinh(x) as (exp(x)-exp(-x))/2,
 *   where exp is calculated as
 *   exp(M*ln2 + ln2*(j/2^k) + r) = 2^M * 2^(j/2^k) * exp(r)
 *
 *   Special cases:
 *
 *   sinh(NaN) = quiet NaN, and raise invalid exception
 *   sinh(INF) = that INF
 *   sinh(x)   = x for subnormals
 *   sinh(x) overflows for big x and returns MAXLOG+log(2)
 *
 */

        .text

        .align    16,0x90
        .globl __svml_sinhs32

__svml_sinhs32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hsinh_data_internal(%rip), %zmm23
        vmovdqu16 64+__svml_hsinh_data_internal(%rip), %zmm29
        vmovdqu16 128+__svml_hsinh_data_internal(%rip), %zmm28
        vmovdqu16 192+__svml_hsinh_data_internal(%rip), %zmm26
        vmovdqu16 256+__svml_hsinh_data_internal(%rip), %zmm5
        vmovdqu16 320+__svml_hsinh_data_internal(%rip), %zmm31
        vmovdqu16 384+__svml_hsinh_data_internal(%rip), %zmm8
        vmovdqu16 448+__svml_hsinh_data_internal(%rip), %zmm24
        vmovdqu16 512+__svml_hsinh_data_internal(%rip), %zmm9
        vmovdqu16 576+__svml_hsinh_data_internal(%rip), %zmm11
        vmovdqu16 640+__svml_hsinh_data_internal(%rip), %zmm30

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__sinh_h:
        cmpq    $31, %rdx
        ja .loaddata__sinh_h
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
        jmp .funcbegin_sinh_h
.loaddata__sinh_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_sinh_h:

        vmovdqu16 %zmm29, %zmm2
        vmovdqu16 %zmm24, %zmm7
        vpandd    %zmm23, %zmm0, %zmm6
        vfmadd213ph {rz-sae}, %zmm28, %zmm6, %zmm2
        vsubph    {rn-sae}, %zmm28, %zmm2, %zmm10

/* hN - 1 */
        vsubph    {rn-sae}, %zmm11, %zmm10, %zmm12

/* save sign */
        vpxord    %zmm0, %zmm6, %zmm1
        vfnmadd231ph {rn-sae}, %zmm10, %zmm26, %zmm6

/* 2^(hN-1) */
        vscalefph {rn-sae}, %zmm12, %zmm11, %zmm25
        vfnmadd231ph {rn-sae}, %zmm10, %zmm5, %zmm6

/* fixup for Inf results */
        vfpclassph $8, %zmm25, %k1

/* 2^(-hN)*2 */
        vrcpph    %zmm25, %zmm27

/* exp(R) -1 */
        vmovaps   %zmm7, %zmm0
        vpandd    %zmm31, %zmm6, %zmm13

/* exp(-R) -1 */
        vfnmadd231ph {rn-sae}, %zmm13, %zmm8, %zmm7
        vfmadd231ph {rn-sae}, %zmm13, %zmm8, %zmm0

/* 2^(-hN)*R */
        vmulph    {rn-sae}, %zmm13, %zmm27, %zmm14

/* 2^hN*R */
        vmulph    {rn-sae}, %zmm13, %zmm25, %zmm2
        vfnmadd213ph {rn-sae}, %zmm9, %zmm13, %zmm7
        vfmadd213ph {rn-sae}, %zmm9, %zmm13, %zmm0
        vfnmadd213ph {rn-sae}, %zmm11, %zmm13, %zmm7
        vfmadd213ph {rn-sae}, %zmm11, %zmm13, %zmm0

/* -Tm*mpoly*0.25 */
        vmulph    {rn-sae}, %zmm14, %zmm7, %zmm15
        vfnmadd213ph {rn-sae}, %zmm25, %zmm30, %zmm27
        vmulph    {rn-sae}, %zmm30, %zmm15, %zmm7

/* T*poly - Tm*mpoly */
        vfmadd213ph {rn-sae}, %zmm7, %zmm2, %zmm0
        vaddph    {rn-sae}, %zmm27, %zmm0, %zmm0

/* fixup */
        vmovdqu16 %zmm25, %zmm0{%k1}

/* fix sign */
        vpxord    %zmm1, %zmm0, %zmm0

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__sinh_h
        ret

        .cfi_endproc

        .type	__svml_sinhs32,@function
        .size	__svml_sinhs32,.-__svml_sinhs32

        .section .rodata, "a"
        .align 64

__svml_hsinh_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x3dc5
	.endr
	.rept	32
        .word	0x6600
	.endr
	.rept	32
        .word	0x398c
	.endr
	.rept	32
        .word	0x8af4
	.endr
	.rept	32
        .word	0xbfff
	.endr
	.rept	32
        .word	0x2976
	.endr
	.rept	32
        .word	0x3176
	.endr
	.rept	32
        .word	0x37ff
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x3400
	.endr
        .type	__svml_hsinh_data_internal,@object
        .size	__svml_hsinh_data_internal,704
	 .section        .note.GNU-stack,"",@progbits
