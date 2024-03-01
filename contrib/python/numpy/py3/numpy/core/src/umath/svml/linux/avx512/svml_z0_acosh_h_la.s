/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Compute acosh(x) as log(x + sqrt(x*x - 1))
 *
 *   Special cases:
 *
 *   acosh(NaN)  = quiet NaN, and raise invalid exception
 *   acosh(-INF) = NaN
 *   acosh(+INF) = +INF
 *   acosh(x)    = NaN if x < 1
 *   acosh(1)    = +0
 *
 */

        .text

        .align    16,0x90
        .globl __svml_acoshs32

__svml_acoshs32:

        .cfi_startproc
        vmovdqu16 __svml_hacosh_data_internal(%rip), %zmm13
        vmovdqu16 64+__svml_hacosh_data_internal(%rip), %zmm1
        vmovdqu16 128+__svml_hacosh_data_internal(%rip), %zmm31
        vmovdqu16 192+__svml_hacosh_data_internal(%rip), %zmm30
        vmovdqu16 256+__svml_hacosh_data_internal(%rip), %zmm29
        vmovdqu16 320+__svml_hacosh_data_internal(%rip), %zmm28
        vmovdqu16 384+__svml_hacosh_data_internal(%rip), %zmm10
        vmovdqu16 448+__svml_hacosh_data_internal(%rip), %zmm12
        vmovdqu16 512+__svml_hacosh_data_internal(%rip), %zmm27
        vmovdqu16 576+__svml_hacosh_data_internal(%rip), %zmm24
        vmovdqu16 640+__svml_hacosh_data_internal(%rip), %zmm26
        vmovdqu16 704+__svml_hacosh_data_internal(%rip), %zmm25
        vmovdqu16 768+__svml_hacosh_data_internal(%rip), %zmm23
        kxnord  %k7, %k7, %k7
/*
 * No callout
 * x in [1,2): acosh(x) ~ poly(x-1)*sqrt(x-1)
 * hY = x-1
 */

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_acosh_h:
        cmpq    $31, %rdx
        ja .loaddata_acosh
/* set up mask %k7 for masked load instruction */
        movl    $1, %eax
        movl    %edx, %ecx
        sall    %cl, %eax
        subl    $1, %eax
        kmovd   %eax, %k7
/* Constant required for masked load */
        movl    $15360, %eax
        vpbroadcastw    %eax, %zmm0
        vmovdqu16 (%rdi), %zmm0{%k7}
        jmp .funcbegin_acosh_h
.loaddata_acosh:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_acosh_h:

/* ReLoad constants */
	vmovdqu16 %zmm31, %zmm8
	vmovdqu16 %zmm29, %zmm6

        vmovaps   %zmm0, %zmm3
        vsubph    {rn-sae}, %zmm13, %zmm3, %zmm5

/*
 * log(x)
 * GetMant(x), normalized to [.75,1.5) for x>=0, NaN for x<0
 */
        vgetmantph $11, {sae}, %zmm3, %zmm14
        vgetexpph {sae}, %zmm3, %zmm15

/*
 * x>=2: acosh(x) = log(x) + log(1+sqrt(1-(1/x)^2))
 * Z ~ 1/x in (0, 0.5]
        vmovdqu16 256+__svml_hacosh_data_internal(%rip), %zmm29
 */
        vrcpph    %zmm3, %zmm11

/* hRS ~ 1/sqrt(x-1) */
        vrsqrtph  %zmm5, %zmm7
        vcmpph    $21, {sae}, %zmm1, %zmm3, %k1
        vmovdqu16 %zmm28, %zmm0
        vfmadd213ph {rn-sae}, %zmm30, %zmm5, %zmm8

/* hS ~ sqrt(x-1) */
        vrcpph    %zmm7, %zmm9

/* log(1+R)/R */
        vmovdqu16 %zmm27, %zmm7
        vfmadd213ph {rn-sae}, %zmm6, %zmm5, %zmm8

/* mantissa - 1 */
        vsubph    {rn-sae}, %zmm13, %zmm14, %zmm6
        vfmadd213ph {rn-sae}, %zmm10, %zmm11, %zmm0

/* exponent correction */
        vgetexpph {sae}, %zmm14, %zmm14

/* poly(x-1)*sqrt(x-1) */
        vmulph    {rn-sae}, %zmm9, %zmm8, %zmm2
        vfmadd213ph {rn-sae}, %zmm12, %zmm11, %zmm0
        vsubph    {rn-sae}, %zmm14, %zmm15, %zmm8
        vmovdqu16 %zmm24, %zmm15
        vfmadd213ph {rn-sae}, %zmm15, %zmm6, %zmm7
        vfmadd213ph {rn-sae}, %zmm26, %zmm6, %zmm7
        vfmadd213ph {rn-sae}, %zmm25, %zmm6, %zmm7

/* log(1+R) + log(1+sqrt(1-z*z)) */
        vfmadd213ph {rn-sae}, %zmm0, %zmm6, %zmm7

/* result for x>=2 */
        vmovdqu16 %zmm23, %zmm0
        vfmadd213ph {rn-sae}, %zmm7, %zmm0, %zmm8

/* result = SelMask?  hPl : hPa */
        vpblendmw %zmm8, %zmm2, %zmm0{%k1}

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_acosh_h
        ret

        .cfi_endproc

        .type	__svml_acoshs32,@function
        .size	__svml_acoshs32,.-__svml_acoshs32

        .section .rodata, "a"
        .align 64

__svml_hacosh_data_internal:
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x4000
	.endr
	.rept	32
        .word	0x24a4
	.endr
	.rept	32
        .word	0xaf5e
	.endr
	.rept	32
        .word	0x3da8
	.endr
	.rept	32
        .word	0xb4d2
	.endr
	.rept	32
        .word	0x231d
	.endr
	.rept	32
        .word	0x398b
	.endr
	.rept	32
        .word	0xb22b
	.endr
	.rept	32
        .word	0x358b
	.endr
	.rept	32
        .word	0xb807
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x398c
	.endr
        .type	__svml_hacosh_data_internal,@object
        .size	__svml_hacosh_data_internal,832
	 .section        .note.GNU-stack,"",@progbits
