/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   log(x) = VGETEXP(x)*log(2) + log(VGETMANT(x))
 *   VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *   log(mx) = -log(RCP(mx)) + log(1 +(mx*RCP(mx)-1))
 *   and the table lookup for log(RCP(mx))
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_log1ps32

__svml_log1ps32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hlog1p_data_internal(%rip), %zmm6
        vmovdqu16 64+__svml_hlog1p_data_internal(%rip), %zmm29
        vmovdqu16 128+__svml_hlog1p_data_internal(%rip), %zmm28
        vmovdqu16 192+__svml_hlog1p_data_internal(%rip), %zmm27
        vmovdqu16 256+__svml_hlog1p_data_internal(%rip), %zmm14
        vmovdqu16 320+__svml_hlog1p_data_internal(%rip), %zmm15
        vmovdqu16 384+__svml_hlog1p_data_internal(%rip), %zmm31
        vmovdqu16 448+__svml_hlog1p_data_internal(%rip), %zmm30

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__log1p_h:
        cmpq    $31, %rdx
        ja .loaddata__log1p_h
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
        jmp .funcbegin_log1p_h
.loaddata__log1p_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_log1p_h:

/* No callout */
        vmovdqu16 %zmm27, %zmm11

/* x+1.0 */
        vaddph    {rn-sae}, %zmm6, %zmm0, %zmm1

/* A = max(x, 1.0) */
        vmaxph    {sae}, %zmm6, %zmm0, %zmm2

/* B = min(x, 1.0); */
        vminph    {sae}, %zmm6, %zmm0, %zmm3

/* input is zero or +Inf? */
        vfpclassph $14, %zmm0, %k1

/* reduce mantissa to [.75, 1.5) */
        vgetmantph $11, {sae}, %zmm1, %zmm5

/* Bh */
        vsubph    {rn-sae}, %zmm2, %zmm1, %zmm4

/* exponent */
        vgetexpph {sae}, %zmm1, %zmm8

/* exponent correction */
        vgetexpph {sae}, %zmm5, %zmm7

/* Mant_low */
        vsubph    {rn-sae}, %zmm4, %zmm3, %zmm9

/* reduced argument */
        vsubph    {rn-sae}, %zmm6, %zmm5, %zmm12

/* -exponent */
        vsubph    {rn-sae}, %zmm8, %zmm7, %zmm1

/* mant_low, whith proper scale */
        vscalefph {rn-sae}, %zmm1, %zmm9, %zmm10

/* fixup in case Mant_low=NaN */
        vpandd %zmm29, %zmm10, %zmm13
        vmovdqu16 %zmm28, %zmm10

/* start polynomial */
        vfmadd213ph {rn-sae}, %zmm11, %zmm12, %zmm10

/* apply correction to reduced argument */
        vaddph    {rn-sae}, %zmm13, %zmm12, %zmm11

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm14, %zmm11, %zmm10
        vfmadd213ph {rn-sae}, %zmm15, %zmm11, %zmm10
        vfmadd213ph {rn-sae}, %zmm31, %zmm11, %zmm10

/* Poly*R+R */
        vfmadd213ph {rn-sae}, %zmm11, %zmm11, %zmm10

/* result:  -m_expon*log(2)+poly */
        vfnmadd213ph {rn-sae}, %zmm10, %zmm30, %zmm1

/* fixup for +0/-0/+Inf */
        vpblendmw %zmm0, %zmm1, %zmm0{%k1}

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__log1p_h
        ret

        .cfi_endproc

        .type	__svml_log1ps32,@function
        .size	__svml_log1ps32,.-__svml_log1ps32

        .section .rodata, "a"
        .align 64

__svml_hlog1p_data_internal:
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0xbfff
	.endr
	.rept	32
        .word	0x3088
	.endr
	.rept	32
        .word	0xb428
	.endr
	.rept	32
        .word	0x356a
	.endr
	.rept	32
        .word	0xb800
	.endr
	.rept	32
        .word	0x833f
	.endr
	.rept	32
        .word	0x398c
	.endr
        .type	__svml_hlog1p_data_internal,@object
        .size	__svml_hlog1p_data_internal,512
	 .section        .note.GNU-stack,"",@progbits
