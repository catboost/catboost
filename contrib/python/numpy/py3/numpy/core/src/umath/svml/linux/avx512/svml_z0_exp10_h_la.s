/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   exp10(x)  = 2^x/log10(2) = 2^n * (1 + T[j]) * (1 + P(y))
 *
 *   x = m*log10(2)/K + y,  y in [-log10(2)/K..log10(2)/K]
 *   m = n*K + j,           m,n,j - signed integer, j in [-K/2..K/2]
 *
 *   values of 2^j/K are tabulated
 *
 *   P(y) is a minimax polynomial approximation of exp10(x)-1
 *   on small interval [-log10(2)/K..log10(2)/K]
 *
 *  Special cases:
 *
 *   exp10(NaN)  = NaN
 *   exp10(+INF) = +INF
 *   exp10(-INF) = 0
 *   exp10(x)    = 1 for subnormals
 *
 */

        .text

        .align    16,0x90
        .globl __svml_exp10s32

__svml_exp10s32:

        .cfi_startproc

/* restrict input range to [-8.0,6.0] */
        vmovdqu16 64+__svml_hexp10_data_internal(%rip), %zmm1
        vmovdqu16 __svml_hexp10_data_internal(%rip), %zmm3

/* (2^11*1.5 + bias) + x*log2(e) */
        vmovdqu16 128+__svml_hexp10_data_internal(%rip), %zmm4
        vmovdqu16 192+__svml_hexp10_data_internal(%rip), %zmm6
        vmovdqu16 256+__svml_hexp10_data_internal(%rip), %zmm7

/* polynomial ~ 2^R */
        vmovdqu16 320+__svml_hexp10_data_internal(%rip), %zmm12
        vmovdqu16 384+__svml_hexp10_data_internal(%rip), %zmm9
        vmovdqu16 448+__svml_hexp10_data_internal(%rip), %zmm10
        vmovdqu16 512+__svml_hexp10_data_internal(%rip), %zmm11

/* fixup for input=NaN */
        vmovdqu16 576+__svml_hexp10_data_internal(%rip), %zmm14
        vminph    {sae}, %zmm1, %zmm0, %zmm2
        vcmpph    $22, {sae}, %zmm14, %zmm0, %k1
        vmaxph    {sae}, %zmm3, %zmm2, %zmm13
        vmovaps   %zmm4, %zmm5
        vfmadd231ph {rn-sae}, %zmm13, %zmm6, %zmm5

/* N = 2*(int)(x*log2(e)/2) */
        vsubph    {rn-sae}, %zmm4, %zmm5, %zmm8

/* 2^(N/2) */
        vpsllw    $10, %zmm5, %zmm15

/* reduced arg.: x*log2(e) - N */
        vfmsub231ph {rn-sae}, %zmm13, %zmm6, %zmm8
        vpblendmw %zmm0, %zmm15, %zmm0{%k1}
        vfmadd213ph {rn-sae}, %zmm8, %zmm7, %zmm13

/* start polynomial */
        vfmadd213ph {rn-sae}, %zmm9, %zmm13, %zmm12
        vfmadd213ph {rn-sae}, %zmm10, %zmm13, %zmm12
        vfmadd213ph {rn-sae}, %zmm11, %zmm13, %zmm12
        vmulph    {rn-sae}, %zmm13, %zmm12, %zmm1
        vfmadd213ph {rn-sae}, %zmm15, %zmm15, %zmm1

/* result:  (1+poly)*2^(N/2)*2^(N/2) */
        vmulph    {rn-sae}, %zmm0, %zmm1, %zmm0
        ret

        .cfi_endproc

        .type	__svml_exp10s32,@function
        .size	__svml_exp10s32,.-__svml_exp10s32

        .section .rodata, "a"
        .align 64

__svml_hexp10_data_internal:
	.rept	32
        .word	0xc800
	.endr
	.rept	32
        .word	0x4600
	.endr
	.rept	32
        .word	0x6a0f
	.endr
	.rept	32
        .word	0x42a5
	.endr
	.rept	32
        .word	0x8d88
	.endr
	.rept	32
        .word	0x2110
	.endr
	.rept	32
        .word	0x2b52
	.endr
	.rept	32
        .word	0x33af
	.endr
	.rept	32
        .word	0x398b
	.endr
	.rept	32
        .word	0x7c00
	.endr
        .type	__svml_hexp10_data_internal,@object
        .size	__svml_hexp10_data_internal,640
	 .section        .note.GNU-stack,"",@progbits
