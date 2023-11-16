/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Implementation reduces argument as:
 *   sX + N*(sNPi1+sNPi2), where sNPi1+sNPi2 ~ -pi/2
 *   RShifter + x*(2/pi) will round to RShifter+N, where N=(int)(x/pi)
 *   To get sign bit we treat sY as integer value to look at last bit
 *   Compute polynomial ~ tan(R)/R
 *   Result = 1/tan(R) when sN = (int)(x/(Pi/2)) is odd
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_tans32

__svml_tans32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovups   __svml_htan_data_internal(%rip), %zmm25
        vmovups   64+__svml_htan_data_internal(%rip), %zmm4
        vmovups   128+__svml_htan_data_internal(%rip), %zmm29
        vmovups   192+__svml_htan_data_internal(%rip), %zmm5
        vmovups   256+__svml_htan_data_internal(%rip), %zmm7
        vmovdqu16 320+__svml_htan_data_internal(%rip), %zmm30
        vmovdqu16 384+__svml_htan_data_internal(%rip), %zmm23
        vmovdqu16 448+__svml_htan_data_internal(%rip), %zmm31
        vmovdqu16 512+__svml_htan_data_internal(%rip), %zmm26

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__tan_h:
        cmpq    $31, %rdx
        ja .loaddata__tan_h
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
        jmp .funcbegin_tan_h
.loaddata__tan_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_tan_h:

        vmovups   %zmm29, %zmm3
        vmovaps   %zmm4, %zmm9

/*
 * No callout
 * Copy argument
 * Needed to set sin(-0)=-0
 */
        vpandd    %zmm25, %zmm0, %zmm1
        vpxord    %zmm1, %zmm0, %zmm0

/* convert to FP32 */
        vextractf32x8 $1, %zmm1, %ymm2
        vcvtph2psx %ymm1, %zmm6
        vcvtph2psx %ymm2, %zmm8
        vfmadd231ps {rn-sae}, %zmm6, %zmm3, %zmm9
        vfmadd213ps {rn-sae}, %zmm4, %zmm8, %zmm3

/* sN = (int)(x/pi) = sY - Rshifter */
        vsubps    {rn-sae}, %zmm4, %zmm9, %zmm10
        vsubps    {rn-sae}, %zmm4, %zmm3, %zmm12

/* sign bit, will treat sY as integer value to look at last bit */
        vpslld    $31, %zmm9, %zmm14
        vpslld    $31, %zmm3, %zmm1
        vfmadd231ps {rn-sae}, %zmm10, %zmm5, %zmm6
        vfmadd231ps {rn-sae}, %zmm12, %zmm5, %zmm8

/* polynomial ~ tan(R)/R */
        vmovdqu16 %zmm23, %zmm9
        vfmadd213ps {rn-sae}, %zmm6, %zmm7, %zmm10
        vfmadd213ps {rn-sae}, %zmm8, %zmm7, %zmm12
        vcvtps2phx %zmm10, %ymm11
        vcvtps2phx %zmm12, %ymm13
        vcvtps2phx %zmm14, %ymm15
        vcvtps2phx %zmm1, %ymm14
        vinsertf32x8 $1, %ymm13, %zmm11, %zmm2

/* hR*hR */
        vmulph    {rn-sae}, %zmm2, %zmm2, %zmm28
        vfmadd231ph {rn-sae}, %zmm28, %zmm30, %zmm9
        vfmadd213ph {rn-sae}, %zmm31, %zmm28, %zmm9
        vfmadd213ph {rn-sae}, %zmm26, %zmm28, %zmm9
        vinsertf32x8 $1, %ymm14, %zmm15, %zmm8

/* result = 1/tan(R) when hN = (int)(x/(Pi/2)) is odd */
        vpmovw2m  %zmm8, %k1

/* add sign to hR */
        vpxord    %zmm8, %zmm2, %zmm27
        vfmadd213ph {rn-sae}, %zmm27, %zmm27, %zmm9
        vrcpph    %zmm9, %zmm9{%k1}

/* needed to set sin(-0)=-0 */
        vpxord    %zmm0, %zmm9, %zmm0

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__tan_h
        ret

        .cfi_endproc

        .type	__svml_tans32,@function
        .size	__svml_tans32,.-__svml_tans32

        .section .rodata, "a"
        .align 64

__svml_htan_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	16
        .long	0x4b000000
	.endr
	.rept	16
        .long	0x3f22f983
	.endr
	.rept	16
        .long	0xbfc90fdb
	.endr
	.rept	16
        .long	0x333bbd2e
	.endr
	.rept	32
        .word	0x2e06
	.endr
	.rept	32
        .word	0x2f6c
	.endr
	.rept	32
        .word	0x355f
	.endr
	.rept	32
        .word	0x82e5
	.endr
        .type	__svml_htan_data_internal,@object
        .size	__svml_htan_data_internal,576
	 .section        .note.GNU-stack,"",@progbits
