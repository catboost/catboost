/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   After computing exp(x), an accurate computation is performed to obtain exp(x)-1
 *   Typical exp() implementation, except that:
 *    - tables are small, allowing for fast gathers
 *    - all arguments processed in the main path
 *        - final VSCALEF assists branch-free design (correct overflow/underflow and special case responses)
 *        - a VAND is used to ensure the reduced argument |R|<2, even for large inputs
 *        - RZ mode used to avoid oveflow to +/-Inf for x*log2(e); helps with special case handling
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_expm1s32

__svml_expm1s32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovups   __svml_hexpm1_data_internal(%rip), %zmm30
        vmovups   64+__svml_hexpm1_data_internal(%rip), %zmm31
        vmovups   128+__svml_hexpm1_data_internal(%rip), %zmm6
        vmovups   192+__svml_hexpm1_data_internal(%rip), %zmm29
        vmovups   256+__svml_hexpm1_data_internal(%rip), %zmm7
        vmovups   320+__svml_hexpm1_data_internal(%rip), %zmm8
        vmovups   384+__svml_hexpm1_data_internal(%rip), %zmm15

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_expm1_h:
        cmpq    $31, %rdx
        ja .loaddata_expm1_h
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
        jmp .funcbegin_expm1_h
.loaddata_expm1_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_expm1_h:

/* No callout */
        vmovups   %zmm30, %zmm2
        vmovups   %zmm29, %zmm11
        vmovups   %zmm31, %zmm5

/* polynomial ~ expm1(R)/R */
        vmovaps   %zmm11, %zmm9
        vextractf32x8 $1, %zmm0, %ymm1
        vcvtph2psx %ymm0, %zmm10
        vcvtph2psx %ymm1, %zmm12

/* Shifter + x * log2(e) */
        vmulps    {rn-sae}, %zmm2, %zmm10, %zmm3
        vmulps    {rn-sae}, %zmm2, %zmm12, %zmm4

/* N ~ x*log2(e) */
        vrndscaleps $8, {sae}, %zmm3, %zmm13
        vrndscaleps $8, {sae}, %zmm4, %zmm14

/* R = x - N*ln(2)_high */
        vfmadd231ps {rn-sae}, %zmm13, %zmm5, %zmm10
        vfmadd231ps {rn-sae}, %zmm14, %zmm5, %zmm12
        vscalefps {rn-sae}, %zmm13, %zmm15, %zmm13
        vscalefps {rn-sae}, %zmm14, %zmm15, %zmm14
        vfmadd231ps {rn-sae}, %zmm10, %zmm6, %zmm9
        vfmadd231ps {rn-sae}, %zmm12, %zmm6, %zmm11

/* fixup for overflow and special cases */
        vfpclassps $14, %zmm13, %k0
        vfpclassps $14, %zmm14, %k2

/* 2^N - 1 */
        vsubps    {rn-sae}, %zmm15, %zmm13, %zmm2
        vsubps    {rn-sae}, %zmm15, %zmm14, %zmm4
        vfmadd213ps {rn-sae}, %zmm7, %zmm10, %zmm9
        vfmadd213ps {rn-sae}, %zmm7, %zmm12, %zmm11
        knotw     %k0, %k1
        knotw     %k2, %k3
        vfmadd213ps {rn-sae}, %zmm8, %zmm10, %zmm9
        vfmadd213ps {rn-sae}, %zmm8, %zmm12, %zmm11

/* exp(R) - 1 */
        vmulps    {rn-sae}, %zmm10, %zmm9, %zmm0
        vmulps    {rn-sae}, %zmm12, %zmm11, %zmm1

/* result */
        vfmadd231ps {rn-sae}, %zmm0, %zmm13, %zmm2{%k1}
        vfmadd231ps {rn-sae}, %zmm1, %zmm14, %zmm4{%k3}
        vcvtps2phx %zmm2, %ymm3
        vcvtps2phx %zmm4, %ymm5
        vinsertf32x8 $1, %ymm5, %zmm3, %zmm0
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_expm1_h

/*
 * #else  _LA_, _EP_
 * #endif  _LA_, _EP_
 */
        ret

        .cfi_endproc

        .type	__svml_expm1s32,@function
        .size	__svml_expm1s32,.-__svml_expm1s32

        .section .rodata, "a"
        .align 64

__svml_hexpm1_data_internal:
	.rept	16
        .long	0x3fb8aa3b
	.endr
	.rept	16
        .long	0xbf317218
	.endr
	.rept	16
        .long	0x3d2bb1be
	.endr
	.rept	16
        .long	0x3e2bb1c1
	.endr
	.rept	16
        .long	0x3efffeaf
	.endr
	.rept	16
        .long	0x3f7fff03
	.endr
	.rept	16
        .long	0x3f800000
	.endr
        .type	__svml_hexpm1_data_internal,@object
        .size	__svml_hexpm1_data_internal,448
	 .section        .note.GNU-stack,"",@progbits
