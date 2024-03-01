/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *     x=2^{3*k+j} * 1.b1 b2 ... b5 b6 ... b52
 *     Let r=(x*2^{-3k-j} - 1.b1 b2 ... b5 1)* rcp[b1 b2 ..b5],
 *     where rcp[b1 b2 .. b5]=1/(1.b1 b2 b3 b4 b5 1) in single precision
 *     cbrtf(2^j * 1. b1 b2 .. b5 1) is approximated as T[j][b1..b5]+D[j][b1..b5]
 *     (T stores the high 24 bits, D stores the low order bits)
 *     Result=2^k*T+(2^k*T*r)*P+2^k*D
 *      where P=p1+p2*r+..
 *
 */

        .text

        .align    16,0x90
        .globl __svml_cbrts32

__svml_cbrts32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hcbrt_data_internal(%rip), %zmm31
        vmovdqu16 64+__svml_hcbrt_data_internal(%rip), %zmm29
        vmovdqu16 128+__svml_hcbrt_data_internal(%rip), %zmm2
        vmovdqu16 192+__svml_hcbrt_data_internal(%rip), %zmm28
        vmovdqu16 256+__svml_hcbrt_data_internal(%rip), %zmm4
        vmovdqu16 320+__svml_hcbrt_data_internal(%rip), %zmm30
        vmovdqu16 384+__svml_hcbrt_data_internal(%rip), %zmm8
        vmovdqu16 448+__svml_hcbrt_data_internal(%rip), %zmm9

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_cbrt_h:
        cmpq    $31, %rdx
        ja .loaddata_cbrth
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
        jmp .funcbegin_cbrt_h
.loaddata_cbrth:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_cbrt_h:

        vmovdqu16 %zmm31, %zmm6
        vmovdqu16 %zmm30, %zmm10
        vgetexpph {sae}, %zmm0, %zmm3

/* mantissa(|x|) */
        vgetmantph $4, {sae}, %zmm0, %zmm1

/* check for +/-Inf, +/-0 */
        vfpclassph $30, %zmm0, %k1
        vaddph    {rn-sae}, %zmm4, %zmm3, %zmm5
        vsubph    {rn-sae}, %zmm2, %zmm1, %zmm11
        vpermt2w  %zmm29, %zmm5, %zmm6
        vfmadd213ph {rn-sae}, %zmm8, %zmm11, %zmm10
        vfmadd213ph {rn-sae}, %zmm9, %zmm11, %zmm10
        vmulph    {rn-sae}, %zmm11, %zmm10, %zmm12

/*
 * No callout
 * sign
 */
        vpandd    %zmm28, %zmm0, %zmm7

/* add sign */
        vpord     %zmm7, %zmm6, %zmm13
        vfmadd231ph {rn-sae}, %zmm13, %zmm12, %zmm13

/* fixup for +/-Inf, +/-0 */
        vmovdqu16 %zmm0, %zmm13{%k1}
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm13, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_cbrt_h
        ret

        .cfi_endproc

        .type	__svml_cbrts32,@function
        .size	__svml_cbrts32,.-__svml_cbrts32

        .section .rodata, "a"
        .align 64

__svml_hcbrt_data_internal:
        .word	0x3c00
        .word	0x3d0a
        .word	0x3e59
        .word	0x4000
        .word	0x410a
        .word	0x4259
        .word	0x4400
        .word	0x450a
        .word	0x4659
        .word	0x4800
        .word	0x490a
        .word	0x4a59
        .word	0x4c00
        .word	0x4d0a
        .word	0x4e59
        .word	0x5000
	.rept	24
        .word	0x0000
	.endr
        .word	0x1c00
        .word	0x1d0a
        .word	0x1e59
        .word	0x2000
        .word	0x210a
        .word	0x2259
        .word	0x2400
        .word	0x250a
        .word	0x2659
        .word	0x2800
        .word	0x290a
        .word	0x2a59
        .word	0x2c00
        .word	0x2d0a
        .word	0x2e59
        .word	0x3000
        .word	0x310a
        .word	0x3259
        .word	0x3400
        .word	0x350a
        .word	0x3659
        .word	0x3800
        .word	0x390a
        .word	0x3a59
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x8000
	.endr
	.rept	32
        .word	0x6600
	.endr
	.rept	32
        .word	0x277c
	.endr
	.rept	32
        .word	0xae84
	.endr
	.rept	32
        .word	0x3554
	.endr
        .type	__svml_hcbrt_data_internal,@object
        .size	__svml_hcbrt_data_internal,512
	 .section        .note.GNU-stack,"",@progbits
