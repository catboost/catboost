/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *    log2(x) = VGETEXP(x) + log2(VGETMANT(x))
 *    VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *    mx = VGETMANT(x) is in [1,2) for all x>=0
 *    log2(mx) = -log2(RCP(mx)) + log2(1 +(mx*RCP(mx)-1))
 *    and the table lookup for log2(RCP(mx))
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_log2s32

__svml_log2s32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hlog2_data_internal(%rip), %zmm1
        vmovdqu16 64+__svml_hlog2_data_internal(%rip), %zmm31
        vmovdqu16 128+__svml_hlog2_data_internal(%rip), %zmm3
        vmovdqu16 192+__svml_hlog2_data_internal(%rip), %zmm6
        vmovdqu16 256+__svml_hlog2_data_internal(%rip), %zmm7
        vmovdqu16 320+__svml_hlog2_data_internal(%rip), %zmm8

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__log2_h:
        cmpq    $31, %rdx
        ja .loaddata__log2_h
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
        jmp .funcbegin_log2_h
.loaddata__log2_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_log2_h:

/* No callout */

/* exponent */
        vgetexpph {sae}, %zmm0, %zmm4

/* reduce mantissa to [.75, 1.5) */
        vgetmantph $11, {sae}, %zmm0, %zmm2

/* reduced argument */
        vsubph    {rn-sae}, %zmm1, %zmm2, %zmm9

/* exponent correction */
        vgetexpph {sae}, %zmm2, %zmm5

/* start polynomial */
        vmovdqu16 %zmm31, %zmm0
        vfmadd213ph {rn-sae}, %zmm3, %zmm9, %zmm0

/* exponent */
        vsubph    {rn-sae}, %zmm5, %zmm4, %zmm10

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm6, %zmm9, %zmm0
        vfmadd213ph {rn-sae}, %zmm7, %zmm9, %zmm0
        vfmadd213ph {rn-sae}, %zmm8, %zmm9, %zmm0

/* Poly*R+expon */
        vfmadd213ph {rn-sae}, %zmm10, %zmm9, %zmm0
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__log2_h
        ret

        .cfi_endproc

        .type	__svml_log2s32,@function
        .size	__svml_log2s32,.-__svml_log2s32

        .section .rodata, "a"
        .align 64

__svml_hlog2_data_internal:
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x328a
	.endr
	.rept	32
        .word	0xb5ff
	.endr
	.rept	32
        .word	0x37cf
	.endr
	.rept	32
        .word	0xb9c5
	.endr
	.rept	32
        .word	0x3dc5
	.endr
	.rept	32
        .word	0x001f
	.endr
	.rept	32
        .word	0x0000
	.endr
	.rept	16
        .long	0x00000001
	.endr
	.rept	16
        .long	0x0000007f
	.endr
	.rept	16
        .long	0x3f800000
	.endr
	.rept	16
        .long	0x3e51367b
	.endr
	.rept	16
        .long	0xbebfd356
	.endr
	.rept	16
        .long	0x3ef9e953
	.endr
	.rept	16
        .long	0xbf389f48
	.endr
	.rept	16
        .long	0x3fb8a7e4
	.endr
	.rept	32
        .word	0xfc00
	.endr
        .type	__svml_hlog2_data_internal,@object
        .size	__svml_hlog2_data_internal,1088
	 .section        .note.GNU-stack,"",@progbits
