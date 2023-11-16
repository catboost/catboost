/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Compute cosh(x) as (exp(x)+exp(-x))/2,
 *   where exp is calculated as
 *   exp(M*ln2 + ln2*(j/2^k) + r) = 2^M * 2^(j/2^k) * exp(r)
 *
 *   Special cases:
 *
 *   cosh(NaN) = quiet NaN, and raise invalid exception
 *   cosh(INF) = that INF
 *   cosh(0)   = 1
 *   cosh(x) overflows for big x and returns MAXLOG+log(2)
 *
 */

        .text

        .align    16,0x90
        .globl __svml_coshs32

__svml_coshs32:

        .cfi_startproc
        vmovdqu16 __svml_hcosh_data_internal(%rip), %zmm31
        vmovdqu16 64+__svml_hcosh_data_internal(%rip), %zmm30
        vmovdqu16 128+__svml_hcosh_data_internal(%rip), %zmm2
        vmovdqu16 192+__svml_hcosh_data_internal(%rip), %zmm3
        vmovdqu16 256+__svml_hcosh_data_internal(%rip), %zmm4
        vmovdqu16 320+__svml_hcosh_data_internal(%rip), %zmm5
        vmovdqu16 384+__svml_hcosh_data_internal(%rip), %zmm29
        vmovdqu16 448+__svml_hcosh_data_internal(%rip), %zmm6
        vmovdqu16 512+__svml_hcosh_data_internal(%rip), %zmm9
        vmovdqu16 576+__svml_hcosh_data_internal(%rip), %zmm12
        kxnord  %k7, %k7, %k7

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_cosh_h:
        cmpq    $31, %rdx
        ja .loaddata_coshh
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
        jmp .funcbegin_cosh_h
.loaddata_coshh:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_cosh_h:

/* Shifter + x*log2(e) */
        vmovdqu16 %zmm30, %zmm1
        vmovdqu16 %zmm29, %zmm10

/* poly + 0.25*mpoly ~ (exp(x)+exp(-x))*0.5 */
        vpandd %zmm31, %zmm0, %zmm7
        vfmadd213ph {rz-sae}, %zmm2, %zmm7, %zmm1
        vsubph    {rn-sae}, %zmm2, %zmm1, %zmm8
        vfnmadd231ph {rn-sae}, %zmm8, %zmm3, %zmm7

/* hN - 1 */
        vsubph    {rn-sae}, %zmm9, %zmm8, %zmm11
        vfnmadd231ph {rn-sae}, %zmm8, %zmm4, %zmm7

/* exp(R) */
        vfmadd231ph {rn-sae}, %zmm7, %zmm5, %zmm10
        vfmadd213ph {rn-sae}, %zmm6, %zmm7, %zmm10
        vfmadd213ph {rn-sae}, %zmm9, %zmm7, %zmm10

/* poly*R+1 */
        vfmadd213ph {rn-sae}, %zmm9, %zmm7, %zmm10
        vscalefph {rn-sae}, %zmm11, %zmm10, %zmm13
        vrcpph    %zmm13, %zmm0
        vfmadd213ph {rn-sae}, %zmm13, %zmm12, %zmm0
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_cosh_h
        ret

        .cfi_endproc

        .type	__svml_coshs32,@function
        .size	__svml_coshs32,.-__svml_coshs32

        .section .rodata, "a"
        .align 64

__svml_hcosh_data_internal:
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
        .word	0x2b17
	.endr
	.rept	32
        .word	0x3122
	.endr
	.rept	32
        .word	0x3802
	.endr
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x3400
	.endr
        .type	__svml_hcosh_data_internal,@object
        .size	__svml_hcosh_data_internal,640
	 .section        .note.GNU-stack,"",@progbits
