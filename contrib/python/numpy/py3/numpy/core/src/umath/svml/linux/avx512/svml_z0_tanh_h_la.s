/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   NOTE: Since the hyperbolic tangent function is odd
 *         (tanh(x) = -tanh(-x)), below algorithm deals with the absolute
 *         value of the argument |x|: tanh(x) = sign(x) * tanh(|x|)
 *
 *   Get absolute value of argument xa = |x| and sign(x)
 *   Limit arguments by threshold xa = min(xa, Thres), Thres<4.0
 *   Get table index as Shifter + xa, RZ mode
 *   Lookup polynomial coefficients: c1, c2 with the index
 *   Compute polynomial and add sign x back: x+x*Poly
 *
 *   IEEE SPECIAL CONDITIONS:
 *   x = [+,-]0, r = [+,-]0
 *   x = +Inf,   r = +1
 *   x = -Inf,   r = -1
 *   x = QNaN,   r = QNaN
 *   x = SNaN,   r = QNaN
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_tanhs32

__svml_tanhs32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_htanh_data_internal(%rip), %zmm31
        vmovdqu16 64+__svml_htanh_data_internal(%rip), %zmm30
        vmovdqu16 128+__svml_htanh_data_internal(%rip), %zmm29
        vmovdqu16 192+__svml_htanh_data_internal(%rip), %zmm1
        vmovdqu16 256+__svml_htanh_data_internal(%rip), %zmm3

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__tanh_h:
        cmpq    $31, %rdx
        ja .loaddata__tanh_h
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
        jmp .funcbegin_tanh_h
.loaddata__tanh_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_tanh_h:

/*
 * No callout
 * xa = |x|
 */
        vpandd    %zmm29, %zmm0, %zmm2
        vminph    {sae}, %zmm2, %zmm1, %zmm6

/* Shifter + xa, RZ mode */
        vaddph    {rz-sae}, %zmm3, %zmm6, %zmm4

/* look up poly coefficients: c1, c2 */
        vpermw    %zmm31, %zmm4, %zmm5

/* sign(x) */
        vpxord    %zmm0, %zmm2, %zmm7
        vpermw    %zmm30, %zmm4, %zmm0

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm5, %zmm6, %zmm0

/* add sign back to x */
        vpxord    %zmm7, %zmm6, %zmm8

/* x+x*Poly */
        vfmadd213ph {rn-sae}, %zmm8, %zmm8, %zmm0

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__tanh_h
        ret

        .cfi_endproc

        .type	__svml_tanhs32,@function
        .size	__svml_tanhs32,.-__svml_tanhs32

        .section .rodata, "a"
        .align 64

__svml_htanh_data_internal:
        .word	0x0c00
        .word	0x216a
        .word	0x273e
        .word	0x2a6c
        .word	0x2c9c
        .word	0x2dc3
        .word	0x2e7c
        .word	0x2eb1
        .word	0x2e5a
        .word	0x2d82
        .word	0x2c3a
        .word	0x2932
        .word	0x21b4
        .word	0xa561
        .word	0xab02
        .word	0xadb5
        .word	0xafe8
        .word	0xb109
        .word	0xb216
        .word	0xb319
        .word	0xb408
        .word	0xb47e
        .word	0xb4ee
        .word	0xb558
        .word	0xb5bc
        .word	0xb61a
        .word	0xb673
        .word	0xb6c6
        .word	0xb715
        .word	0xb75f
        .word	0xb7a5
        .word	0xb7e7
        .word	0xa94d
        .word	0xafc2
        .word	0xb228
        .word	0xb404
        .word	0xb4b8
        .word	0xb52f
        .word	0xb56d
        .word	0xb57d
        .word	0xb567
        .word	0xb537
        .word	0xb4f6
        .word	0xb4aa
        .word	0xb45a
        .word	0xb409
        .word	0xb373
        .word	0xb2dd
        .word	0xb250
        .word	0xb1ce
        .word	0xb156
        .word	0xb0e9
        .word	0xb086
        .word	0xb02c
        .word	0xafb6
        .word	0xaf22
        .word	0xae9d
        .word	0xae25
        .word	0xadb7
        .word	0xad54
        .word	0xacfa
        .word	0xaca9
        .word	0xac5e
        .word	0xac1a
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x42e3
	.endr
	.rept	32
        .word	0x5a00
	.endr
        .type	__svml_htanh_data_internal,@object
        .size	__svml_htanh_data_internal,320
	 .section        .note.GNU-stack,"",@progbits
