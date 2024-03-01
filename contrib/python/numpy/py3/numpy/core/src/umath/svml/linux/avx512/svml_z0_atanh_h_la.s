/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *   Compute atanh(x) as 0.5 * log((1 + x)/(1 - x))
 *
 *   Special cases:
 *
 *   atanh(0)  = 0
 *   atanh(+1) = +INF
 *   atanh(-1) = -INF
 *   atanh(x)  = NaN if |x| > 1, or if x is a NaN or INF
 *
 */

        .text

        .align    16,0x90
        .globl __svml_atanhs32

__svml_atanhs32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hatanh_data_internal(%rip), %zmm31
        vmovdqu16 64+__svml_hatanh_data_internal(%rip), %zmm30
        vmovdqu16 128+__svml_hatanh_data_internal(%rip), %zmm29
        vmovdqu16 192+__svml_hatanh_data_internal(%rip), %zmm28
        vmovdqu16 256+__svml_hatanh_data_internal(%rip), %zmm27
        vmovdqu16 320+__svml_hatanh_data_internal(%rip), %zmm26

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_atanh_h:
        cmpq    $31, %rdx
        ja .loaddata_atanhh
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
        jmp .funcbegin_atanh_h
.loaddata_atanhh:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_atanh_h:

/*
 * reduced argument: 4*(1-|x|)
 * scaling by 4 to get around limited FP16 range for coefficient representation
 */
        vmovdqu16 %zmm27, %zmm3

/* prepare special case mask */
        vpandd    %zmm28, %zmm0, %zmm2
        vfnmadd213ph {rn-sae}, %zmm3, %zmm2, %zmm3

/* get table index */
        vpsrlw    $9, %zmm3, %zmm4

/* Special result */
        vrsqrtph  %zmm3, %zmm7
        vcmpph    $18, {sae}, %zmm26, %zmm3, %k1

/* look up poly coefficients: c1, c2, c3 */
        vpermw    %zmm29, %zmm4, %zmm10
        vpermw    %zmm30, %zmm4, %zmm5
        vpermw    %zmm31, %zmm4, %zmm6

/* polynomial */
        vfmadd213ph {rn-sae}, %zmm5, %zmm3, %zmm10
        vfmadd213ph {rn-sae}, %zmm6, %zmm3, %zmm10

/* x+x*Poly */
        vfmadd213ph {rn-sae}, %zmm0, %zmm0, %zmm10
        vpandnd   %zmm0, %zmm28, %zmm8
        vpord     %zmm8, %zmm7, %zmm11
        vpblendmw %zmm11, %zmm10, %zmm0{%k1}
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_atanh_h
        ret

        .cfi_endproc

        .type	__svml_atanhs32,@function
        .size	__svml_atanhs32,.-__svml_atanhs32

        .section .rodata, "a"
        .align 64

__svml_hatanh_data_internal:
        .word	0x3770
        .word	0x35ab
	.rept	10
        .word	0x0000
	.endr
        .word	0x439e
        .word	0x4345
        .word	0x42ed
        .word	0x4294
        .word	0x423c
        .word	0x41e3
        .word	0x418a
        .word	0x4132
        .word	0x40da
        .word	0x4081
        .word	0x4029
        .word	0x3fa2
        .word	0x3ef4
        .word	0x3e46
        .word	0x3d9a
        .word	0x3cef
        .word	0x3c48
        .word	0x3b47
        .word	0x3a08
        .word	0x38d4
        .word	0xb412
        .word	0xb1b1
	.rept	10
        .word	0x0000
	.endr
        .word	0xde80
        .word	0xdc99
        .word	0xda7d
        .word	0xd896
        .word	0xd677
        .word	0xd491
        .word	0xd26e
        .word	0xd088
        .word	0xce5d
        .word	0xcc79
        .word	0xca41
        .word	0xc85f
        .word	0xc613
        .word	0xc437
        .word	0xc1cc
        .word	0xbff0
        .word	0xbd60
        .word	0xbb38
        .word	0xb8c7
        .word	0xb641
        .word	0x288f
        .word	0x25b7
	.rept	10
        .word	0x0000
	.endr
        .word	0x7940
        .word	0x754a
        .word	0x713f
        .word	0x6d49
        .word	0x693d
        .word	0x6546
        .word	0x613a
        .word	0x5d42
        .word	0x5934
        .word	0x553a
        .word	0x5129
        .word	0x4d2b
        .word	0x4915
        .word	0x4510
        .word	0x40f3
        .word	0x3ce4
        .word	0x38bd
        .word	0x34a5
        .word	0x307b
        .word	0x2c6f
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x4400
	.endr
	.rept	32
        .word	0x0000
	.endr
        .type	__svml_hatanh_data_internal,@object
        .size	__svml_hatanh_data_internal,384
	 .section        .note.GNU-stack,"",@progbits
