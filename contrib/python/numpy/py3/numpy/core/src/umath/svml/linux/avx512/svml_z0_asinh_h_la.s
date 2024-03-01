/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *    Compute log(x+sqrt(x*x+1)) using RSQRT for starting the
 *    square root approximation, and small table lookups for log
 *
 *
 */

        .text

        .align    16,0x90
        .globl __svml_asinhs32

__svml_asinhs32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hasinh_data_internal(%rip), %zmm22
        vmovdqu16 64+__svml_hasinh_data_internal(%rip), %zmm31
        vmovdqu16 128+__svml_hasinh_data_internal(%rip), %zmm24
        vmovdqu16 192+__svml_hasinh_data_internal(%rip), %zmm25
        vmovdqu16 256+__svml_hasinh_data_internal(%rip), %zmm23
        vmovdqu16 320+__svml_hasinh_data_internal(%rip), %zmm21
        vmovdqu16 384+__svml_hasinh_data_internal(%rip), %zmm27
        vmovdqu16 448+__svml_hasinh_data_internal(%rip), %zmm8
        vmovdqu16 512+__svml_hasinh_data_internal(%rip), %zmm9
        vmovdqu16 576+__svml_hasinh_data_internal(%rip), %zmm11
        vmovdqu16 640+__svml_hasinh_data_internal(%rip), %zmm30
        vmovdqu16 704+__svml_hasinh_data_internal(%rip), %zmm15
        vmovdqu16 768+__svml_hasinh_data_internal(%rip), %zmm29
        vmovdqu16 832+__svml_hasinh_data_internal(%rip), %zmm28
        vmovdqu16 896+__svml_hasinh_data_internal(%rip), %zmm26

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray_asinh_h:
        cmpq    $31, %rdx
        ja .loaddata_asinh
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
        jmp .funcbegin_asinh_h
.loaddata_asinh:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi
        
.funcbegin_asinh_h:

        vmovdqu16 %zmm24, %zmm2
        vmovdqu16 %zmm22, %zmm1
        vmovdqu16 %zmm21, %zmm7

/*
 * No callout
 * |x|
 */
        vpandd    %zmm31, %zmm0, %zmm4
        vfmadd213ph {rn-sae}, %zmm25, %zmm4, %zmm2

/*
 * log(|x|)
 * GetMant(x), normalized to [.75,1.5) for x>=0, NaN for x<0
 */
        vgetmantph $11, {sae}, %zmm4, %zmm12

/*
 * |x|>=1:  result = log(x) + log(1+sqrt(1+(1/x)^2))
 * y = 1/|x|
 */
        vrcpph    %zmm4, %zmm10
        vgetexpph {sae}, %zmm4, %zmm13
        vcmpph    $21, {sae}, %zmm1, %zmm4, %k1

/* exponent correction */
        vgetexpph {sae}, %zmm12, %zmm14
        vfmadd213ph {rn-sae}, %zmm23, %zmm4, %zmm2

/* mantissa - 1 */
        vsubph    {rn-sae}, %zmm1, %zmm12, %zmm5

/* log(1+R)/R */
        vmovdqu16 %zmm30, %zmm6
        vfmadd213ph {rn-sae}, %zmm7, %zmm4, %zmm2
        vsubph    {rn-sae}, %zmm14, %zmm13, %zmm7
        vfmadd213ph {rn-sae}, %zmm15, %zmm5, %zmm6
        vfmadd213ph {rn-sae}, %zmm4, %zmm4, %zmm2
        vfmadd213ph {rn-sae}, %zmm29, %zmm5, %zmm6
        vfmadd213ph {rn-sae}, %zmm28, %zmm5, %zmm6

/* save sign */
        vpxord    %zmm4, %zmm0, %zmm3
        vmovdqu16 %zmm27, %zmm0
        vfmadd213ph {rn-sae}, %zmm8, %zmm10, %zmm0
        vfmadd213ph {rn-sae}, %zmm9, %zmm10, %zmm0
        vfmadd213ph {rn-sae}, %zmm11, %zmm10, %zmm0

/* log(1+R) + log(1+sqrt(1+y*y)) */
        vfmadd213ph {rn-sae}, %zmm0, %zmm5, %zmm6

/* result for x>=2 */
        vfmadd213ph {rn-sae}, %zmm6, %zmm26, %zmm7

/* result = SelMask?  hPl : hPa */
        vpblendmw %zmm7, %zmm2, %zmm1{%k1}

/* set sign */
        vpxord    %zmm3, %zmm1, %zmm0
/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray_asinh_h
        ret

        .cfi_endproc

        .type	__svml_asinhs32,@function
        .size	__svml_asinhs32,.-__svml_asinhs32

        .section .rodata, "a"
        .align 64

__svml_hasinh_data_internal:
	.rept	32
        .word	0x3c00
	.endr
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x2c52
	.endr
	.rept	32
        .word	0xb206
	.endr
	.rept	32
        .word	0x185d
	.endr
	.rept	32
        .word	0x8057
	.endr
	.rept	32
        .word	0xadb6
	.endr
	.rept	32
        .word	0x347f
	.endr
	.rept	32
        .word	0x9b9e
	.endr
	.rept	32
        .word	0x398c
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
        .type	__svml_hasinh_data_internal,@object
        .size	__svml_hasinh_data_internal,960
	 .section        .note.GNU-stack,"",@progbits
