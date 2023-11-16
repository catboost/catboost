/*******************************************
* Copyright (C) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *
 *  1) Range reduction to [-Pi/2; +Pi/2] interval
 *     a) We remove sign using AND operation
 *     b) Add Pi/2 value to argument X for Cos to Sin transformation
 *     c) Getting octant Y by 1/Pi multiplication
 *     d) Add "Right Shifter" value
 *     e) Treat obtained value as integer for destination sign setting.
 *        Shift first bit of this value to the last (sign) position
 *     f) Subtract "Right Shifter"  value
 *     g) Subtract 0.5 from result for octant correction
 *     h) Subtract Y*PI from X argument, where PI divided to 4 parts:
 *        X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
 *  2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
 *     a) Calculate X^2 = X * X
 *     b) Calculate polynomial:
 *        R = X + X * X^2 * (A3 + x^2 * (A5 +
 *  3) Destination sign setting
 *     a) Set shifted destination sign using XOR operation:
 *        R = XOR( R, S );
 *
 */

        .text

        .align    16,0x90
        .globl __svml_coss32

__svml_coss32:

        .cfi_startproc
        kxnord  %k7, %k7, %k7
        vmovdqu16 __svml_hcos_data_internal(%rip), %zmm30
        vmovdqu16 64+__svml_hcos_data_internal(%rip), %zmm29
        vmovdqu16 128+__svml_hcos_data_internal(%rip), %zmm31
        vmovdqu16 192+__svml_hcos_data_internal(%rip), %zmm28
        vmovdqu16 256+__svml_hcos_data_internal(%rip), %zmm27
        vmovdqu16 320+__svml_hcos_data_internal(%rip), %zmm26
        vmovdqu16 384+__svml_hcos_data_internal(%rip), %zmm22
        vmovdqu16 448+__svml_hcos_data_internal(%rip), %zmm25
        vmovdqu16 576+__svml_hcos_data_internal(%rip), %zmm24
        vmovdqu16 640+__svml_hcos_data_internal(%rip), %zmm21
        vmovdqu16 704+__svml_hcos_data_internal(%rip), %zmm23
        vmovdqu16 768+__svml_hcos_data_internal(%rip), %zmm20
        vmovdqu16 896+__svml_hcos_data_internal(%rip), %zmm19
        vmovdqu16 832+__svml_hcos_data_internal(%rip), %zmm18
        vmovdqu16 960+__svml_hcos_data_internal(%rip), %zmm17
        vmovdqu16 1024+__svml_hcos_data_internal(%rip), %zmm16

/* npy_half* in -> %rdi, npy_half* out -> %rsi, size_t N -> %rdx */
.looparray__cos_h:
        cmpq    $31, %rdx
        ja .loaddata__cos_h
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
        jmp .funcbegin_cos_h
.loaddata__cos_h:
        vmovdqu16 (%rdi), %zmm0
        addq $64, %rdi

.funcbegin_cos_h:

/* Check for +/- INF */
        vfpclassph $24, %zmm0, %k6
/* Dummy vsubph instruction to raise an invalid */
        vsubph  %zmm0, %zmm0, %zmm8{%k6}{z}

        vmovdqu16 %zmm22, %zmm8
        vmovdqu16 %zmm21, %zmm15

/*
 * Variables:
 * H
 * S
 * HM
 * No callout
 * Copy argument
 */
        vpandd    %zmm30, %zmm0, %zmm1
        vaddph    {rn-sae}, %zmm31, %zmm1, %zmm9

/* |sX| > threshold? */
        vpcmpgtw  %zmm29, %zmm1, %k1
        vfmadd213ph {rn-sae}, %zmm28, %zmm27, %zmm9

/* fN0 = (int)(sX/pi)*(2^(-5)) */
        vsubph    {rn-sae}, %zmm28, %zmm9, %zmm5

/*
 * sign bit, will treat hY as integer value to look at last bit
 * shift to FP16 sign position
 */
        vpsllw    $15, %zmm9, %zmm11

/* hN = ((int)(sX/pi)-0.5)*(2^(-5)) */
        vsubph    {rn-sae}, %zmm26, %zmm5, %zmm10
        vfmadd213ph {rn-sae}, %zmm1, %zmm10, %zmm8
        vfmadd213ph {rn-sae}, %zmm8, %zmm25, %zmm10

/* hR*hR */
        vmulph    {rn-sae}, %zmm10, %zmm10, %zmm14
        vfmadd231ph {rn-sae}, %zmm14, %zmm24, %zmm15
        vfmadd213ph {rn-sae}, %zmm23, %zmm14, %zmm15

/* short path */
        kortestd  %k1, %k1

/* set sign of R */
        vpxord    %zmm11, %zmm10, %zmm2

/* hR2*hR */
        vmulph    {rn-sae}, %zmm2, %zmm14, %zmm0

/* hR + hR3*fPoly */
        vfmadd213ph {rn-sae}, %zmm2, %zmm15, %zmm0

/* Go to exit */
        jne       .LBL_EXIT
                                # LOE rbx rbp r12 r13 r14 r15 zmm0 zmm1 k1

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__cos_h

        ret

/* Restore registers
 * and exit the function
 */

.LBL_EXIT:
        vmovdqu16 %zmm20, %zmm4
        vmovdqu16   1152+__svml_hcos_data_internal(%rip), %zmm12
        vmovdqu16   %zmm4, %zmm11

/* convert to FP32 */
        vextractf32x8 $1, %zmm1, %ymm3
        vcvtph2psx %ymm1, %zmm8
        vcvtph2psx %ymm3, %zmm10
        vfmadd231ps {rz-sae}, %zmm8, %zmm19, %zmm11
        vfmadd231ps {rz-sae}, %zmm10, %zmm19, %zmm4
        vsubps    {rn-sae}, %zmm18, %zmm11, %zmm2
        vsubps    {rn-sae}, %zmm18, %zmm4, %zmm1

/* sign bit, will treat sY as integer value to look at last bit */
        vpslld    $31, %zmm4, %zmm5
        vpslld    $31, %zmm11, %zmm3
        vfmsub231ps {rn-sae}, %zmm2, %zmm17, %zmm8
        vfmsub231ps {rn-sae}, %zmm1, %zmm17, %zmm10

/* c2*sR2 + c1 */
        vmovdqu16   1088+__svml_hcos_data_internal(%rip), %zmm4
        vfmadd213ps {rn-sae}, %zmm8, %zmm16, %zmm2
        vfmadd213ps {rn-sae}, %zmm10, %zmm16, %zmm1

/* sR*sR */
        vmulps    {rn-sae}, %zmm2, %zmm2, %zmm13
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm14
        vmovdqu16   %zmm4, %zmm15
        vfmadd231ps {rn-sae}, %zmm13, %zmm12, %zmm15
        vfmadd231ps {rn-sae}, %zmm14, %zmm12, %zmm4

/* sR2*sR */
        vmulps    {rn-sae}, %zmm2, %zmm13, %zmm12
        vmulps    {rn-sae}, %zmm1, %zmm14, %zmm13

/* sR + sR3*sPoly */
        vfmadd213ps {rn-sae}, %zmm2, %zmm15, %zmm12
        vfmadd213ps {rn-sae}, %zmm1, %zmm4, %zmm13

/* add sign bit to result (logical XOR between sPoly and i32_sgn bits) */
        vxorps    %zmm3, %zmm12, %zmm1
        vxorps    %zmm5, %zmm13, %zmm6
        vcvtps2phx %zmm1, %ymm2
        vcvtps2phx %zmm6, %ymm7
        vinsertf32x8 $1, %ymm7, %zmm2, %zmm8

/*
 * ensure results are always exactly the same for common arguments
 * (return fast path result for common args)
 */
        vpblendmw %zmm8, %zmm0, %zmm0{%k1}

/* store result to our array and adjust pointers */
        vmovdqu16 %zmm0, (%rsi){%k7}
        addq $64, %rsi
        subq $32, %rdx
        cmpq $0, %rdx
        jg .looparray__cos_h
        ret
                                # LOE rbx rbp r12 r13 r14 r15 zmm0
        .cfi_endproc

        .type	__svml_coss32,@function
        .size	__svml_coss32,.-__svml_coss32

        .section .rodata, "a"
        .align 64

__svml_hcos_data_internal:
	.rept	32
        .word	0x7fff
	.endr
	.rept	32
        .word	0x4bd9
	.endr
	.rept	32
        .word	0x3e48
	.endr
	.rept	32
        .word	0x5200
	.endr
	.rept	32
        .word	0x2118
	.endr
	.rept	32
        .word	0x2400
	.endr
	.rept	32
        .word	0xd648
	.endr
	.rept	32
        .word	0xa7ed
	.endr
	.rept	32
        .word	0x0001
	.endr
	.rept	32
        .word	0x8a2d
	.endr
	.rept	32
        .word	0x2042
	.endr
	.rept	32
        .word	0xb155
	.endr
	.rept	16
        .long	0x4b000000
	.endr
	.rept	16
        .long	0x4affffff
	.endr
	.rept	16
        .long	0x3ea2f983
	.endr
	.rept	16
        .long	0x40490fdb
	.endr
	.rept	16
        .long	0xb3bbbd2e
	.endr
	.rept	16
        .long	0xbe2a026e
	.endr
	.rept	16
        .long	0x3bf9f9b6
	.endr
	.rept	16
        .long	0x3f000000
	.endr
	.rept	16
        .long	0x3fc90fdb
	.endr
        .type	__svml_hcos_data_internal,@object
        .size	__svml_hcos_data_internal,1344
	 .section        .note.GNU-stack,"",@progbits
