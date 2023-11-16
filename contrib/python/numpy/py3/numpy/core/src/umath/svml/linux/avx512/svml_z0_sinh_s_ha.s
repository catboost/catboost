/*******************************************************************************
* INTEL CONFIDENTIAL
* Copyright 1996-2023 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/


	.text
.L_2__routine_start___svml_sinhf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_sinhf16_ha

__svml_sinhf16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   1152+__svml_ssinh_ha_data_internal(%rip), %zmm10

/*
 * ............... Load argument ............................
 * dM = x/log(2) + RShifter
 */
        vmovups   960+__svml_ssinh_ha_data_internal(%rip), %zmm9
        vmovups   576+__svml_ssinh_ha_data_internal(%rip), %zmm5
        vmovups   1024+__svml_ssinh_ha_data_internal(%rip), %zmm1
        vmovups   1088+__svml_ssinh_ha_data_internal(%rip), %zmm8

/* ... */
        vmovups   896+__svml_ssinh_ha_data_internal(%rip), %zmm12

/* x^2 */
        vmovups   832+__svml_ssinh_ha_data_internal(%rip), %zmm13

/*
 * ...............Check for overflow\underflow .............
 * MORE faster than GE?
 */
        vpternlogd $255, %zmm4, %zmm4, %zmm4
        vmovaps   %zmm0, %zmm15

/*
 * ----------------------------------- Implementation  ---------------------
 * ............... Abs argument ............................
 */
        vandps    %zmm15, %zmm10, %zmm14
        vmovups   512+__svml_ssinh_ha_data_internal(%rip), %zmm0
        vxorps    %zmm15, %zmm14, %zmm11
        vfmadd213ps {rn-sae}, %zmm0, %zmm11, %zmm9
        vpcmpd    $2, 704+__svml_ssinh_ha_data_internal(%rip), %zmm11, %k1
        vcmpps    $0, {sae}, %zmm9, %zmm5, %k2
        vmovups   256+__svml_ssinh_ha_data_internal(%rip), %zmm5
        vblendmps %zmm0, %zmm9, %zmm6{%k2}

/* sN = sM - RShifter */
        vsubps    {rn-sae}, %zmm0, %zmm6, %zmm3
        vmovups   128+__svml_ssinh_ha_data_internal(%rip), %zmm0
        vpandnd   %zmm11, %zmm11, %zmm4{%k1}

/*
 * ................... R ...................................
 * sR = sX - sN*Log2_hi
 */
        vfnmadd231ps {rn-sae}, %zmm1, %zmm3, %zmm11
        vptestmd  %zmm4, %zmm4, %k0
        vmovups   __svml_ssinh_ha_data_internal(%rip), %zmm4

/* sR = (sX - sN*Log2_hi) - sN*Log2_lo */
        vfnmadd231ps {rn-sae}, %zmm8, %zmm3, %zmm11
        kmovw     %k0, %edx

/* sR2 = sR^2 */
        vmulps    {rn-sae}, %zmm11, %zmm11, %zmm9
        vmulps    {rn-sae}, %zmm9, %zmm12, %zmm12

/* sSinh_r = r + r*(r^2*(a3)) */
        vfmadd213ps {rn-sae}, %zmm11, %zmm11, %zmm12

/* sOut = r^2*(a2) */
        vmulps    {rn-sae}, %zmm9, %zmm13, %zmm11

/* ............... G1,G2 2^N,2^(-N) ........... */
        vpandd    640+__svml_ssinh_ha_data_internal(%rip), %zmm6, %zmm7
        vpxord    %zmm7, %zmm6, %zmm2
        vpslld    $18, %zmm2, %zmm8
        vmovups   384+__svml_ssinh_ha_data_internal(%rip), %zmm2
        vpermt2ps 320+__svml_ssinh_ha_data_internal(%rip), %zmm7, %zmm5
        vpermt2ps 64+__svml_ssinh_ha_data_internal(%rip), %zmm7, %zmm4
        vpermt2ps 448+__svml_ssinh_ha_data_internal(%rip), %zmm7, %zmm2
        vpsubd    %zmm8, %zmm5, %zmm3
        vpaddd    %zmm8, %zmm4, %zmm6
        vpermt2ps 192+__svml_ssinh_ha_data_internal(%rip), %zmm7, %zmm0

/* if |sTnl| < sM, then scaled sTnl = 0 */
        vandnps   %zmm2, %zmm10, %zmm7
        vsubps    {rn-sae}, %zmm3, %zmm6, %zmm5
        vpaddd    %zmm8, %zmm0, %zmm1
        vpsubd    %zmm8, %zmm2, %zmm0
        vsubps    {rn-sae}, %zmm6, %zmm5, %zmm10
        vaddps    {rn-sae}, %zmm3, %zmm6, %zmm4
        vcmpps    $17, {sae}, %zmm8, %zmm7, %k3
        vaddps    {rn-sae}, %zmm10, %zmm3, %zmm2
        vxorps    %zmm0, %zmm0, %zmm0{%k3}
        vsubps    {rn-sae}, %zmm2, %zmm1, %zmm1
        vsubps    {rn-sae}, %zmm0, %zmm1, %zmm0

/* res = sG1*(r + r*(r^2*(a3))) + sG2*(1+r^2*(a2)) */
        vfmadd213ps {rn-sae}, %zmm0, %zmm5, %zmm11
        vfmadd213ps {rn-sae}, %zmm11, %zmm12, %zmm4
        vaddps    {rn-sae}, %zmm5, %zmm4, %zmm13

/* ................... Ret H ...................... */
        vorps     %zmm13, %zmm14, %zmm0
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm15, 64(%rsp)
        vmovups   %zmm0, 128(%rsp)
        je        .LBL_1_2


        xorl      %eax, %eax


        vzeroupper
        kmovw     %k4, 24(%rsp)
        kmovw     %k5, 16(%rsp)
        kmovw     %k6, 8(%rsp)
        kmovw     %k7, (%rsp)
        movq      %rsi, 40(%rsp)
        movq      %rdi, 32(%rsp)
        movq      %r12, 56(%rsp)
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 48(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_7:

        btl       %r12d, %r13d
        jc        .LBL_1_10

.LBL_1_8:

        incl      %r12d
        cmpl      $16, %r12d
        jl        .LBL_1_7


        kmovw     24(%rsp), %k4
	.cfi_restore 122
        kmovw     16(%rsp), %k5
	.cfi_restore 123
        kmovw     8(%rsp), %k6
	.cfi_restore 124
        kmovw     (%rsp), %k7
	.cfi_restore 125
        vmovups   128(%rsp), %zmm0
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_2
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_10:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi

        call      __svml_ssinh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sinhf16_ha,@function
	.size	__svml_sinhf16_ha,.-__svml_sinhf16_ha
..LN__svml_sinhf16_ha.0:

.L_2__routine_start___svml_ssinh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_ssinh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r9
        movzwl    2(%rdi), %edx
        xorl      %eax, %eax
        andl      $32640, %edx
        movss     (%rdi), %xmm2
        cmpl      $32640, %edx
        je        .LBL_2_17


        cvtss2sd  %xmm2, %xmm2
        movsd     %xmm2, -8(%rsp)
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        movsd     %xmm2, -32(%rsp)
        shrl      $4, %edx
        andb      $127, -25(%rsp)
        testl     %edx, %edx
        jle       .LBL_2_16


        cmpl      $969, %edx
        jle       .LBL_2_14


        movsd     -32(%rsp), %xmm0
        movsd     1136+__ssinh_ha_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_13


        movsd     1184+__ssinh_ha_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        comisd    1176+__ssinh_ha_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1112+__ssinh_ha_CoutTab(%rip), %xmm3
        lea       __ssinh_ha_CoutTab(%rip), %rcx
        mulsd     %xmm0, %xmm3
        movsd     1144+__ssinh_ha_CoutTab(%rip), %xmm10
        movq      8+__ssinh_ha_CoutTab(%rip), %r10
        movq      %r10, %rsi
        shrq      $48, %rsi
        addsd     1120+__ssinh_ha_CoutTab(%rip), %xmm3
        movsd     %xmm3, -40(%rsp)
        andl      $-32753, %esi
        movsd     -40(%rsp), %xmm13
        movl      -40(%rsp), %r8d
        movl      %r8d, %r11d
        shrl      $6, %r11d
        andl      $63, %r8d
        movq      %r10, -16(%rsp)
        subsd     1120+__ssinh_ha_CoutTab(%rip), %xmm13
        mulsd     %xmm13, %xmm10
        lea       1023(%r11), %edi
        xorps     .L_2il0floatpacket.96(%rip), %xmm13
        addl      $1022, %r11d
        mulsd     1152+__ssinh_ha_CoutTab(%rip), %xmm13
        subsd     %xmm10, %xmm0
        movaps    %xmm0, %xmm5
        movaps    %xmm0, %xmm11
        andl      $2047, %r11d
        lea       (%r8,%r8), %edx
        negl      %edi
        lea       1(%r8,%r8), %r8d
        movsd     (%rcx,%rdx,8), %xmm8
        negl      %edx
        shll      $4, %r11d
        addl      $-4, %edi
        orl       %r11d, %esi
        andl      $2047, %edi
        movw      %si, -10(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        addsd     %xmm13, %xmm5
        movsd     %xmm5, -24(%rsp)
        orl       %edi, %esi
        movsd     -24(%rsp), %xmm7
        movsd     1128+__ssinh_ha_CoutTab(%rip), %xmm5
        subsd     %xmm7, %xmm11
        movsd     %xmm11, -56(%rsp)
        movsd     -24(%rsp), %xmm4
        movsd     -56(%rsp), %xmm12
        movsd     (%rcx,%r8,8), %xmm6
        addsd     %xmm12, %xmm4
        movsd     %xmm4, -48(%rsp)
        movsd     -56(%rsp), %xmm9
        movsd     -16(%rsp), %xmm4
        addsd     %xmm9, %xmm13
        mulsd     %xmm4, %xmm8
        mulsd     %xmm4, %xmm6
        movsd     %xmm13, -56(%rsp)
        movaps    %xmm8, %xmm9
        movsd     -48(%rsp), %xmm15
        movw      %si, -10(%rsp)
        lea       128(%rdx), %esi
        movsd     -16(%rsp), %xmm14
        addl      $129, %edx
        subsd     %xmm15, %xmm0
        movaps    %xmm8, %xmm15
        movsd     %xmm0, -48(%rsp)
        movsd     -56(%rsp), %xmm3
        movsd     -48(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        movsd     %xmm3, -48(%rsp)
        movsd     -24(%rsp), %xmm10
        mulsd     %xmm10, %xmm5
        movaps    %xmm10, %xmm2
        mulsd     %xmm10, %xmm2
        movsd     -48(%rsp), %xmm3
        movaps    %xmm10, %xmm1
        movsd     %xmm5, -24(%rsp)
        movsd     -24(%rsp), %xmm7
        subsd     %xmm10, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -24(%rsp), %xmm12
        movsd     -56(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     1064+__ssinh_ha_CoutTab(%rip), %xmm11
        mulsd     %xmm2, %xmm11
        movsd     %xmm12, -24(%rsp)
        movsd     1072+__ssinh_ha_CoutTab(%rip), %xmm12
        mulsd     %xmm2, %xmm12
        addsd     1048+__ssinh_ha_CoutTab(%rip), %xmm11
        mulsd     %xmm2, %xmm11
        addsd     1056+__ssinh_ha_CoutTab(%rip), %xmm12
        mulsd     %xmm2, %xmm12
        mulsd     %xmm10, %xmm11
        addsd     1040+__ssinh_ha_CoutTab(%rip), %xmm12
        addsd     %xmm11, %xmm10
        mulsd     %xmm2, %xmm12
        movsd     (%rcx,%rsi,8), %xmm2
        mulsd     %xmm14, %xmm2
        movsd     -24(%rsp), %xmm0
        subsd     %xmm2, %xmm9
        subsd     %xmm0, %xmm1
        movsd     %xmm1, -56(%rsp)
        movsd     -24(%rsp), %xmm7
        movsd     -56(%rsp), %xmm5
        movsd     %xmm9, -24(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     (%rcx,%rdx,8), %xmm1
        subsd     %xmm13, %xmm15
        mulsd     %xmm14, %xmm1
        subsd     %xmm2, %xmm15
        movsd     %xmm15, -56(%rsp)
        movaps    %xmm8, %xmm13
        movsd     -24(%rsp), %xmm14
        addsd     %xmm2, %xmm13
        movsd     -56(%rsp), %xmm9
        movaps    %xmm14, %xmm0
        movb      -1(%rsp), %cl
        addsd     %xmm6, %xmm9
        addsd     %xmm1, %xmm6
        subsd     %xmm1, %xmm9
        andb      $-128, %cl
        addsd     %xmm9, %xmm0
        movsd     %xmm0, -24(%rsp)
        movsd     -24(%rsp), %xmm4
        subsd     %xmm4, %xmm14
        addsd     %xmm14, %xmm9
        movsd     %xmm9, -56(%rsp)
        movsd     -24(%rsp), %xmm9
        movsd     -56(%rsp), %xmm0
        movsd     %xmm13, -24(%rsp)
        movsd     -24(%rsp), %xmm15
        subsd     %xmm15, %xmm8
        addsd     %xmm8, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -24(%rsp), %xmm2
        movsd     -56(%rsp), %xmm4
        addsd     %xmm6, %xmm4
        movaps    %xmm2, %xmm6
        addsd     %xmm4, %xmm6
        movsd     %xmm6, -24(%rsp)
        movsd     -24(%rsp), %xmm8
        movsd     1128+__ssinh_ha_CoutTab(%rip), %xmm6
        subsd     %xmm8, %xmm2
        addsd     %xmm2, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -24(%rsp), %xmm1
        mulsd     %xmm1, %xmm6
        movsd     -56(%rsp), %xmm2
        movsd     %xmm6, -24(%rsp)
        movaps    %xmm1, %xmm6
        movsd     -24(%rsp), %xmm14
        mulsd     %xmm2, %xmm10
        subsd     %xmm1, %xmm14
        movsd     %xmm14, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm8
        subsd     %xmm8, %xmm13
        movsd     %xmm13, -24(%rsp)
        movaps    %xmm11, %xmm13
        movsd     -24(%rsp), %xmm15
        mulsd     %xmm1, %xmm13
        subsd     %xmm15, %xmm6
        mulsd     %xmm3, %xmm1
        mulsd     %xmm2, %xmm3
        movaps    %xmm12, %xmm15
        movaps    %xmm13, %xmm4
        mulsd     %xmm9, %xmm15
        mulsd     %xmm0, %xmm12
        addsd     %xmm15, %xmm4
        addsd     %xmm0, %xmm12
        movsd     %xmm6, -56(%rsp)
        addsd     %xmm1, %xmm12
        movsd     -24(%rsp), %xmm8
        addsd     %xmm3, %xmm12
        movsd     -56(%rsp), %xmm6
        movsd     %xmm4, -24(%rsp)
        movsd     -24(%rsp), %xmm14
        subsd     %xmm14, %xmm13
        addsd     %xmm13, %xmm15
        movsd     %xmm15, -56(%rsp)
        movaps    %xmm7, %xmm15
        mulsd     %xmm8, %xmm15
        mulsd     %xmm5, %xmm8
        mulsd     %xmm6, %xmm5
        mulsd     %xmm6, %xmm7
        movsd     -24(%rsp), %xmm14
        movaps    %xmm14, %xmm13
        movsd     -56(%rsp), %xmm4
        addsd     %xmm15, %xmm13
        addsd     %xmm8, %xmm4
        movsd     %xmm13, -24(%rsp)
        addsd     %xmm5, %xmm4
        movsd     -24(%rsp), %xmm13
        addsd     %xmm7, %xmm4
        subsd     %xmm13, %xmm15
        addsd     %xmm4, %xmm12
        addsd     %xmm15, %xmm14
        movsd     %xmm14, -56(%rsp)
        movaps    %xmm9, %xmm15
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm14
        addsd     %xmm13, %xmm15
        addsd     %xmm14, %xmm12
        movsd     %xmm15, -24(%rsp)
        movsd     -24(%rsp), %xmm15
        subsd     %xmm15, %xmm9
        addsd     %xmm9, %xmm13
        movsd     %xmm13, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm9
        addsd     %xmm9, %xmm12
        addsd     %xmm12, %xmm13
        addsd     %xmm13, %xmm10
        movsd     %xmm10, -32(%rsp)
        movb      -25(%rsp), %dil
        andb      $127, %dil
        orb       %cl, %dil
        movb      %dil, -25(%rsp)
        movsd     -32(%rsp), %xmm10
        cvtsd2ss  %xmm10, %xmm10
        movss     %xmm10, (%r9)
        ret

.LBL_2_8:

        movaps    %xmm0, %xmm2
        mulsd     %xmm0, %xmm2
        movsd     1104+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        movb      -1(%rsp), %dl
        andb      $-128, %dl
        addsd     1096+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1088+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1080+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm2
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -32(%rsp)
        movb      -25(%rsp), %cl
        andb      $127, %cl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movsd     -32(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_9:

        movsd     1112+__ssinh_ha_CoutTab(%rip), %xmm1
        lea       __ssinh_ha_CoutTab(%rip), %r8
        mulsd     %xmm0, %xmm1
        movsd     1144+__ssinh_ha_CoutTab(%rip), %xmm2
        movsd     1152+__ssinh_ha_CoutTab(%rip), %xmm3
        movq      8+__ssinh_ha_CoutTab(%rip), %rdx
        movq      %rdx, -16(%rsp)
        addsd     1120+__ssinh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm4
        movsd     1072+__ssinh_ha_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1120+__ssinh_ha_CoutTab(%rip), %xmm4
        mulsd     %xmm4, %xmm2
        lea       (%rsi,%rsi), %ecx
        mulsd     %xmm3, %xmm4
        subsd     %xmm2, %xmm0
        movsd     (%r8,%rcx,8), %xmm5
        lea       1(%rsi,%rsi), %edi
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1064+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1048+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1040+__ssinh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        mulsd     %xmm5, %xmm1
        addsd     (%r8,%rdi,8), %xmm1
        addsd     %xmm5, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_11


        movq      8+__ssinh_ha_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -10(%rsp)
        movsd     -16(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        movsd     %xmm0, -32(%rsp)
        jmp       .LBL_2_12

.LBL_2_11:

        decl      %edx
        andl      $2047, %edx
        movzwl    -10(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -10(%rsp)
        movsd     -16(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        mulsd     1024+__ssinh_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -32(%rsp)

.LBL_2_12:

        movb      -25(%rsp), %cl
        movb      -1(%rsp), %dl
        andb      $127, %cl
        andb      $-128, %dl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movsd     -32(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_13:

        movsd     1168+__ssinh_ha_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm2, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_14:

        movsd     __ssinh_ha_CoutTab(%rip), %xmm0
        addsd     1160+__ssinh_ha_CoutTab(%rip), %xmm0
        mulsd     %xmm2, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)


        ret

.LBL_2_16:

        movsd     1160+__ssinh_ha_CoutTab(%rip), %xmm0
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, -24(%rsp)
        pxor      %xmm2, %xmm2
        cvtss2sd  (%rdi), %xmm2
        movsd     -24(%rsp), %xmm1
        movq      8+__ssinh_ha_CoutTab(%rip), %rdx
        addsd     %xmm1, %xmm2
        cvtsd2ss  %xmm2, %xmm2
        movq      %rdx, -16(%rsp)
        movss     %xmm2, (%r9)
        ret

.LBL_2_17:

        addss     %xmm2, %xmm2
        movss     %xmm2, (%r9)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_ssinh_ha_cout_rare_internal,@function
	.size	__svml_ssinh_ha_cout_rare_internal,.-__svml_ssinh_ha_cout_rare_internal
..LN__svml_ssinh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_ssinh_ha_data_internal:
	.long	1056964608
	.long	1057148295
	.long	1057336003
	.long	1057527823
	.long	1057723842
	.long	1057924154
	.long	1058128851
	.long	1058338032
	.long	1058551792
	.long	1058770234
	.long	1058993458
	.long	1059221571
	.long	1059454679
	.long	1059692891
	.long	1059936319
	.long	1060185078
	.long	1060439283
	.long	1060699055
	.long	1060964516
	.long	1061235789
	.long	1061513002
	.long	1061796286
	.long	1062085772
	.long	1062381598
	.long	1062683901
	.long	1062992824
	.long	1063308511
	.long	1063631111
	.long	1063960775
	.long	1064297658
	.long	1064641917
	.long	1064993715
	.long	0
	.long	2999887785
	.long	852465809
	.long	3003046475
	.long	2984291233
	.long	3001644133
	.long	854021668
	.long	2997748242
	.long	849550193
	.long	2995541347
	.long	851518274
	.long	809701978
	.long	2997656926
	.long	2996185864
	.long	2980965110
	.long	3002882728
	.long	844097402
	.long	848217591
	.long	2999013352
	.long	2992006718
	.long	831170615
	.long	3002278818
	.long	833158180
	.long	3000769962
	.long	2991891850
	.long	2999994908
	.long	2979965785
	.long	2982419430
	.long	2982221534
	.long	2999469642
	.long	833168438
	.long	2987538264
	.long	1056964608
	.long	1056605107
	.long	1056253309
	.long	1055909050
	.long	1055572167
	.long	1055242503
	.long	1054919903
	.long	1054604216
	.long	1054295293
	.long	1053992990
	.long	1053697164
	.long	1053407678
	.long	1053124394
	.long	1052847181
	.long	1052575908
	.long	1052310447
	.long	1052050675
	.long	1051796470
	.long	1051547711
	.long	1051304283
	.long	1051066071
	.long	1050832963
	.long	1050604850
	.long	1050381626
	.long	1050163184
	.long	1049949424
	.long	1049740243
	.long	1049535546
	.long	1049335234
	.long	1049139215
	.long	1048947395
	.long	1048759687
	.long	0
	.long	2979149656
	.long	824779830
	.long	2991081034
	.long	2973832926
	.long	2974030822
	.long	2971577177
	.long	2991606300
	.long	2983503242
	.long	2992381354
	.long	824769572
	.long	2993890210
	.long	822782007
	.long	2983618110
	.long	2990624744
	.long	839828983
	.long	835708794
	.long	2994494120
	.long	2972576502
	.long	2987797256
	.long	2989268318
	.long	801313370
	.long	843129666
	.long	2987152739
	.long	841161585
	.long	2989359634
	.long	845633060
	.long	2993255525
	.long	2975902625
	.long	2994657867
	.long	844077201
	.long	2991499177
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	1118922496
	.long	0
	.long	687887406
	.long	2915115070
	.long	1042983726
	.long	929258934
	.long	980813922
	.long	1018266026
	.long	1042992474
	.long	954428950
	.long	997598593
	.long	1026665631
	.long	1043023968
	.long	968883188
	.long	1007325380
	.long	1032156897
	.long	1043076692
	.long	979611573
	.long	1014406071
	.long	1035098005
	.long	1043150518
	.long	987662767
	.long	1019277074
	.long	1038061036
	.long	1043245588
	.long	994080931
	.long	1024140990
	.long	1040619474
	.long	1043361758
	.long	999883086
	.long	1027459341
	.long	1042131339
	.long	1043501068
	.long	1004844750
	.long	1031304911
	.long	1043662546
	.long	1043660792
	.long	1008932267
	.long	1033741849
	.long	1045216058
	.long	1043844218
	.long	1012931568
	.long	1036203181
	.long	1046794669
	.long	1044047389
	.long	1016426573
	.long	1038940459
	.long	1048401520
	.long	1044276784
	.long	1019375189
	.long	1041073109
	.long	1049307698
	.long	1044526186
	.long	1022871607
	.long	1042725668
	.long	1050143823
	.long	1044801107
	.long	1025188112
	.long	1044524705
	.long	1050998533
	.long	1045100274
	.long	1027560014
	.long	1046473595
	.long	1051873886
	.long	1045430084
	.long	1030282880
	.long	1048576000
	.long	1052770896
	.long	1045780079
	.long	1032591216
	.long	1049705931
	.long	1053691782
	.long	1046163095
	.long	1034344377
	.long	1050916716
	.long	1054637154
	.long	1046554568
	.long	1036314518
	.long	1052210623
	.long	1055610247
	.long	1046982038
	.long	1038516252
	.long	1053590081
	.long	1056612417
	.long	1047444104
	.long	1040576009
	.long	1055057680
	.long	1057305094
	.long	1047938203
	.long	1041931271
	.long	1056616175
	.long	1057838147
	.long	1048464140
	.long	1043425610
	.long	1057616551
	.long	1058387990
	.long	1048791818
	.long	1045067287
	.long	1058491172
	.long	1058956436
	.long	1049080411
	.long	1046864840
	.long	1059415895
	.long	1059545021
	.long	1049393183
	.long	1048701551
	.long	1060392458
	.long	1060153606
	.long	1049717660
	.long	1049769606
	.long	1061422692
	.long	1060784342
	.long	1050066322
	.long	1050929319
	.long	1062508534
	.long	1061437519
	.long	1050422447
	.long	1052185595
	.long	1063652019
	.long	1062114959
	.long	1050803760
	.long	1053543521
	.long	1064855295
	.long	1062817471
	.long	1051202252
	.long	1055008374
	.long	1065736918
	.long	1063547051
	.long	1051622601
	.long	3097084200
	.long	1074266112
	.long	1064305255
	.long	1052071435
	.long	3097592230
	.long	1074615279
	.long	1065092533
	.long	1052543428
	.long	3098127090
	.long	1074981832
	.long	1065632015
	.long	1053027915
	.long	3098657586
	.long	1075366458
	.long	1066057424
	.long	1053547140
	.long	3099216842
	.long	1075769880
	.long	1066499901
	.long	1054080955
	.long	3099820420
	.long	1076192855
	.long	1066960277
	.long	1054635449
	.long	3100431607
	.long	1076636176
	.long	1067439415
	.long	1055231108
	.long	3101072121
	.long	1077100676
	.long	1067938215
	.long	1055851490
	.long	3101734019
	.long	1077587227
	.long	1068457613
	.long	1056495628
	.long	3102420416
	.long	1078096742
	.long	1068998584
	.long	1057067604
	.long	3103151062
	.long	1078630177
	.long	1069562144
	.long	1057425150
	.long	3103842417
	.long	1079188534
	.long	1070149350
	.long	1057796175
	.long	3104239345
	.long	1079772860
	.long	1070761305
	.long	1058192335
	.long	3104632042
	.long	1080384254
	.long	1071399156
	.long	1058592040
	.long	3105065708
	.long	1081023861
	.long	1072064103
	.long	1059022895
	.long	3105522352
	.long	1081692883
	.long	1072757393
	.long	1059471212
	.long	3105980727
	.long	1082261504
	.long	1073480326
	.long	1059935747
	.long	3106458228
	.long	1082627342
	.long	1073988042
	.long	1060431367
	.long	3106985545
	.long	1083009859
	.long	1074381218
	.long	1060942660
	.long	3107497595
	.long	1083409773
	.long	1074791339
	.long	1061470753
	.long	3108033911
	.long	1083827834
	.long	1075219176
	.long	1062030223
	.long	3108625747
	.long	1084264827
	.long	1075665533
	.long	1062616535
	.long	3109213903
	.long	1084721573
	.long	1076131246
	.long	1063215716
	.long	3109826597
	.long	1085198928
	.long	1076617190
	.long	1063856328
	.long	3110492915
	.long	1085697789
	.long	1077124278
	.long	1064519640
	.long	3111153932
	.long	1086219092
	.long	1077653460
	.long	1065214942
	.long	3111866338
	.long	1086763816
	.long	1078205731
	.long	1065643458
	.long	3112375523
	.long	1087332983
	.long	1078782126
	.long	1066020158
	.long	3112765050
	.long	1087927661
	.long	1079383729
	.long	1066418599
	.long	3113168833
	.long	1088548967
	.long	1080011668
	.long	1066831834
	.long	3113609244
	.long	1089198066
	.long	1080667123
	.long	1067273229
	.long	3114032535
	.long	1089876179
	.long	1081351321
	.long	1067717011
	.long	3114484025
	.long	1090551808
	.long	1082065549
	.long	1068193280
	.long	3114970280
	.long	1090921814
	.long	1082470790
	.long	1068689694
	.long	3115467036
	.long	1091308322
	.long	1082859974
	.long	1069207685
	.long	3115974474
	.long	1091712058
	.long	1083266273
	.long	1069737995
	.long	3116537826
	.long	1092133779
	.long	1083690451
	.long	1070298768
	.long	3117085761
	.long	1092574277
	.long	1084133302
	.long	1070882802
	.long	3117709126
	.long	1093034378
	.long	1084595660
	.long	1071508439
	.long	3118314866
	.long	1093514947
	.long	1085078390
	.long	1072148994
	.long	3118933130
	.long	1094016886
	.long	1085582399
	.long	1072806866
	.long	3119628767
	.long	1094541136
	.long	1086108635
	.long	1073507255
	.long	3120312034
	.long	1095088682
	.long	1086658083
	.long	1073986932
	.long	3120816642
	.long	1095660551
	.long	1087231777
	.long	1074370169
	.long	3121187932
	.long	1096257817
	.long	1087830791
	.long	1074769178
	.long	3121594488
	.long	1096881601
	.long	1088456252
	.long	1075185795
	.long	3122020198
	.long	1097533074
	.long	1089109333
	.long	1075619595
	.long	3122451537
	.long	1098213459
	.long	1089791259
	.long	1076069917
	.long	3122905402
	.long	1098915840
	.long	1090503311
	.long	1076549119
	.long	3123389748
	.long	1099286888
	.long	1090882933
	.long	1077045731
	.long	3123878864
	.long	1099674394
	.long	1091271119
	.long	1077560283
	.long	3124401536
	.long	1100079085
	.long	1091676463
	.long	1078101378
	.long	3124930682
	.long	1100501721
	.long	1092099725
	.long	1078662472
	.long	3125516800
	.long	1100943095
	.long	1092541701
	.long	1079251056
	.long	3126075229
	.long	1101404036
	.long	1093003218
	.long	1079857728
	.long	3126728388
	.long	1101885408
	.long	1093485146
	.long	1080508502
	.long	3127359219
	.long	1102388116
	.long	1093988386
	.long	1081175245
	.long	3128014352
	.long	1102913103
	.long	1094513884
	.long	1081871787
	.long	3128747686
	.long	1103461354
	.long	1095062628
	.long	1082369373
	.long	3129206088
	.long	1104033899
	.long	1095635645
	.long	1082749126
	.long	3129593301
	.long	1104631812
	.long	1096234013
	.long	1083148279
	.long	3130008743
	.long	1105256215
	.long	1096858855
	.long	1083570858
	.long	3130406199
	.long	1105908282
	.long	1097511341
	.long	1083997642
	.long	3130855937
	.long	1106589234
	.long	1098192700
	.long	1084459829
	.long	3131310395
	.long	1107298304
	.long	1098904208
	.long	1084929536
	.long	3131761492
	.long	1107669613
	.long	1099277424
	.long	1085415965
	.long	3132265084
	.long	1108057368
	.long	1099665361
	.long	1085939887
	.long	3132783371
	.long	1108462298
	.long	1100070466
	.long	1086478564
	.long	3133369511
	.long	1108885162
	.long	1100493501
	.long	1087055088
	.long	3133891436
	.long	1109326756
	.long	1100935256
	.long	1087624344
	.long	3134507369
	.long	1109787906
	.long	1101396565
	.long	1088246740
	.long	3135123225
	.long	1110269479
	.long	1101878291
	.long	1088894950
	.long	3135765391
	.long	1110772379
	.long	1102381339
	.long	1089569026
	.long	3136459557
	.long	1111297550
	.long	1102906654
	.long	1090269725
	.long	3137139863
	.long	1111845978
	.long	1103455220
	.long	1090756438
	.long	3137594905
	.long	1112418692
	.long	1104028068
	.long	1091135322
	.long	3137977906
	.long	1113016767
	.long	1104626274
	.long	1091531952
	.long	3138391473
	.long	1113641325
	.long	1105250961
	.long	1091953464
	.long	3138794156
	.long	1114293540
	.long	1105903299
	.long	1092383610
	.long	3139244396
	.long	1114974634
	.long	1106584516
	.long	1092846205
	.long	3139699003
	.long	1115685376
	.long	1107295888
	.long	1093316096
	.long	3140154077
	.long	1116056750
	.long	1107667503
	.long	1093805095
	.long	3140669482
	.long	1116444567
	.long	1108055378
	.long	1094336475
	.long	3141178479
	.long	1116849557
	.long	1108460423
	.long	1094869431
	.long	3141737901
	.long	1117272479
	.long	1108883400
	.long	1095429351
	.long	3142284745
	.long	1117714127
	.long	1109325101
	.long	1096014237
	.long	3142915054
	.long	1118175329
	.long	1109786358
	.long	1096645678
	.long	3143505197
	.long	1118656953
	.long	1110268033
	.long	1097277902
	.long	3144150196
	.long	1119159901
	.long	1110771033
	.long	1097953811
	.long	3144845928
	.long	1119685118
	.long	1111296302
	.long	1098655549
	.long	3145529363
	.long	1120233590
	.long	1111844824
	.long	1099144661
	.long	3145987662
	.long	1120806346
	.long	1112417630
	.long	1099525884
	.long	3146377804
	.long	1121404461
	.long	1113015796
	.long	1099927000
	.long	3146786805
	.long	1122029058
	.long	1113640444
	.long	1100345687
	.long	3147190794
	.long	1122681310
	.long	1114292745
	.long	1100776673
	.long	3147632967
	.long	1123362440
	.long	1114973926
	.long	1101234255
	.long	3148087611
	.long	1124073600
	.long	1115685064
	.long	1101704192
	.long	3148551873
	.long	1124444990
	.long	1116056479
	.long	1102198949
	.long	3149053844
	.long	1124832823
	.long	1116444338
	.long	1102721963
	.long	3149560519
	.long	1125237828
	.long	1116849368
	.long	1103253489
	.long	3150129648
	.long	1125660764
	.long	1117272331
	.long	1103819489
	.long	3150699108
	.long	1126102425
	.long	1117714019
	.long	1104418512
	.long	3151300238
	.long	1126563641
	.long	1118175262
	.long	1105031754
	.long	3151908533
	.long	1127045277
	.long	1118656925
	.long	1105675327
	.long	3152521467
	.long	1127548238
	.long	1119159912
	.long	1106331233
	.long	3153233976
	.long	1128073466
	.long	1119685170
	.long	1107043461
	.long	3153918194
	.long	1128621949
	.long	1120233681
	.long	1107533172
	.long	3154369114
	.long	1129194716
	.long	1120806476
	.long	1107909865
	.long	3154761041
	.long	1129792841
	.long	1121404632
	.long	1108312102
	.long	3155164804
	.long	1130417448
	.long	1122029270
	.long	1108727526
	.long	3155573216
	.long	1131069709
	.long	1122681562
	.long	1109161279
	.long	3156013372
	.long	1131750848
	.long	1123362734
	.long	1109617608
	.long	3156476219
	.long	1132462112
	.long	1124073768
	.long	1110092672
	.long	3156942778
	.long	1132833506
	.long	1124445179
	.long	1110588868
	.long	3157441390
	.long	1133221343
	.long	1124833034
	.long	1111109791
	.long	3157939291
	.long	1133626352
	.long	1125238060
	.long	1111635844
	.long	3158527234
	.long	1134049291
	.long	1125661020
	.long	1112213594
	.long	3159077768
	.long	1134490956
	.long	1126102704
	.long	1112800807
	.long	3159687990
	.long	1134952175
	.long	1126563944
	.long	1113419729
	.long	3160268049
	.long	1135433815
	.long	1127045603
	.long	1114045678
	.long	3160913934
	.long	1135936778
	.long	1127548588
	.long	1114722160
	.long	3161622444
	.long	1136462009
	.long	1128073843
	.long	1115431895
	.long	3162298664
	.long	1137010495
	.long	1128622351
	.long	1115919199
	.long	3162764127
	.long	1137583264
	.long	1129195144
	.long	1116302432
	.long	3163148306
	.long	1138181392
	.long	1129793297
	.long	1116699834
	.long	3163558953
	.long	1138806001
	.long	1130417933
	.long	1117119556
	.long	3163972568
	.long	1139458264
	.long	1131070223
	.long	1117556560
	.long	3164399930
	.long	1140139406
	.long	1131751392
	.long	1118004903
	.long	3164864827
	.long	1140850696
	.long	1132462400
	.long	1118481248
	.long	3165331960
	.long	1141222091
	.long	1132833810
	.long	1118977804
	.long	3165829733
	.long	1141609929
	.long	1133221664
	.long	1119498204
	.long	3166325440
	.long	1142014939
	.long	1133626689
	.long	1120022889
	.long	3166909893
	.long	1142437879
	.long	1134049648
	.long	1120598461
	.long	3167455696
	.long	1142879545
	.long	1134491331
	.long	1121182721
	.long	3168059997
	.long	1143340765
	.long	1134952570
	.long	1121797948
	.long	3168665771
	.long	1143822405
	.long	1135434229
	.long	1122439952
	.long	3169303507
	.long	1144325369
	.long	1135937213
	.long	1123111348
	.long	3170002824
	.long	1144850601
	.long	1136462467
	.long	1123815345
	.long	3170701624
	.long	1145399087
	.long	1137010975
	.long	1124312276
	.long	3171154336
	.long	1145971857
	.long	1137583767
	.long	1124692030
	.long	3171532482
	.long	1146569986
	.long	1138181919
	.long	1125085665
	.long	3171936657
	.long	1147194596
	.long	1138806554
	.long	1125501347
	.long	3172359765
	.long	1147846859
	.long	1139458844
	.long	1125944278
	.long	3172796218
	.long	1148528001
	.long	1140140013
	.long	1126398298
	.long	3173253435
	.long	1149239298
	.long	1140851014
	.long	1126869848
	.long	3173728905
	.long	1149610693
	.long	1141222424
	.long	1127371609
	.long	3174201888
	.long	1149998532
	.long	1141610277
	.long	1127876532
	.long	3174738014
	.long	1150403541
	.long	1142015303
	.long	1128426452
	.long	3175297014
	.long	1150826482
	.long	1142438261
	.long	1128986134
	.long	3175849827
	.long	1151268148
	.long	1142879944
	.long	1129574771
	.long	3176460842
	.long	1151729368
	.long	1143341183
	.long	1130194189
	.long	3177073044
	.long	1152211008
	.long	1143822842
	.long	1130840207
	.long	3177684163
	.long	1152713973
	.long	1144325825
	.long	1131494986
	.long	3178389375
	.long	1153239205
	.long	1144851079
	.long	1132202663
	.long	3179093821
	.long	1153787691
	.long	1145399587
	.long	1132702002
	.long	3179547441
	.long	1154360461
	.long	1145972379
	.long	1133083443
	.long	3179928175
	.long	1154958590
	.long	1146570531
	.long	1133478694
	.long	3180334828
	.long	1155583200
	.long	1147195166
	.long	1133895924
	.long	3180743924
	.long	1156235464
	.long	1147847455
	.long	1134330106
	.long	3181182650
	.long	1156916606
	.long	1148528624
	.long	1134785545
	.long	3181625656
	.long	1157627905
	.long	1149239623
	.long	1135248223
	.long	3182103210
	.long	1157999300
	.long	1149611033
	.long	1135751286
	.long	3182610963
	.long	1158387138
	.long	1149998887
	.long	1136277916
	.long	3183116226
	.long	1158792148
	.long	1150403912
	.long	1136808568
	.long	3183677057
	.long	1159215089
	.long	1150826870
	.long	1137369393
	.long	3184231622
	.long	1159656755
	.long	1151268553
	.long	1137959124
	.long	3184844315
	.long	1160117975
	.long	1151729792
	.long	1138579590
	.long	3185458125
	.long	1160599615
	.long	1152211451
	.long	1139226612
	.long	3186070783
	.long	1161102580
	.long	1152714434
	.long	1139882351
	.long	3186777469
	.long	1161627812
	.long	1153239688
	.long	1140590948
	.long	3187483326
	.long	1162176298
	.long	1153788196
	.long	1141090889
	.long	3187937173
	.long	1162749068
	.long	1154360988
	.long	1141472752
	.long	3188318554
	.long	1163347197
	.long	1154959140
	.long	1141868407
	.long	3188725827
	.long	1163971807
	.long	1155583775
	.long	1142286024
	.long	3189135516
	.long	1164624071
	.long	1156236064
	.long	1142720577
	.long	3189574810
	.long	1165305213
	.long	1156917233
	.long	1143176370
	.long	3190034748
	.long	1166016512
	.long	1157628232
	.long	1143649619
	.long	3190480049
	.long	1166387908
	.long	1157999641
	.long	1144132546
	.long	3190988301
	.long	1166775746
	.long	1158387495
	.long	1144659488
	.long	3191494042
	.long	1167180756
	.long	1158792520
	.long	1145190438
	.long	3192088104
	.long	1167603696
	.long	1159215479
	.long	1145772009
	.long	3192610334
	.long	1168045363
	.long	1159657161
	.long	1146341553
	.long	3193223446
	.long	1168506583
	.long	1160118400
	.long	1146962281
	.long	3193837658
	.long	1168988223
	.long	1160600059
	.long	1147609554
	.long	3194483474
	.long	1169491187
	.long	1161103043
	.long	1148285994
	.long	3195157755
	.long	1170016420
	.long	1161628296
	.long	1148974361
	.long	3195863964
	.long	1170564906
	.long	1162176804
	.long	1149477009
	.long	3196321965
	.long	1171137676
	.long	1162749596
	.long	1149858978
	.long	3196703508
	.long	1171735805
	.long	1163347748
	.long	1150254734
	.long	3197110936
	.long	1172360415
	.long	1163972383
	.long	1150672447
	.long	3197520774
	.long	1173012679
	.long	1164624672
	.long	1151107093
	.long	3197960210
	.long	1173693821
	.long	1165305841
	.long	1151562975
	.long	3198420283
	.long	1174405120
	.long	1166016840
	.long	1152036309
	.long	3198898489
	.long	1174776515
	.long	1166388250
	.long	1152539778
	.long	3199374091
	.long	1175164354
	.long	1166776103
	.long	1153046337
	.long	3199879952
	.long	1175569364
	.long	1167181128
	.long	1153577361
	.long	3200474128
	.long	1175992304
	.long	1167604087
	.long	1154159004
	.long	3201029241
	.long	1176433970
	.long	1168045770
	.long	1154749077
	.long	3201609685
	.long	1176895191
	.long	1168507008
	.long	1155349410
	.long	3202223997
	.long	1177376831
	.long	1168988667
	.long	1155996745
	.long	3202869909
	.long	1177879795
	.long	1169491651
	.long	1156673246
	.long	3203544282
	.long	1178405028
	.long	1170016904
	.long	1157361670
	.long	3204250580
	.long	1178953514
	.long	1170565412
	.long	1157864996
	.long	3204709619
	.long	1179526284
	.long	1171138204
	.long	1158246990
	.long	3205091203
	.long	1180124413
	.long	1171736356
	.long	1158642772
	.long	3205498670
	.long	1180749023
	.long	1172360991
	.long	1159060509
	.long	3205908544
	.long	1181401287
	.long	1173013280
	.long	1159495178
	.long	3206348016
	.long	1182082429
	.long	1173694449
	.long	1159951082
	.long	3206808123
	.long	1182793728
	.long	1174405448
	.long	1160424437
	.long	3207286361
	.long	1183165123
	.long	1174776858
	.long	1160927927
	.long	3207761995
	.long	1183552962
	.long	1175164711
	.long	1161434505
	.long	3208300659
	.long	1183957971
	.long	1175569737
	.long	1161986009
	.long	3208862090
	.long	1184380912
	.long	1175992695
	.long	1162547209
	.long	3209417231
	.long	1184822578
	.long	1176434378
	.long	1163137299
	.long	3209997701
	.long	1185283799
	.long	1176895616
	.long	1163737648
	.long	3210612038
	.long	1185765439
	.long	1177377275
	.long	1164384999
	.long	3211257974
	.long	1186268403
	.long	1177880259
	.long	1165061514
	.long	3211932370
	.long	1186793636
	.long	1178405512
	.long	1165749953
	.long	3212638690
	.long	1187342122
	.long	1178954020
	.long	1166253448
	.long	3213097989
	.long	1187914892
	.long	1179526812
	.long	1166635449
	.long	3213479583
	.long	1188513021
	.long	1180124964
	.long	1167031237
	.long	3213887059
	.long	1189137631
	.long	1180749599
	.long	1167448981
	.long	3214296943
	.long	1189789895
	.long	1181401888
	.long	1167883655
	.long	3214736423
	.long	1190471037
	.long	1182083057
	.long	1168339565
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.long	32
	.type	__svml_ssinh_ha_data_internal,@object
	.size	__svml_ssinh_ha_data_internal,5632
	.align 32
__ssinh_ha_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	1398474845
	.long	3161559171
	.long	3541402996
	.long	1072716208
	.long	2759177317
	.long	1015903202
	.long	410360776
	.long	1072727877
	.long	1269990655
	.long	1013024446
	.long	1828292879
	.long	1072739672
	.long	1255956747
	.long	1016636974
	.long	852742562
	.long	1072751596
	.long	667253587
	.long	1010842135
	.long	3490863953
	.long	1072763649
	.long	960797498
	.long	3163997456
	.long	2930322912
	.long	1072775834
	.long	2599499422
	.long	3163762623
	.long	1014845819
	.long	1072788152
	.long	3117910646
	.long	3162607681
	.long	3949972341
	.long	1072800603
	.long	2068408548
	.long	1015962444
	.long	828946858
	.long	1072813191
	.long	10642492
	.long	1016988014
	.long	2288159958
	.long	1072825915
	.long	2169144469
	.long	1015924597
	.long	1853186616
	.long	1072838778
	.long	3066496371
	.long	1016705150
	.long	1709341917
	.long	1072851781
	.long	2571168217
	.long	1015201075
	.long	4112506593
	.long	1072864925
	.long	2947355221
	.long	1015419624
	.long	2799960843
	.long	1072878213
	.long	1423655381
	.long	1016070727
	.long	171030293
	.long	1072891646
	.long	3526460132
	.long	1015477354
	.long	2992903935
	.long	1072905224
	.long	2218154406
	.long	1016276769
	.long	926591435
	.long	1072918951
	.long	3208833762
	.long	3163962090
	.long	887463927
	.long	1072932827
	.long	3596744163
	.long	3161842742
	.long	1276261410
	.long	1072946854
	.long	300981948
	.long	1015732745
	.long	569847338
	.long	1072961034
	.long	472945272
	.long	3160339305
	.long	1617004845
	.long	1072975368
	.long	82804944
	.long	1011391354
	.long	3049340112
	.long	1072989858
	.long	3062915824
	.long	1014219171
	.long	3577096743
	.long	1073004506
	.long	2951496418
	.long	1014842263
	.long	1990012071
	.long	1073019314
	.long	3529070563
	.long	3163861769
	.long	1453150082
	.long	1073034283
	.long	498154669
	.long	3162536638
	.long	917841882
	.long	1073049415
	.long	18715565
	.long	1016707884
	.long	3712504873
	.long	1073064711
	.long	88491949
	.long	1016476236
	.long	363667784
	.long	1073080175
	.long	813753950
	.long	1016833785
	.long	2956612997
	.long	1073095806
	.long	2118169751
	.long	3163784129
	.long	2186617381
	.long	1073111608
	.long	2270764084
	.long	3164321289
	.long	1719614413
	.long	1073127582
	.long	330458198
	.long	3164331316
	.long	1013258799
	.long	1073143730
	.long	1748797611
	.long	3161177658
	.long	3907805044
	.long	1073160053
	.long	2257091225
	.long	3162598983
	.long	1447192521
	.long	1073176555
	.long	1462857171
	.long	3163563097
	.long	1944781191
	.long	1073193236
	.long	3993278767
	.long	3162772855
	.long	919555682
	.long	1073210099
	.long	3121969534
	.long	1013996802
	.long	2571947539
	.long	1073227145
	.long	3558159064
	.long	3164425245
	.long	2604962541
	.long	1073244377
	.long	2614425274
	.long	3164587768
	.long	1110089947
	.long	1073261797
	.long	1451641639
	.long	1016523249
	.long	2568320822
	.long	1073279406
	.long	2732824428
	.long	1015401491
	.long	2966275557
	.long	1073297207
	.long	2176155324
	.long	3160891335
	.long	2682146384
	.long	1073315202
	.long	2082178513
	.long	3164411995
	.long	2191782032
	.long	1073333393
	.long	2960257726
	.long	1014791238
	.long	2069751141
	.long	1073351782
	.long	1562170675
	.long	3163773257
	.long	2990417245
	.long	1073370371
	.long	3683467745
	.long	3164417902
	.long	1434058175
	.long	1073389163
	.long	251133233
	.long	1016134345
	.long	2572866477
	.long	1073408159
	.long	878562433
	.long	1016570317
	.long	3092190715
	.long	1073427362
	.long	814012168
	.long	3160571998
	.long	4076559943
	.long	1073446774
	.long	2119478331
	.long	3161806927
	.long	2420883922
	.long	1073466398
	.long	2049810052
	.long	1015168464
	.long	3716502172
	.long	1073486235
	.long	2303740125
	.long	1015091301
	.long	777507147
	.long	1073506289
	.long	4282924205
	.long	1016236109
	.long	3706687593
	.long	1073526560
	.long	3521726939
	.long	1014301643
	.long	1242007932
	.long	1073547053
	.long	1132034716
	.long	3164388407
	.long	3707479175
	.long	1073567768
	.long	3613079303
	.long	1015213314
	.long	64696965
	.long	1073588710
	.long	1768797490
	.long	1016865536
	.long	863738719
	.long	1073609879
	.long	1326992220
	.long	3163661773
	.long	3884662774
	.long	1073631278
	.long	2158611599
	.long	1015258761
	.long	2728693978
	.long	1073652911
	.long	396109971
	.long	3164511267
	.long	3999357479
	.long	1073674779
	.long	2258941616
	.long	1016973300
	.long	1533953344
	.long	1073696886
	.long	769171851
	.long	1016714209
	.long	2174652632
	.long	1073719233
	.long	4087714590
	.long	1015498835
	.long	0
	.long	1073741824
	.long	0
	.long	0
	.long	0
	.long	1071644672
	.long	1431652600
	.long	1069897045
	.long	1431670732
	.long	1067799893
	.long	984555731
	.long	1065423122
	.long	472530941
	.long	1062650218
	.long	1431655765
	.long	1069897045
	.long	286331153
	.long	1065423121
	.long	436314138
	.long	1059717536
	.long	2773927732
	.long	1053236707
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
	.long	33554432
	.long	1101004800
	.long	2684354560
	.long	1079401119
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	1
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	3671843104
	.long	1067178892
	.long	3875694624
	.long	1077247184
	.type	__ssinh_ha_CoutTab,@object
	.size	__ssinh_ha_CoutTab,1192
	.space 8, 0x00 	
	.align 16
.L_2il0floatpacket.96:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.96,@object
	.size	.L_2il0floatpacket.96,16
      	.section        .note.GNU-stack,"",@progbits
