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
.L_2__routine_start___svml_cosh8_ha_z0_0:

	.align    16,0x90
	.globl __svml_cosh8_ha

__svml_cosh8_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   4672+__svml_dcosh_ha_data_internal(%rip), %zmm1
        vmovups   384+__svml_dcosh_ha_data_internal(%rip), %zmm8

/*
 * ............... Load argument ...........................
 * dM = x*2^K/log(2) + RShifter
 */
        vmovups   3968+__svml_dcosh_ha_data_internal(%rip), %zmm7
        vmovups   4032+__svml_dcosh_ha_data_internal(%rip), %zmm9
        vmovups   4096+__svml_dcosh_ha_data_internal(%rip), %zmm6
        vmovups   128+__svml_dcosh_ha_data_internal(%rip), %zmm4
        vmovups   256+__svml_dcosh_ha_data_internal(%rip), %zmm5
        vmovups   832+__svml_dcosh_ha_data_internal(%rip), %zmm11
        vmovups   768+__svml_dcosh_ha_data_internal(%rip), %zmm13
        vmovups   512+__svml_dcosh_ha_data_internal(%rip), %zmm14
        vmovups   576+__svml_dcosh_ha_data_internal(%rip), %zmm12
        vmovaps   %zmm0, %zmm15

/* ............... Abs argument ............................ */
        vandnpd   %zmm15, %zmm1, %zmm10
        vfmadd213pd {rn-sae}, %zmm8, %zmm10, %zmm7

/*
 * ...............Check for overflow\underflow .............
 * 
 */
        vpsrlq    $32, %zmm10, %zmm0

/* dN = dM - RShifter */
        vsubpd    {rn-sae}, %zmm8, %zmm7, %zmm8
        vpmovqd   %zmm0, %ymm2
        vpermt2pd 192+__svml_dcosh_ha_data_internal(%rip), %zmm7, %zmm4
        vpermt2pd 320+__svml_dcosh_ha_data_internal(%rip), %zmm7, %zmm5

/* dR = dX - dN*Log2_hi/2^K */
        vfnmadd231pd {rn-sae}, %zmm9, %zmm8, %zmm10

/* dR = dX - dN*Log2_hi/2^K */
        vfnmadd231pd {rn-sae}, %zmm6, %zmm8, %zmm10
        vpsllq    $48, %zmm7, %zmm6
        vpcmpgtd  4736+__svml_dcosh_ha_data_internal(%rip), %ymm2, %ymm3
        vmulpd    {rn-sae}, %zmm10, %zmm10, %zmm2
        vmovmskps %ymm3, %edx

/* .............. Index and lookup ......................... */
        vmovups   __svml_dcosh_ha_data_internal(%rip), %zmm3
        vpermt2pd 64+__svml_dcosh_ha_data_internal(%rip), %zmm7, %zmm3

/* lM now is an EXP(2^N) */
        vpandq    4608+__svml_dcosh_ha_data_internal(%rip), %zmm6, %zmm7
        vpaddq    %zmm7, %zmm4, %zmm0
        vpaddq    %zmm7, %zmm3, %zmm1
        vpsubq    %zmm7, %zmm5, %zmm3

/*
 * poly(r) = Gmjp(1 + a2*r^2 + a4*r^4) + Gmjn*(r+ a3*r^3 +a5*r^5)       =
 * = Gmjp_h +Gmjp_l+ Gmjp*r^2*(a2 + a4*r^2) + Gmjn*(r+ r^3*(a3 +a5*r^2)
 */
        vmovups   704+__svml_dcosh_ha_data_internal(%rip), %zmm4
        vaddpd    {rn-sae}, %zmm3, %zmm1, %zmm5
        vsubpd    {rn-sae}, %zmm3, %zmm1, %zmm9
        vfmadd231pd {rn-sae}, %zmm2, %zmm11, %zmm4
        vmovups   640+__svml_dcosh_ha_data_internal(%rip), %zmm11
        vfmadd213pd {rn-sae}, %zmm12, %zmm2, %zmm4
        vfmadd231pd {rn-sae}, %zmm2, %zmm13, %zmm11

/* dM=r^2*(a3 +a5*r^2) */
        vmulpd    {rn-sae}, %zmm2, %zmm4, %zmm13
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm11

/* dM= r + r^3*(a3 +a5*r^2) */
        vfmadd213pd {rn-sae}, %zmm10, %zmm10, %zmm13

/* dOut=r^2*(a2 + a4*r^2) */
        vmulpd    {rn-sae}, %zmm2, %zmm11, %zmm12

/* dOut=Gmjp*r^2*(a2 + a4*r^2)+Gmjp_l(1) */
        vfmadd213pd {rn-sae}, %zmm0, %zmm5, %zmm12

/* dOut=Gmjp*r^2*(a2 + a4*r^2)+Gmjp_l(1) + Gmjn*(r+ r^3*(a3+a5*r^2) */
        vfmadd213pd {rn-sae}, %zmm12, %zmm9, %zmm13

/* dOut=Gmjp*r^2*(a2 + a4*r^2)+Gmjp_l(1) + Gmjn*(r+ r^3*(a3+a5*r^2) + Gmjp_l(2) */
        vaddpd    {rn-sae}, %zmm3, %zmm13, %zmm10

/* Gmjp_h +Gmjp_l+ Gmjp*r^2*(a2 + a4*r^2) + Gmjn*(r+ r^3*(a3+a5*r^2) */
        vaddpd    {rn-sae}, %zmm1, %zmm10, %zmm0
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
        cmpl      $8, %r12d
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

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi

        call      __svml_dcosh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_cosh8_ha,@function
	.size	__svml_cosh8_ha,.-__svml_cosh8_ha
..LN__svml_cosh8_ha.0:

.L_2__routine_start___svml_dcosh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_dcosh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r8
        movzwl    6(%rdi), %edx
        xorl      %eax, %eax
        andl      $32752, %edx
        cmpl      $32752, %edx
        je        .LBL_2_12


        movq      (%rdi), %rdx
        movq      %rdx, -8(%rsp)
        shrq      $56, %rdx
        andl      $127, %edx
        movb      %dl, -1(%rsp)
        movzwl    -2(%rsp), %ecx
        andl      $32752, %ecx
        cmpl      $15504, %ecx
        jle       .LBL_2_10


        movsd     -8(%rsp), %xmm0
        movsd     1096+__dcosh_ha_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        movq      1128+__dcosh_ha_CoutTab(%rip), %rdx
        movq      %rdx, -8(%rsp)
        comisd    1144+__dcosh_ha_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1040+__dcosh_ha_CoutTab(%rip), %xmm1
        lea       __dcosh_ha_CoutTab(%rip), %r9
        mulsd     %xmm0, %xmm1
        addsd     1048+__dcosh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movsd     1088+__dcosh_ha_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1048+__dcosh_ha_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       (%rsi,%rsi), %ecx
        movsd     -32(%rsp), %xmm3
        lea       1(%rsi,%rsi), %edi
        mulsd     1104+__dcosh_ha_CoutTab(%rip), %xmm3
        movsd     -32(%rsp), %xmm4
        subsd     %xmm3, %xmm0
        mulsd     1112+__dcosh_ha_CoutTab(%rip), %xmm4
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1080+__dcosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1072+__dcosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1064+__dcosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__dcosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        movsd     (%r9,%rcx,8), %xmm0
        mulsd     %xmm0, %xmm1
        addsd     (%r9,%rdi,8), %xmm1
        addsd     %xmm0, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_7


        movq      1128+__dcosh_ha_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        movsd     %xmm0, (%r8)
        ret

.LBL_2_7:

        decl      %edx
        andl      $2047, %edx
        movzwl    -2(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        mulsd     1024+__dcosh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, (%r8)
        ret

.LBL_2_8:

        movsd     1040+__dcosh_ha_CoutTab(%rip), %xmm1
        lea       __dcosh_ha_CoutTab(%rip), %rcx
        movzwl    -2(%rsp), %esi
        andl      $-32753, %esi
        movsd     1080+__dcosh_ha_CoutTab(%rip), %xmm14
        mulsd     %xmm0, %xmm1
        addsd     1048+__dcosh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movl      -40(%rsp), %r10d
        movl      %r10d, %r9d
        shrl      $6, %r9d
        subsd     1048+__dcosh_ha_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       1023(%r9), %edi
        andl      $63, %r10d
        addl      $1022, %r9d
        movsd     -32(%rsp), %xmm3
        andl      $2047, %r9d
        negl      %edi
        shll      $4, %r9d
        addl      $-4, %edi
        mulsd     1104+__dcosh_ha_CoutTab(%rip), %xmm3
        lea       (%r10,%r10), %edx
        movsd     (%rcx,%rdx,8), %xmm15
        negl      %edx
        movsd     -32(%rsp), %xmm4
        orl       %r9d, %esi
        andl      $2047, %edi
        lea       1(%r10,%r10), %r11d
        mulsd     1112+__dcosh_ha_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm0
        movw      %si, -2(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        subsd     %xmm4, %xmm0
        movsd     -8(%rsp), %xmm6
        orl       %edi, %esi
        movw      %si, -2(%rsp)
        lea       128(%rdx), %esi
        mulsd     %xmm6, %xmm15
        movaps    %xmm0, %xmm5
        mulsd     %xmm0, %xmm5
        movsd     -8(%rsp), %xmm7
        movaps    %xmm15, %xmm8
        movsd     (%rcx,%rsi,8), %xmm11
        addl      $129, %edx
        mulsd     %xmm7, %xmm11
        movaps    %xmm15, %xmm10
        mulsd     %xmm5, %xmm14
        addsd     %xmm11, %xmm8
        subsd     %xmm11, %xmm15
        addsd     1064+__dcosh_ha_CoutTab(%rip), %xmm14
        movsd     %xmm8, -24(%rsp)
        movsd     (%rcx,%r11,8), %xmm12
        movsd     (%rcx,%rdx,8), %xmm13
        movsd     -24(%rsp), %xmm9
        mulsd     %xmm6, %xmm12
        subsd     %xmm9, %xmm10
        mulsd     %xmm7, %xmm13
        mulsd     %xmm5, %xmm14
        addsd     %xmm11, %xmm10
        mulsd     %xmm0, %xmm14
        movsd     1088+__dcosh_ha_CoutTab(%rip), %xmm1
        movaps    %xmm12, %xmm11
        mulsd     %xmm5, %xmm1
        subsd     %xmm13, %xmm12
        mulsd     %xmm15, %xmm14
        mulsd     %xmm0, %xmm12
        addsd     1072+__dcosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm15, %xmm0
        mulsd     %xmm5, %xmm1
        addsd     %xmm12, %xmm11
        movsd     %xmm10, -16(%rsp)
        addsd     %xmm13, %xmm11
        addsd     1056+__dcosh_ha_CoutTab(%rip), %xmm1
        addsd     %xmm14, %xmm11
        mulsd     %xmm5, %xmm1
        addsd     %xmm0, %xmm11
        movsd     -24(%rsp), %xmm3
        mulsd     %xmm3, %xmm1
        movsd     -16(%rsp), %xmm2
        addsd     %xmm1, %xmm11
        addsd     %xmm2, %xmm11
        movsd     %xmm11, -24(%rsp)
        movsd     -24(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        movsd     %xmm3, (%r8)
        ret

.LBL_2_9:

        movsd     1120+__dcosh_ha_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%r8)
        ret

.LBL_2_10:

        movsd     1136+__dcosh_ha_CoutTab(%rip), %xmm0
        addsd     -8(%rsp), %xmm0
        movsd     %xmm0, (%r8)


        ret

.LBL_2_12:

        movsd     (%rdi), %xmm0
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%r8)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dcosh_ha_cout_rare_internal,@function
	.size	__svml_dcosh_ha_cout_rare_internal,.-__svml_dcosh_ha_cout_rare_internal
..LN__svml_dcosh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dcosh_ha_data_internal:
	.long	0
	.long	1071644672
	.long	1828292879
	.long	1071691096
	.long	1014845819
	.long	1071739576
	.long	1853186616
	.long	1071790202
	.long	171030293
	.long	1071843070
	.long	1276261410
	.long	1071898278
	.long	3577096743
	.long	1071955930
	.long	3712504873
	.long	1072016135
	.long	1719614413
	.long	1072079006
	.long	1944781191
	.long	1072144660
	.long	1110089947
	.long	1072213221
	.long	2191782032
	.long	1072284817
	.long	2572866477
	.long	1072359583
	.long	3716502172
	.long	1072437659
	.long	3707479175
	.long	1072519192
	.long	2728693978
	.long	1072604335
	.long	0
	.long	0
	.long	1255956747
	.long	1015588398
	.long	3117910646
	.long	3161559105
	.long	3066496371
	.long	1015656574
	.long	3526460132
	.long	1014428778
	.long	300981948
	.long	1014684169
	.long	2951496418
	.long	1013793687
	.long	88491949
	.long	1015427660
	.long	330458198
	.long	3163282740
	.long	3993278767
	.long	3161724279
	.long	1451641639
	.long	1015474673
	.long	2960257726
	.long	1013742662
	.long	878562433
	.long	1015521741
	.long	2303740125
	.long	1014042725
	.long	3613079303
	.long	1014164738
	.long	396109971
	.long	3163462691
	.long	0
	.long	1071644672
	.long	2728693978
	.long	1071555759
	.long	3707479175
	.long	1071470616
	.long	3716502172
	.long	1071389083
	.long	2572866477
	.long	1071311007
	.long	2191782032
	.long	1071236241
	.long	1110089947
	.long	1071164645
	.long	1944781191
	.long	1071096084
	.long	1719614413
	.long	1071030430
	.long	3712504873
	.long	1070967559
	.long	3577096743
	.long	1070907354
	.long	1276261410
	.long	1070849702
	.long	171030293
	.long	1070794494
	.long	1853186616
	.long	1070741626
	.long	1014845819
	.long	1070691000
	.long	1828292879
	.long	1070642520
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	0
	.long	1071644672
	.long	0
	.long	0
	.long	0
	.long	1071644672
	.long	2851812149
	.long	1071650365
	.long	2595802551
	.long	1015767337
	.long	730821105
	.long	1071633346
	.long	1048019041
	.long	1071656090
	.long	1398474845
	.long	3160510595
	.long	2174652632
	.long	1071622081
	.long	3899555717
	.long	1071661845
	.long	427280750
	.long	3162546972
	.long	2912730644
	.long	1071610877
	.long	3541402996
	.long	1071667632
	.long	2759177317
	.long	1014854626
	.long	1533953344
	.long	1071599734
	.long	702412510
	.long	1071673451
	.long	3803266087
	.long	3162280415
	.long	929806999
	.long	1071588651
	.long	410360776
	.long	1071679301
	.long	1269990655
	.long	1011975870
	.long	3999357479
	.long	1071577627
	.long	3402036099
	.long	1071685182
	.long	405889334
	.long	1015105656
	.long	764307441
	.long	1071566664
	.long	1828292879
	.long	1071691096
	.long	1255956747
	.long	1015588398
	.long	2728693978
	.long	1071555759
	.long	728909815
	.long	1071697042
	.long	383930225
	.long	1015029468
	.long	4224142467
	.long	1071544913
	.long	852742562
	.long	1071703020
	.long	667253587
	.long	1009793559
	.long	3884662774
	.long	1071534126
	.long	2952712987
	.long	1071709030
	.long	3293494651
	.long	3160120301
	.long	351641897
	.long	1071523398
	.long	3490863953
	.long	1071715073
	.long	960797498
	.long	3162948880
	.long	863738719
	.long	1071512727
	.long	3228316108
	.long	1071721149
	.long	3010241991
	.long	3158422804
	.long	4076975200
	.long	1071502113
	.long	2930322912
	.long	1071727258
	.long	2599499422
	.long	3162714047
	.long	64696965
	.long	1071491558
	.long	3366293073
	.long	1071733400
	.long	3119426314
	.long	1014120554
	.long	382305176
	.long	1071481059
	.long	1014845819
	.long	1071739576
	.long	3117910646
	.long	3161559105
	.long	3707479175
	.long	1071470616
	.long	948735466
	.long	1071745785
	.long	3516338028
	.long	3162574883
	.long	135105010
	.long	1071460231
	.long	3949972341
	.long	1071752027
	.long	2068408548
	.long	1014913868
	.long	1242007932
	.long	1071449901
	.long	2214878420
	.long	1071758304
	.long	892270087
	.long	3163116422
	.long	1432208378
	.long	1071439627
	.long	828946858
	.long	1071764615
	.long	10642492
	.long	1015939438
	.long	3706687593
	.long	1071429408
	.long	586995997
	.long	1071770960
	.long	41662348
	.long	3162627992
	.long	2483480501
	.long	1071419245
	.long	2288159958
	.long	1071777339
	.long	2169144469
	.long	1014876021
	.long	777507147
	.long	1071409137
	.long	2440944790
	.long	1071783753
	.long	2492769774
	.long	1014147454
	.long	1610600570
	.long	1071399083
	.long	1853186616
	.long	1071790202
	.long	3066496371
	.long	1015656574
	.long	3716502172
	.long	1071389083
	.long	1337108031
	.long	1071796686
	.long	3203724452
	.long	1014677845
	.long	1540824585
	.long	1071379138
	.long	1709341917
	.long	1071803205
	.long	2571168217
	.long	1014152499
	.long	2420883922
	.long	1071369246
	.long	3790955393
	.long	1071809759
	.long	2352942462
	.long	3163180090
	.long	815859274
	.long	1071359408
	.long	4112506593
	.long	1071816349
	.long	2947355221
	.long	1014371048
	.long	4076559943
	.long	1071349622
	.long	3504003472
	.long	1071822975
	.long	3594001060
	.long	3157330652
	.long	2380618042
	.long	1071339890
	.long	2799960843
	.long	1071829637
	.long	1423655381
	.long	1015022151
	.long	3092190715
	.long	1071330210
	.long	2839424854
	.long	1071836335
	.long	1171596163
	.long	1013041679
	.long	697153126
	.long	1071320583
	.long	171030293
	.long	1071843070
	.long	3526460132
	.long	1014428778
	.long	2572866477
	.long	1071311007
	.long	4232894513
	.long	1071849840
	.long	2383938684
	.long	1014668519
	.long	3218338682
	.long	1071301483
	.long	2992903935
	.long	1071856648
	.long	2218154406
	.long	1015228193
	.long	1434058175
	.long	1071292011
	.long	1603444721
	.long	1071863493
	.long	1548633640
	.long	3162201326
	.long	321958744
	.long	1071282590
	.long	926591435
	.long	1071870375
	.long	3208833762
	.long	3162913514
	.long	2990417245
	.long	1071273219
	.long	1829099622
	.long	1071877294
	.long	1016661181
	.long	3163461005
	.long	3964284211
	.long	1071263899
	.long	887463927
	.long	1071884251
	.long	3596744163
	.long	3160794166
	.long	2069751141
	.long	1071254630
	.long	3272845541
	.long	1071891245
	.long	928852419
	.long	3163488248
	.long	434316067
	.long	1071245411
	.long	1276261410
	.long	1071898278
	.long	300981948
	.long	1014684169
	.long	2191782032
	.long	1071236241
	.long	78413852
	.long	1071905349
	.long	4183226867
	.long	3163017251
	.long	1892288442
	.long	1071227121
	.long	569847338
	.long	1071912458
	.long	472945272
	.long	3159290729
	.long	2682146384
	.long	1071218050
	.long	3645941911
	.long	1071919605
	.long	3814685081
	.long	3161573341
	.long	3418903055
	.long	1071209028
	.long	1617004845
	.long	1071926792
	.long	82804944
	.long	1010342778
	.long	2966275557
	.long	1071200055
	.long	3978100823
	.long	1071934017
	.long	3513027190
	.long	1015845963
	.long	194117574
	.long	1071191131
	.long	3049340112
	.long	1071941282
	.long	3062915824
	.long	1013170595
	.long	2568320822
	.long	1071182254
	.long	4040676318
	.long	1071948586
	.long	4090609238
	.long	1015663458
	.long	380978316
	.long	1071173426
	.long	3577096743
	.long	1071955930
	.long	2951496418
	.long	1013793687
	.long	1110089947
	.long	1071164645
	.long	2583551245
	.long	1071963314
	.long	3161094195
	.long	1015606491
	.long	3649726105
	.long	1071155911
	.long	1990012071
	.long	1071970738
	.long	3529070563
	.long	3162813193
	.long	2604962541
	.long	1071147225
	.long	2731501122
	.long	1071978202
	.long	1774031855
	.long	3162470021
	.long	1176749997
	.long	1071138586
	.long	1453150082
	.long	1071985707
	.long	498154669
	.long	3161488062
	.long	2571947539
	.long	1071129993
	.long	3395129871
	.long	1071993252
	.long	4025345435
	.long	3162335388
	.long	1413356050
	.long	1071121447
	.long	917841882
	.long	1072000839
	.long	18715565
	.long	1015659308
	.long	919555682
	.long	1071112947
	.long	3566716925
	.long	1072008466
	.long	1536826856
	.long	1014142433
	.long	19972402
	.long	1071104493
	.long	3712504873
	.long	1072016135
	.long	88491949
	.long	1015427660
	.long	1944781191
	.long	1071096084
	.long	2321106615
	.long	1072023846
	.long	2171176610
	.long	1009535771
	.long	1339972927
	.long	1071087721
	.long	363667784
	.long	1072031599
	.long	813753950
	.long	1015785209
	.long	1447192521
	.long	1071079403
	.long	3111574537
	.long	1072039393
	.long	2606161479
	.long	3162759746
	.long	1218806132
	.long	1071071130
	.long	2956612997
	.long	1072047230
	.long	2118169751
	.long	3162735553
	.long	3907805044
	.long	1071062901
	.long	885834528
	.long	1072055110
	.long	1973258547
	.long	3162261564
	.long	4182873220
	.long	1071054717
	.long	2186617381
	.long	1072063032
	.long	2270764084
	.long	3163272713
	.long	1013258799
	.long	1071046578
	.long	3561793907
	.long	1072070997
	.long	1157054053
	.long	1011890350
	.long	1963711167
	.long	1071038482
	.long	1719614413
	.long	1072079006
	.long	330458198
	.long	3163282740
	.long	1719614413
	.long	1071030430
	.long	1963711167
	.long	1072087058
	.long	1744767757
	.long	3160574294
	.long	3561793907
	.long	1071022421
	.long	1013258799
	.long	1072095154
	.long	1748797611
	.long	3160129082
	.long	2186617381
	.long	1071014456
	.long	4182873220
	.long	1072103293
	.long	629542646
	.long	3161996303
	.long	885834528
	.long	1071006534
	.long	3907805044
	.long	1072111477
	.long	2257091225
	.long	3161550407
	.long	2956612997
	.long	1070998654
	.long	1218806132
	.long	1072119706
	.long	1818613052
	.long	3162548441
	.long	3111574537
	.long	1070990817
	.long	1447192521
	.long	1072127979
	.long	1462857171
	.long	3162514521
	.long	363667784
	.long	1070983023
	.long	1339972927
	.long	1072136297
	.long	167908909
	.long	1015572152
	.long	2321106615
	.long	1070975270
	.long	1944781191
	.long	1072144660
	.long	3993278767
	.long	3161724279
	.long	3712504873
	.long	1070967559
	.long	19972402
	.long	1072153069
	.long	3507899862
	.long	1016009292
	.long	3566716925
	.long	1070959890
	.long	919555682
	.long	1072161523
	.long	3121969534
	.long	1012948226
	.long	917841882
	.long	1070952263
	.long	1413356050
	.long	1072170023
	.long	1651349291
	.long	3162668166
	.long	3395129871
	.long	1070944676
	.long	2571947539
	.long	1072178569
	.long	3558159064
	.long	3163376669
	.long	1453150082
	.long	1070937131
	.long	1176749997
	.long	1072187162
	.long	2738998779
	.long	3162035844
	.long	2731501122
	.long	1070929626
	.long	2604962541
	.long	1072195801
	.long	2614425274
	.long	3163539192
	.long	1990012071
	.long	1070922162
	.long	3649726105
	.long	1072204487
	.long	4085036346
	.long	1015649474
	.long	2583551245
	.long	1070914738
	.long	1110089947
	.long	1072213221
	.long	1451641639
	.long	1015474673
	.long	3577096743
	.long	1070907354
	.long	380978316
	.long	1072222002
	.long	854188970
	.long	3160462686
	.long	4040676318
	.long	1070900010
	.long	2568320822
	.long	1072230830
	.long	2732824428
	.long	1014352915
	.long	3049340112
	.long	1070892706
	.long	194117574
	.long	1072239707
	.long	777528612
	.long	3163412089
	.long	3978100823
	.long	1070885441
	.long	2966275557
	.long	1072248631
	.long	2176155324
	.long	3159842759
	.long	1617004845
	.long	1070878216
	.long	3418903055
	.long	1072257604
	.long	2527457337
	.long	3160820604
	.long	3645941911
	.long	1070871029
	.long	2682146384
	.long	1072266626
	.long	2082178513
	.long	3163363419
	.long	569847338
	.long	1070863882
	.long	1892288442
	.long	1072275697
	.long	2446255666
	.long	3162600381
	.long	78413852
	.long	1070856773
	.long	2191782032
	.long	1072284817
	.long	2960257726
	.long	1013742662
	.long	1276261410
	.long	1070849702
	.long	434316067
	.long	1072293987
	.long	2028358766
	.long	1013458122
	.long	3272845541
	.long	1070842669
	.long	2069751141
	.long	1072303206
	.long	1562170675
	.long	3162724681
	.long	887463927
	.long	1070835675
	.long	3964284211
	.long	1072312475
	.long	2111583915
	.long	1015427164
	.long	1829099622
	.long	1070828718
	.long	2990417245
	.long	1072321795
	.long	3683467745
	.long	3163369326
	.long	926591435
	.long	1070821799
	.long	321958744
	.long	1072331166
	.long	3401933767
	.long	1015794558
	.long	1603444721
	.long	1070814917
	.long	1434058175
	.long	1072340587
	.long	251133233
	.long	1015085769
	.long	2992903935
	.long	1070808072
	.long	3218338682
	.long	1072350059
	.long	3404164304
	.long	3162477108
	.long	4232894513
	.long	1070801264
	.long	2572866477
	.long	1072359583
	.long	878562433
	.long	1015521741
	.long	171030293
	.long	1070794494
	.long	697153126
	.long	1072369159
	.long	1283515429
	.long	3163283189
	.long	2839424854
	.long	1070787759
	.long	3092190715
	.long	1072378786
	.long	814012168
	.long	3159523422
	.long	2799960843
	.long	1070781061
	.long	2380618042
	.long	1072388466
	.long	3149557219
	.long	3163320799
	.long	3504003472
	.long	1070774399
	.long	4076559943
	.long	1072398198
	.long	2119478331
	.long	3160758351
	.long	4112506593
	.long	1070767773
	.long	815859274
	.long	1072407984
	.long	240396590
	.long	3163487443
	.long	3790955393
	.long	1070761183
	.long	2420883922
	.long	1072417822
	.long	2049810052
	.long	1014119888
	.long	1709341917
	.long	1070754629
	.long	1540824585
	.long	1072427714
	.long	1064017011
	.long	3163487690
	.long	1337108031
	.long	1070748110
	.long	3716502172
	.long	1072437659
	.long	2303740125
	.long	1014042725
	.long	1853186616
	.long	1070741626
	.long	1610600570
	.long	1072447659
	.long	3766732298
	.long	1015760183
	.long	2440944790
	.long	1070735177
	.long	777507147
	.long	1072457713
	.long	4282924205
	.long	1015187533
	.long	2288159958
	.long	1070728763
	.long	2483480501
	.long	1072467821
	.long	1216371780
	.long	1013034172
	.long	586995997
	.long	1070722384
	.long	3706687593
	.long	1072477984
	.long	3521726939
	.long	1013253067
	.long	828946858
	.long	1070716039
	.long	1432208378
	.long	1072488203
	.long	1401068914
	.long	3162363963
	.long	2214878420
	.long	1070709728
	.long	1242007932
	.long	1072498477
	.long	1132034716
	.long	3163339831
	.long	3949972341
	.long	1070703451
	.long	135105010
	.long	1072508807
	.long	1906148728
	.long	3163375739
	.long	948735466
	.long	1070697209
	.long	3707479175
	.long	1072519192
	.long	3613079303
	.long	1014164738
	.long	1014845819
	.long	1070691000
	.long	382305176
	.long	1072529635
	.long	2347622376
	.long	3162578625
	.long	3366293073
	.long	1070684824
	.long	64696965
	.long	1072540134
	.long	1768797490
	.long	1015816960
	.long	2930322912
	.long	1070678682
	.long	4076975200
	.long	1072550689
	.long	2029000899
	.long	1015208535
	.long	3228316108
	.long	1070672573
	.long	863738719
	.long	1072561303
	.long	1326992220
	.long	3162613197
	.long	3490863953
	.long	1070666497
	.long	351641897
	.long	1072571974
	.long	2172261526
	.long	3163010599
	.long	2952712987
	.long	1070660454
	.long	3884662774
	.long	1072582702
	.long	2158611599
	.long	1014210185
	.long	852742562
	.long	1070654444
	.long	4224142467
	.long	1072593489
	.long	3389820386
	.long	1015207202
	.long	728909815
	.long	1070648466
	.long	2728693978
	.long	1072604335
	.long	396109971
	.long	3163462691
	.long	1828292879
	.long	1070642520
	.long	764307441
	.long	1072615240
	.long	3021057420
	.long	3163329523
	.long	3402036099
	.long	1070636606
	.long	3999357479
	.long	1072626203
	.long	2258941616
	.long	1015924724
	.long	410360776
	.long	1070630725
	.long	929806999
	.long	1072637227
	.long	3205336643
	.long	1015259557
	.long	702412510
	.long	1070624875
	.long	1533953344
	.long	1072648310
	.long	769171851
	.long	1015665633
	.long	3541402996
	.long	1070619056
	.long	2912730644
	.long	1072659453
	.long	3490067722
	.long	3163405074
	.long	3899555717
	.long	1070613269
	.long	2174652632
	.long	1072670657
	.long	4087714590
	.long	1014450259
	.long	1048019041
	.long	1070607514
	.long	730821105
	.long	1072681922
	.long	2523232743
	.long	1012067188
	.long	2851812149
	.long	1070601789
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	127
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	4294967128
	.long	1071644671
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	1431655910
	.long	1069897045
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	2898925341
	.long	1067799893
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	564252519
	.long	1065423121
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.type	__svml_dcosh_ha_data_internal,@object
	.size	__svml_dcosh_ha_data_internal,4800
	.align 32
__dcosh_ha_CoutTab:
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
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
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
	.long	2411329662
	.long	1082536910
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	3875694624
	.long	1077247184
	.type	__dcosh_ha_CoutTab,@object
	.size	__dcosh_ha_CoutTab,1152
      	.section        .note.GNU-stack,"",@progbits
