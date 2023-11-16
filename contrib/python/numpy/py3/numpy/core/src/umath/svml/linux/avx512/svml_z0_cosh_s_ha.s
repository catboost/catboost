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
.L_2__routine_start___svml_coshf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_coshf16_ha

__svml_coshf16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   1024+__svml_scosh_ha_data_internal(%rip), %zmm5
        vmovups   384+__svml_scosh_ha_data_internal(%rip), %zmm7

/*
 * ............... Load argument ............................
 * dM = x/log(2) + RShifter
 */
        vmovups   768+__svml_scosh_ha_data_internal(%rip), %zmm11
        vmovups   896+__svml_scosh_ha_data_internal(%rip), %zmm8
        vmovups   960+__svml_scosh_ha_data_internal(%rip), %zmm10

/* x^2 */
        vmovups   640+__svml_scosh_ha_data_internal(%rip), %zmm3

/* ... */
        vmovups   704+__svml_scosh_ha_data_internal(%rip), %zmm2

/* ............... G1,G2 2^N,2^(-N) ........... */
        vmovups   __svml_scosh_ha_data_internal(%rip), %zmm13
        vmovups   256+__svml_scosh_ha_data_internal(%rip), %zmm15
        vmovups   128+__svml_scosh_ha_data_internal(%rip), %zmm14

/* ...............Check for overflow\underflow ............. */
        vpternlogd $255, %zmm6, %zmm6, %zmm6
        vmovaps   %zmm0, %zmm4

/*
 * -------------------- Implementation  -------------------
 * ............... Abs argument ............................
 */
        vandnps   %zmm4, %zmm5, %zmm1
        vfmadd213ps {rn-sae}, %zmm7, %zmm1, %zmm11
        vpcmpd    $1, 512+__svml_scosh_ha_data_internal(%rip), %zmm1, %k1

/* iM now is an EXP(2^N) */
        vpslld    $18, %zmm11, %zmm12

/*
 * ................... R ...................................
 * sN = sM - RShifter
 */
        vsubps    {rn-sae}, %zmm7, %zmm11, %zmm9
        vpermt2ps 64+__svml_scosh_ha_data_internal(%rip), %zmm11, %zmm13
        vpermt2ps 320+__svml_scosh_ha_data_internal(%rip), %zmm11, %zmm15
        vpermt2ps 192+__svml_scosh_ha_data_internal(%rip), %zmm11, %zmm14
        vpandnd   %zmm1, %zmm1, %zmm6{%k1}

/* sR = sX - sN*Log2_hi */
        vfnmadd231ps {rn-sae}, %zmm8, %zmm9, %zmm1
        vptestmd  %zmm6, %zmm6, %k0

/* sR = (sX - sN*Log2_hi) - sN*Log2_lo */
        vfnmadd231ps {rn-sae}, %zmm10, %zmm9, %zmm1
        kmovw     %k0, %edx
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm0
        vmulps    {rn-sae}, %zmm0, %zmm2, %zmm2

/* sOut = r^2*(a2) */
        vmulps    {rn-sae}, %zmm0, %zmm3, %zmm0

/* sSinh_r = r + r*(r^2*(a3)) */
        vfmadd213ps {rn-sae}, %zmm1, %zmm1, %zmm2
        vpandd    1216+__svml_scosh_ha_data_internal(%rip), %zmm12, %zmm5
        vpaddd    %zmm5, %zmm13, %zmm8
        vpsubd    %zmm5, %zmm15, %zmm7
        vpaddd    %zmm5, %zmm14, %zmm14

/* sG2 = 2^N*Th + 2^(-N)*T_h */
        vaddps    {rn-sae}, %zmm7, %zmm8, %zmm15

/* sG1 = 2^N*Th - 2^(-N)*T_h */
        vsubps    {rn-sae}, %zmm7, %zmm8, %zmm6

/* res = sG1*(r + r*(r^2*(a3))) + sG2*(1+r^2*(a2)) */
        vfmadd213ps {rn-sae}, %zmm14, %zmm15, %zmm0
        vfmadd213ps {rn-sae}, %zmm0, %zmm6, %zmm2
        vaddps    {rn-sae}, %zmm7, %zmm2, %zmm1
        vaddps    {rn-sae}, %zmm8, %zmm1, %zmm0
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

        vmovups   %zmm4, 64(%rsp)
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

        call      __svml_scosh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_coshf16_ha,@function
	.size	__svml_coshf16_ha,.-__svml_coshf16_ha
..LN__svml_coshf16_ha.0:

.L_2__routine_start___svml_scosh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_scosh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r8
        movzwl    2(%rdi), %edx
        xorl      %eax, %eax
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_12


        pxor      %xmm0, %xmm0
        cvtss2sd  (%rdi), %xmm0
        movsd     %xmm0, -8(%rsp)
        andb      $127, -1(%rsp)
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        cmpl      $15504, %edx
        jle       .LBL_2_10


        movsd     -8(%rsp), %xmm0
        movsd     1096+__scosh_ha_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        movq      1128+__scosh_ha_CoutTab(%rip), %rdx
        movq      %rdx, -8(%rsp)
        comisd    1144+__scosh_ha_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1040+__scosh_ha_CoutTab(%rip), %xmm1
        lea       __scosh_ha_CoutTab(%rip), %r9
        mulsd     %xmm0, %xmm1
        addsd     1048+__scosh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movsd     1088+__scosh_ha_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1048+__scosh_ha_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       (%rsi,%rsi), %ecx
        movsd     -32(%rsp), %xmm3
        lea       1(%rsi,%rsi), %edi
        mulsd     1104+__scosh_ha_CoutTab(%rip), %xmm3
        movsd     -32(%rsp), %xmm4
        subsd     %xmm3, %xmm0
        mulsd     1112+__scosh_ha_CoutTab(%rip), %xmm4
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1080+__scosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1072+__scosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1064+__scosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__scosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        movsd     (%r9,%rcx,8), %xmm0
        mulsd     %xmm0, %xmm1
        addsd     (%r9,%rdi,8), %xmm1
        addsd     %xmm0, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_7


        movq      1128+__scosh_ha_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
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
        mulsd     1024+__scosh_ha_CoutTab(%rip), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        ret

.LBL_2_8:

        movsd     1040+__scosh_ha_CoutTab(%rip), %xmm1
        lea       __scosh_ha_CoutTab(%rip), %rcx
        movzwl    -2(%rsp), %esi
        andl      $-32753, %esi
        movsd     1080+__scosh_ha_CoutTab(%rip), %xmm14
        mulsd     %xmm0, %xmm1
        addsd     1048+__scosh_ha_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movl      -40(%rsp), %r10d
        movl      %r10d, %r9d
        shrl      $6, %r9d
        subsd     1048+__scosh_ha_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       1023(%r9), %edi
        movsd     -32(%rsp), %xmm3
        addl      $1022, %r9d
        mulsd     1104+__scosh_ha_CoutTab(%rip), %xmm3
        andl      $63, %r10d
        movsd     -32(%rsp), %xmm4
        lea       (%r10,%r10), %edx
        mulsd     1112+__scosh_ha_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm0
        andl      $2047, %r9d
        negl      %edi
        movsd     (%rcx,%rdx,8), %xmm15
        negl      %edx
        shll      $4, %r9d
        addl      $-4, %edi
        orl       %r9d, %esi
        andl      $2047, %edi
        movw      %si, -2(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        lea       1(%r10,%r10), %r11d
        movsd     -8(%rsp), %xmm6
        orl       %edi, %esi
        movw      %si, -2(%rsp)
        lea       128(%rdx), %esi
        addl      $129, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm6, %xmm15
        movaps    %xmm0, %xmm5
        movaps    %xmm15, %xmm8
        mulsd     %xmm0, %xmm5
        movaps    %xmm15, %xmm10
        movsd     (%rcx,%r11,8), %xmm2
        mulsd     %xmm6, %xmm2
        mulsd     %xmm5, %xmm14
        movsd     -8(%rsp), %xmm7
        movaps    %xmm2, %xmm12
        movsd     (%rcx,%rdx,8), %xmm13
        mulsd     %xmm7, %xmm13
        addsd     1064+__scosh_ha_CoutTab(%rip), %xmm14
        movsd     1088+__scosh_ha_CoutTab(%rip), %xmm1
        subsd     %xmm13, %xmm12
        mulsd     %xmm5, %xmm1
        mulsd     %xmm5, %xmm14
        mulsd     %xmm0, %xmm12
        addsd     1072+__scosh_ha_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm14
        addsd     %xmm12, %xmm2
        mulsd     %xmm5, %xmm1
        addsd     %xmm13, %xmm2
        addsd     1056+__scosh_ha_CoutTab(%rip), %xmm1
        movsd     (%rcx,%rsi,8), %xmm11
        mulsd     %xmm7, %xmm11
        mulsd     %xmm5, %xmm1
        addsd     %xmm11, %xmm8
        subsd     %xmm11, %xmm15
        movsd     %xmm8, -24(%rsp)
        movsd     -24(%rsp), %xmm9
        mulsd     %xmm15, %xmm14
        subsd     %xmm9, %xmm10
        mulsd     %xmm15, %xmm0
        addsd     %xmm11, %xmm10
        addsd     %xmm14, %xmm2
        movsd     %xmm10, -16(%rsp)
        addsd     %xmm0, %xmm2
        movsd     -24(%rsp), %xmm3
        mulsd     %xmm3, %xmm1
        movsd     -16(%rsp), %xmm6
        addsd     %xmm1, %xmm2
        addsd     %xmm6, %xmm2
        movsd     %xmm2, -24(%rsp)
        movsd     -24(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        cvtsd2ss  %xmm3, %xmm3
        movss     %xmm3, (%r8)
        ret

.LBL_2_9:

        movsd     1120+__scosh_ha_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        ret

.LBL_2_10:

        movsd     1136+__scosh_ha_CoutTab(%rip), %xmm0
        addsd     -8(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)


        ret

.LBL_2_12:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%r8)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_scosh_ha_cout_rare_internal,@function
	.size	__svml_scosh_ha_cout_rare_internal,.-__svml_scosh_ha_cout_rare_internal
..LN__svml_scosh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_scosh_ha_data_internal:
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
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
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
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
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
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
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
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
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
	.long	944570348
	.long	870537889
	.long	1056963788
	.long	988584323
	.long	3089368227
	.long	1026654286
	.long	1056972809
	.long	1005362723
	.long	3089410886
	.long	1035053812
	.long	1056996444
	.long	1013759196
	.long	3089450701
	.long	1040545168
	.long	1057035884
	.long	1018294210
	.long	3089519489
	.long	1043486152
	.long	1057091204
	.long	1022210002
	.long	3089622651
	.long	1046449073
	.long	1057162508
	.long	1024792095
	.long	3089732783
	.long	1049007747
	.long	1057249929
	.long	1026787500
	.long	3089879760
	.long	1050519514
	.long	1057353632
	.long	1028802193
	.long	3090009552
	.long	1052050675
	.long	1057473810
	.long	1030843673
	.long	3090201654
	.long	1053604104
	.long	1057610691
	.long	1032358162
	.long	3090393038
	.long	1055182718
	.long	1057764530
	.long	1033401816
	.long	3090624519
	.long	1056789478
	.long	1057935617
	.long	1034476232
	.long	3090859136
	.long	1057696005
	.long	1058124272
	.long	1035562860
	.long	3091126256
	.long	1058532085
	.long	1058330850
	.long	1036689182
	.long	3091401474
	.long	1059386854
	.long	1058555738
	.long	1037824061
	.long	3091713853
	.long	1060261915
	.long	1058799359
	.long	1038999406
	.long	3092054410
	.long	1061158912
	.long	1059062170
	.long	1040187520
	.long	3092413532
	.long	1062079528
	.long	1059344664
	.long	1040796570
	.long	3092816174
	.long	1063025490
	.long	1059647372
	.long	1041432479
	.long	3093223701
	.long	1063998575
	.long	1059970861
	.long	1042082428
	.long	3093662789
	.long	1065000609
	.long	1060315739
	.long	1042753182
	.long	3094122539
	.long	1065693345
	.long	1060682653
	.long	1043434554
	.long	3094645738
	.long	1066226161
	.long	1061072293
	.long	1044155985
	.long	3095155406
	.long	1066776362
	.long	1061485388
	.long	1044890780
	.long	3095550555
	.long	1067344981
	.long	1061922715
	.long	1045635453
	.long	3095847386
	.long	1067933084
	.long	1062385095
	.long	1046418690
	.long	3096168298
	.long	1068541775
	.long	1062873396
	.long	1047240047
	.long	3096488137
	.long	1069172198
	.long	1063388533
	.long	1048071426
	.long	3096841182
	.long	1069825535
	.long	1063931475
	.long	1048758942
	.long	3097209475
	.long	1070503013
	.long	1064503240
	.long	1049207926
	.long	3097589791
	.long	1071205903
	.long	1065104901
	.long	1049678351
	.long	3097993402
	.long	1071935525
	.long	1065545402
	.long	1050164645
	.long	3098411341
	.long	1072693248
	.long	1065877852
	.long	1050673310
	.long	3098859808
	.long	1073480495
	.long	1066227033
	.long	1051198081
	.long	3099325394
	.long	1074020284
	.long	1066593600
	.long	1051736997
	.long	3099839474
	.long	1074445677
	.long	1066978242
	.long	1052300332
	.long	3100370328
	.long	1074888136
	.long	1067381680
	.long	1052909383
	.long	3100909820
	.long	1075348494
	.long	1067804671
	.long	1053514627
	.long	3101459594
	.long	1075827613
	.long	1068248009
	.long	1054160592
	.long	3102047769
	.long	1076326394
	.long	1068712527
	.long	1054814464
	.long	3102677758
	.long	1076845772
	.long	1069199097
	.long	1055502910
	.long	3103340170
	.long	1077386722
	.long	1069708632
	.long	1056225281
	.long	3103903569
	.long	1077950259
	.long	1070242088
	.long	1056977834
	.long	3104249593
	.long	1078537443
	.long	1070800466
	.long	1057360587
	.long	3104632246
	.long	1079149373
	.long	1071384816
	.long	1057776467
	.long	3105038122
	.long	1079787200
	.long	1071996234
	.long	1058202023
	.long	3105440616
	.long	1080452121
	.long	1072635866
	.long	1058640522
	.long	3105862938
	.long	1081145383
	.long	1073304914
	.long	1059104028
	.long	3106308416
	.long	1081868288
	.long	1073873229
	.long	1059586215
	.long	3106787412
	.long	1082376312
	.long	1074239082
	.long	1060097588
	.long	3107276928
	.long	1082769472
	.long	1074621614
	.long	1060619929
	.long	3107776680
	.long	1083179578
	.long	1075021543
	.long	1061153935
	.long	3108330475
	.long	1083607398
	.long	1075439621
	.long	1061737331
	.long	3108881710
	.long	1084053737
	.long	1075876631
	.long	1062331214
	.long	3109487286
	.long	1084519432
	.long	1076333395
	.long	1062953203
	.long	3110070509
	.long	1085005358
	.long	1076810768
	.long	1063586843
	.long	3110728850
	.long	1085512425
	.long	1077309649
	.long	1064276575
	.long	3111383871
	.long	1086041587
	.long	1077830972
	.long	1064978612
	.long	3112084118
	.long	1086593836
	.long	1078375717
	.long	1065536743
	.long	3112493703
	.long	1087170210
	.long	1078944906
	.long	1065913820
	.long	3112867371
	.long	1087771789
	.long	1079539607
	.long	1066317189
	.long	3113278547
	.long	1088399703
	.long	1080160938
	.long	1066739445
	.long	3113690682
	.long	1089055131
	.long	1080810063
	.long	1067177635
	.long	3114113585
	.long	1089739304
	.long	1081488201
	.long	1067625214
	.long	3114565947
	.long	1090453504
	.long	1082163529
	.long	1068105897
	.long	3115052575
	.long	1090859057
	.long	1082533550
	.long	1068596020
	.long	3115539880
	.long	1091248226
	.long	1082920073
	.long	1069111659
	.long	3116077017
	.long	1091654509
	.long	1083323825
	.long	1069663909
	.long	3116603774
	.long	1092078670
	.long	1083745562
	.long	1070225544
	.long	3117166138
	.long	1092521504
	.long	1084186077
	.long	1070821702
	.long	3117769278
	.long	1092983843
	.long	1084646197
	.long	1071437696
	.long	3118359457
	.long	1093466555
	.long	1085126784
	.long	1072071392
	.long	3119000307
	.long	1093970545
	.long	1085628742
	.long	1072746100
	.long	3119686251
	.long	1094496760
	.long	1086153013
	.long	1073443058
	.long	3120382865
	.long	1095046187
	.long	1086700580
	.long	1073960254
	.long	3120829800
	.long	1095619858
	.long	1087272471
	.long	1074341025
	.long	3121221705
	.long	1096218849
	.long	1087869761
	.long	1074743826
	.long	3121630109
	.long	1096844285
	.long	1088493570
	.long	1075162699
	.long	3122040558
	.long	1097497340
	.long	1089145068
	.long	1075598254
	.long	3122471799
	.long	1098179240
	.long	1089825479
	.long	1076049525
	.long	3122921786
	.long	1098891264
	.long	1090527560
	.long	1076527273
	.long	3123410322
	.long	1099271199
	.long	1090898623
	.long	1077017199
	.long	3123905268
	.long	1099659370
	.long	1091286144
	.long	1077536277
	.long	3124427171
	.long	1100064698
	.long	1091690851
	.long	1078077742
	.long	3124955362
	.long	1100487944
	.long	1092113503
	.long	1078639053
	.long	3125512315
	.long	1100929902
	.long	1092554894
	.long	1079230664
	.long	3126114846
	.long	1101391402
	.long	1093015853
	.long	1079845159
	.long	3126723150
	.long	1101873310
	.long	1093497244
	.long	1080489100
	.long	3127384205
	.long	1102376531
	.long	1093999972
	.long	1081154940
	.long	3128045109
	.long	1102902009
	.long	1094524979
	.long	1081855739
	.long	3128757202
	.long	1103450730
	.long	1095073252
	.long	1082365260
	.long	3129233957
	.long	1104023725
	.long	1095645820
	.long	1082749515
	.long	3129593552
	.long	1104622070
	.long	1096243755
	.long	1083141940
	.long	3130009456
	.long	1105246886
	.long	1096868184
	.long	1083565083
	.long	3130431772
	.long	1105899348
	.long	1097520276
	.long	1083997423
	.long	3130861002
	.long	1106580680
	.long	1098201255
	.long	1084447059
	.long	3131310395
	.long	1107292160
	.long	1098910024
	.long	1084924074
	.long	3131783023
	.long	1107665690
	.long	1099281347
	.long	1085424177
	.long	3132296264
	.long	1108053612
	.long	1099669118
	.long	1085933889
	.long	3132789780
	.long	1108458701
	.long	1100074063
	.long	1086477769
	.long	3133359295
	.long	1108881718
	.long	1100496945
	.long	1087044117
	.long	3133914895
	.long	1109323457
	.long	1100938555
	.long	1087634592
	.long	3134525467
	.long	1109784747
	.long	1101399724
	.long	1088253827
	.long	3135105529
	.long	1110266455
	.long	1101881315
	.long	1088879869
	.long	3135755251
	.long	1110769483
	.long	1102384235
	.long	1089558833
	.long	3136442666
	.long	1111294777
	.long	1102909427
	.long	1090255482
	.long	3137142241
	.long	1111843322
	.long	1103457876
	.long	1090755410
	.long	3137605970
	.long	1112416148
	.long	1104030612
	.long	1091140533
	.long	3137986162
	.long	1113014331
	.long	1104628710
	.long	1091535483
	.long	3138387555
	.long	1113638993
	.long	1105253293
	.long	1091949463
	.long	3138804646
	.long	1114291306
	.long	1105905533
	.long	1092388670
	.long	3139233372
	.long	1114972496
	.long	1106586654
	.long	1092837897
	.long	3139699003
	.long	1115683840
	.long	1107297096
	.long	1093314730
	.long	3140167653
	.long	1116055769
	.long	1107668484
	.long	1093812263
	.long	3140669084
	.long	1116443628
	.long	1108056317
	.long	1094334974
	.long	3141171888
	.long	1116848658
	.long	1108461322
	.long	1094864117
	.long	3141735347
	.long	1117271618
	.long	1108884261
	.long	1095426609
	.long	3142298803
	.long	1117713302
	.long	1109325926
	.long	1096021914
	.long	3142894998
	.long	1118174540
	.long	1109787147
	.long	1096632105
	.long	3143500773
	.long	1118656197
	.long	1110268789
	.long	1097274132
	.long	3144147662
	.long	1119159177
	.long	1110771757
	.long	1097951263
	.long	3144833512
	.long	1119684425
	.long	1111296995
	.long	1098646873
	.long	3145529957
	.long	1120232926
	.long	1111845488
	.long	1099144404
	.long	3145990428
	.long	1120805710
	.long	1112418266
	.long	1099527187
	.long	3146379868
	.long	1121403852
	.long	1113016405
	.long	1099927882
	.long	3146785826
	.long	1122028475
	.long	1113641027
	.long	1100344686
	.long	3147185223
	.long	1122680752
	.long	1114293303
	.long	1100772823
	.long	3147622018
	.long	1123361906
	.long	1114974460
	.long	1101227063
	.long	3148087611
	.long	1124073216
	.long	1115685320
	.long	1101703851
	.long	3148547074
	.long	1124444745
	.long	1116056724
	.long	1102195626
	.long	3149061936
	.long	1124832589
	.long	1116444573
	.long	1102706245
	.long	3149567064
	.long	1125237603
	.long	1116849593
	.long	1103257276
	.long	3150120816
	.long	1125660549
	.long	1117272546
	.long	1103813688
	.long	3150694429
	.long	1126102219
	.long	1117714225
	.long	1104415316
	.long	3151287031
	.long	1126563444
	.long	1118175459
	.long	1105023245
	.long	3151907427
	.long	1127045088
	.long	1118657114
	.long	1105674384
	.long	3152520833
	.long	1127548057
	.long	1119160093
	.long	1106330596
	.long	3153222679
	.long	1128073293
	.long	1119685343
	.long	1107036177
	.long	3153918342
	.long	1128621783
	.long	1120233847
	.long	1107533108
	.long	3154369806
	.long	1129194557
	.long	1120806635
	.long	1107910191
	.long	3154757460
	.long	1129792689
	.long	1121404784
	.long	1108309765
	.long	3155168656
	.long	1130417302
	.long	1122029416
	.long	1108729833
	.long	3155580017
	.long	1131069569
	.long	1122681702
	.long	1109165432
	.long	3156018828
	.long	1131750714
	.long	1123362868
	.long	1109620926
	.long	3156476219
	.long	1132462016
	.long	1124073832
	.long	1110092587
	.long	3156933385
	.long	1132833445
	.long	1124445240
	.long	1110582922
	.long	3157451606
	.long	1133221285
	.long	1124833093
	.long	1111095633
	.long	3157965508
	.long	1133626295
	.long	1125238117
	.long	1111652137
	.long	3158533220
	.long	1134049237
	.long	1125661074
	.long	1112217259
	.long	3159060211
	.long	1134490905
	.long	1126102755
	.long	1112789777
	.long	3159676495
	.long	1134952126
	.long	1126563993
	.long	1113412486
	.long	3160292353
	.long	1135433767
	.long	1127045651
	.long	1114060788
	.long	3160905582
	.long	1135936733
	.long	1127548633
	.long	1114716886
	.long	3161611427
	.long	1136461966
	.long	1128073886
	.long	1115424959
	.long	3162315088
	.long	1137010453
	.long	1128622393
	.long	1115924298
	.long	3162768396
	.long	1137583224
	.long	1129195184
	.long	1116305071
	.long	3163147411
	.long	1138181354
	.long	1129793335
	.long	1116699250
	.long	3163551723
	.long	1138805965
	.long	1130417969
	.long	1117115018
	.long	3163974268
	.long	1139458229
	.long	1131070258
	.long	1117557598
	.long	3164409487
	.long	1140139372
	.long	1131751426
	.long	1118010847
	.long	3164864827
	.long	1140850672
	.long	1132462416
	.long	1118481227
	.long	3165321418
	.long	1141222076
	.long	1132833825
	.long	1118971202
	.long	3165840479
	.long	1141609915
	.long	1133221679
	.long	1119484436
	.long	3166356575
	.long	1142014924
	.long	1133626704
	.long	1120042308
	.long	3166895003
	.long	1142437866
	.long	1134049661
	.long	1120589147
	.long	3167459500
	.long	1142879532
	.long	1134491344
	.long	1121185079
	.long	3168048930
	.long	1143340753
	.long	1134952582
	.long	1121791022
	.long	3168671847
	.long	1143822393
	.long	1135434241
	.long	1122443730
	.long	3169293226
	.long	1144325358
	.long	1135937224
	.long	1123104914
	.long	3170008263
	.long	1144850590
	.long	1136462478
	.long	1123818726
	.long	3170689344
	.long	1145399077
	.long	1137010985
	.long	1124308436
	.long	3171155403
	.long	1145971847
	.long	1137583777
	.long	1124692689
	.long	3171540451
	.long	1146569976
	.long	1138181929
	.long	1125090634
	.long	3171951236
	.long	1147194586
	.long	1138806564
	.long	1125510443
	.long	3172347900
	.long	1147846851
	.long	1139458852
	.long	1125936865
	.long	3172790414
	.long	1148527993
	.long	1140140021
	.long	1126394668
	.long	3173253435
	.long	1149239292
	.long	1140851018
	.long	1126869843
	.long	3173701689
	.long	1149610690
	.long	1141222427
	.long	1127354613
	.long	3174212768
	.long	1149998528
	.long	1141610281
	.long	1127883320
	.long	3174721217
	.long	1150403538
	.long	1142015306
	.long	1128415961
	.long	3175285098
	.long	1150826479
	.long	1142438264
	.long	1128978690
	.long	3175842584
	.long	1151268145
	.long	1142879947
	.long	1129570245
	.long	3176458075
	.long	1151729365
	.long	1143341186
	.long	1130192458
	.long	3177074563
	.long	1152211005
	.long	1143822845
	.long	1130841152
	.long	3177689786
	.long	1152713970
	.long	1144325828
	.long	1131498492
	.long	3178398928
	.long	1153239202
	.long	1144851082
	.long	1132208623
	.long	3179074364
	.long	1153787689
	.long	1145399589
	.long	1132695927
	.long	3179539514
	.long	1154360459
	.long	1145972381
	.long	1133078492
	.long	3179921974
	.long	1154958588
	.long	1146570533
	.long	1133474821
	.long	3180330280
	.long	1155583198
	.long	1147195168
	.long	1133893083
	.long	3180740958
	.long	1156235462
	.long	1147847457
	.long	1134328253
	.long	3181181199
	.long	1156916604
	.long	1148528626
	.long	1134784637
	.long	3181625657
	.long	1157627903
	.long	1149239624
	.long	1135258451
	.long	3182104600
	.long	1157999299
	.long	1149611034
	.long	1135752152
	.long	3182613683
	.long	1158387137
	.long	1149998888
	.long	1136279613
	.long	3183120221
	.long	1158792147
	.long	1150403913
	.long	1136811061
	.long	3183682271
	.long	1159215088
	.long	1150826871
	.long	1137372647
	.long	3184238005
	.long	1159656754
	.long	1151268554
	.long	1137963108
	.long	3184851817
	.long	1160117974
	.long	1151729793
	.long	1138584273
	.long	3185433925
	.long	1160599615
	.long	1152211451
	.long	1139211502
	.long	3186080382
	.long	1161102579
	.long	1152714435
	.long	1139888343
	.long	3186788050
	.long	1161627811
	.long	1153239689
	.long	1140597554
	.long	3187462075
	.long	1162176298
	.long	1153788196
	.long	1141084255
	.long	3187926998
	.long	1162749068
	.long	1154360988
	.long	1141466399
	.long	3188308811
	.long	1163347197
	.long	1154959140
	.long	1141862324
	.long	3188716497
	.long	1163971807
	.long	1155583775
	.long	1142280199
	.long	3189126581
	.long	1164624071
	.long	1156236064
	.long	1142714999
	.long	3189566254
	.long	1165305213
	.long	1156917233
	.long	1143171028
	.long	3190026555
	.long	1166016512
	.long	1157628232
	.long	1143644503
	.long	3190504977
	.long	1166387907
	.long	1157999642
	.long	1144148108
	.long	3190980787
	.long	1166775746
	.long	1158387495
	.long	1144654797
	.long	3191519621
	.long	1167180755
	.long	1158792521
	.long	1145206407
	.long	3192081214
	.long	1167603696
	.long	1159215479
	.long	1145767708
	.long	3192636510
	.long	1168045362
	.long	1159657162
	.long	1146357895
	.long	3193217128
	.long	1168506583
	.long	1160118400
	.long	1146958337
	.long	3193831608
	.long	1168988223
	.long	1160600059
	.long	1147605777
	.long	3194477680
	.long	1169491187
	.long	1161103043
	.long	1148282377
	.long	3195152207
	.long	1170016420
	.long	1161628296
	.long	1148970897
	.long	3195858652
	.long	1170564906
	.long	1162176804
	.long	1149475351
	.long	3196319422
	.long	1171137676
	.long	1162749596
	.long	1149857389
	.long	3196701072
	.long	1171735805
	.long	1163347748
	.long	1150253213
	.long	3197108604
	.long	1172360415
	.long	1163972383
	.long	1150670991
	.long	3197518540
	.long	1173012679
	.long	1164624672
	.long	1151105698
	.long	3197958071
	.long	1173693821
	.long	1165305841
	.long	1151561639
	.long	3198418235
	.long	1174405120
	.long	1166016840
	.long	1152035030
	.long	3198896527
	.long	1174776515
	.long	1166388250
	.long	1152538553
	.long	3199372213
	.long	1175164354
	.long	1166776103
	.long	1153045164
	.long	3199910927
	.long	1175569363
	.long	1167181129
	.long	1153596699
	.long	3200472406
	.long	1175992304
	.long	1167604087
	.long	1154157929
	.long	3201027592
	.long	1176433970
	.long	1168045770
	.long	1154748047
	.long	3201608106
	.long	1176895191
	.long	1168507008
	.long	1155348424
	.long	3202222485
	.long	1177376831
	.long	1168988667
	.long	1155995801
	.long	3202868461
	.long	1177879795
	.long	1169491651
	.long	1156672341
	.long	3203542895
	.long	1178405028
	.long	1170016904
	.long	1157360804
	.long	3204249252
	.long	1178953514
	.long	1170565412
	.long	1157864581
	.long	3204708983
	.long	1179526284
	.long	1171138204
	.long	1158246593
	.long	3205090594
	.long	1180124413
	.long	1171736356
	.long	1158642392
	.long	3205498087
	.long	1180749023
	.long	1172360991
	.long	1159060145
	.long	3205907986
	.long	1181401287
	.long	1173013280
	.long	1159494829
	.long	3206347481
	.long	1182082429
	.long	1173694449
	.long	1159950748
	.long	3206807611
	.long	1182793728
	.long	1174405448
	.long	1160424117
	.long	3207285871
	.long	1183165123
	.long	1174776858
	.long	1160927621
	.long	3207761525
	.long	1183552962
	.long	1175164711
	.long	1161434212
	.long	3208300209
	.long	1183957971
	.long	1175569737
	.long	1161985728
	.long	3208861660
	.long	1184380912
	.long	1175992695
	.long	1162546940
	.long	3209416818
	.long	1184822578
	.long	1176434378
	.long	1163137042
	.long	3209997306
	.long	1185283799
	.long	1176895616
	.long	1163737402
	.long	3210611660
	.long	1185765439
	.long	1177377275
	.long	1164384763
	.long	3211257612
	.long	1186268403
	.long	1177880259
	.long	1165061288
	.long	3211932023
	.long	1186793636
	.long	1178405512
	.long	1165749736
	.long	3212638358
	.long	1187342122
	.long	1178954020
	.long	1166253344
	.long	3213097830
	.long	1187914892
	.long	1179526812
	.long	1166635350
	.long	3213479430
	.long	1188513021
	.long	1180124964
	.long	1167031142
	.long	3213886913
	.long	1189137631
	.long	1180749599
	.long	1167448890
	.long	3214296803
	.long	1189789895
	.long	1181401888
	.long	1167883568
	.long	3214736289
	.long	1190471037
	.long	1182083057
	.long	1168339481
	.type	__svml_scosh_ha_data_internal,@object
	.size	__svml_scosh_ha_data_internal,5568
	.align 32
__scosh_ha_CoutTab:
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
	.long	2684354560
	.long	1079401119
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
	.type	__scosh_ha_CoutTab,@object
	.size	__scosh_ha_CoutTab,1152
      	.section        .note.GNU-stack,"",@progbits
