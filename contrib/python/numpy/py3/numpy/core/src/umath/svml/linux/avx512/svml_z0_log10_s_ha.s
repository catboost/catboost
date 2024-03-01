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

/*
 * ALGORITHM DESCRIPTION:
 *  *  log10(x) = VGETEXP(x)*log10(2) + log10(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log10(mx) = -log10(RCP(mx)) + log10(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 4 fractional bits,
 *  *      and the table lookup for log(RCP(mx)) is based on a small permute instruction
 *  *
 *  *   LA, EP versions use interval interpolation (16 intervals)
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log10f16_ha_z0_0:

	.align    16,0x90
	.globl __svml_log10f16_ha

__svml_log10f16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovaps   %zmm0, %zmm2

/* GetMant(x), normalized to [1,2) for x>=0, NaN for x<0 */
        vgetmantps $10, {sae}, %zmm2, %zmm4

/* Reduced argument: R = SglRcp*Mantissa - 1 */
        vmovups   256+__svml_slog10_ha_data_internal_avx512(%rip), %zmm1

/* GetExp(x) */
        vgetexpps {sae}, %zmm2, %zmm7

/* Table lookup */
        vmovups   __svml_slog10_ha_data_internal_avx512(%rip), %zmm0

/* Start polynomial evaluation */
        vmovups   320+__svml_slog10_ha_data_internal_avx512(%rip), %zmm12
        vmovups   640+__svml_slog10_ha_data_internal_avx512(%rip), %zmm9
        vmovups   448+__svml_slog10_ha_data_internal_avx512(%rip), %zmm13
        vmovups   512+__svml_slog10_ha_data_internal_avx512(%rip), %zmm15
        vmovups   128+__svml_slog10_ha_data_internal_avx512(%rip), %zmm11

/* K*log10(2)_low+Tl */
        vmovups   704+__svml_slog10_ha_data_internal_avx512(%rip), %zmm10

/* SglRcp ~ 1/Mantissa */
        vrcp14ps  %zmm4, %zmm3

/* x<=0? */
        vfpclassps $94, %zmm2, %k1

/* round SglRcp to 5 fractional bits (RN mode, no Precision exception) */
        vrndscaleps $88, {sae}, %zmm3, %zmm5
        kmovw     %k1, %edx
        vgetexpps {sae}, %zmm5, %zmm8
        vfmsub231ps {rn-sae}, %zmm5, %zmm4, %zmm1
        vmovups   384+__svml_slog10_ha_data_internal_avx512(%rip), %zmm4

/* Prepare table index */
        vpsrld    $18, %zmm5, %zmm6

/* K*log(2)_high+Th */
        vsubps    {rn-sae}, %zmm8, %zmm7, %zmm3
        vfmadd231ps {rn-sae}, %zmm1, %zmm12, %zmm4

/* K+Th+(R*c1)_h */
        vmovups   576+__svml_slog10_ha_data_internal_avx512(%rip), %zmm12
        vfmadd231ps {rn-sae}, %zmm1, %zmm13, %zmm15
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm14
        vpermt2ps 64+__svml_slog10_ha_data_internal_avx512(%rip), %zmm6, %zmm0
        vpermt2ps 192+__svml_slog10_ha_data_internal_avx512(%rip), %zmm6, %zmm11

/* poly */
        vfmadd213ps {rn-sae}, %zmm15, %zmm14, %zmm4
        vfmadd231ps {rn-sae}, %zmm3, %zmm9, %zmm0
        vfmadd213ps {rn-sae}, %zmm11, %zmm10, %zmm3
        vmovaps   %zmm12, %zmm5
        vfmadd213ps {rn-sae}, %zmm0, %zmm1, %zmm5
        vfmadd213ps {rn-sae}, %zmm3, %zmm1, %zmm4

/* (R*c1)_h */
        vsubps    {rn-sae}, %zmm0, %zmm5, %zmm0
        vxorps    %zmm0, %zmm0, %zmm0{%k1}

/* (R*c1h)_l */
        vfmsub213ps {rn-sae}, %zmm0, %zmm12, %zmm1
        vaddps    {rn-sae}, %zmm1, %zmm4, %zmm1
        vaddps    {rn-sae}, %zmm1, %zmm5, %zmm0
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

        vmovups   %zmm2, 64(%rsp)
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

        call      __svml_slog10_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log10f16_ha,@function
	.size	__svml_log10f16_ha,.-__svml_log10f16_ha
..LN__svml_log10f16_ha.0:

.L_2__routine_start___svml_slog10_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_slog10_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_12


        movss     (%rdi), %xmm2
        xorl      %ecx, %ecx
        movss     %xmm2, -16(%rsp)
        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_4


        mulss     .L_2il0floatpacket.83(%rip), %xmm2
        movl      $-40, %ecx
        movss     %xmm2, -16(%rsp)

.LBL_2_4:

        pxor      %xmm0, %xmm0
        comiss    %xmm0, %xmm2
        jbe       .LBL_2_8


        movaps    %xmm2, %xmm1
        subss     .L_2il0floatpacket.99(%rip), %xmm1
        movss     %xmm1, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.84(%rip), %xmm0
        jbe       .LBL_2_7


        movss     %xmm2, -20(%rsp)
        pxor      %xmm8, %xmm8
        movzwl    -18(%rsp), %edi
        lea       __slog10_ha_CoutTab(%rip), %r10
        andl      $-32641, %edi
        addl      $16256, %edi
        movw      %di, -18(%rsp)
        movss     -20(%rsp), %xmm3
        movaps    %xmm3, %xmm0
        movss     .L_2il0floatpacket.86(%rip), %xmm2
        movaps    %xmm2, %xmm1
        addss     .L_2il0floatpacket.85(%rip), %xmm0
        addss     %xmm3, %xmm1
        movss     %xmm0, -24(%rsp)
        movl      -24(%rsp), %r8d
        movss     %xmm1, -24(%rsp)
        andl      $127, %r8d
        movss     -24(%rsp), %xmm9
        movss     .L_2il0floatpacket.95(%rip), %xmm6
        subss     %xmm2, %xmm9
        movzwl    -14(%rsp), %edx
        lea       (%r8,%r8,2), %r9d
        movss     (%r10,%r9,4), %xmm7
        andl      $32640, %edx
        shrl      $7, %edx
        subss     %xmm9, %xmm3
        mulss     %xmm7, %xmm9
        mulss     %xmm3, %xmm7
        subss     .L_2il0floatpacket.87(%rip), %xmm9
        movaps    %xmm9, %xmm4
        lea       -127(%rcx,%rdx), %ecx
        cvtsi2ss  %ecx, %xmm8
        addss     %xmm7, %xmm4
        mulss     %xmm4, %xmm6
        movss     .L_2il0floatpacket.96(%rip), %xmm10
        mulss     %xmm8, %xmm10
        addss     .L_2il0floatpacket.94(%rip), %xmm6
        addss     4(%r10,%r9,4), %xmm10
        mulss     %xmm4, %xmm6
        addss     %xmm9, %xmm10
        addss     .L_2il0floatpacket.93(%rip), %xmm6
        mulss     %xmm4, %xmm6
        movss     .L_2il0floatpacket.97(%rip), %xmm5
        mulss     %xmm5, %xmm8
        addss     .L_2il0floatpacket.92(%rip), %xmm6
        addss     8(%r10,%r9,4), %xmm8
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.91(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.90(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.89(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.88(%rip), %xmm6
        mulss     %xmm6, %xmm9
        mulss     %xmm7, %xmm6
        addss     %xmm6, %xmm8
        addss     %xmm7, %xmm8
        addss     %xmm8, %xmm9
        addss     %xmm9, %xmm10
        movss     %xmm10, (%rsi)
        ret

.LBL_2_7:

        movss     .L_2il0floatpacket.87(%rip), %xmm0
        mulss     %xmm0, %xmm1
        movss     .L_2il0floatpacket.95(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.94(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.93(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.92(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.91(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.90(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.89(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.88(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     %xmm1, %xmm2
        movss     %xmm2, (%rsi)
        ret

.LBL_2_8:

        ucomiss   %xmm0, %xmm2
        jp        .LBL_2_9
        je        .LBL_2_11

.LBL_2_9:

        divss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_11:

        movss     .L_2il0floatpacket.98(%rip), %xmm1
        movl      $2, %eax
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_12:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_14

.LBL_2_13:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_14:

        testl     $8388607, (%rdi)
        jne       .LBL_2_13


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog10_ha_cout_rare_internal,@function
	.size	__svml_slog10_ha_cout_rare_internal,.-__svml_slog10_ha_cout_rare_internal
..LN__svml_slog10_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog10_ha_data_internal_avx512:
	.long	1050288128
	.long	1049839616
	.long	1049405440
	.long	1048981504
	.long	1048567808
	.long	1047769088
	.long	1046990848
	.long	1046233088
	.long	1045495808
	.long	1044779008
	.long	1044074496
	.long	1043390464
	.long	1042718720
	.long	1042063360
	.long	1041424384
	.long	1040797696
	.long	1040179200
	.long	1038974976
	.long	1037803520
	.long	1036648448
	.long	1035509760
	.long	1034403840
	.long	1033314304
	.long	1032241152
	.long	1030586368
	.long	1028521984
	.long	1026490368
	.long	1024507904
	.long	1021673472
	.long	1017839616
	.long	1013055488
	.long	1004535808
	.long	916096252
	.long	922116432
	.long	3080225825
	.long	937573077
	.long	3035959522
	.long	907149878
	.long	932083293
	.long	937852153
	.long	932873468
	.long	3083585687
	.long	923506544
	.long	3080111720
	.long	922618327
	.long	929046259
	.long	3073359178
	.long	3075170456
	.long	3027570914
	.long	932146099
	.long	3086063431
	.long	3082882946
	.long	937977627
	.long	3064614024
	.long	3065085585
	.long	934503987
	.long	923757491
	.long	927716856
	.long	937794272
	.long	3074368796
	.long	929297206
	.long	3083361151
	.long	3065901602
	.long	912917177
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
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	3185469915
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	1041515222
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	3193854937
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	2990071104
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1054759897
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	1050288128
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	916096252
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	8388608
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.type	__svml_slog10_ha_data_internal_avx512,@object
	.size	__svml_slog10_ha_data_internal_avx512,960
	.align 32
__slog10_ha_CoutTab:
	.long	1121868800
	.long	0
	.long	0
	.long	1121641104
	.long	1004535808
	.long	912917177
	.long	1121413408
	.long	1013055488
	.long	3065901602
	.long	1121185712
	.long	1017839616
	.long	3083361151
	.long	1120958016
	.long	1021673472
	.long	929297206
	.long	1120844168
	.long	1023524864
	.long	3077496589
	.long	1120616472
	.long	1025499136
	.long	3070500046
	.long	1120388776
	.long	1027506176
	.long	912271551
	.long	1120274928
	.long	1028521984
	.long	927716856
	.long	1120047232
	.long	1030586368
	.long	923757491
	.long	1119933384
	.long	1031634944
	.long	3056752848
	.long	1119705688
	.long	1032775680
	.long	917029265
	.long	1119591840
	.long	1033314304
	.long	3065085585
	.long	1119364144
	.long	1034403840
	.long	3064614024
	.long	1119250296
	.long	1034954752
	.long	921091539
	.long	1119136448
	.long	1035513856
	.long	3057436454
	.long	1118908752
	.long	1036644352
	.long	922468856
	.long	1118794904
	.long	1037219840
	.long	3049155845
	.long	1118681056
	.long	1037799424
	.long	904301451
	.long	1118567208
	.long	1038385152
	.long	908617625
	.long	1118453360
	.long	1038977024
	.long	905362229
	.long	1118225664
	.long	1040179200
	.long	3027570914
	.long	1118111816
	.long	1040488448
	.long	882280038
	.long	1117997968
	.long	1040796672
	.long	911375775
	.long	1117884120
	.long	1041108480
	.long	904500572
	.long	1117770272
	.long	1041423872
	.long	3057579304
	.long	1117656424
	.long	1041742336
	.long	3053334705
	.long	1117542576
	.long	1042064384
	.long	3053389931
	.long	1117428728
	.long	1042390016
	.long	3051561465
	.long	1117314880
	.long	1042719232
	.long	3011187895
	.long	1117201032
	.long	1043052544
	.long	3059907089
	.long	1117087184
	.long	1043389440
	.long	3057005374
	.long	1116973336
	.long	1043729920
	.long	911932638
	.long	1116859488
	.long	1044075008
	.long	892958461
	.long	1116859488
	.long	1044075008
	.long	892958461
	.long	1116745640
	.long	1044424192
	.long	3048660547
	.long	1116631792
	.long	1044777472
	.long	3049032043
	.long	1116517944
	.long	1045134848
	.long	906867152
	.long	1116404096
	.long	1045496832
	.long	911484894
	.long	1116404096
	.long	1045496832
	.long	911484894
	.long	1116290248
	.long	1045863424
	.long	912580963
	.long	1116176400
	.long	1046235136
	.long	3058440244
	.long	1116062552
	.long	1046610944
	.long	895945194
	.long	1116062552
	.long	1046610944
	.long	895945194
	.long	1115948704
	.long	1046991872
	.long	904357324
	.long	1115834856
	.long	1047377920
	.long	902293870
	.long	1115721008
	.long	1047769088
	.long	907149878
	.long	1115721008
	.long	1047769088
	.long	907149878
	.long	1115529456
	.long	1048165888
	.long	3052029263
	.long	1115301760
	.long	1048567808
	.long	3035959522
	.long	1115301760
	.long	1048567808
	.long	3035959522
	.long	1115074064
	.long	1048775680
	.long	892998645
	.long	1115074064
	.long	1048775680
	.long	892998645
	.long	1114846368
	.long	1048982400
	.long	881767775
	.long	1114618672
	.long	1049192064
	.long	893839142
	.long	1114618672
	.long	1049192064
	.long	893839142
	.long	1114390976
	.long	1049404800
	.long	896498651
	.long	1114390976
	.long	1049404800
	.long	896498651
	.long	1114163280
	.long	1049620736
	.long	3033695903
	.long	1114163280
	.long	1049620736
	.long	3033695903
	.long	1113935584
	.long	1049839872
	.long	3029986056
	.long	1113935584
	.long	1049839872
	.long	3029986056
	.long	1113707888
	.long	1050062336
	.long	884671939
	.long	1113707888
	.long	1050062336
	.long	884671939
	.long	1113480192
	.long	1050288256
	.long	894707678
	.long	1050279936
	.long	964848148
	.long	1207959616
	.long	1174405120
	.long	1002438656
	.long	1400897536
	.long	0
	.long	1065353216
	.long	1121868800
	.long	3212771328
	.long	3079888218
	.long	870463078
	.long	2957202361
	.long	749987585
	.long	2838272395
	.long	631921661
	.long	2720751022
	.type	__slog10_ha_CoutTab,@object
	.size	__slog10_ha_CoutTab,848
	.align 4
.L_2il0floatpacket.83:
	.long	0x53800000
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0x3bc00000
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0x48000040
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0x46000000
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0x42de5c00
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0xbf7f0000
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0xb7935d5a
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x33e23666
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
	.align 4
.L_2il0floatpacket.91:
	.long	0xb04353b9
	.type	.L_2il0floatpacket.91,@object
	.size	.L_2il0floatpacket.91,4
	.align 4
.L_2il0floatpacket.92:
	.long	0x2cb3e701
	.type	.L_2il0floatpacket.92,@object
	.size	.L_2il0floatpacket.92,4
	.align 4
.L_2il0floatpacket.93:
	.long	0xa92c998b
	.type	.L_2il0floatpacket.93,@object
	.size	.L_2il0floatpacket.93,4
	.align 4
.L_2il0floatpacket.94:
	.long	0x25aa5bfd
	.type	.L_2il0floatpacket.94,@object
	.size	.L_2il0floatpacket.94,4
	.align 4
.L_2il0floatpacket.95:
	.long	0xa22b5dae
	.type	.L_2il0floatpacket.95,@object
	.size	.L_2il0floatpacket.95,4
	.align 4
.L_2il0floatpacket.96:
	.long	0x3e9a0000
	.type	.L_2il0floatpacket.96,@object
	.size	.L_2il0floatpacket.96,4
	.align 4
.L_2il0floatpacket.97:
	.long	0x39826a14
	.type	.L_2il0floatpacket.97,@object
	.size	.L_2il0floatpacket.97,4
	.align 4
.L_2il0floatpacket.98:
	.long	0xbf800000
	.type	.L_2il0floatpacket.98,@object
	.size	.L_2il0floatpacket.98,4
	.align 4
.L_2il0floatpacket.99:
	.long	0x3f800000
	.type	.L_2il0floatpacket.99,@object
	.size	.L_2il0floatpacket.99,4
      	.section        .note.GNU-stack,"",@progbits
