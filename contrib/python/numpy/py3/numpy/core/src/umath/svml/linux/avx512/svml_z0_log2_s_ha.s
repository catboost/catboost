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
 *  *  log2(x) = VGETEXP(x) + log2(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log2(mx) = -log2(RCP(mx)) + log2(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 4 fractional bits,
 *  *      and the table lookup for log2(RCP(mx)) is based on a small permute instruction
 *  *
 *  *   LA, EP versions use interval interpolation (16 intervals)
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log2f16_ha_z0_0:

	.align    16,0x90
	.globl __svml_log2f16_ha

__svml_log2f16_ha:


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
        vmovups   256+__svml_slog2_ha_data_internal_avx512(%rip), %zmm1

/* GetExp(x) */
        vgetexpps {sae}, %zmm2, %zmm7

/* Table lookup */
        vmovups   __svml_slog2_ha_data_internal_avx512(%rip), %zmm10

/* Start polynomial evaluation */
        vmovups   320+__svml_slog2_ha_data_internal_avx512(%rip), %zmm11
        vmovups   448+__svml_slog2_ha_data_internal_avx512(%rip), %zmm12
        vmovups   512+__svml_slog2_ha_data_internal_avx512(%rip), %zmm14
        vmovups   128+__svml_slog2_ha_data_internal_avx512(%rip), %zmm0

/* SglRcp ~ 1/Mantissa */
        vrcp14ps  %zmm4, %zmm3

/* x<=0? */
        vfpclassps $94, %zmm2, %k1

/* round SglRcp to 5 fractional bits (RN mode, no Precision exception) */
        vrndscaleps $88, {sae}, %zmm3, %zmm5
        vmovups   384+__svml_slog2_ha_data_internal_avx512(%rip), %zmm3
        kmovw     %k1, %edx
        vgetexpps {sae}, %zmm5, %zmm8
        vfmsub231ps {rn-sae}, %zmm5, %zmm4, %zmm1

/* Prepare table index */
        vpsrld    $18, %zmm5, %zmm6

/* K*log(2)_high+Th */
        vsubps    {rn-sae}, %zmm8, %zmm7, %zmm9
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm13
        vfmadd231ps {rn-sae}, %zmm1, %zmm11, %zmm3

/* K+Th+(R*c1)_h */
        vmovups   576+__svml_slog2_ha_data_internal_avx512(%rip), %zmm11
        vfmadd231ps {rn-sae}, %zmm1, %zmm12, %zmm14
        vpermt2ps 64+__svml_slog2_ha_data_internal_avx512(%rip), %zmm6, %zmm10
        vpermt2ps 192+__svml_slog2_ha_data_internal_avx512(%rip), %zmm6, %zmm0

/* poly */
        vfmadd213ps {rn-sae}, %zmm14, %zmm13, %zmm3
        vaddps    {rn-sae}, %zmm10, %zmm9, %zmm15
        vfmadd213ps {rn-sae}, %zmm0, %zmm1, %zmm3
        vmovaps   %zmm15, %zmm4
        vfmadd231ps {rn-sae}, %zmm1, %zmm11, %zmm4

/* (R*c1)_h */
        vsubps    {rn-sae}, %zmm15, %zmm4, %zmm13
        vxorps    %zmm13, %zmm13, %zmm13{%k1}

/* (R*c1h)_l */
        vfmsub213ps {rn-sae}, %zmm13, %zmm11, %zmm1
        vaddps    {rn-sae}, %zmm1, %zmm3, %zmm0
        vaddps    {rn-sae}, %zmm0, %zmm4, %zmm0
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

        call      __svml_slog2_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log2f16_ha,@function
	.size	__svml_log2f16_ha,.-__svml_log2f16_ha
..LN__svml_log2f16_ha.0:

.L_2__routine_start___svml_slog2_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_slog2_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_13


        movss     (%rdi), %xmm2
        xorl      %ecx, %ecx
        pxor      %xmm1, %xmm1
        movss     %xmm2, -16(%rsp)
        ucomiss   %xmm1, %xmm2
        jp        .LBL_2_3
        je        .LBL_2_5

.LBL_2_3:

        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_5


        movss     .L_2il0floatpacket.81(%rip), %xmm0
        movl      $-27, %ecx
        mulss     %xmm0, %xmm2
        movss     %xmm2, -16(%rsp)

.LBL_2_5:

        comiss    %xmm1, %xmm2
        jbe       .LBL_2_9


        movaps    %xmm2, %xmm1
        subss     .L_2il0floatpacket.95(%rip), %xmm1
        movss     %xmm1, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.82(%rip), %xmm0
        jbe       .LBL_2_8


        movzwl    -14(%rsp), %edx
        pxor      %xmm8, %xmm8
        andl      $32640, %edx
        lea       __slog2_ha_CoutTab(%rip), %r10
        shrl      $7, %edx
        movss     %xmm2, -20(%rsp)
        movss     .L_2il0floatpacket.84(%rip), %xmm2
        movaps    %xmm2, %xmm1
        movss     .L_2il0floatpacket.93(%rip), %xmm6
        lea       -127(%rcx,%rdx), %r9d
        movzwl    -18(%rsp), %ecx
        andl      $-32641, %ecx
        addl      $16256, %ecx
        movw      %cx, -18(%rsp)
        movss     -20(%rsp), %xmm3
        movaps    %xmm3, %xmm0
        addss     %xmm3, %xmm1
        addss     .L_2il0floatpacket.83(%rip), %xmm0
        cvtsi2ss  %r9d, %xmm8
        movss     %xmm0, -24(%rsp)
        movl      -24(%rsp), %edi
        movss     %xmm1, -24(%rsp)
        andl      $127, %edi
        movss     -24(%rsp), %xmm7
        subss     %xmm2, %xmm7
        lea       (%rdi,%rdi,2), %r8d
        movss     (%r10,%r8,4), %xmm5
        subss     %xmm7, %xmm3
        addss     4(%r10,%r8,4), %xmm8
        mulss     %xmm5, %xmm7
        mulss     %xmm3, %xmm5
        subss     .L_2il0floatpacket.85(%rip), %xmm7
        movaps    %xmm7, %xmm4
        addss     %xmm7, %xmm8
        addss     %xmm5, %xmm4
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.92(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.91(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.90(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.89(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.88(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.87(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.86(%rip), %xmm6
        mulss     %xmm6, %xmm7
        mulss     %xmm5, %xmm6
        addss     8(%r10,%r8,4), %xmm6
        addss     %xmm5, %xmm6
        addss     %xmm6, %xmm7
        addss     %xmm7, %xmm8
        movss     %xmm8, (%rsi)
        ret

.LBL_2_8:

        movss     .L_2il0floatpacket.85(%rip), %xmm0
        mulss     %xmm0, %xmm1
        movss     .L_2il0floatpacket.93(%rip), %xmm2
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
        addss     .L_2il0floatpacket.87(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.86(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     %xmm1, %xmm2
        movss     %xmm2, (%rsi)
        ret

.LBL_2_9:

        ucomiss   %xmm1, %xmm2
        jp        .LBL_2_10
        je        .LBL_2_12

.LBL_2_10:

        divss     %xmm1, %xmm1
        movss     %xmm1, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_12:

        movss     .L_2il0floatpacket.94(%rip), %xmm0
        movl      $2, %eax
        divss     %xmm1, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_13:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_15

.LBL_2_14:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_15:

        testl     $8388607, (%rdi)
        jne       .LBL_2_14


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog2_ha_cout_rare_internal,@function
	.size	__svml_slog2_ha_cout_rare_internal,.-__svml_slog2_ha_cout_rare_internal
..LN__svml_slog2_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog2_ha_data_internal_avx512:
	.long	1065353216
	.long	1064608768
	.long	1063885824
	.long	1063184384
	.long	1062502400
	.long	1061838848
	.long	1061193728
	.long	1060564992
	.long	1059952640
	.long	1059354624
	.long	1058770944
	.long	1058201600
	.long	1057645568
	.long	1057100800
	.long	1056174080
	.long	1055133696
	.long	1054113792
	.long	1053116416
	.long	1052137472
	.long	1051179008
	.long	1050238976
	.long	1049317376
	.long	1048248320
	.long	1046470656
	.long	1044725760
	.long	1043013632
	.long	1041330176
	.long	1039163392
	.long	1035911168
	.long	1032708096
	.long	1027309568
	.long	1018822656
	.long	0
	.long	3082083684
	.long	890262383
	.long	3073448373
	.long	3058814943
	.long	933352011
	.long	3056977939
	.long	3052757441
	.long	3085997978
	.long	3070703220
	.long	931851714
	.long	916473404
	.long	3081224294
	.long	938815082
	.long	3055557319
	.long	3083106903
	.long	3050426335
	.long	3083856513
	.long	913737309
	.long	3045697063
	.long	3029223305
	.long	3078533744
	.long	3063765991
	.long	927944704
	.long	932711104
	.long	3062847489
	.long	3072621226
	.long	3076855565
	.long	3086857368
	.long	3075299840
	.long	937767542
	.long	930849160
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
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	3199776222
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	1056326045
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
	.long	848473495
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
	.type	__svml_slog2_ha_data_internal_avx512,@object
	.size	__svml_slog2_ha_data_internal_avx512,832
	.align 32
__slog2_ha_CoutTab:
	.long	1136175680
	.long	0
	.long	0
	.long	1135986583
	.long	1018822656
	.long	930849160
	.long	1135809305
	.long	1026916352
	.long	941737263
	.long	1135632026
	.long	1032306688
	.long	936581683
	.long	1135466566
	.long	1035100160
	.long	929197062
	.long	1135301106
	.long	1037934592
	.long	897678483
	.long	1135135647
	.long	1040498688
	.long	3059980496
	.long	1134982005
	.long	1041852416
	.long	908010313
	.long	1134828364
	.long	1043226624
	.long	3073739761
	.long	1134686541
	.long	1044510720
	.long	918631281
	.long	1134538809
	.long	1045868544
	.long	3062817788
	.long	1134402896
	.long	1047134208
	.long	3064656237
	.long	1134266982
	.long	1048416256
	.long	3029590737
	.long	1134131069
	.long	1049145856
	.long	903671587
	.long	1134001065
	.long	1049775616
	.long	911388989
	.long	1133876970
	.long	1050384896
	.long	3069885983
	.long	1133752875
	.long	1051001344
	.long	3037530952
	.long	1133634689
	.long	1051596288
	.long	3069922038
	.long	1133516503
	.long	1052198400
	.long	3070222063
	.long	1133404227
	.long	1052776960
	.long	919559368
	.long	1133291951
	.long	1053363200
	.long	840060372
	.long	1133185584
	.long	1053924864
	.long	915603033
	.long	1133079217
	.long	1054493184
	.long	921334924
	.long	1132978759
	.long	1055036416
	.long	896601826
	.long	1132872392
	.long	1055618048
	.long	908913293
	.long	1132777843
	.long	1056141312
	.long	3065728751
	.long	1132677386
	.long	1056702976
	.long	909020429
	.long	1132582837
	.long	1057101312
	.long	3048020321
	.long	1132494198
	.long	1057354752
	.long	3038815896
	.long	1132337219
	.long	1057628160
	.long	3068137421
	.long	1132159940
	.long	1057887232
	.long	3069993595
	.long	1131994480
	.long	1058131456
	.long	3054354312
	.long	1131817202
	.long	1058395904
	.long	910223436
	.long	1131651742
	.long	1058645504
	.long	3046952660
	.long	1131486282
	.long	1058897664
	.long	3057670844
	.long	1131332641
	.long	1059133952
	.long	924929721
	.long	1131178999
	.long	1059373056
	.long	3068093797
	.long	1131025358
	.long	1059614208
	.long	3058851683
	.long	1130871717
	.long	1059857920
	.long	3069897752
	.long	1130729894
	.long	1060084736
	.long	924446297
	.long	1130576253
	.long	1060333312
	.long	903058075
	.long	1130434430
	.long	1060564992
	.long	3052757441
	.long	1130304426
	.long	1060779264
	.long	3045479197
	.long	1130162603
	.long	1061015040
	.long	924699798
	.long	1130032599
	.long	1061233664
	.long	3070937808
	.long	1129890776
	.long	1061473792
	.long	925912756
	.long	1129772591
	.long	1061676032
	.long	923952205
	.long	1129642586
	.long	1061900544
	.long	906547304
	.long	1129512582
	.long	1062127104
	.long	3050351427
	.long	1129394397
	.long	1062334976
	.long	3070601694
	.long	1129276211
	.long	1062544384
	.long	900519722
	.long	1129158025
	.long	1062755840
	.long	3055774932
	.long	1129039840
	.long	1062969088
	.long	3053661845
	.long	1128921654
	.long	1063184384
	.long	3073448373
	.long	1128815287
	.long	1063379456
	.long	907090876
	.long	1128697101
	.long	1063598336
	.long	881051555
	.long	1128590734
	.long	1063796992
	.long	898320955
	.long	1128484367
	.long	1063997440
	.long	3068804107
	.long	1128378000
	.long	1064199168
	.long	923531617
	.long	1128283452
	.long	1064380416
	.long	3070994608
	.long	1128177085
	.long	1064585472
	.long	901920533
	.long	1128082536
	.long	1064769536
	.long	3071653428
	.long	1127976169
	.long	1064977920
	.long	903017594
	.long	1127881621
	.long	1065164800
	.long	911713416
	.long	1127787072
	.long	1065353216
	.long	0
	.long	1065353216
	.long	0
	.long	1207959616
	.long	1174405120
	.long	1002438656
	.long	1291845632
	.long	0
	.long	1065353216
	.long	1136175680
	.long	3212771328
	.long	3065082383
	.long	841219731
	.long	2913632803
	.long	691870088
	.long	2765780188
	.long	545377693
	.long	2619180638
	.type	__slog2_ha_CoutTab,@object
	.size	__slog2_ha_CoutTab,848
	.align 4
.L_2il0floatpacket.81:
	.long	0x4d000000
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,4
	.align 4
.L_2il0floatpacket.82:
	.long	0x3bc00000
	.type	.L_2il0floatpacket.82,@object
	.size	.L_2il0floatpacket.82,4
	.align 4
.L_2il0floatpacket.83:
	.long	0x48000040
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0x46000000
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0x43b8aa40
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0xbf7f0000
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0xb6b1720f
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0x3223fe93
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0xadaa8223
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x293d1988
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
	.align 4
.L_2il0floatpacket.91:
	.long	0xa4da74dc
	.type	.L_2il0floatpacket.91,@object
	.size	.L_2il0floatpacket.91,4
	.align 4
.L_2il0floatpacket.92:
	.long	0x2081cd9d
	.type	.L_2il0floatpacket.92,@object
	.size	.L_2il0floatpacket.92,4
	.align 4
.L_2il0floatpacket.93:
	.long	0x9c1d865e
	.type	.L_2il0floatpacket.93,@object
	.size	.L_2il0floatpacket.93,4
	.align 4
.L_2il0floatpacket.94:
	.long	0xbf800000
	.type	.L_2il0floatpacket.94,@object
	.size	.L_2il0floatpacket.94,4
	.align 4
.L_2il0floatpacket.95:
	.long	0x3f800000
	.type	.L_2il0floatpacket.95,@object
	.size	.L_2il0floatpacket.95,4
      	.section        .note.GNU-stack,"",@progbits
