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
 *  *  Compute 1+_VARG1 in high-low parts.  The low part will be
 *  *  incorporated in the reduced argument (with proper scaling).
 *  *  log(x) = VGETEXP(x)*log(2) + log(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log(mx) = -log(RCP(mx)) + log(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 5 fractional bits,
 *  *      and the table lookup for log(RCP(mx)) is based on a small permute instruction
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log1pf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_log1pf16_ha

__svml_log1pf16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm4
        vmovups   320+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm11
        vmovaps   %zmm0, %zmm6

/* compute 1+x as high, low parts */
        vmaxps    {sae}, %zmm6, %zmm4, %zmm7
        vminps    {sae}, %zmm6, %zmm4, %zmm9
        vaddps    {rn-sae}, %zmm6, %zmm4, %zmm8
        vandps    %zmm11, %zmm6, %zmm5

/* GetMant(x), normalized to [1,2) for x>=0, NaN for x<0 */
        vgetmantps $8, {sae}, %zmm8, %zmm3

/* GetExp(x) */
        vgetexpps {sae}, %zmm8, %zmm2
        vsubps    {rn-sae}, %zmm7, %zmm8, %zmm10

/* SglRcp ~ 1/Mantissa */
        vrcp14ps  %zmm3, %zmm12

/* Xl */
        vsubps    {rn-sae}, %zmm10, %zmm9, %zmm13

/* -expon */
        vxorps    %zmm11, %zmm2, %zmm14

/* round SglRcp to 5 fractional bits (RN mode, no Precision exception) */
        vrndscaleps $104, {sae}, %zmm12, %zmm7
        vmovups   128+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm10
        vmovups   640+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm9

/* Start polynomial evaluation */
        vmovups   448+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm12

/* Xl*2^(-Expon) */
        vscalefps {rn-sae}, %zmm14, %zmm13, %zmm15

/* Reduced argument: R = SglRcp*Mantissa - 1 */
        vfmsub213ps {rn-sae}, %zmm4, %zmm7, %zmm3
        vgetexpps {sae}, %zmm7, %zmm8
        vmovups   576+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm13

/* K*log(2)_low + Tl */
        vmovups   704+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm14
        vmulps    {rn-sae}, %zmm15, %zmm7, %zmm0

/* exponent correction */
        vsubps    {rn-sae}, %zmm4, %zmm2, %zmm4

/* Prepare table index */
        vpsrld    $18, %zmm7, %zmm2
        vaddps    {rn-sae}, %zmm0, %zmm3, %zmm1
        vpermt2ps 192+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm2, %zmm10
        vsubps    {rn-sae}, %zmm3, %zmm1, %zmm3
        vsubps    {rn-sae}, %zmm3, %zmm0, %zmm11

/* Table lookup */
        vmovups   __svml_slog1p_ha_data_internal_avx512(%rip), %zmm0
        vmovups   512+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm3

/* Rl+Tl */
        vaddps    {rn-sae}, %zmm11, %zmm10, %zmm15
        vpermt2ps 64+__svml_slog1p_ha_data_internal_avx512(%rip), %zmm2, %zmm0

/* K*log(2)_high+Th */
        vsubps    {rn-sae}, %zmm8, %zmm4, %zmm2
        vfmadd231ps {rn-sae}, %zmm1, %zmm12, %zmm3
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm12
        vfmadd231ps {rn-sae}, %zmm2, %zmm9, %zmm0
        vfmadd213ps {rn-sae}, %zmm13, %zmm1, %zmm3
        vfmadd213ps {rn-sae}, %zmm15, %zmm14, %zmm2

/* K*log(2)_high+Th+Rh */
        vaddps    {rn-sae}, %zmm1, %zmm0, %zmm4
        vfmadd213ps {rn-sae}, %zmm2, %zmm12, %zmm3

/* Rh */
        vsubps    {rn-sae}, %zmm0, %zmm4, %zmm0

/* Rl */
        vsubps    {rn-sae}, %zmm0, %zmm1, %zmm1
        vcmpps    $4, {sae}, %zmm1, %zmm1, %k0
        vaddps    {rn-sae}, %zmm1, %zmm3, %zmm0
        kmovw     %k0, %edx
        vaddps    {rn-sae}, %zmm0, %zmm4, %zmm1
        vorps     %zmm5, %zmm1, %zmm0
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

        vmovups   %zmm6, 64(%rsp)
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

        call      __svml_slog1p_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log1pf16_ha,@function
	.size	__svml_log1pf16_ha,.-__svml_log1pf16_ha
..LN__svml_log1pf16_ha.0:

.L_2__routine_start___svml_slog1p_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_slog1p_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movss     .L_2il0floatpacket.93(%rip), %xmm1
        xorb      %r8b, %r8b
        movss     (%rdi), %xmm5
        addss     %xmm1, %xmm5
        movss     %xmm5, -20(%rsp)
        movzwl    -18(%rsp), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_15


        movss     %xmm5, -16(%rsp)
        xorl      %ecx, %ecx
        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_4


        mulss     .L_2il0floatpacket.78(%rip), %xmm5
        movb      $1, %r8b
        movss     %xmm5, -16(%rsp)
        movl      $-40, %ecx

.LBL_2_4:

        pxor      %xmm3, %xmm3
        comiss    %xmm3, %xmm5
        jbe       .LBL_2_10


        movaps    %xmm5, %xmm2
        subss     %xmm1, %xmm2
        movss     %xmm2, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.79(%rip), %xmm0
        jbe       .LBL_2_9


        movzwl    -14(%rsp), %edx
        pxor      %xmm6, %xmm6
        andl      $32640, %edx
        shrl      $7, %edx
        lea       -127(%rcx,%rdx), %ecx
        cvtsi2ss  %ecx, %xmm6
        cmpb      $1, %r8b
        je        .LBL_2_13


        movss     .L_2il0floatpacket.89(%rip), %xmm4
        movss     .L_2il0floatpacket.90(%rip), %xmm0
        mulss     %xmm6, %xmm4
        mulss     %xmm0, %xmm6

.LBL_2_8:

        movss     %xmm5, -20(%rsp)
        movaps    %xmm4, %xmm9
        movzwl    -18(%rsp), %edx
        lea       __slog1p_ha_CoutTab(%rip), %r8
        andl      $-32641, %edx
        addl      $16256, %edx
        movw      %dx, -18(%rsp)
        movss     -20(%rsp), %xmm8
        movaps    %xmm8, %xmm2
        movss     .L_2il0floatpacket.92(%rip), %xmm7
        addss     .L_2il0floatpacket.91(%rip), %xmm2
        movss     %xmm2, -24(%rsp)
        movl      -24(%rsp), %ecx
        andl      $127, %ecx
        lea       (%rcx,%rcx,2), %edi
        movss     4(%r8,%rdi,4), %xmm5
        movss     (%r8,%rdi,4), %xmm0
        addss     %xmm5, %xmm9
        addss     8(%r8,%rdi,4), %xmm6
        movaps    %xmm9, %xmm3
        subss     %xmm4, %xmm3
        movss     %xmm3, -24(%rsp)
        movss     -24(%rsp), %xmm4
        subss     %xmm4, %xmm5
        movss     %xmm5, -24(%rsp)
        movss     -24(%rsp), %xmm10
        addss     %xmm6, %xmm10
        movaps    %xmm7, %xmm6
        addss     %xmm8, %xmm6
        movss     %xmm6, -24(%rsp)
        movss     -24(%rsp), %xmm12
        subss     %xmm7, %xmm12
        subss     %xmm12, %xmm8
        mulss     %xmm0, %xmm12
        subss     %xmm1, %xmm12
        mulss     %xmm8, %xmm0
        movaps    %xmm0, %xmm15
        movaps    %xmm12, %xmm2
        addss     %xmm10, %xmm15
        addss     %xmm9, %xmm12
        addss     %xmm0, %xmm2
        movaps    %xmm15, %xmm1
        movaps    %xmm12, %xmm13
        subss     %xmm10, %xmm1
        addss     %xmm15, %xmm13
        movss     %xmm1, -24(%rsp)
        movss     -24(%rsp), %xmm11
        subss     %xmm11, %xmm0
        movss     %xmm0, -24(%rsp)
        movss     -24(%rsp), %xmm0
        movss     %xmm13, (%rsi)
        subss     %xmm12, %xmm13
        movss     .L_2il0floatpacket.86(%rip), %xmm12
        mulss     %xmm2, %xmm12
        movss     %xmm13, -24(%rsp)
        movss     -24(%rsp), %xmm14
        addss     .L_2il0floatpacket.85(%rip), %xmm12
        subss     %xmm14, %xmm15
        mulss     %xmm2, %xmm12
        movss     %xmm15, -24(%rsp)
        movss     -24(%rsp), %xmm1
        addss     .L_2il0floatpacket.84(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.83(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.82(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.81(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.80(%rip), %xmm12
        mulss     %xmm2, %xmm12
        mulss     %xmm2, %xmm12
        addss     %xmm12, %xmm0
        addss     %xmm0, %xmm1
        movss     %xmm1, -24(%rsp)
        movss     -24(%rsp), %xmm3
        addss     (%rsi), %xmm3
        movss     %xmm3, (%rsi)
        ret

.LBL_2_9:

        movss     .L_2il0floatpacket.86(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.85(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.84(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.83(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.82(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.81(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.80(%rip), %xmm0
        mulss     %xmm2, %xmm0
        mulss     %xmm2, %xmm0
        addss     %xmm2, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_10:

        ucomiss   %xmm3, %xmm5
        jp        .LBL_2_11
        je        .LBL_2_14

.LBL_2_11:

        divss     %xmm3, %xmm3
        movss     %xmm3, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_13:

        movss     .L_2il0floatpacket.88(%rip), %xmm0
        mulss     %xmm0, %xmm6
        movaps    %xmm6, %xmm4
        movaps    %xmm3, %xmm6
        jmp       .LBL_2_8

.LBL_2_14:

        movss     .L_2il0floatpacket.87(%rip), %xmm0
        movl      $2, %eax
        divss     %xmm3, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_15:

        movb      -17(%rsp), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_17

.LBL_2_16:

        mulss     %xmm5, %xmm5
        movss     %xmm5, (%rsi)
        ret

.LBL_2_17:

        testl     $8388607, -20(%rsp)
        jne       .LBL_2_16


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog1p_ha_cout_rare_internal,@function
	.size	__svml_slog1p_ha_cout_rare_internal,.-__svml_slog1p_ha_cout_rare_internal
..LN__svml_slog1p_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog1p_ha_data_internal_avx512:
	.long	1060205056
	.long	1059688960
	.long	1059187712
	.long	1058701824
	.long	1058229248
	.long	1057769472
	.long	1057321984
	.long	1056807936
	.long	1055958016
	.long	1055129600
	.long	1054320640
	.long	1053531136
	.long	1052760064
	.long	1052006400
	.long	1051268096
	.long	1050547200
	.long	1049840640
	.long	1049148416
	.long	1048365056
	.long	1047035904
	.long	1045733376
	.long	1044455424
	.long	1043200000
	.long	1041969152
	.long	1040760832
	.long	1038958592
	.long	1036623872
	.long	1034330112
	.long	1032073216
	.long	1027907584
	.long	1023541248
	.long	1015087104
	.long	901758606
	.long	3071200204
	.long	931108809
	.long	3074069268
	.long	3077535321
	.long	3071146094
	.long	3063010043
	.long	3072147991
	.long	908173938
	.long	3049723733
	.long	925190435
	.long	923601997
	.long	3048768765
	.long	3076457870
	.long	926424291
	.long	3073778483
	.long	3069146713
	.long	912794238
	.long	912483742
	.long	920635797
	.long	3054902185
	.long	3069864633
	.long	922801832
	.long	3033791132
	.long	3076717488
	.long	3076037756
	.long	3072434855
	.long	3077481184
	.long	3066991812
	.long	917116064
	.long	925811956
	.long	900509991
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
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.type	__svml_slog1p_ha_data_internal_avx512,@object
	.size	__svml_slog1p_ha_data_internal_avx512,768
	.align 32
__slog1p_ha_CoutTab:
	.long	1065353216
	.long	0
	.long	0
	.long	1065091072
	.long	1015087104
	.long	900509991
	.long	1064828928
	.long	1023541248
	.long	925811956
	.long	1064566784
	.long	1027915776
	.long	3084221144
	.long	1064304640
	.long	1032073216
	.long	3066991812
	.long	1064173568
	.long	1033195520
	.long	882149603
	.long	1063911424
	.long	1035468800
	.long	928189163
	.long	1063649280
	.long	1037783040
	.long	927501741
	.long	1063518208
	.long	1038958592
	.long	3076037756
	.long	1063256064
	.long	1040759808
	.long	904405630
	.long	1063124992
	.long	1041361920
	.long	3052231524
	.long	1062862848
	.long	1042581504
	.long	922094799
	.long	1062731776
	.long	1043201024
	.long	3070120623
	.long	1062469632
	.long	1044455424
	.long	3069864633
	.long	1062338560
	.long	1045091328
	.long	3063188516
	.long	1062207488
	.long	1045733376
	.long	3054902185
	.long	1061945344
	.long	1047035904
	.long	920635797
	.long	1061814272
	.long	1047697408
	.long	904920689
	.long	1061683200
	.long	1048365056
	.long	912483742
	.long	1061552128
	.long	1048807936
	.long	3052664405
	.long	1061421056
	.long	1049148416
	.long	912794238
	.long	1061158912
	.long	1049840384
	.long	889474359
	.long	1061027840
	.long	1050191872
	.long	3059868362
	.long	1060896768
	.long	1050546944
	.long	3059256525
	.long	1060765696
	.long	1050905600
	.long	912008988
	.long	1060634624
	.long	1051268352
	.long	912290698
	.long	1060503552
	.long	1051635200
	.long	3037211048
	.long	1060372480
	.long	1052005888
	.long	906226119
	.long	1060241408
	.long	1052380928
	.long	3052480305
	.long	1060110336
	.long	1052760064
	.long	3048768765
	.long	1059979264
	.long	1053143552
	.long	3049975450
	.long	1059848192
	.long	1053531392
	.long	894485718
	.long	1059717120
	.long	1053923840
	.long	897598623
	.long	1059586048
	.long	1054320896
	.long	907355277
	.long	1059586048
	.long	1054320896
	.long	907355277
	.long	1059454976
	.long	1054722816
	.long	881705073
	.long	1059323904
	.long	1055129600
	.long	3049723733
	.long	1059192832
	.long	1055541248
	.long	890353599
	.long	1059061760
	.long	1055958016
	.long	908173938
	.long	1059061760
	.long	1055958016
	.long	908173938
	.long	1058930688
	.long	1056380160
	.long	883644938
	.long	1058799616
	.long	1056807680
	.long	3052015799
	.long	1058668544
	.long	1057102592
	.long	884897284
	.long	1058668544
	.long	1057102592
	.long	884897284
	.long	1058537472
	.long	1057321920
	.long	3037632470
	.long	1058406400
	.long	1057544128
	.long	865017195
	.long	1058275328
	.long	1057769344
	.long	3042936546
	.long	1058275328
	.long	1057769344
	.long	3042936546
	.long	1058144256
	.long	1057997568
	.long	903344518
	.long	1058013184
	.long	1058228992
	.long	897862967
	.long	1058013184
	.long	1058228992
	.long	897862967
	.long	1057882112
	.long	1058463680
	.long	3047822280
	.long	1057882112
	.long	1058463680
	.long	3047822280
	.long	1057751040
	.long	1058701632
	.long	883793293
	.long	1057619968
	.long	1058943040
	.long	851667963
	.long	1057619968
	.long	1058943040
	.long	851667963
	.long	1057488896
	.long	1059187968
	.long	3000004036
	.long	1057488896
	.long	1059187968
	.long	3000004036
	.long	1057357824
	.long	1059436544
	.long	3047430717
	.long	1057357824
	.long	1059436544
	.long	3047430717
	.long	1057226752
	.long	1059688832
	.long	3043802308
	.long	1057226752
	.long	1059688832
	.long	3043802308
	.long	1057095680
	.long	1059944960
	.long	876113044
	.long	1057095680
	.long	1059944960
	.long	876113044
	.long	1056964608
	.long	1060205056
	.long	901758606
	.long	1060205056
	.long	901758606
	.long	1207959616
	.long	1174405120
	.long	1008730112
	.long	1400897536
	.long	0
	.long	1065353216
	.long	3204448256
	.long	1051372203
	.long	3196059648
	.long	1045220557
	.long	3190467243
	.long	1041387009
	.long	3187672480
	.type	__slog1p_ha_CoutTab,@object
	.size	__slog1p_ha_CoutTab,840
	.align 4
.L_2il0floatpacket.78:
	.long	0x53800000
	.type	.L_2il0floatpacket.78,@object
	.size	.L_2il0floatpacket.78,4
	.align 4
.L_2il0floatpacket.79:
	.long	0x3c200000
	.type	.L_2il0floatpacket.79,@object
	.size	.L_2il0floatpacket.79,4
	.align 4
.L_2il0floatpacket.80:
	.long	0xbf000000
	.type	.L_2il0floatpacket.80,@object
	.size	.L_2il0floatpacket.80,4
	.align 4
.L_2il0floatpacket.81:
	.long	0x3eaaaaab
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,4
	.align 4
.L_2il0floatpacket.82:
	.long	0xbe800000
	.type	.L_2il0floatpacket.82,@object
	.size	.L_2il0floatpacket.82,4
	.align 4
.L_2il0floatpacket.83:
	.long	0x3e4ccccd
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0xbe2aaaab
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0x3e124e01
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0xbe0005a0
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0xbf800000
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0x3f317218
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0x3f317200
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x35bfbe8e
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
	.align 4
.L_2il0floatpacket.91:
	.long	0x48000040
	.type	.L_2il0floatpacket.91,@object
	.size	.L_2il0floatpacket.91,4
	.align 4
.L_2il0floatpacket.92:
	.long	0x46000000
	.type	.L_2il0floatpacket.92,@object
	.size	.L_2il0floatpacket.92,4
	.align 4
.L_2il0floatpacket.93:
	.long	0x3f800000
	.type	.L_2il0floatpacket.93,@object
	.size	.L_2il0floatpacket.93,4
      	.section        .note.GNU-stack,"",@progbits
