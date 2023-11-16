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
.L_2__routine_start___svml_exp2f16_ha_z0_0:

	.align    16,0x90
	.globl __svml_exp2f16_ha

__svml_exp2f16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   384+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm3
        vmovups   320+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm5
        vmovups   256+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm11

/*
 * Reduced argument
 * where VREDUCE is available
 */
        vreduceps $65, {sae}, %zmm0, %zmm7
        vmovups   192+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm6
        vmovups   128+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm8
        vmovups   512+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm2

/* c4*r + c3 */
        vfmadd231ps {rn-sae}, %zmm7, %zmm5, %zmm11

/*
 * Integer form of K+0.b1b2b3b4 in lower bits - call K_plus_f0
 * Mantisssa of normalized single precision FP: 1.b1b2...b23
 */
        vaddps    {rd-sae}, %zmm3, %zmm0, %zmm4
        vandps    448+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm0, %zmm1

/* c4*r^2+c3*r+c2 */
        vfmadd213ps {rn-sae}, %zmm6, %zmm7, %zmm11
        vcmpps    $30, {sae}, %zmm2, %zmm1, %k0

/* c4*r^3+c3*r^2+c2*r+c1 */
        vfmadd213ps {rn-sae}, %zmm8, %zmm7, %zmm11

/* Table value: 2^(0.b1b2b3b4) */
        vpermps   __svml_sexp2_ha_data_internal_avx512(%rip), %zmm4, %zmm12
        vpermps   64+__svml_sexp2_ha_data_internal_avx512(%rip), %zmm4, %zmm10
        kmovw     %k0, %edx

/* Th*r */
        vmulps    {rn-sae}, %zmm7, %zmm12, %zmm9

/* Tl + (Th*r*(c4*r^3+c3*r^2+c2*r+c1)) */
        vfmadd213ps {rn-sae}, %zmm10, %zmm9, %zmm11

/* Th + (Tl + (Th*r*(c4*r^3+c3*r^2+c2*r+c1))) */
        vaddps    {rn-sae}, %zmm12, %zmm11, %zmm13

/* Scaling placed at the end to avoid accuracy loss when T*r*scale underflows */
        vscalefps {rn-sae}, %zmm0, %zmm13, %zmm1

/*
 * ===================================================================
 * =============================== LA / EP ===========================
 * ===================================================================
 */
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm1, 128(%rsp)
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
        vmovups   128(%rsp), %zmm1
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

        call      __svml_sexp2_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_exp2f16_ha,@function
	.size	__svml_exp2f16_ha,.-__svml_exp2f16_ha
..LN__svml_exp2f16_ha.0:

.L_2__routine_start___svml_sexp2_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_sexp2_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        pxor      %xmm7, %xmm7
        movzwl    2(%rdi), %edx
        xorl      %eax, %eax
        andl      $32640, %edx
        movss     (%rdi), %xmm0
        cvtss2sd  %xmm0, %xmm7
        cmpl      $32640, %edx
        je        .LBL_2_8


        movsd     1072+__sexp2_ha__imldExp2HATab(%rip), %xmm0
        comisd    %xmm7, %xmm0
        jbe       .LBL_2_6


        comisd    1088+__sexp2_ha__imldExp2HATab(%rip), %xmm7
        jbe       .LBL_2_5


        movsd     1024+__sexp2_ha__imldExp2HATab(%rip), %xmm1
        movaps    %xmm7, %xmm4
        movsd     1064+__sexp2_ha__imldExp2HATab(%rip), %xmm6
        addsd     %xmm7, %xmm1
        cmpltsd   1080+__sexp2_ha__imldExp2HATab(%rip), %xmm7
        movsd     %xmm1, -16(%rsp)
        lea       __sexp2_ha__imldExp2HATab(%rip), %rdi
        movsd     -16(%rsp), %xmm2
        movl      -16(%rsp), %r8d
        movl      %r8d, %edx
        shrl      $6, %r8d
        subsd     1024+__sexp2_ha__imldExp2HATab(%rip), %xmm2
        movsd     %xmm2, -8(%rsp)
        addl      $1023, %r8d
        movsd     -8(%rsp), %xmm3
        andl      $2047, %r8d
        andl      $63, %edx
        subsd     %xmm3, %xmm4
        mulsd     %xmm4, %xmm6
        lea       1(%rdx,%rdx), %ecx
        movzwl    1118+__sexp2_ha__imldExp2HATab(%rip), %r9d
        lea       (%rdx,%rdx), %eax
        movsd     (%rdi,%rax,8), %xmm5
        andl      $-32753, %r9d
        movsd     1112+__sexp2_ha__imldExp2HATab(%rip), %xmm0
        shll      $4, %r8d
        addsd     1056+__sexp2_ha__imldExp2HATab(%rip), %xmm6
        movd      %xmm7, %eax
        movsd     %xmm0, -24(%rsp)
        mulsd     %xmm4, %xmm6
        orl       %r8d, %r9d
        movw      %r9w, -18(%rsp)
        andl      $4, %eax
        addsd     1048+__sexp2_ha__imldExp2HATab(%rip), %xmm6
        mulsd     %xmm4, %xmm6
        addsd     1040+__sexp2_ha__imldExp2HATab(%rip), %xmm6
        mulsd     %xmm4, %xmm6
        addsd     1032+__sexp2_ha__imldExp2HATab(%rip), %xmm6
        mulsd     %xmm4, %xmm6
        addsd     (%rdi,%rcx,8), %xmm6
        mulsd     %xmm5, %xmm6
        addsd     %xmm5, %xmm6
        mulsd     -24(%rsp), %xmm6
        cvtsd2ss  %xmm6, %xmm6
        movss     %xmm6, (%rsi)
        ret

.LBL_2_5:

        movsd     1096+__sexp2_ha__imldExp2HATab(%rip), %xmm0
        movl      $4, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_6:

        movsd     1104+__sexp2_ha__imldExp2HATab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)


        ret

.LBL_2_8:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_10

.LBL_2_9:

        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_10:

        testl     $8388607, (%rdi)
        jne       .LBL_2_9


        movsd     1112+__sexp2_ha__imldExp2HATab(%rip), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sexp2_ha_cout_rare_internal,@function
	.size	__svml_sexp2_ha_cout_rare_internal,.-__svml_sexp2_ha_cout_rare_internal
..LN__svml_sexp2_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sexp2_ha_data_internal_avx512:
	.long	1065353216
	.long	1065724611
	.long	1066112450
	.long	1066517459
	.long	1066940400
	.long	1067382066
	.long	1067843287
	.long	1068324927
	.long	1068827891
	.long	1069353124
	.long	1069901610
	.long	1070474380
	.long	1071072509
	.long	1071697119
	.long	1072349383
	.long	1073030525
	.long	0
	.long	860854417
	.long	2992679841
	.long	862410276
	.long	857938801
	.long	859906882
	.long	3006045534
	.long	2989353718
	.long	852486010
	.long	3007401960
	.long	839559223
	.long	841546788
	.long	3000280458
	.long	2988354393
	.long	2990610142
	.long	841557046
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1047920129
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1029917730
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
	.long	1008798169
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
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.type	__svml_sexp2_ha_data_internal_avx512,@object
	.size	__svml_sexp2_ha_data_internal_avx512,576
	.align 32
__sexp2_ha__imldExp2HATab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	2631457885
	.long	3161546771
	.long	3541402996
	.long	1072716208
	.long	896005651
	.long	1015861842
	.long	410360776
	.long	1072727877
	.long	1642514529
	.long	1012987726
	.long	1828292879
	.long	1072739672
	.long	1568897901
	.long	1016568486
	.long	852742562
	.long	1072751596
	.long	1882168529
	.long	1010744893
	.long	3490863953
	.long	1072763649
	.long	707771662
	.long	3163903570
	.long	2930322912
	.long	1072775834
	.long	3117806614
	.long	3163670819
	.long	1014845819
	.long	1072788152
	.long	3936719688
	.long	3162512149
	.long	3949972341
	.long	1072800603
	.long	1058231231
	.long	1015777676
	.long	828946858
	.long	1072813191
	.long	1044000608
	.long	1016786167
	.long	2288159958
	.long	1072825915
	.long	1151779725
	.long	1015705409
	.long	1853186616
	.long	1072838778
	.long	3819481236
	.long	1016499965
	.long	1709341917
	.long	1072851781
	.long	2552227826
	.long	1015039787
	.long	4112506593
	.long	1072864925
	.long	1829350193
	.long	1015216097
	.long	2799960843
	.long	1072878213
	.long	1913391796
	.long	1015756674
	.long	171030293
	.long	1072891646
	.long	1303423926
	.long	1015238005
	.long	2992903935
	.long	1072905224
	.long	1574172746
	.long	1016061241
	.long	926591435
	.long	1072918951
	.long	3427487848
	.long	3163704045
	.long	887463927
	.long	1072932827
	.long	1049900754
	.long	3161575912
	.long	1276261410
	.long	1072946854
	.long	2804567149
	.long	1015390024
	.long	569847338
	.long	1072961034
	.long	1209502043
	.long	3159926671
	.long	1617004845
	.long	1072975368
	.long	1623370769
	.long	1011049453
	.long	3049340112
	.long	1072989858
	.long	3667985273
	.long	1013894369
	.long	3577096743
	.long	1073004506
	.long	3145379760
	.long	1014403278
	.long	1990012071
	.long	1073019314
	.long	7447438
	.long	3163526196
	.long	1453150082
	.long	1073034283
	.long	3171891295
	.long	3162037958
	.long	917841882
	.long	1073049415
	.long	419288974
	.long	1016280325
	.long	3712504873
	.long	1073064711
	.long	3793507337
	.long	1016095713
	.long	363667784
	.long	1073080175
	.long	728023093
	.long	1016345318
	.long	2956612997
	.long	1073095806
	.long	1005538728
	.long	3163304901
	.long	2186617381
	.long	1073111608
	.long	2018924632
	.long	3163803357
	.long	1719614413
	.long	1073127582
	.long	3210617384
	.long	3163796463
	.long	1013258799
	.long	1073143730
	.long	3094194670
	.long	3160631279
	.long	3907805044
	.long	1073160053
	.long	2119843535
	.long	3161988964
	.long	1447192521
	.long	1073176555
	.long	508946058
	.long	3162904882
	.long	1944781191
	.long	1073193236
	.long	3108873501
	.long	3162190556
	.long	919555682
	.long	1073210099
	.long	2882956373
	.long	1013312481
	.long	2571947539
	.long	1073227145
	.long	4047189812
	.long	3163777462
	.long	2604962541
	.long	1073244377
	.long	3631372142
	.long	3163870288
	.long	1110089947
	.long	1073261797
	.long	3253791412
	.long	1015920431
	.long	2568320822
	.long	1073279406
	.long	1509121860
	.long	1014756995
	.long	2966275557
	.long	1073297207
	.long	2339118633
	.long	3160254904
	.long	2682146384
	.long	1073315202
	.long	586480042
	.long	3163702083
	.long	2191782032
	.long	1073333393
	.long	730975783
	.long	1014083580
	.long	2069751141
	.long	1073351782
	.long	576856675
	.long	3163014404
	.long	2990417245
	.long	1073370371
	.long	3552361237
	.long	3163667409
	.long	1434058175
	.long	1073389163
	.long	1853053619
	.long	1015310724
	.long	2572866477
	.long	1073408159
	.long	2462790535
	.long	1015814775
	.long	3092190715
	.long	1073427362
	.long	1457303226
	.long	3159737305
	.long	4076559943
	.long	1073446774
	.long	950899508
	.long	3160987380
	.long	2420883922
	.long	1073466398
	.long	174054861
	.long	1014300631
	.long	3716502172
	.long	1073486235
	.long	816778419
	.long	1014197934
	.long	777507147
	.long	1073506289
	.long	3507050924
	.long	1015341199
	.long	3706687593
	.long	1073526560
	.long	1821514088
	.long	1013410604
	.long	1242007932
	.long	1073547053
	.long	1073740399
	.long	3163532637
	.long	3707479175
	.long	1073567768
	.long	2789017511
	.long	1014276997
	.long	64696965
	.long	1073588710
	.long	3586233004
	.long	1015962192
	.long	863738719
	.long	1073609879
	.long	129252895
	.long	3162690849
	.long	3884662774
	.long	1073631278
	.long	1614448851
	.long	1014281732
	.long	2728693978
	.long	1073652911
	.long	2413007344
	.long	3163551506
	.long	3999357479
	.long	1073674779
	.long	1101668360
	.long	1015989180
	.long	1533953344
	.long	1073696886
	.long	835814894
	.long	1015702697
	.long	2174652632
	.long	1073719233
	.long	1301400989
	.long	1014466875
	.long	0
	.long	1121452032
	.long	4277811695
	.long	1072049730
	.long	4286751290
	.long	1070514109
	.long	3607585384
	.long	1068264200
	.long	871937163
	.long	1065595565
	.long	3302507530
	.long	1062590576
	.long	0
	.long	1080033280
	.long	0
	.long	3227484160
	.long	0
	.long	3227697152
	.long	1
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	1073741824
	.long	0
	.long	967835648
	.long	0
	.long	0
	.type	__sexp2_ha__imldExp2HATab,@object
	.size	__sexp2_ha__imldExp2HATab,1152
      	.section        .note.GNU-stack,"",@progbits
