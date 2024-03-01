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
 * 
 *   NOTE: Since the hyperbolic tangent function is odd
 *         (tanh(x) = -tanh(-x)), below algorithm deals with the absolute
 *         value of the argument |x|: tanh(x) = sign(x) * tanh(|x|)
 * 
 *   We use a table lookup method to compute tanh(|x|).
 *   The basic idea is to split the input range into a number of subintervals
 *   and to approximate tanh(.) with a polynomial on each of them.
 * 
 *   IEEE SPECIAL CONDITIONS:
 *   x = [+,-]0, r = [+,-]0
 *   x = +Inf,   r = +1
 *   x = -Inf,   r = -1
 *   x = QNaN,   r = QNaN
 *   x = SNaN,   r = QNaN
 * 
 * 
 *   ALGORITHM DETAILS
 *   We handle special values in a callout function, aside from main path
 *   computations. "Special" for this algorithm are:
 *   INF, NAN, |x| > HUGE_THRESHOLD
 * 
 * 
 *   Main path computations are organized as follows:
 *   Actually we split the interval [0, SATURATION_THRESHOLD)
 *   into a number of subintervals.  On each subinterval we approximate tanh(.)
 *   with a minimax polynomial of pre-defined degree. Polynomial coefficients
 *   are computed beforehand and stored in table. We also use
 * 
 *       y := |x| + B,
 * 
 *   here B depends on subinterval and is used to make argument
 *   closer to zero.
 *   We also add large fake interval [SATURATION_THRESHOLD, HUGE_THRESHOLD],
 *   where 1.0 + 0.0*y + 0.0*y^2 ... coefficients are stored - just to
 *   preserve main path computation logic but return 1.0 for all arguments.
 * 
 *   Hence reconstruction looks as follows:
 *   we extract proper polynomial and range reduction coefficients
 *        (Pj and B), corresponding to subinterval, to which |x| belongs,
 *        and return
 * 
 *       r := sign(x) * (P0 + P1 * y + ... + Pn * y^n)
 * 
 *   NOTE: we use multiprecision technique to multiply and sum the first
 *         K terms of the polynomial. So Pj, j = 0..K are stored in
 *         table each as a pair of target precision numbers (Pj and PLj) to
 *         achieve wider than target precision.
 * 
 * --
 * 
 */


	.text
.L_2__routine_start___svml_tanhf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_tanhf16_ha

__svml_tanhf16_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovaps   %zmm0, %zmm1
        vmovups   __svml_stanh_ha_data_internal(%rip), %zmm9
        vmovups   896+__svml_stanh_ha_data_internal(%rip), %zmm10
        vmovups   1024+__svml_stanh_ha_data_internal(%rip), %zmm15
        vmovups   768+__svml_stanh_ha_data_internal(%rip), %zmm11
        vmovups   640+__svml_stanh_ha_data_internal(%rip), %zmm12
        vmovups   512+__svml_stanh_ha_data_internal(%rip), %zmm13
        vandps    3136+__svml_stanh_ha_data_internal(%rip), %zmm1, %zmm8
        vpternlogd $255, %zmm2, %zmm2, %zmm2
        vandps    3072+__svml_stanh_ha_data_internal(%rip), %zmm1, %zmm0

/* Here huge arguments, INF and NaNs are filtered out to callout. */
        vpandd    1152+__svml_stanh_ha_data_internal(%rip), %zmm1, %zmm3
        vpsubd    1216+__svml_stanh_ha_data_internal(%rip), %zmm3, %zmm4
        vpcmpd    $2, 3264+__svml_stanh_ha_data_internal(%rip), %zmm3, %k1

/*
 * * small table specific variables *
 * **********************************
 * -------------------- Constant loading -------------------
 */
        vpxord    %zmm5, %zmm5, %zmm5

/* if VMIN, VMAX is defined for I type */
        vpmaxsd   %zmm5, %zmm4, %zmm6
        vpminsd   1280+__svml_stanh_ha_data_internal(%rip), %zmm6, %zmm7
        vpsrld    $21, %zmm7, %zmm14
        vmovups   128+__svml_stanh_ha_data_internal(%rip), %zmm4
        vpermt2ps 64+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm9
        vpermt2ps 960+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm10
        vpermt2ps 1088+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm15
        vpermt2ps 832+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm11
        vpermt2ps 704+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm12
        vpermt2ps 576+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm13
        vpermt2ps 192+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm4
        vpandnd   %zmm3, %zmm3, %zmm2{%k1}
        vsubps    {rn-sae}, %zmm9, %zmm8, %zmm3
        vptestmd  %zmm2, %zmm2, %k0
        vmovups   384+__svml_stanh_ha_data_internal(%rip), %zmm2
        vfmadd213ps {rn-sae}, %zmm10, %zmm3, %zmm15

/* sP[1] is the lower part of constant term sP[0] */
        vmovups   256+__svml_stanh_ha_data_internal(%rip), %zmm10
        vpermt2ps 448+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm2
        vfmadd213ps {rn-sae}, %zmm11, %zmm3, %zmm15
        vpermt2ps 320+__svml_stanh_ha_data_internal(%rip), %zmm14, %zmm10
        kmovw     %k0, %edx
        vfmadd213ps {rn-sae}, %zmm12, %zmm3, %zmm15
        vfmadd213ps {rn-sae}, %zmm13, %zmm3, %zmm15
        vmulps    {rn-sae}, %zmm3, %zmm15, %zmm11
        vfmadd213ps {rn-sae}, %zmm10, %zmm3, %zmm11
        vfmadd213ps {rn-sae}, %zmm11, %zmm2, %zmm3
        vaddps    {rn-sae}, %zmm4, %zmm3, %zmm5
        vorps     %zmm0, %zmm5, %zmm0
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

        vmovups   %zmm1, 64(%rsp)
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

        call      __svml_stanh_ha_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_tanhf16_ha,@function
	.size	__svml_tanhf16_ha,.-__svml_tanhf16_ha
..LN__svml_tanhf16_ha.0:

.L_2__routine_start___svml_stanh_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_stanh_ha_cout_rare_internal:


	.cfi_startproc
..L53:

        lea       __stanh_ha__imlsTanhTab(%rip), %rdx
        movb      3(%rdi), %al
        andb      $-128, %al
        shrb      $7, %al
        movzbl    %al, %ecx
        movzwl    2(%rdi), %r8d
        andl      $32640, %r8d
        movl      (%rdx,%rcx,4), %eax
        cmpl      $32640, %r8d
        je        .LBL_2_4

.LBL_2_2:

        movl      %eax, (%rsi)

.LBL_2_3:

        xorl      %eax, %eax
        ret

.LBL_2_4:

        testl     $8388607, (%rdi)
        je        .LBL_2_2


        movss     (%rdi), %xmm0
        addss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        jmp       .LBL_2_3
	.align    16,0x90

	.cfi_endproc

	.type	__svml_stanh_ha_cout_rare_internal,@function
	.size	__svml_stanh_ha_cout_rare_internal,.-__svml_stanh_ha_cout_rare_internal
..LN__svml_stanh_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_stanh_ha_data_internal:
	.long	0
	.long	1030750208
	.long	1032847360
	.long	1034944512
	.long	1037041664
	.long	1039138816
	.long	1041235968
	.long	1043333120
	.long	1045430272
	.long	1047527424
	.long	1049624576
	.long	1051721728
	.long	1053818880
	.long	1055916032
	.long	1058013184
	.long	1060110336
	.long	1062207488
	.long	1064304640
	.long	1066401792
	.long	1068498944
	.long	1070596096
	.long	1072693248
	.long	1074790400
	.long	1076887552
	.long	1078984704
	.long	1081081856
	.long	1083179008
	.long	1085276160
	.long	1087373312
	.long	1089470464
	.long	1091567616
	.long	0
	.long	0
	.long	1030732233
	.long	1032831839
	.long	1034916201
	.long	1036994987
	.long	1039067209
	.long	1041174248
	.long	1043220868
	.long	1045245838
	.long	1047245614
	.long	1049383373
	.long	1051287907
	.long	1053115377
	.long	1054857013
	.long	1057129528
	.long	1058581488
	.long	1059832960
	.long	1060891676
	.long	1062153819
	.long	1063337043
	.long	1064100733
	.long	1064582223
	.long	1064984555
	.long	1065216645
	.long	1065302845
	.long	1065334668
	.long	1065349076
	.long	1065352656
	.long	1065353140
	.long	1065353206
	.long	1065353215
	.long	1065353216
	.long	0
	.long	2963361822
	.long	2971470750
	.long	2945658640
	.long	821708412
	.long	824483568
	.long	824941280
	.long	2984085072
	.long	2957298688
	.long	838449816
	.long	2966046080
	.long	2988320324
	.long	2989804564
	.long	842626356
	.long	3000013710
	.long	2972725824
	.long	3002017674
	.long	853753500
	.long	2987104448
	.long	3000350914
	.long	855535800
	.long	852410906
	.long	851608946
	.long	2988641656
	.long	2997011000
	.long	2989576736
	.long	3000884068
	.long	2999984336
	.long	840950056
	.long	2995215280
	.long	855269702
	.long	0
	.long	1065353216
	.long	1065295748
	.long	1065270545
	.long	1065229919
	.long	1065181343
	.long	1065124909
	.long	1065025765
	.long	1064867200
	.long	1064679597
	.long	1064464345
	.long	1064093083
	.long	1063517074
	.long	1062862743
	.long	1062146519
	.long	1060992371
	.long	1059386208
	.long	1057800167
	.long	1055660649
	.long	1051764737
	.long	1046959010
	.long	1041444634
	.long	1035462611
	.long	1026689093
	.long	1015337940
	.long	1002731447
	.long	990958554
	.long	973168670
	.long	948705851
	.long	924299482
	.long	899955662
	.long	864224966
	.long	0
	.long	2956213371
	.long	3178161821
	.long	3180268967
	.long	3182315389
	.long	3184339487
	.long	3186337805
	.long	3188474939
	.long	3190373619
	.long	3192189570
	.long	3193910865
	.long	3196176320
	.long	3197556682
	.long	3198679950
	.long	3199536798
	.long	3200331518
	.long	3200564882
	.long	3200049264
	.long	3199029518
	.long	3197040598
	.long	3192620804
	.long	3188208183
	.long	3182392393
	.long	3173916356
	.long	3162750726
	.long	3150176437
	.long	3138431708
	.long	3120650203
	.long	3096189170
	.long	3071783062
	.long	3047439278
	.long	3011707180
	.long	0
	.long	3198855845
	.long	3198879250
	.long	3198677023
	.long	3198476576
	.long	3198388151
	.long	3198245218
	.long	3197982711
	.long	3197594458
	.long	3197117197
	.long	3196587519
	.long	3195304371
	.long	3192667528
	.long	3189843074
	.long	3186330810
	.long	3177085101
	.long	1013669486
	.long	1032032579
	.long	1036132065
	.long	1038305199
	.long	1036774550
	.long	1033498413
	.long	1028927137
	.long	1021175553
	.long	1009568359
	.long	998361895
	.long	985691041
	.long	967585842
	.long	943363289
	.long	919210013
	.long	895139148
	.long	858471606
	.long	0
	.long	3077428921
	.long	3189516141
	.long	1008586543
	.long	1036101517
	.long	1033304453
	.long	1034073627
	.long	1036071831
	.long	1037235824
	.long	1039436298
	.long	1040631208
	.long	1041906362
	.long	1042793477
	.long	1043232976
	.long	1043086916
	.long	1042100375
	.long	1039444212
	.long	1034126600
	.long	1026638186
	.long	995501655
	.long	3165579977
	.long	3167654937
	.long	3165317828
	.long	3158960080
	.long	3148291549
	.long	3137354510
	.long	3124730373
	.long	3106670759
	.long	3082457650
	.long	3058305807
	.long	3034235241
	.long	2997581996
	.long	0
	.long	1040781545
	.long	1131811139
	.long	1097198812
	.long	3247503190
	.long	3230402941
	.long	3224086547
	.long	3212798938
	.long	1059790272
	.long	1053691997
	.long	1061317268
	.long	3134918084
	.long	1034173207
	.long	3176246152
	.long	3165561405
	.long	3174788493
	.long	3178015405
	.long	3178847213
	.long	3177176538
	.long	3171127099
	.long	3155996003
	.long	985352038
	.long	999682315
	.long	998398067
	.long	989522534
	.long	977926264
	.long	966355955
	.long	948911724
	.long	924561635
	.long	900244966
	.long	875993879
	.long	841254832
	.long	0
	.long	3155046246
	.long	1175181842
	.long	1138112751
	.long	3286309950
	.long	3267011817
	.long	3259619885
	.long	3246758786
	.long	1088248663
	.long	1078543936
	.long	1086795944
	.long	3205436942
	.long	1043392367
	.long	3198686087
	.long	3182586396
	.long	3174374999
	.long	3142320544
	.long	1008565243
	.long	1014115537
	.long	1016545052
	.long	1010017051
	.long	998649588
	.long	975680464
	.long	3124451591
	.long	3121544226
	.long	3112148751
	.long	3100159824
	.long	3082673659
	.long	3058641232
	.long	3034613169
	.long	3010665978
	.long	2975473412
	.long	0
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	0
	.long	0
	.long	434973
	.long	1072693248
	.long	2462381979
	.long	3194875870
	.long	1922363613
	.long	3218429247
	.long	3436033793
	.long	3207031997
	.long	1164049177
	.long	1069640674
	.long	2766840751
	.long	3214171076
	.long	0
	.long	0
	.long	1640345390
	.long	3194496887
	.long	1662417413
	.long	1072693250
	.long	3716526453
	.long	3205098446
	.long	1853512117
	.long	3218426760
	.long	3156517427
	.long	3211911078
	.long	3184519871
	.long	1069779119
	.long	3280598482
	.long	3215502338
	.long	0
	.long	0
	.long	1962245523
	.long	3196442075
	.long	366239963
	.long	1072693255
	.long	3337913224
	.long	3206521562
	.long	1328356923
	.long	3218424220
	.long	1470245799
	.long	3212690354
	.long	922782103
	.long	1069841533
	.long	705136934
	.long	3215769608
	.long	0
	.long	0
	.long	917120191
	.long	3198018856
	.long	2270262052
	.long	1072693264
	.long	21785897
	.long	3207561752
	.long	3853373788
	.long	3218420654
	.long	3959915849
	.long	3213220134
	.long	839274685
	.long	1069902529
	.long	3478609944
	.long	3215984949
	.long	0
	.long	0
	.long	321264669
	.long	3199232231
	.long	3507921756
	.long	1072693279
	.long	855596455
	.long	3208292995
	.long	4197403487
	.long	3218416395
	.long	1260192796
	.long	3213688235
	.long	509545499
	.long	1069956190
	.long	4001843557
	.long	3216067072
	.long	0
	.long	0
	.long	2572895834
	.long	3200373196
	.long	4238319527
	.long	1072693307
	.long	1589084946
	.long	3209032256
	.long	323547252
	.long	3218410632
	.long	129829396
	.long	3214058556
	.long	2665301683
	.long	1070009663
	.long	3805267410
	.long	3216137363
	.long	0
	.long	0
	.long	1373918637
	.long	3199925337
	.long	2391440540
	.long	1072693299
	.long	3494583150
	.long	3208925835
	.long	2192964039
	.long	3218411256
	.long	579095213
	.long	3214044622
	.long	3432431090
	.long	1070009041
	.long	3870858437
	.long	3216138421
	.long	0
	.long	0
	.long	3062447777
	.long	1055926683
	.long	3334650904
	.long	1072692790
	.long	3497776375
	.long	1062371871
	.long	4014660983
	.long	3218436927
	.long	1708666466
	.long	3212333537
	.long	648260668
	.long	1069902577
	.long	1156520282
	.long	3216044909
	.long	0
	.long	0
	.long	4186264729
	.long	1058462985
	.long	3883474621
	.long	1072690745
	.long	4001630278
	.long	1065042042
	.long	484659484
	.long	3218507007
	.long	301873053
	.long	1066864880
	.long	2426783364
	.long	1069685380
	.long	3518509994
	.long	3215777524
	.long	0
	.long	0
	.long	1324317639
	.long	1061009204
	.long	1677646538
	.long	1072681882
	.long	781584286
	.long	1067165904
	.long	3649499968
	.long	3218726741
	.long	2264952365
	.long	1069102871
	.long	2344790854
	.long	1068835622
	.long	4047770869
	.long	3215138580
	.long	0
	.long	0
	.long	70848422
	.long	1063287485
	.long	1930391614
	.long	1072650795
	.long	586495590
	.long	1068891644
	.long	2415479819
	.long	3219189888
	.long	2049892606
	.long	1070582148
	.long	1783689851
	.long	3213584996
	.long	2396151379
	.long	3213355995
	.long	0
	.long	0
	.long	2764829776
	.long	1064683280
	.long	95861817
	.long	1072595436
	.long	350241294
	.long	1069957747
	.long	1429983818
	.long	3219518543
	.long	2046078110
	.long	1071248730
	.long	2818409407
	.long	3216573116
	.long	351621961
	.long	1065184929
	.long	0
	.long	0
	.long	818345493
	.long	1065579544
	.long	47166764
	.long	1072535009
	.long	2931635641
	.long	1070624305
	.long	2472163867
	.long	3219785146
	.long	898647657
	.long	1071677167
	.long	2840881315
	.long	3217227676
	.long	1213275070
	.long	1066490976
	.long	0
	.long	0
	.long	3770339094
	.long	1065664250
	.long	4021094867
	.long	1072525054
	.long	3250137669
	.long	1070683759
	.long	3067647579
	.long	3219831010
	.long	706412794
	.long	1071716084
	.long	3457985438
	.long	3217296958
	.long	693681995
	.long	1066592455
	.long	0
	.long	0
	.long	794345931
	.long	3214229553
	.long	674007974
	.long	1072761769
	.long	1339296402
	.long	3213866766
	.long	2063412275
	.long	3219199437
	.long	3042293216
	.long	1071038746
	.long	1218111703
	.long	3216613854
	.long	1828949834
	.long	1065778789
	.long	0
	.long	0
	.long	3709362262
	.long	3216572138
	.long	1704472411
	.long	1073083731
	.long	334125080
	.long	3219185499
	.long	3643953259
	.long	3216245823
	.long	972935809
	.long	1069563300
	.long	4262764539
	.long	3215188513
	.long	3947124972
	.long	1064363655
	.long	0
	.long	0
	.long	684725320
	.long	3217602215
	.long	2059930851
	.long	1073428282
	.long	6923247
	.long	3220175349
	.long	1962536238
	.long	1070738118
	.long	2626892535
	.long	3214818472
	.long	1541908021
	.long	3211168932
	.long	1264782098
	.long	1061514036
	.long	0
	.long	0
	.long	4193183898
	.long	3218211722
	.long	2527318106
	.long	1073704783
	.long	1779267795
	.long	3220520390
	.long	2178062862
	.long	1071649373
	.long	2371270354
	.long	3216802466
	.long	214503718
	.long	1066134183
	.long	2527651537
	.long	3209129722
	.long	0
	.long	0
	.long	1145099230
	.long	3217848868
	.long	1219675578
	.long	1073564173
	.long	3377824792
	.long	3220387400
	.long	1294161399
	.long	1071386209
	.long	535756989
	.long	3216499614
	.long	3414431292
	.long	1065769858
	.long	3872552752
	.long	3208765586
	.long	0
	.long	0
	.long	3432152680
	.long	3212471108
	.long	3481247728
	.long	1073111648
	.long	2087872556
	.long	3219843286
	.long	1539630695
	.long	1070713931
	.long	2045031161
	.long	3215666774
	.long	1438917333
	.long	1064738520
	.long	2997200424
	.long	3207590169
	.long	0
	.long	0
	.long	157024952
	.long	1070614475
	.long	1896115811
	.long	1072588717
	.long	1533634146
	.long	3219167457
	.long	3479089950
	.long	1069795336
	.long	294041664
	.long	3214609167
	.long	3323703207
	.long	1063520882
	.long	1200470279
	.long	3206092743
	.long	0
	.long	0
	.long	780145450
	.long	1071804775
	.long	3436973384
	.long	1071541223
	.long	1373298557
	.long	3217881162
	.long	616458359
	.long	1068360186
	.long	1012488256
	.long	3212939359
	.long	3381328826
	.long	1061569412
	.long	3619594050
	.long	3203906531
	.long	0
	.long	0
	.long	3555024088
	.long	1072352823
	.long	703965661
	.long	1069801815
	.long	68876051
	.long	3215985072
	.long	4285546012
	.long	1066131701
	.long	1692571309
	.long	3210444434
	.long	2250664999
	.long	1058874117
	.long	2757518980
	.long	3200902424
	.long	0
	.long	0
	.long	4088530245
	.long	1072580854
	.long	2571880719
	.long	1067895848
	.long	4091013897
	.long	3213873796
	.long	4246435429
	.long	1063770948
	.long	92905889
	.long	3207872058
	.long	248987709
	.long	1056074614
	.long	2369951583
	.long	3197898922
	.long	0
	.long	0
	.long	3580076556
	.long	1072660066
	.long	1353576036
	.long	1065860878
	.long	2410885661
	.long	3211602990
	.long	2989427096
	.long	1061369430
	.long	3886685439
	.long	3205273864
	.long	529712074
	.long	1053215589
	.long	3764845364
	.long	3194905549
	.long	0
	.long	0
	.long	660908647
	.long	1072688177
	.long	2675542798
	.long	1062777930
	.long	772498083
	.long	3208233517
	.long	377295306
	.long	1057798793
	.long	162648032
	.long	3201438006
	.long	623489458
	.long	1049119366
	.long	3651746243
	.long	3190506519
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
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
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	1038090240
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.long	54525952
	.type	__svml_stanh_ha_data_internal,@object
	.size	__svml_stanh_ha_data_internal,3456
	.align 4
__stanh_ha__imlsTanhTab:
	.long	1065353216
	.long	3212836864
	.type	__stanh_ha__imlsTanhTab,@object
	.size	__stanh_ha__imlsTanhTab,8
      	.section        .note.GNU-stack,"",@progbits
