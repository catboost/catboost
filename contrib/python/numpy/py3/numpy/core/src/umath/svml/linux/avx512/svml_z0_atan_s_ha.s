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
 *      For    0.0    <= x <=  7.0/16.0: atan(x) = atan(0.0) + atan(s), where s=(x-0.0)/(1.0+0.0*x)
 *      For  7.0/16.0 <= x <= 11.0/16.0: atan(x) = atan(0.5) + atan(s), where s=(x-0.5)/(1.0+0.5*x)
 *      For 11.0/16.0 <= x <= 19.0/16.0: atan(x) = atan(1.0) + atan(s), where s=(x-1.0)/(1.0+1.0*x)
 *      For 19.0/16.0 <= x <= 39.0/16.0: atan(x) = atan(1.5) + atan(s), where s=(x-1.5)/(1.0+1.5*x)
 *      For 39.0/16.0 <= x <=    inf   : atan(x) = atan(inf) + atan(s), where s=-1.0/x
 *      Where atan(s) ~= s+s^3*Poly11(s^2) on interval |s|<7.0/0.16.
 * 
 */


	.text
.L_2__routine_start___svml_atanf16_ha_z0_0:

	.align    16,0x90
	.globl __svml_atanf16_ha

__svml_atanf16_ha:


	.cfi_startproc
..L2:

        vmovups   128+__svml_satan_ha_data_internal_avx512(%rip), %zmm4

/* saturate X range, initialize Y */
        vmovups   320+__svml_satan_ha_data_internal_avx512(%rip), %zmm5
        vmovups   64+__svml_satan_ha_data_internal_avx512(%rip), %zmm6
        vmovups   256+__svml_satan_ha_data_internal_avx512(%rip), %zmm10

/* round to 2 bits after binary point */
        vmovups   192+__svml_satan_ha_data_internal_avx512(%rip), %zmm13

/* table lookup sequence */
        vmovups   448+__svml_satan_ha_data_internal_avx512(%rip), %zmm15

        vandps    __svml_satan_ha_data_internal_avx512(%rip), %zmm0, %zmm11
        vcmpps    $17, {sae}, %zmm4, %zmm11, %k1
        vminps    {sae}, %zmm11, %zmm5, %zmm2
        vmovups   576+__svml_satan_ha_data_internal_avx512(%rip), %zmm4
        vaddps    {rn-sae}, %zmm6, %zmm11, %zmm3

/* initialize DiffX=-1 */
        vxorps    %zmm0, %zmm11, %zmm1
        vreduceps $40, {sae}, %zmm11, %zmm13{%k1}
        vsubps    {rn-sae}, %zmm6, %zmm3, %zmm7
        vpermt2ps 512+__svml_satan_ha_data_internal_avx512(%rip), %zmm3, %zmm15
        vpermt2ps 640+__svml_satan_ha_data_internal_avx512(%rip), %zmm3, %zmm4
        vfmadd213ps {rn-sae}, %zmm10, %zmm7, %zmm2{%k1}
        vmovups   832+__svml_satan_ha_data_internal_avx512(%rip), %zmm3
        vmovups   896+__svml_satan_ha_data_internal_avx512(%rip), %zmm6

/* (x*x0)_high */
        vsubps    {rn-sae}, %zmm10, %zmm2, %zmm8

/* R+Rl = DiffX/Y */
        vrcp14ps  %zmm2, %zmm12

/* (x*x0)_low */
        vfmsub213ps {rn-sae}, %zmm8, %zmm7, %zmm11
        knotw     %k1, %k2

/* eps=1-Y*Rcp */
        vmovaps   %zmm10, %zmm9
        vfnmadd231ps {rn-sae}, %zmm2, %zmm12, %zmm9

/* set table value to Pi/2 for large X */
        vblendmps 704+__svml_satan_ha_data_internal_avx512(%rip), %zmm15, %zmm7{%k2}
        vblendmps 768+__svml_satan_ha_data_internal_avx512(%rip), %zmm4, %zmm5{%k2}

/* Rcp+Rcp*eps */
        vfmadd213ps {rn-sae}, %zmm12, %zmm9, %zmm12

/* Diff*Rcp1 */
        vmulps    {rn-sae}, %zmm13, %zmm12, %zmm0

/* (x*x0)_low*R */
        vmulps    {rn-sae}, %zmm12, %zmm11, %zmm14

/* eps1=1-Y*Rcp1 */
        vfnmadd213ps {rn-sae}, %zmm10, %zmm12, %zmm2

/* (Diff*Rcp1)_low */
        vfmsub213ps {rn-sae}, %zmm0, %zmm12, %zmm13

/* polynomial evaluation */
        vmulps    {rn-sae}, %zmm0, %zmm0, %zmm12

/* R_low = (Diff*Rcp1)_low + eps1*(DiffX*Rcp1) */
        vfmadd213ps {rn-sae}, %zmm13, %zmm0, %zmm2
        vaddps    {rn-sae}, %zmm0, %zmm7, %zmm13
        vmulps    {rn-sae}, %zmm0, %zmm12, %zmm10

/* R_low = R_low - (X*X0)_low*Rcp1*(Diff*Rcp1) */
        vfnmadd231ps {rn-sae}, %zmm0, %zmm14, %zmm2{%k1}
        vfmadd231ps {rn-sae}, %zmm12, %zmm3, %zmm6
        vsubps    {rn-sae}, %zmm7, %zmm13, %zmm8
        vaddps    {rn-sae}, %zmm5, %zmm2, %zmm9
        vmovups   960+__svml_satan_ha_data_internal_avx512(%rip), %zmm2
        vsubps    {rn-sae}, %zmm8, %zmm0, %zmm0
        vfmadd213ps {rn-sae}, %zmm2, %zmm6, %zmm12
        vaddps    {rn-sae}, %zmm0, %zmm9, %zmm11
        vfmadd213ps {rn-sae}, %zmm11, %zmm10, %zmm12
        vaddps    {rn-sae}, %zmm13, %zmm12, %zmm14
        vxorps    %zmm1, %zmm14, %zmm0

/* no invcbrt in libm, so taking it out here */
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanf16_ha,@function
	.size	__svml_atanf16_ha,.-__svml_atanf16_ha
..LN__svml_atanf16_ha.0:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_satan_ha_data_internal_avx512:
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
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
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
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
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
	.long	0
	.long	0
	.long	0
	.long	1048239024
	.long	1055744824
	.long	1059372157
	.long	1061752795
	.long	1063609315
	.long	1065064543
	.long	1065786489
	.long	1066252045
	.long	1066633083
	.long	1066949484
	.long	1067215699
	.long	1067442363
	.long	1067637412
	.long	1067806856
	.long	1067955311
	.long	1068086373
	.long	1068202874
	.long	1068307075
	.long	1068400798
	.long	1068485529
	.long	1068562486
	.long	1068632682
	.long	1068696961
	.long	1068756035
	.long	1068810506
	.long	1068860887
	.long	1068907620
	.long	1068951084
	.long	1068991608
	.long	1069029480
	.long	1069064949
	.long	0
	.long	2975494116
	.long	833369962
	.long	835299256
	.long	2998648110
	.long	2995239174
	.long	3000492182
	.long	860207626
	.long	3008447516
	.long	3005590622
	.long	3000153675
	.long	860754741
	.long	859285590
	.long	844944488
	.long	2993069463
	.long	858157665
	.long	3006142000
	.long	3007693206
	.long	3009342234
	.long	847469400
	.long	3006114683
	.long	852829553
	.long	847325583
	.long	860305056
	.long	846145135
	.long	2997638646
	.long	855837703
	.long	2979047222
	.long	2995344192
	.long	854092798
	.long	3000498637
	.long	859965578
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.type	__svml_satan_ha_data_internal_avx512,@object
	.size	__svml_satan_ha_data_internal_avx512,1024
      	.section        .note.GNU-stack,"",@progbits
