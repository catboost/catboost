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
 *      For    0.0    <= x <=  7.0/16.0: atan(x) = atan(0.0) + atan(s), where s=(x-0.0)/(1.0+0.0*x)
 *      For  7.0/16.0 <= x <= 11.0/16.0: atan(x) = atan(0.5) + atan(s), where s=(x-0.5)/(1.0+0.5*x)
 *      For 11.0/16.0 <= x <= 19.0/16.0: atan(x) = atan(1.0) + atan(s), where s=(x-1.0)/(1.0+1.0*x)
 *      For 19.0/16.0 <= x <= 39.0/16.0: atan(x) = atan(1.5) + atan(s), where s=(x-1.5)/(1.0+1.5*x)
 *      For 39.0/16.0 <= x <=    inf   : atan(x) = atan(inf) + atan(s), where s=-1.0/x
 *      Where atan(s) ~= s+s^3*Poly11(s^2) on interval |s|<7.0/0.16.
 * --
 * 
 */


	.text
.L_2__routine_start___svml_atan28_ha_z0_0:

	.align    16,0x90
	.globl __svml_atan28_ha

__svml_atan28_ha:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp
        xorl      %edx, %edx

/*
 * #define NO_VECTOR_ZERO_ATAN2_ARGS
 * -------------------- Declarations -----------------------
 * Variables
 * Constants
 * -------------- The end of declarations ----------------
 * ---------------------- Implementation ---------------------
 * Get r0~=1/B
 * Cannot be replaced by VQRCP(D, dR0, dB);
 * Argument Absolute values
 */
        vmovups   896+__svml_datan2_ha_data_internal(%rip), %zmm8

/* Get PiOrZero = Pi (if x<0), or Zero (if x>0) */
        vmovups   2880+__svml_datan2_ha_data_internal(%rip), %zmm10
        vmovdqu   2688+__svml_datan2_ha_data_internal(%rip), %ymm2
        kmovw     %k7, 56(%rsp)
        kmovw     %k5, 48(%rsp)
        vmovaps   %zmm1, %zmm11
        vandpd    %zmm8, %zmm11, %zmm9
        vandpd    %zmm8, %zmm0, %zmm8
        vcmppd    $17, {sae}, %zmm10, %zmm11, %k7
        vmovdqu   2752+__svml_datan2_ha_data_internal(%rip), %ymm10

/*
 * Determining table index (to which interval is subject y/x)
 * Reduction of |y/x| to the interval [0;7/16].
 * 0. If 39/16 < |y/x| <  Inf  then a=0*y-  1*x, b=0*x+  1*y, AtanHi+AtanLo=atan(Inf).
 * 1. If 19/16 < |y/x| < 39/16 then a=1*y-1.5*x, b=1*x+1.5*y, AtanHi+AtanLo=atan(3/2).
 * 2. If 11/16 < |y/x| < 19/16 then a=1*y-1.0*x, b=1*x+1.0*y, AtanHi+AtanLo=atan( 1 ).
 * 3. If  7/16 < |y/x| < 11/16 then a=1*y-0.5*x, b=1*x+0.5*y, AtanHi+AtanLo=atan(1/2).
 * 4. If   0   < |y/x| <  7/16 then a=1*y-0  *x, b=1*x+0  *y, AtanHi+AtanLo=atan( 0 ).
 * Hence common formulas are:       a=c*y-  d*x, b=c*x+  d*y (c is mask in our case)
 * (b is always positive)
 */
        vmovups   1152+__svml_datan2_ha_data_internal(%rip), %zmm1

/* Check if y and x are on main path. */
        vpsrlq    $32, %zmm9, %zmm5
        vpsrlq    $32, %zmm8, %zmm13

/* Argument signs */
        vxorpd    %zmm9, %zmm11, %zmm6
        vpmovqd   %zmm5, %ymm4
        vpmovqd   %zmm13, %ymm3
        vxorpd    %zmm8, %zmm0, %zmm7
        vpsubd    %ymm2, %ymm4, %ymm15
        vpsubd    %ymm2, %ymm3, %ymm5
        vmovups   1792+__svml_datan2_ha_data_internal(%rip), %zmm2
        vpcmpgtd  %ymm10, %ymm15, %ymm12
        vpcmpeqd  %ymm10, %ymm15, %ymm14
        vpcmpgtd  %ymm10, %ymm5, %ymm4
        vpcmpeqd  %ymm10, %ymm5, %ymm10
        vpor      %ymm14, %ymm12, %ymm13
        vpor      %ymm10, %ymm4, %ymm3
        vmovups   1344+__svml_datan2_ha_data_internal(%rip), %zmm15
        vmovups   1280+__svml_datan2_ha_data_internal(%rip), %zmm14
        vmovups   1216+__svml_datan2_ha_data_internal(%rip), %zmm12
        vmulpd    {rn-sae}, %zmm1, %zmm9, %zmm4
        vmulpd    {rn-sae}, %zmm15, %zmm9, %zmm5
        vmulpd    {rn-sae}, %zmm12, %zmm9, %zmm15
        vmovups   768+__svml_datan2_ha_data_internal(%rip), %zmm1
        vmovups   192+__svml_datan2_ha_data_internal(%rip), %zmm12
        vcmppd    $17, {sae}, %zmm8, %zmm5, %k1
        vcmppd    $17, {sae}, %zmm8, %zmm4, %k5
        vcmppd    $17, {sae}, %zmm8, %zmm15, %k2
        vmovups   512+__svml_datan2_ha_data_internal(%rip), %zmm15
        vpor      %ymm3, %ymm13, %ymm10
        vblendmpd 320+__svml_datan2_ha_data_internal(%rip), %zmm12, %zmm5{%k1}
        vmulpd    {rn-sae}, %zmm14, %zmm9, %zmm3
        vmovups   704+__svml_datan2_ha_data_internal(%rip), %zmm13
        vmovups   256+__svml_datan2_ha_data_internal(%rip), %zmm14
        vcmppd    $17, {sae}, %zmm8, %zmm3, %k3
        vblendmpd %zmm2, %zmm13, %zmm3{%k1}
        vblendmpd 384+__svml_datan2_ha_data_internal(%rip), %zmm14, %zmm4{%k1}
        vmovups   448+__svml_datan2_ha_data_internal(%rip), %zmm13
        vblendmpd %zmm2, %zmm1, %zmm1{%k3}
        vblendmpd 576+__svml_datan2_ha_data_internal(%rip), %zmm13, %zmm14{%k3}
        vblendmpd %zmm1, %zmm3, %zmm3{%k2}
        vblendmpd 640+__svml_datan2_ha_data_internal(%rip), %zmm15, %zmm13{%k3}
        vmovups   1984+__svml_datan2_ha_data_internal(%rip), %zmm1
        vblendmpd %zmm14, %zmm5, %zmm5{%k2}
        vblendmpd %zmm13, %zmm4, %zmm4{%k2}
        vmovaps   %zmm9, %zmm12
        vxorpd    %zmm9, %zmm9, %zmm12{%k3}
        vfmadd231pd {rn-sae}, %zmm8, %zmm3, %zmm12{%k5}

/*
 * Divide r:=a*(1/b), where a==AHi+ALo, b==BHi+BLo, 1/b~=InvHi+InvLo
 * Get r0~=1/BHi
 */
        vrcp14pd  %zmm12, %zmm14

/*
 * Now refine r0 by several iterations (hidden in polynomial)
 * e = 1-Bhi*r0
 */
        vmovaps   %zmm2, %zmm15
        vfnmadd231pd {rn-sae}, %zmm12, %zmm14, %zmm15

/* r0 ~= 1/Bhi*(1-e)(1+e) or 1/Bhi*(1-e)(1+e+e^2) */
        vfmadd213pd {rn-sae}, %zmm14, %zmm15, %zmm14

/* e' = 1-Bhi*r0 */
        vfnmadd231pd {rn-sae}, %zmm12, %zmm14, %zmm2
        vmovaps   %zmm8, %zmm13
        vxorpd    %zmm8, %zmm8, %zmm13{%k3}

/* r0 ~= 1/Bhi*(1-e')(1+e') = 1/Bhi(1-e'^2) */
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm14
        vfnmadd231pd {rn-sae}, %zmm9, %zmm3, %zmm13{%k5}

/*
 * Now 1/B ~= R0 + InvLo
 * Get r:=a*(1/b), where a==AHi+ALo, 1/b~=InvHi+InvLo
 */
        vmulpd    {rn-sae}, %zmm13, %zmm14, %zmm3
        vmulpd    {rn-sae}, %zmm3, %zmm3, %zmm15
        vfnmadd213pd {rn-sae}, %zmm13, %zmm3, %zmm12
        vmulpd    {rn-sae}, %zmm15, %zmm15, %zmm2
        vmulpd    {rn-sae}, %zmm14, %zmm12, %zmm13

/* Atan polynomial approximation */
        vmovups   1856+__svml_datan2_ha_data_internal(%rip), %zmm12
        vmovups   1920+__svml_datan2_ha_data_internal(%rip), %zmm14
        vaddpd    {rn-sae}, %zmm4, %zmm13, %zmm13{%k5}
        vfmadd231pd {rn-sae}, %zmm2, %zmm12, %zmm1
        vmovups   2048+__svml_datan2_ha_data_internal(%rip), %zmm12
        vfmadd231pd {rn-sae}, %zmm2, %zmm14, %zmm12
        vmovups   2112+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm1
        vmovups   2176+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm12
        vmovups   2240+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm1
        vmovups   2304+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm12
        vmovups   2368+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm1
        vmovups   2432+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm12
        vmovups   2496+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm1
        vmovups   2560+__svml_datan2_ha_data_internal(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm14, %zmm2, %zmm12
        vfmadd213pd {rn-sae}, %zmm12, %zmm15, %zmm1
        vmulpd    {rn-sae}, %zmm15, %zmm1, %zmm15

/*
 * Res = ( (RLo + AtanBaseLo + PiOrZeroLo*sx + Poly(R^2)*R^3 + RHi + AtanBaseHi) * sx + PiOrZeroHi) * sy
 * Get PiOrZero = Pi (if x<0), or Zero (if x>0)
 */
        vxorpd    1024+__svml_datan2_ha_data_internal(%rip), %zmm6, %zmm1
        vaddpd    {rn-sae}, %zmm1, %zmm13, %zmm13{%k7}
        vfmadd213pd {rn-sae}, %zmm13, %zmm3, %zmm15
        vaddpd    {rn-sae}, %zmm15, %zmm3, %zmm3
        vaddpd    {rn-sae}, %zmm5, %zmm3, %zmm3{%k5}
        vxorpd    %zmm6, %zmm3, %zmm1
        vmovups   960+__svml_datan2_ha_data_internal(%rip), %zmm6
        vaddpd    {rn-sae}, %zmm6, %zmm1, %zmm1{%k7}
        vmovmskps %ymm10, %eax
        vorpd     %zmm7, %zmm1, %zmm7

/* =========== Special branch for fast (vector) processing of zero arguments ================ */
        testl     %eax, %eax
        jne       .LBL_1_12
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22

.LBL_1_2:


/*
 * =========== Special branch for fast (vector) processing of zero arguments ================
 * -------------- The end of implementation ----------------
 */
        testl     %edx, %edx
        jne       .LBL_1_4

.LBL_1_3:


/* no invcbrt in libm, so taking it out here */
        kmovw     48(%rsp), %k5
	.cfi_restore 123
        kmovw     56(%rsp), %k7
	.cfi_restore 125
        vmovaps   %zmm7, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22

.LBL_1_4:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm11, 128(%rsp)
        vmovups   %zmm7, 192(%rsp)
        je        .LBL_1_3


        xorl      %eax, %eax


        vzeroupper
        kmovw     %k4, 8(%rsp)
        kmovw     %k6, (%rsp)
        movq      %rsi, 24(%rsp)
        movq      %rdi, 16(%rsp)
        movq      %r12, 40(%rsp)
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 32(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

.LBL_1_8:

        btl       %r12d, %r13d
        jc        .LBL_1_11

.LBL_1_9:

        incl      %r12d
        cmpl      $8, %r12d
        jl        .LBL_1_8


        kmovw     8(%rsp), %k4
	.cfi_restore 122
        kmovw     (%rsp), %k6
	.cfi_restore 124
        vmovups   192(%rsp), %zmm7
        movq      24(%rsp), %rsi
	.cfi_restore 4
        movq      16(%rsp), %rdi
	.cfi_restore 5
        movq      40(%rsp), %r12
	.cfi_restore 12
        movq      32(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_3
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

.LBL_1_11:

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi
        lea       192(%rsp,%r12,8), %rdx

        call      __svml_datan2_ha_cout_rare_internal
        jmp       .LBL_1_9
	.cfi_restore 4
	.cfi_restore 5
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 122
	.cfi_restore 124

.LBL_1_12:


/* Check if both X & Y are not NaNs:  iXYnotNAN */
        vcmppd    $3, {sae}, %zmm11, %zmm11, %k2
        vcmppd    $3, {sae}, %zmm0, %zmm0, %k3

/*
 * 1) If y<x then PIO2=0
 * 2) If y>x then PIO2=Pi/2
 */
        vcmppd    $17, {sae}, %zmm9, %zmm8, %k1
        vmovups   832+__svml_datan2_ha_data_internal(%rip), %zmm14
        vmovups   3136+__svml_datan2_ha_data_internal(%rip), %zmm4
        vmovups   3072+__svml_datan2_ha_data_internal(%rip), %zmm12
        vblendmpd %zmm9, %zmm8, %zmm3{%k1}
        vpbroadcastq .L_2il0floatpacket.40(%rip), %zmm5
        vandpd    %zmm14, %zmm11, %zmm13
        vandpd    %zmm14, %zmm0, %zmm6
        vxorpd    %zmm4, %zmm4, %zmm4{%k1}
        vmovaps   %zmm5, %zmm15
        vmovaps   %zmm5, %zmm2

/* Check if at least on of Y or Y is zero: iAXAYZERO */
        vmovaps   %zmm5, %zmm14
        vpandnq   %zmm11, %zmm11, %zmm15{%k2}
        vpandnq   %zmm0, %zmm0, %zmm2{%k3}
        vandpd    %zmm2, %zmm15, %zmm1
        vmovups   2880+__svml_datan2_ha_data_internal(%rip), %zmm2
        vpsrlq    $32, %zmm1, %zmm1
        vcmppd    $4, {sae}, %zmm2, %zmm9, %k5
        vcmppd    $4, {sae}, %zmm2, %zmm8, %k7

/*
 * -------- Path for zero arguments (at least one of both) --------------
 * Check if both args are zeros (den. is zero)
 */
        vcmppd    $4, {sae}, %zmm2, %zmm3, %k1

/* Res = sign(Y)*(X<0)?(PIO2+PI):PIO2 */
        vpcmpgtq  %zmm11, %zmm2, %k2
        vpmovqd   %zmm1, %ymm1
        vpandnq   %zmm9, %zmm9, %zmm14{%k5}
        vmovaps   %zmm5, %zmm9
        vpandnq   %zmm8, %zmm8, %zmm9{%k7}
        vorpd     %zmm9, %zmm14, %zmm8
        vpsrlq    $32, %zmm8, %zmm9
        vpmovqd   %zmm9, %ymm14

/* Check if at least on of Y or Y is zero and not NaN: iAXAYZEROnotNAN */
        vpand     %ymm1, %ymm14, %ymm8

/* Exclude from previous callout mask zero (and not NaN) arguments */
        vpandn    %ymm10, %ymm8, %ymm10

/* Merge results from main and spec path */
        vpmovzxdq %ymm8, %zmm8

/* Go to callout */
        vmovmskps %ymm10, %edx
        vpandnq   %zmm3, %zmm3, %zmm5{%k1}

/* Set sPIO2 to zero if den. is zero */
        vpandnq   %zmm4, %zmm5, %zmm1
        vpandq    %zmm5, %zmm2, %zmm3
        vporq     %zmm3, %zmm1, %zmm4
        vorpd     %zmm13, %zmm4, %zmm9
        vpsllq    $32, %zmm8, %zmm2
        vaddpd    {rn-sae}, %zmm12, %zmm9, %zmm9{%k2}
        vorpd     %zmm6, %zmm9, %zmm6
        vpord     %zmm2, %zmm8, %zmm10
        vpandnq   %zmm7, %zmm10, %zmm7
        vpandq    %zmm10, %zmm6, %zmm1
        vporq     %zmm1, %zmm7, %zmm7
        jmp       .LBL_1_2
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atan28_ha,@function
	.size	__svml_atan28_ha,.-__svml_atan28_ha
..LN__svml_atan28_ha.0:

.L_2__routine_start___svml_datan2_ha_cout_rare_internal_1:

	.align    16,0x90

__svml_datan2_ha_cout_rare_internal:


	.cfi_startproc
..L58:

        movq      %rdx, %rcx
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm1
        movsd     (%rdi), %xmm2
        movsd     (%rsi), %xmm0
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm1
        movsd     %xmm2, -48(%rsp)
        movsd     %xmm1, -40(%rsp)
        movzwl    -42(%rsp), %r9d
        andl      $32752, %r9d
        movb      -33(%rsp), %al
        movzwl    -34(%rsp), %r8d
        andb      $-128, %al
        andl      $32752, %r8d
        shrl      $4, %r9d
        movb      -41(%rsp), %dl
        shrb      $7, %dl
        shrb      $7, %al
        shrl      $4, %r8d
        cmpl      $2047, %r9d
        je        .LBL_2_49


        cmpl      $2047, %r8d
        je        .LBL_2_38


        testl     %r9d, %r9d
        jne       .LBL_2_6


        testl     $1048575, -44(%rsp)
        jne       .LBL_2_6


        cmpl      $0, -48(%rsp)
        je        .LBL_2_31

.LBL_2_6:

        testl     %r8d, %r8d
        jne       .LBL_2_9


        testl     $1048575, -36(%rsp)
        jne       .LBL_2_9


        cmpl      $0, -40(%rsp)
        je        .LBL_2_29

.LBL_2_9:

        negl      %r8d
        movsd     %xmm2, -48(%rsp)
        addl      %r9d, %r8d
        movsd     %xmm1, -40(%rsp)
        movb      -41(%rsp), %dil
        movb      -33(%rsp), %sil
        andb      $127, %dil
        andb      $127, %sil
        cmpl      $-54, %r8d
        jle       .LBL_2_24


        cmpl      $54, %r8d
        jge       .LBL_2_21


        movb      %sil, -33(%rsp)
        movb      %dil, -41(%rsp)
        testb     %al, %al
        jne       .LBL_2_13


        movsd     1976+__datan2_ha_CoutTab(%rip), %xmm1
        movaps    %xmm1, %xmm0
        jmp       .LBL_2_14

.LBL_2_13:

        movsd     1936+__datan2_ha_CoutTab(%rip), %xmm1
        movsd     1944+__datan2_ha_CoutTab(%rip), %xmm0

.LBL_2_14:

        movsd     -48(%rsp), %xmm4
        movsd     -40(%rsp), %xmm2
        movaps    %xmm4, %xmm5
        divsd     %xmm2, %xmm5
        movzwl    -42(%rsp), %esi
        movsd     %xmm5, -16(%rsp)
        testl     %r9d, %r9d
        jle       .LBL_2_37


        cmpl      $2046, %r9d
        jge       .LBL_2_17


        andl      $-32753, %esi
        addl      $-1023, %r9d
        movsd     %xmm4, -48(%rsp)
        addl      $16368, %esi
        movw      %si, -42(%rsp)
        jmp       .LBL_2_18

.LBL_2_17:

        movsd     1992+__datan2_ha_CoutTab(%rip), %xmm3
        movl      $1022, %r9d
        mulsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)

.LBL_2_18:

        negl      %r9d
        addl      $1023, %r9d
        andl      $2047, %r9d
        movzwl    1894+__datan2_ha_CoutTab(%rip), %esi
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm3
        andl      $-32753, %esi
        shll      $4, %r9d
        movsd     %xmm3, -40(%rsp)
        orl       %r9d, %esi
        movw      %si, -34(%rsp)
        movsd     -40(%rsp), %xmm4
        mulsd     %xmm4, %xmm2
        comisd    1880+__datan2_ha_CoutTab(%rip), %xmm5
        jb        .LBL_2_20


        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm12
        movaps    %xmm2, %xmm3
        mulsd     %xmm2, %xmm12
        movsd     %xmm12, -72(%rsp)
        movsd     -72(%rsp), %xmm13
        movsd     %xmm5, -24(%rsp)
        subsd     %xmm2, %xmm13
        movsd     %xmm13, -64(%rsp)
        movsd     -72(%rsp), %xmm15
        movsd     -64(%rsp), %xmm14
        movl      -20(%rsp), %r8d
        movl      %r8d, %r9d
        andl      $-524288, %r8d
        andl      $-1048576, %r9d
        addl      $262144, %r8d
        subsd     %xmm14, %xmm15
        movsd     %xmm15, -72(%rsp)
        andl      $1048575, %r8d
        movsd     -72(%rsp), %xmm4
        orl       %r8d, %r9d
        movl      $0, -24(%rsp)
        subsd     %xmm4, %xmm3
        movl      %r9d, -20(%rsp)
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -24(%rsp), %xmm11
        movsd     -64(%rsp), %xmm9
        mulsd     %xmm11, %xmm5
        mulsd     %xmm11, %xmm9
        movsd     1968+__datan2_ha_CoutTab(%rip), %xmm8
        mulsd     %xmm8, %xmm5
        mulsd     %xmm8, %xmm9
        movaps    %xmm5, %xmm7
        movzwl    -10(%rsp), %edi
        addsd     %xmm9, %xmm7
        movsd     %xmm7, -72(%rsp)
        andl      $32752, %edi
        movsd     -72(%rsp), %xmm6
        shrl      $4, %edi
        subsd     %xmm6, %xmm5
        movl      -12(%rsp), %esi
        addsd     %xmm5, %xmm9
        movsd     %xmm9, -64(%rsp)
        andl      $1048575, %esi
        movsd     -48(%rsp), %xmm9
        movsd     -72(%rsp), %xmm3
        movaps    %xmm9, %xmm12
        movsd     -64(%rsp), %xmm10
        movaps    %xmm9, %xmm14
        movaps    %xmm9, %xmm6
        addsd     %xmm3, %xmm12
        movsd     %xmm12, -72(%rsp)
        movsd     -72(%rsp), %xmm13
        shll      $20, %edi
        subsd     %xmm13, %xmm14
        movsd     %xmm14, -64(%rsp)
        orl       %esi, %edi
        movsd     -72(%rsp), %xmm4
        addl      $-1069547520, %edi
        movsd     -64(%rsp), %xmm15
        movl      $113, %esi
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm13
        addsd     %xmm15, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -64(%rsp), %xmm8
        sarl      $19, %edi
        addsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        cmpl      $113, %edi
        movsd     -56(%rsp), %xmm7
        cmovl     %edi, %esi
        subsd     %xmm7, %xmm6
        movsd     %xmm6, -56(%rsp)
        addl      %esi, %esi
        movsd     -64(%rsp), %xmm12
        lea       __datan2_ha_CoutTab(%rip), %rdi
        movsd     -56(%rsp), %xmm5
        movslq    %esi, %rsi
        addsd     %xmm5, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -72(%rsp), %xmm7
        mulsd     %xmm7, %xmm13
        movsd     -56(%rsp), %xmm8
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm10, %xmm8
        movsd     -72(%rsp), %xmm4
        movaps    %xmm9, %xmm10
        mulsd     2000+__datan2_ha_CoutTab(%rip), %xmm10
        subsd     %xmm7, %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -72(%rsp), %xmm3
        movsd     -64(%rsp), %xmm14
        subsd     %xmm14, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm15
        subsd     %xmm15, %xmm7
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm7
        movsd     -64(%rsp), %xmm4
        movsd     %xmm10, -72(%rsp)
        movaps    %xmm2, %xmm10
        addsd     %xmm4, %xmm8
        movsd     -72(%rsp), %xmm4
        subsd     -48(%rsp), %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm3
        subsd     %xmm3, %xmm6
        movaps    %xmm2, %xmm3
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        subsd     %xmm5, %xmm9
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm12
        movsd     -64(%rsp), %xmm9
        mulsd     %xmm11, %xmm12
        mulsd     %xmm11, %xmm9
        movaps    %xmm12, %xmm11
        addsd     %xmm9, %xmm11
        movsd     %xmm11, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        subsd     %xmm4, %xmm12
        addsd     %xmm9, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm15
        movsd     -64(%rsp), %xmm6
        addsd     %xmm15, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm3
        subsd     %xmm5, %xmm10
        movsd     %xmm10, -64(%rsp)
        movsd     -72(%rsp), %xmm13
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm13
        movsd     %xmm13, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm13
        addsd     %xmm14, %xmm15
        movsd     %xmm15, -64(%rsp)
        movsd     -56(%rsp), %xmm4
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm14
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -64(%rsp), %xmm4
        movsd     -56(%rsp), %xmm2
        addsd     %xmm2, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -72(%rsp), %xmm12
        mulsd     %xmm12, %xmm3
        movsd     -56(%rsp), %xmm5
        movsd     %xmm3, -72(%rsp)
        addsd     %xmm6, %xmm5
        movsd     -72(%rsp), %xmm9
        subsd     %xmm12, %xmm9
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm2
        subsd     %xmm2, %xmm10
        movsd     %xmm10, -72(%rsp)
        movsd     -72(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm9
        divsd     %xmm9, %xmm14
        mulsd     %xmm14, %xmm13
        movsd     -64(%rsp), %xmm10
        movsd     %xmm13, -64(%rsp)
        addsd     %xmm10, %xmm5
        movsd     -64(%rsp), %xmm15
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm12
        subsd     %xmm14, %xmm15
        movsd     %xmm15, -56(%rsp)
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm4
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm13
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -56(%rsp), %xmm3
        mulsd     %xmm3, %xmm9
        movsd     -56(%rsp), %xmm11
        subsd     %xmm9, %xmm12
        mulsd     %xmm11, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -64(%rsp), %xmm5
        subsd     %xmm5, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -64(%rsp), %xmm2
        movq      -56(%rsp), %r10
        movsd     -64(%rsp), %xmm6
        movsd     -56(%rsp), %xmm4
        movq      %r10, -40(%rsp)
        movsd     -40(%rsp), %xmm3
        movaps    %xmm3, %xmm5
        addsd     1888+__datan2_ha_CoutTab(%rip), %xmm2
        mulsd     %xmm7, %xmm5
        mulsd     %xmm6, %xmm2
        mulsd     %xmm4, %xmm2
        mulsd     %xmm2, %xmm7
        mulsd     %xmm8, %xmm2
        mulsd     %xmm3, %xmm8
        addsd     %xmm2, %xmm7
        movsd     1872+__datan2_ha_CoutTab(%rip), %xmm3
        addsd     %xmm8, %xmm7
        movsd     %xmm7, -72(%rsp)
        movaps    %xmm5, %xmm7
        movsd     -72(%rsp), %xmm4
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm6
        addsd     %xmm4, %xmm7
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm8
        subsd     %xmm8, %xmm5
        addsd     %xmm4, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -72(%rsp), %xmm11
        movaps    %xmm11, %xmm2
        mulsd     %xmm11, %xmm2
        mulsd     %xmm11, %xmm6
        mulsd     %xmm2, %xmm3
        movsd     -64(%rsp), %xmm4
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm7
        addsd     1864+__datan2_ha_CoutTab(%rip), %xmm3
        subsd     %xmm11, %xmm7
        mulsd     %xmm2, %xmm3
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm9
        movsd     -64(%rsp), %xmm8
        addsd     1856+__datan2_ha_CoutTab(%rip), %xmm3
        subsd     %xmm8, %xmm9
        mulsd     %xmm2, %xmm3
        movsd     %xmm9, -72(%rsp)
        movsd     -72(%rsp), %xmm10
        addsd     1848+__datan2_ha_CoutTab(%rip), %xmm3
        subsd     %xmm10, %xmm11
        mulsd     %xmm2, %xmm3
        movsd     %xmm11, -64(%rsp)
        addsd     1840+__datan2_ha_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        addsd     1832+__datan2_ha_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        addsd     1824+__datan2_ha_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        mulsd     %xmm3, %xmm13
        movsd     -72(%rsp), %xmm2
        movsd     -64(%rsp), %xmm12
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm12, %xmm4
        movsd     -72(%rsp), %xmm14
        subsd     %xmm3, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm5
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        subsd     %xmm6, %xmm3
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movaps    %xmm6, %xmm12
        movaps    %xmm5, %xmm3
        mulsd     %xmm4, %xmm6
        mulsd     %xmm4, %xmm3
        mulsd     %xmm2, %xmm5
        mulsd     %xmm2, %xmm12
        addsd     %xmm3, %xmm6
        movaps    %xmm12, %xmm7
        movaps    %xmm12, %xmm8
        addsd     %xmm5, %xmm6
        addsd     %xmm2, %xmm7
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        subsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -56(%rsp), %xmm2
        subsd     %xmm2, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     -56(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        movsd     %xmm14, -56(%rsp)
        movq      -72(%rsp), %r11
        movsd     -56(%rsp), %xmm15
        movq      %r11, -40(%rsp)
        addsd     %xmm15, %xmm4
        movsd     -40(%rsp), %xmm8
        addsd     %xmm5, %xmm4
        movsd     %xmm4, -32(%rsp)
        movaps    %xmm8, %xmm4
        movaps    %xmm8, %xmm2
        addsd     (%rdi,%rsi,8), %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm3
        addsd     %xmm3, %xmm5
        movsd     %xmm5, -56(%rsp)
        movsd     -64(%rsp), %xmm6
        addsd     (%rdi,%rsi,8), %xmm6
        movsd     %xmm6, -64(%rsp)
        movsd     -56(%rsp), %xmm7
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -56(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -56(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movq      -72(%rsp), %r8
        movq      %r8, -40(%rsp)


        movsd     -56(%rsp), %xmm2
        movaps    %xmm1, %xmm3
        shrq      $56, %r8
        addsd     -32(%rsp), %xmm2
        shlb      $7, %dl
        addsd     8(%rdi,%rsi,8), %xmm2
        movb      %al, %sil
        andb      $127, %r8b
        shlb      $7, %sil
        movsd     %xmm2, -32(%rsp)
        orb       %sil, %r8b
        movb      %r8b, -33(%rsp)
        movsd     -40(%rsp), %xmm9
        movaps    %xmm9, %xmm5
        addsd     %xmm9, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        movb      -25(%rsp), %dil
        movb      %dil, %r9b
        shrb      $7, %dil
        subsd     %xmm4, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -72(%rsp), %xmm7
        movsd     -64(%rsp), %xmm6
        xorb      %dil, %al
        andb      $127, %r9b
        shlb      $7, %al
        addsd     %xmm6, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -64(%rsp), %xmm8
        addsd     %xmm8, %xmm1
        movsd     %xmm1, -64(%rsp)
        orb       %al, %r9b
        movsd     -56(%rsp), %xmm1
        movb      %r9b, -25(%rsp)
        subsd     %xmm1, %xmm9
        movsd     %xmm9, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        movsd     -56(%rsp), %xmm10
        addsd     %xmm10, %xmm11
        movsd     %xmm11, -56(%rsp)
        movq      -72(%rsp), %rax
        movsd     -56(%rsp), %xmm12
        movq      %rax, -40(%rsp)
        addsd     %xmm12, %xmm0
        movsd     -40(%rsp), %xmm13
        addsd     -32(%rsp), %xmm0
        movsd     %xmm0, -32(%rsp)
        addsd     %xmm0, %xmm13
        movsd     %xmm13, -24(%rsp)
        movb      -17(%rsp), %r10b
        andb      $127, %r10b
        orb       %dl, %r10b
        movb      %r10b, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_20:

        movsd     -48(%rsp), %xmm12
        movb      %al, %r8b
        movaps    %xmm12, %xmm7
        mulsd     2000+__datan2_ha_CoutTab(%rip), %xmm7
        shlb      $7, %r8b
        shlb      $7, %dl
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm13
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm7
        mulsd     %xmm2, %xmm13
        subsd     -48(%rsp), %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        subsd     %xmm9, %xmm10
        movsd     %xmm10, -72(%rsp)
        movsd     -72(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movsd     %xmm13, -72(%rsp)
        movsd     -72(%rsp), %xmm14
        subsd     %xmm2, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm4
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm12
        divsd     %xmm12, %xmm7
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm2
        mulsd     %xmm7, %xmm2
        movsd     -64(%rsp), %xmm14
        movsd     %xmm2, -64(%rsp)
        movsd     -64(%rsp), %xmm8
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -56(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -56(%rsp), %xmm9
        subsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -56(%rsp), %xmm11
        mulsd     %xmm11, %xmm12
        movsd     -56(%rsp), %xmm13
        subsd     %xmm12, %xmm4
        mulsd     %xmm13, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -64(%rsp), %xmm15
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm13
        subsd     %xmm15, %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -64(%rsp), %xmm7
        movq      -56(%rsp), %rsi
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm3
        movq      %rsi, -40(%rsp)
        movsd     -40(%rsp), %xmm8
        movaps    %xmm8, %xmm9
        addsd     1888+__datan2_ha_CoutTab(%rip), %xmm7
        mulsd     %xmm6, %xmm9
        mulsd     %xmm5, %xmm8
        mulsd     %xmm2, %xmm7
        movsd     -16(%rsp), %xmm2
        mulsd     %xmm2, %xmm2
        mulsd     %xmm3, %xmm7
        movsd     1872+__datan2_ha_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        mulsd     %xmm7, %xmm6
        mulsd     %xmm5, %xmm7
        addsd     1864+__datan2_ha_CoutTab(%rip), %xmm3
        addsd     %xmm7, %xmm6
        mulsd     %xmm2, %xmm3
        addsd     %xmm8, %xmm6
        addsd     1856+__datan2_ha_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        movaps    %xmm9, %xmm5
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        addsd     1848+__datan2_ha_CoutTab(%rip), %xmm3
        addsd     %xmm4, %xmm5
        mulsd     %xmm2, %xmm3
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     2000+__datan2_ha_CoutTab(%rip), %xmm5
        subsd     %xmm6, %xmm9
        addsd     1840+__datan2_ha_CoutTab(%rip), %xmm3
        addsd     %xmm4, %xmm9
        mulsd     %xmm2, %xmm3
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm11
        mulsd     %xmm11, %xmm5
        addsd     1832+__datan2_ha_CoutTab(%rip), %xmm3
        movsd     -64(%rsp), %xmm4
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm7
        mulsd     %xmm2, %xmm3
        subsd     %xmm11, %xmm7
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     -64(%rsp), %xmm6
        addsd     1824+__datan2_ha_CoutTab(%rip), %xmm3
        subsd     %xmm6, %xmm8
        mulsd     %xmm2, %xmm3
        movsd     %xmm8, -72(%rsp)
        movsd     -72(%rsp), %xmm10
        mulsd     %xmm3, %xmm13
        subsd     %xmm10, %xmm11
        movsd     %xmm11, -64(%rsp)
        movsd     -72(%rsp), %xmm2
        movsd     -64(%rsp), %xmm12
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm12, %xmm4
        movsd     -72(%rsp), %xmm14
        subsd     %xmm3, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm5
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        subsd     %xmm6, %xmm3
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movaps    %xmm6, %xmm12
        movaps    %xmm5, %xmm3
        mulsd     %xmm4, %xmm6
        mulsd     %xmm4, %xmm3
        mulsd     %xmm2, %xmm5
        mulsd     %xmm2, %xmm12
        addsd     %xmm3, %xmm6
        movaps    %xmm12, %xmm7
        movaps    %xmm12, %xmm8
        addsd     %xmm5, %xmm6
        addsd     %xmm2, %xmm7
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        subsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -56(%rsp), %xmm2
        subsd     %xmm2, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     -56(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        movsd     %xmm14, -56(%rsp)
        movq      -72(%rsp), %rdi
        movsd     -56(%rsp), %xmm15
        movq      %rdi, -40(%rsp)
        addsd     %xmm15, %xmm4
        shrq      $56, %rdi
        addsd     %xmm5, %xmm4
        andb      $127, %dil
        orb       %r8b, %dil
        movb      %dil, -33(%rsp)
        movsd     %xmm4, -32(%rsp)
        movaps    %xmm1, %xmm4
        movsd     -40(%rsp), %xmm7
        movaps    %xmm7, %xmm2
        addsd     %xmm7, %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        movb      -25(%rsp), %r9b
        movb      %r9b, %r10b
        shrb      $7, %r9b
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm3
        xorb      %r9b, %al
        andb      $127, %r10b
        shlb      $7, %al
        addsd     %xmm3, %xmm5
        movsd     %xmm5, -56(%rsp)
        movsd     -64(%rsp), %xmm6
        addsd     %xmm6, %xmm1
        movsd     %xmm1, -64(%rsp)
        orb       %al, %r10b
        movsd     -56(%rsp), %xmm1
        movb      %r10b, -25(%rsp)
        subsd     %xmm1, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm1
        addsd     %xmm1, %xmm2
        movsd     %xmm2, -56(%rsp)
        movq      -72(%rsp), %rax
        movsd     -56(%rsp), %xmm3
        movq      %rax, -40(%rsp)
        addsd     %xmm3, %xmm0
        movsd     -40(%rsp), %xmm4
        addsd     -32(%rsp), %xmm0
        movsd     %xmm0, -32(%rsp)
        addsd     %xmm0, %xmm4
        movsd     %xmm4, -24(%rsp)
        movb      -17(%rsp), %r11b
        andb      $127, %r11b
        orb       %dl, %r11b
        movb      %r11b, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_21:

        cmpl      $74, %r8d
        jge       .LBL_2_53


        movb      %dil, -41(%rsp)
        divsd     -48(%rsp), %xmm1
        movsd     1928+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        subsd     %xmm1, %xmm0
        addsd     1920+__datan2_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_24:

        testb     %al, %al
        jne       .LBL_2_35


        movb      %dil, -41(%rsp)
        movb      %sil, -33(%rsp)
        movsd     -48(%rsp), %xmm2
        divsd     -40(%rsp), %xmm2
        movsd     %xmm2, -24(%rsp)
        movzwl    -18(%rsp), %eax
        testl     $32752, %eax
        je        .LBL_2_27


        movsd     1888+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -72(%rsp)
        movsd     -72(%rsp), %xmm1
        mulsd     %xmm1, %xmm2
        movsd     %xmm2, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_27:

        mulsd     %xmm2, %xmm2
        shlb      $7, %dl
        movsd     %xmm2, -72(%rsp)
        movsd     -72(%rsp), %xmm0
        addsd     -24(%rsp), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_29:

        testl     %r9d, %r9d
        jne       .LBL_2_53


        testl     $1048575, -44(%rsp)
        jne       .LBL_2_53
        jmp       .LBL_2_57

.LBL_2_31:

        jne       .LBL_2_53

.LBL_2_33:

        testb     %al, %al
        jne       .LBL_2_35

.LBL_2_34:

        shlb      $7, %dl
        movq      1976+__datan2_ha_CoutTab(%rip), %rax
        movq      %rax, -24(%rsp)
        shrq      $56, %rax
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_35:

        movsd     1936+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        addsd     1944+__datan2_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)

.LBL_2_36:

        xorl      %eax, %eax
        ret

.LBL_2_37:

        movsd     1984+__datan2_ha_CoutTab(%rip), %xmm3
        movl      $-1022, %r9d
        mulsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)
        jmp       .LBL_2_18

.LBL_2_38:

        cmpl      $2047, %r9d
        je        .LBL_2_49

.LBL_2_39:

        testl     $1048575, -36(%rsp)
        jne       .LBL_2_41


        cmpl      $0, -40(%rsp)
        je        .LBL_2_42

.LBL_2_41:

        addsd     %xmm1, %xmm2
        movsd     %xmm2, (%rcx)
        jmp       .LBL_2_36

.LBL_2_42:

        cmpl      $2047, %r9d
        je        .LBL_2_46


        testb     %al, %al
        je        .LBL_2_34
        jmp       .LBL_2_35

.LBL_2_46:

        testb     %al, %al
        jne       .LBL_2_48


        movsd     1904+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        addsd     1912+__datan2_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_48:

        movsd     1952+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        addsd     1960+__datan2_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_49:

        testl     $1048575, -44(%rsp)
        jne       .LBL_2_41


        cmpl      $0, -48(%rsp)
        jne       .LBL_2_41


        cmpl      $2047, %r8d
        je        .LBL_2_39

.LBL_2_53:

        movsd     1920+__datan2_ha_CoutTab(%rip), %xmm0
        shlb      $7, %dl
        addsd     1928+__datan2_ha_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %dl, %al
        movb      %al, -17(%rsp)
        movq      -24(%rsp), %rdx
        movq      %rdx, (%rcx)
        jmp       .LBL_2_36

.LBL_2_57:

        cmpl      $0, -48(%rsp)
        jne       .LBL_2_53
        jmp       .LBL_2_33
	.align    16,0x90

	.cfi_endproc

	.type	__svml_datan2_ha_cout_rare_internal,@function
	.size	__svml_datan2_ha_cout_rare_internal,.-__svml_datan2_ha_cout_rare_internal
..LN__svml_datan2_ha_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_datan2_ha_data_internal:
	.long	0
	.long	1072693248
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	0
	.long	0
	.long	0
	.long	1073217536
	.long	3531732635
	.long	1072657163
	.long	2062601149
	.long	1013974920
	.long	4294967295
	.long	4294967295
	.long	0
	.long	1072693248
	.long	1413754136
	.long	1072243195
	.long	856972295
	.long	1015129638
	.long	4294967295
	.long	4294967295
	.long	0
	.long	1071644672
	.long	90291023
	.long	1071492199
	.long	573531618
	.long	1014639487
	.long	4294967295
	.long	4294967295
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	4294967295
	.long	4294967295
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	90291023
	.long	1071492199
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	573531618
	.long	1014639487
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	1413754136
	.long	1072243195
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	856972295
	.long	1015129638
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	3531732635
	.long	1072657163
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	2062601149
	.long	1013974920
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
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
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	0
	.long	1017226816
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	4160749568
	.long	4294967295
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1071382528
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1072889856
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1073971200
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	0
	.long	1072037888
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
	.long	4293918720
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
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
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
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
	.long	133169152
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
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	3866310424
	.long	1066132731
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	529668190
	.long	3214953687
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1493047753
	.long	1067887957
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	1554070819
	.long	3215629941
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	3992437651
	.long	1068372721
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	845965549
	.long	3216052365
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	3073500986
	.long	1068740914
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	426211919
	.long	3216459217
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	435789718
	.long	1069314503
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2453936673
	.long	3217180964
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	2576977731
	.long	1070176665
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
	.long	1431655762
	.long	3218429269
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
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	2150629376
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4258267136
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
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
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	0
	.long	4294967295
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
	.long	1005584384
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
	.type	__svml_datan2_ha_data_internal,@object
	.size	__svml_datan2_ha_data_internal,3328
	.align 32
__datan2_ha_CoutTab:
	.long	3892314112
	.long	1069799150
	.long	2332892550
	.long	1039715405
	.long	1342177280
	.long	1070305495
	.long	270726690
	.long	1041535749
	.long	939524096
	.long	1070817911
	.long	2253973841
	.long	3188654726
	.long	3221225472
	.long	1071277294
	.long	3853927037
	.long	1043226911
	.long	2818572288
	.long	1071767563
	.long	2677759107
	.long	1044314101
	.long	3355443200
	.long	1072103591
	.long	1636578514
	.long	3191094734
	.long	1476395008
	.long	1072475260
	.long	1864703685
	.long	3188646936
	.long	805306368
	.long	1072747407
	.long	192551812
	.long	3192726267
	.long	2013265920
	.long	1072892781
	.long	2240369452
	.long	1043768538
	.long	0
	.long	1072999953
	.long	3665168337
	.long	3192705970
	.long	402653184
	.long	1073084787
	.long	1227953434
	.long	3192313277
	.long	2013265920
	.long	1073142981
	.long	3853283127
	.long	1045277487
	.long	805306368
	.long	1073187261
	.long	1676192264
	.long	3192868861
	.long	134217728
	.long	1073217000
	.long	4290763938
	.long	1042034855
	.long	671088640
	.long	1073239386
	.long	994303084
	.long	3189643768
	.long	402653184
	.long	1073254338
	.long	1878067156
	.long	1042652475
	.long	1610612736
	.long	1073265562
	.long	670314820
	.long	1045138554
	.long	3221225472
	.long	1073273048
	.long	691126919
	.long	3189987794
	.long	3489660928
	.long	1073278664
	.long	1618990832
	.long	3188194509
	.long	1207959552
	.long	1073282409
	.long	2198872939
	.long	1044806069
	.long	3489660928
	.long	1073285217
	.long	2633982383
	.long	1042307894
	.long	939524096
	.long	1073287090
	.long	1059367786
	.long	3189114230
	.long	2281701376
	.long	1073288494
	.long	3158525533
	.long	1044484961
	.long	3221225472
	.long	1073289430
	.long	286581777
	.long	1044893263
	.long	4026531840
	.long	1073290132
	.long	2000245215
	.long	3191647611
	.long	134217728
	.long	1073290601
	.long	4205071590
	.long	1045035927
	.long	536870912
	.long	1073290952
	.long	2334392229
	.long	1043447393
	.long	805306368
	.long	1073291186
	.long	2281458177
	.long	3188885569
	.long	3087007744
	.long	1073291361
	.long	691611507
	.long	1044733832
	.long	3221225472
	.long	1073291478
	.long	1816229550
	.long	1044363390
	.long	2281701376
	.long	1073291566
	.long	1993843750
	.long	3189837440
	.long	134217728
	.long	1073291625
	.long	3654754496
	.long	1044970837
	.long	4026531840
	.long	1073291668
	.long	3224300229
	.long	3191935390
	.long	805306368
	.long	1073291698
	.long	2988777976
	.long	3188950659
	.long	536870912
	.long	1073291720
	.long	1030371341
	.long	1043402665
	.long	3221225472
	.long	1073291734
	.long	1524463765
	.long	1044361356
	.long	3087007744
	.long	1073291745
	.long	2754295320
	.long	1044731036
	.long	134217728
	.long	1073291753
	.long	3099629057
	.long	1044970710
	.long	2281701376
	.long	1073291758
	.long	962914160
	.long	3189838838
	.long	805306368
	.long	1073291762
	.long	3543908206
	.long	3188950786
	.long	4026531840
	.long	1073291764
	.long	1849909620
	.long	3191935434
	.long	3221225472
	.long	1073291766
	.long	1641333636
	.long	1044361352
	.long	536870912
	.long	1073291768
	.long	1373968792
	.long	1043402654
	.long	134217728
	.long	1073291769
	.long	2033191599
	.long	1044970710
	.long	3087007744
	.long	1073291769
	.long	4117947437
	.long	1044731035
	.long	805306368
	.long	1073291770
	.long	315378368
	.long	3188950787
	.long	2281701376
	.long	1073291770
	.long	2428571750
	.long	3189838838
	.long	3221225472
	.long	1073291770
	.long	1608007466
	.long	1044361352
	.long	4026531840
	.long	1073291770
	.long	1895711420
	.long	3191935434
	.long	134217728
	.long	1073291771
	.long	2031108713
	.long	1044970710
	.long	536870912
	.long	1073291771
	.long	1362518342
	.long	1043402654
	.long	805306368
	.long	1073291771
	.long	317461253
	.long	3188950787
	.long	939524096
	.long	1073291771
	.long	4117231784
	.long	1044731035
	.long	1073741824
	.long	1073291771
	.long	1607942376
	.long	1044361352
	.long	1207959552
	.long	1073291771
	.long	2428929577
	.long	3189838838
	.long	1207959552
	.long	1073291771
	.long	2031104645
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1895722602
	.long	3191935434
	.long	1342177280
	.long	1073291771
	.long	317465322
	.long	3188950787
	.long	1342177280
	.long	1073291771
	.long	1362515546
	.long	1043402654
	.long	1342177280
	.long	1073291771
	.long	1607942248
	.long	1044361352
	.long	1342177280
	.long	1073291771
	.long	4117231610
	.long	1044731035
	.long	1342177280
	.long	1073291771
	.long	2031104637
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1540251232
	.long	1045150466
	.long	1342177280
	.long	1073291771
	.long	2644671394
	.long	1045270303
	.long	1342177280
	.long	1073291771
	.long	2399244691
	.long	1045360181
	.long	1342177280
	.long	1073291771
	.long	803971124
	.long	1045420100
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192879152
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192849193
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192826724
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192811744
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192800509
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192793019
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192787402
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192783657
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192780848
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192778976
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192777572
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192776635
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192775933
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192775465
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192775114
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774880
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774704
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774587
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774500
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774441
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774397
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774368
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774346
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774331
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774320
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774313
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774308
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774304
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774301
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774299
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774298
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774297
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1466225875
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1343512524
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1251477510
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1190120835
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1144103328
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1113424990
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1090416237
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1075077068
	.long	3192774295
	.long	1431655765
	.long	3218429269
	.long	2576978363
	.long	1070176665
	.long	2453154343
	.long	3217180964
	.long	4189149139
	.long	1069314502
	.long	1775019125
	.long	3216459198
	.long	273199057
	.long	1068739452
	.long	874748308
	.long	3215993277
	.long	0
	.long	1069547520
	.long	0
	.long	1072693248
	.long	0
	.long	1073741824
	.long	1413754136
	.long	1072243195
	.long	856972295
	.long	1015129638
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	1413754136
	.long	1074340347
	.long	856972295
	.long	1017226790
	.long	2134057426
	.long	1073928572
	.long	1285458442
	.long	1016756537
	.long	0
	.long	3220176896
	.long	0
	.long	0
	.long	0
	.long	2144337920
	.long	0
	.long	1048576
	.long	33554432
	.long	1101004800
	.type	__datan2_ha_CoutTab,@object
	.size	__datan2_ha_CoutTab,2008
	.align 8
.L_2il0floatpacket.40:
	.long	0xffffffff,0xffffffff
	.type	.L_2il0floatpacket.40,@object
	.size	.L_2il0floatpacket.40,8
      	.section        .note.GNU-stack,"",@progbits
