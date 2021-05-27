OPTION	DOTNAME
.text$	SEGMENT ALIGN(256) 'CODE'

PUBLIC	x25519_fe51_mul

ALIGN	32
x25519_fe51_mul	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe51_mul::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	lea	rsp,QWORD PTR[((-40))+rsp]

$L$fe51_mul_body::

	mov	rax,QWORD PTR[rsi]
	mov	r11,QWORD PTR[rdx]
	mov	r12,QWORD PTR[8+rdx]
	mov	r13,QWORD PTR[16+rdx]
	mov	rbp,QWORD PTR[24+rdx]
	mov	r14,QWORD PTR[32+rdx]

	mov	QWORD PTR[32+rsp],rdi
	mov	rdi,rax
	mul	r11
	mov	QWORD PTR[rsp],r11
	mov	rbx,rax
	mov	rax,rdi
	mov	rcx,rdx
	mul	r12
	mov	QWORD PTR[8+rsp],r12
	mov	r8,rax
	mov	rax,rdi
	lea	r15,QWORD PTR[r14*8+r14]
	mov	r9,rdx
	mul	r13
	mov	QWORD PTR[16+rsp],r13
	mov	r10,rax
	mov	rax,rdi
	lea	rdi,QWORD PTR[r15*2+r14]
	mov	r11,rdx
	mul	rbp
	mov	r12,rax
	mov	rax,QWORD PTR[rsi]
	mov	r13,rdx
	mul	r14
	mov	r14,rax
	mov	rax,QWORD PTR[8+rsi]
	mov	r15,rdx

	mul	rdi
	add	rbx,rax
	mov	rax,QWORD PTR[16+rsi]
	adc	rcx,rdx
	mul	rdi
	add	r8,rax
	mov	rax,QWORD PTR[24+rsi]
	adc	r9,rdx
	mul	rdi
	add	r10,rax
	mov	rax,QWORD PTR[32+rsi]
	adc	r11,rdx
	mul	rdi
	imul	rdi,rbp,19
	add	r12,rax
	mov	rax,QWORD PTR[8+rsi]
	adc	r13,rdx
	mul	rbp
	mov	rbp,QWORD PTR[16+rsp]
	add	r14,rax
	mov	rax,QWORD PTR[16+rsi]
	adc	r15,rdx

	mul	rdi
	add	rbx,rax
	mov	rax,QWORD PTR[24+rsi]
	adc	rcx,rdx
	mul	rdi
	add	r8,rax
	mov	rax,QWORD PTR[32+rsi]
	adc	r9,rdx
	mul	rdi
	imul	rdi,rbp,19
	add	r10,rax
	mov	rax,QWORD PTR[8+rsi]
	adc	r11,rdx
	mul	rbp
	add	r12,rax
	mov	rax,QWORD PTR[16+rsi]
	adc	r13,rdx
	mul	rbp
	mov	rbp,QWORD PTR[8+rsp]
	add	r14,rax
	mov	rax,QWORD PTR[24+rsi]
	adc	r15,rdx

	mul	rdi
	add	rbx,rax
	mov	rax,QWORD PTR[32+rsi]
	adc	rcx,rdx
	mul	rdi
	add	r8,rax
	mov	rax,QWORD PTR[8+rsi]
	adc	r9,rdx
	mul	rbp
	imul	rdi,rbp,19
	add	r10,rax
	mov	rax,QWORD PTR[16+rsi]
	adc	r11,rdx
	mul	rbp
	add	r12,rax
	mov	rax,QWORD PTR[24+rsi]
	adc	r13,rdx
	mul	rbp
	mov	rbp,QWORD PTR[rsp]
	add	r14,rax
	mov	rax,QWORD PTR[32+rsi]
	adc	r15,rdx

	mul	rdi
	add	rbx,rax
	mov	rax,QWORD PTR[8+rsi]
	adc	rcx,rdx
	mul	rbp
	add	r8,rax
	mov	rax,QWORD PTR[16+rsi]
	adc	r9,rdx
	mul	rbp
	add	r10,rax
	mov	rax,QWORD PTR[24+rsi]
	adc	r11,rdx
	mul	rbp
	add	r12,rax
	mov	rax,QWORD PTR[32+rsi]
	adc	r13,rdx
	mul	rbp
	add	r14,rax
	adc	r15,rdx

	mov	rdi,QWORD PTR[32+rsp]
	jmp	$L$reduce51
$L$fe51_mul_epilogue::

$L$SEH_end_x25519_fe51_mul::
x25519_fe51_mul	ENDP

PUBLIC	x25519_fe51_sqr

ALIGN	32
x25519_fe51_sqr	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe51_sqr::
	mov	rdi,rcx
	mov	rsi,rdx



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	lea	rsp,QWORD PTR[((-40))+rsp]

$L$fe51_sqr_body::

	mov	rax,QWORD PTR[rsi]
	mov	r15,QWORD PTR[16+rsi]
	mov	rbp,QWORD PTR[32+rsi]

	mov	QWORD PTR[32+rsp],rdi
	lea	r14,QWORD PTR[rax*1+rax]
	mul	rax
	mov	rbx,rax
	mov	rax,QWORD PTR[8+rsi]
	mov	rcx,rdx
	mul	r14
	mov	r8,rax
	mov	rax,r15
	mov	QWORD PTR[rsp],r15
	mov	r9,rdx
	mul	r14
	mov	r10,rax
	mov	rax,QWORD PTR[24+rsi]
	mov	r11,rdx
	imul	rdi,rbp,19
	mul	r14
	mov	r12,rax
	mov	rax,rbp
	mov	r13,rdx
	mul	r14
	mov	r14,rax
	mov	rax,rbp
	mov	r15,rdx

	mul	rdi
	add	r12,rax
	mov	rax,QWORD PTR[8+rsi]
	adc	r13,rdx

	mov	rsi,QWORD PTR[24+rsi]
	lea	rbp,QWORD PTR[rax*1+rax]
	mul	rax
	add	r10,rax
	mov	rax,QWORD PTR[rsp]
	adc	r11,rdx
	mul	rbp
	add	r12,rax
	mov	rax,rbp
	adc	r13,rdx
	mul	rsi
	add	r14,rax
	mov	rax,rbp
	adc	r15,rdx
	imul	rbp,rsi,19
	mul	rdi
	add	rbx,rax
	lea	rax,QWORD PTR[rsi*1+rsi]
	adc	rcx,rdx

	mul	rdi
	add	r10,rax
	mov	rax,rsi
	adc	r11,rdx
	mul	rbp
	add	r8,rax
	mov	rax,QWORD PTR[rsp]
	adc	r9,rdx

	lea	rsi,QWORD PTR[rax*1+rax]
	mul	rax
	add	r14,rax
	mov	rax,rbp
	adc	r15,rdx
	mul	rsi
	add	rbx,rax
	mov	rax,rsi
	adc	rcx,rdx
	mul	rdi
	add	r8,rax
	adc	r9,rdx

	mov	rdi,QWORD PTR[32+rsp]
	jmp	$L$reduce51

ALIGN	32
$L$reduce51::
	mov	rbp,07ffffffffffffh

	mov	rdx,r10
	shr	r10,51
	shl	r11,13
	and	rdx,rbp
	or	r11,r10
	add	r12,r11
	adc	r13,0

	mov	rax,rbx
	shr	rbx,51
	shl	rcx,13
	and	rax,rbp
	or	rcx,rbx
	add	r8,rcx
	adc	r9,0

	mov	rbx,r12
	shr	r12,51
	shl	r13,13
	and	rbx,rbp
	or	r13,r12
	add	r14,r13
	adc	r15,0

	mov	rcx,r8
	shr	r8,51
	shl	r9,13
	and	rcx,rbp
	or	r9,r8
	add	rdx,r9

	mov	r10,r14
	shr	r14,51
	shl	r15,13
	and	r10,rbp
	or	r15,r14

	lea	r14,QWORD PTR[r15*8+r15]
	lea	r15,QWORD PTR[r14*2+r15]
	add	rax,r15

	mov	r8,rdx
	and	rdx,rbp
	shr	r8,51
	add	rbx,r8

	mov	r9,rax
	and	rax,rbp
	shr	r9,51
	add	rcx,r9

	mov	QWORD PTR[rdi],rax
	mov	QWORD PTR[8+rdi],rcx
	mov	QWORD PTR[16+rdi],rdx
	mov	QWORD PTR[24+rdi],rbx
	mov	QWORD PTR[32+rdi],r10

	mov	r15,QWORD PTR[40+rsp]

	mov	r14,QWORD PTR[48+rsp]

	mov	r13,QWORD PTR[56+rsp]

	mov	r12,QWORD PTR[64+rsp]

	mov	rbx,QWORD PTR[72+rsp]

	mov	rbp,QWORD PTR[80+rsp]

	lea	rsp,QWORD PTR[88+rsp]

$L$fe51_sqr_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe51_sqr::
x25519_fe51_sqr	ENDP

PUBLIC	x25519_fe51_mul121666

ALIGN	32
x25519_fe51_mul121666	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe51_mul121666::
	mov	rdi,rcx
	mov	rsi,rdx



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	lea	rsp,QWORD PTR[((-40))+rsp]

$L$fe51_mul121666_body::
	mov	eax,121666

	mul	QWORD PTR[rsi]
	mov	rbx,rax
	mov	eax,121666
	mov	rcx,rdx
	mul	QWORD PTR[8+rsi]
	mov	r8,rax
	mov	eax,121666
	mov	r9,rdx
	mul	QWORD PTR[16+rsi]
	mov	r10,rax
	mov	eax,121666
	mov	r11,rdx
	mul	QWORD PTR[24+rsi]
	mov	r12,rax
	mov	eax,121666
	mov	r13,rdx
	mul	QWORD PTR[32+rsi]
	mov	r14,rax
	mov	r15,rdx

	jmp	$L$reduce51
$L$fe51_mul121666_epilogue::

$L$SEH_end_x25519_fe51_mul121666::
x25519_fe51_mul121666	ENDP
EXTERN	OPENSSL_ia32cap_P:NEAR
PUBLIC	x25519_fe64_eligible

ALIGN	32
x25519_fe64_eligible	PROC PUBLIC

	mov	ecx,DWORD PTR[((OPENSSL_ia32cap_P+8))]
	xor	eax,eax
	and	ecx,080100h
	cmp	ecx,080100h
	cmove	eax,ecx
	DB	0F3h,0C3h		;repret

x25519_fe64_eligible	ENDP

PUBLIC	x25519_fe64_mul

ALIGN	32
x25519_fe64_mul	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_mul::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	push	rdi

	lea	rsp,QWORD PTR[((-16))+rsp]

$L$fe64_mul_body::

	mov	rax,rdx
	mov	rbp,QWORD PTR[rdx]
	mov	rdx,QWORD PTR[rsi]
	mov	rcx,QWORD PTR[8+rax]
	mov	r14,QWORD PTR[16+rax]
	mov	r15,QWORD PTR[24+rax]

	mulx	rax,r8,rbp
	xor	edi,edi
	mulx	rbx,r9,rcx
	adcx	r9,rax
	mulx	rax,r10,r14
	adcx	r10,rbx
	mulx	r12,r11,r15
	mov	rdx,QWORD PTR[8+rsi]
	adcx	r11,rax
	mov	QWORD PTR[rsp],r14
	adcx	r12,rdi

	mulx	rbx,rax,rbp
	adox	r9,rax
	adcx	r10,rbx
	mulx	rbx,rax,rcx
	adox	r10,rax
	adcx	r11,rbx
	mulx	rbx,rax,r14
	adox	r11,rax
	adcx	r12,rbx
	mulx	r13,rax,r15
	mov	rdx,QWORD PTR[16+rsi]
	adox	r12,rax
	adcx	r13,rdi
	adox	r13,rdi

	mulx	rbx,rax,rbp
	adcx	r10,rax
	adox	r11,rbx
	mulx	rbx,rax,rcx
	adcx	r11,rax
	adox	r12,rbx
	mulx	rbx,rax,r14
	adcx	r12,rax
	adox	r13,rbx
	mulx	r14,rax,r15
	mov	rdx,QWORD PTR[24+rsi]
	adcx	r13,rax
	adox	r14,rdi
	adcx	r14,rdi

	mulx	rbx,rax,rbp
	adox	r11,rax
	adcx	r12,rbx
	mulx	rbx,rax,rcx
	adox	r12,rax
	adcx	r13,rbx
	mulx	rbx,rax,QWORD PTR[rsp]
	adox	r13,rax
	adcx	r14,rbx
	mulx	r15,rax,r15
	mov	edx,38
	adox	r14,rax
	adcx	r15,rdi
	adox	r15,rdi

	jmp	$L$reduce64
$L$fe64_mul_epilogue::

$L$SEH_end_x25519_fe64_mul::
x25519_fe64_mul	ENDP

PUBLIC	x25519_fe64_sqr

ALIGN	32
x25519_fe64_sqr	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_sqr::
	mov	rdi,rcx
	mov	rsi,rdx



	push	rbp

	push	rbx

	push	r12

	push	r13

	push	r14

	push	r15

	push	rdi

	lea	rsp,QWORD PTR[((-16))+rsp]

$L$fe64_sqr_body::

	mov	rdx,QWORD PTR[rsi]
	mov	rcx,QWORD PTR[8+rsi]
	mov	rbp,QWORD PTR[16+rsi]
	mov	rsi,QWORD PTR[24+rsi]


	mulx	r15,r8,rdx
	mulx	rax,r9,rcx
	xor	edi,edi
	mulx	rbx,r10,rbp
	adcx	r10,rax
	mulx	r12,r11,rsi
	mov	rdx,rcx
	adcx	r11,rbx
	adcx	r12,rdi


	mulx	rbx,rax,rbp
	adox	r11,rax
	adcx	r12,rbx
	mulx	r13,rax,rsi
	mov	rdx,rbp
	adox	r12,rax
	adcx	r13,rdi


	mulx	r14,rax,rsi
	mov	rdx,rcx
	adox	r13,rax
	adcx	r14,rdi
	adox	r14,rdi

	adcx	r9,r9
	adox	r9,r15
	adcx	r10,r10
	mulx	rbx,rax,rdx
	mov	rdx,rbp
	adcx	r11,r11
	adox	r10,rax
	adcx	r12,r12
	adox	r11,rbx
	mulx	rbx,rax,rdx
	mov	rdx,rsi
	adcx	r13,r13
	adox	r12,rax
	adcx	r14,r14
	adox	r13,rbx
	mulx	r15,rax,rdx
	mov	edx,38
	adox	r14,rax
	adcx	r15,rdi
	adox	r15,rdi
	jmp	$L$reduce64

ALIGN	32
$L$reduce64::
	mulx	rbx,rax,r12
	adcx	r8,rax
	adox	r9,rbx
	mulx	rbx,rax,r13
	adcx	r9,rax
	adox	r10,rbx
	mulx	rbx,rax,r14
	adcx	r10,rax
	adox	r11,rbx
	mulx	r12,rax,r15
	adcx	r11,rax
	adox	r12,rdi
	adcx	r12,rdi

	mov	rdi,QWORD PTR[16+rsp]
	imul	r12,rdx

	add	r8,r12
	adc	r9,0
	adc	r10,0
	adc	r11,0

	sbb	rax,rax
	and	rax,38

	add	r8,rax
	mov	QWORD PTR[8+rdi],r9
	mov	QWORD PTR[16+rdi],r10
	mov	QWORD PTR[24+rdi],r11
	mov	QWORD PTR[rdi],r8

	mov	r15,QWORD PTR[24+rsp]

	mov	r14,QWORD PTR[32+rsp]

	mov	r13,QWORD PTR[40+rsp]

	mov	r12,QWORD PTR[48+rsp]

	mov	rbx,QWORD PTR[56+rsp]

	mov	rbp,QWORD PTR[64+rsp]

	lea	rsp,QWORD PTR[72+rsp]

$L$fe64_sqr_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe64_sqr::
x25519_fe64_sqr	ENDP

PUBLIC	x25519_fe64_mul121666

ALIGN	32
x25519_fe64_mul121666	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_mul121666::
	mov	rdi,rcx
	mov	rsi,rdx


$L$fe64_mul121666_body::

	mov	edx,121666
	mulx	rcx,r8,QWORD PTR[rsi]
	mulx	rax,r9,QWORD PTR[8+rsi]
	add	r9,rcx
	mulx	rcx,r10,QWORD PTR[16+rsi]
	adc	r10,rax
	mulx	rax,r11,QWORD PTR[24+rsi]
	adc	r11,rcx
	adc	rax,0

	imul	rax,rax,38

	add	r8,rax
	adc	r9,0
	adc	r10,0
	adc	r11,0

	sbb	rax,rax
	and	rax,38

	add	r8,rax
	mov	QWORD PTR[8+rdi],r9
	mov	QWORD PTR[16+rdi],r10
	mov	QWORD PTR[24+rdi],r11
	mov	QWORD PTR[rdi],r8

$L$fe64_mul121666_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe64_mul121666::
x25519_fe64_mul121666	ENDP

PUBLIC	x25519_fe64_add

ALIGN	32
x25519_fe64_add	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_add::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8


$L$fe64_add_body::

	mov	r8,QWORD PTR[rsi]
	mov	r9,QWORD PTR[8+rsi]
	mov	r10,QWORD PTR[16+rsi]
	mov	r11,QWORD PTR[24+rsi]

	add	r8,QWORD PTR[rdx]
	adc	r9,QWORD PTR[8+rdx]
	adc	r10,QWORD PTR[16+rdx]
	adc	r11,QWORD PTR[24+rdx]

	sbb	rax,rax
	and	rax,38

	add	r8,rax
	adc	r9,0
	adc	r10,0
	mov	QWORD PTR[8+rdi],r9
	adc	r11,0
	mov	QWORD PTR[16+rdi],r10
	sbb	rax,rax
	mov	QWORD PTR[24+rdi],r11
	and	rax,38

	add	r8,rax
	mov	QWORD PTR[rdi],r8

$L$fe64_add_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe64_add::
x25519_fe64_add	ENDP

PUBLIC	x25519_fe64_sub

ALIGN	32
x25519_fe64_sub	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_sub::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8


$L$fe64_sub_body::

	mov	r8,QWORD PTR[rsi]
	mov	r9,QWORD PTR[8+rsi]
	mov	r10,QWORD PTR[16+rsi]
	mov	r11,QWORD PTR[24+rsi]

	sub	r8,QWORD PTR[rdx]
	sbb	r9,QWORD PTR[8+rdx]
	sbb	r10,QWORD PTR[16+rdx]
	sbb	r11,QWORD PTR[24+rdx]

	sbb	rax,rax
	and	rax,38

	sub	r8,rax
	sbb	r9,0
	sbb	r10,0
	mov	QWORD PTR[8+rdi],r9
	sbb	r11,0
	mov	QWORD PTR[16+rdi],r10
	sbb	rax,rax
	mov	QWORD PTR[24+rdi],r11
	and	rax,38

	sub	r8,rax
	mov	QWORD PTR[rdi],r8

$L$fe64_sub_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe64_sub::
x25519_fe64_sub	ENDP

PUBLIC	x25519_fe64_tobytes

ALIGN	32
x25519_fe64_tobytes	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_x25519_fe64_tobytes::
	mov	rdi,rcx
	mov	rsi,rdx


$L$fe64_to_body::

	mov	r8,QWORD PTR[rsi]
	mov	r9,QWORD PTR[8+rsi]
	mov	r10,QWORD PTR[16+rsi]
	mov	r11,QWORD PTR[24+rsi]


	lea	rax,QWORD PTR[r11*1+r11]
	sar	r11,63
	shr	rax,1
	and	r11,19
	add	r11,19

	add	r8,r11
	adc	r9,0
	adc	r10,0
	adc	rax,0

	lea	r11,QWORD PTR[rax*1+rax]
	sar	rax,63
	shr	r11,1
	not	rax
	and	rax,19

	sub	r8,rax
	sbb	r9,0
	sbb	r10,0
	sbb	r11,0

	mov	QWORD PTR[rdi],r8
	mov	QWORD PTR[8+rdi],r9
	mov	QWORD PTR[16+rdi],r10
	mov	QWORD PTR[24+rdi],r11

$L$fe64_to_epilogue::
	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_x25519_fe64_tobytes::
x25519_fe64_tobytes	ENDP
DB	88,50,53,53,49,57,32,112,114,105,109,105,116,105,118,101
DB	115,32,102,111,114,32,120,56,54,95,54,52,44,32,67,82
DB	89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112
DB	114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0
EXTERN	__imp_RtlVirtualUnwind:NEAR


ALIGN	16
short_handler	PROC PRIVATE
	push	rsi
	push	rdi
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	pushfq
	sub	rsp,64

	mov	rax,QWORD PTR[120+r8]
	mov	rbx,QWORD PTR[248+r8]

	mov	rsi,QWORD PTR[8+r9]
	mov	r11,QWORD PTR[56+r9]

	mov	r10d,DWORD PTR[r11]
	lea	r10,QWORD PTR[r10*1+rsi]
	cmp	rbx,r10
	jb	$L$common_seh_tail

	mov	rax,QWORD PTR[152+r8]
	jmp	$L$common_seh_tail
short_handler	ENDP


ALIGN	16
full_handler	PROC PRIVATE
	push	rsi
	push	rdi
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
	pushfq
	sub	rsp,64

	mov	rax,QWORD PTR[120+r8]
	mov	rbx,QWORD PTR[248+r8]

	mov	rsi,QWORD PTR[8+r9]
	mov	r11,QWORD PTR[56+r9]

	mov	r10d,DWORD PTR[r11]
	lea	r10,QWORD PTR[r10*1+rsi]
	cmp	rbx,r10
	jb	$L$common_seh_tail

	mov	rax,QWORD PTR[152+r8]

	mov	r10d,DWORD PTR[4+r11]
	lea	r10,QWORD PTR[r10*1+rsi]
	cmp	rbx,r10
	jae	$L$common_seh_tail

	mov	r10d,DWORD PTR[8+r11]
	lea	rax,QWORD PTR[r10*1+rax]

	mov	rbp,QWORD PTR[((-8))+rax]
	mov	rbx,QWORD PTR[((-16))+rax]
	mov	r12,QWORD PTR[((-24))+rax]
	mov	r13,QWORD PTR[((-32))+rax]
	mov	r14,QWORD PTR[((-40))+rax]
	mov	r15,QWORD PTR[((-48))+rax]
	mov	QWORD PTR[144+r8],rbx
	mov	QWORD PTR[160+r8],rbp
	mov	QWORD PTR[216+r8],r12
	mov	QWORD PTR[224+r8],r13
	mov	QWORD PTR[232+r8],r14
	mov	QWORD PTR[240+r8],r15

$L$common_seh_tail::
	mov	rdi,QWORD PTR[8+rax]
	mov	rsi,QWORD PTR[16+rax]
	mov	QWORD PTR[152+r8],rax
	mov	QWORD PTR[168+r8],rsi
	mov	QWORD PTR[176+r8],rdi

	mov	rdi,QWORD PTR[40+r9]
	mov	rsi,r8
	mov	ecx,154
	DD	0a548f3fch

	mov	rsi,r9
	xor	rcx,rcx
	mov	rdx,QWORD PTR[8+rsi]
	mov	r8,QWORD PTR[rsi]
	mov	r9,QWORD PTR[16+rsi]
	mov	r10,QWORD PTR[40+rsi]
	lea	r11,QWORD PTR[56+rsi]
	lea	r12,QWORD PTR[24+rsi]
	mov	QWORD PTR[32+rsp],r10
	mov	QWORD PTR[40+rsp],r11
	mov	QWORD PTR[48+rsp],r12
	mov	QWORD PTR[56+rsp],rcx
	call	QWORD PTR[__imp_RtlVirtualUnwind]

	mov	eax,1
	add	rsp,64
	popfq
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	pop	rbx
	pop	rdi
	pop	rsi
	DB	0F3h,0C3h		;repret
full_handler	ENDP

.text$	ENDS
.pdata	SEGMENT READONLY ALIGN(4)
ALIGN	4
	DD	imagerel $L$SEH_begin_x25519_fe51_mul
	DD	imagerel $L$SEH_end_x25519_fe51_mul
	DD	imagerel $L$SEH_info_x25519_fe51_mul

	DD	imagerel $L$SEH_begin_x25519_fe51_sqr
	DD	imagerel $L$SEH_end_x25519_fe51_sqr
	DD	imagerel $L$SEH_info_x25519_fe51_sqr

	DD	imagerel $L$SEH_begin_x25519_fe51_mul121666
	DD	imagerel $L$SEH_end_x25519_fe51_mul121666
	DD	imagerel $L$SEH_info_x25519_fe51_mul121666
	DD	imagerel $L$SEH_begin_x25519_fe64_mul
	DD	imagerel $L$SEH_end_x25519_fe64_mul
	DD	imagerel $L$SEH_info_x25519_fe64_mul

	DD	imagerel $L$SEH_begin_x25519_fe64_sqr
	DD	imagerel $L$SEH_end_x25519_fe64_sqr
	DD	imagerel $L$SEH_info_x25519_fe64_sqr

	DD	imagerel $L$SEH_begin_x25519_fe64_mul121666
	DD	imagerel $L$SEH_end_x25519_fe64_mul121666
	DD	imagerel $L$SEH_info_x25519_fe64_mul121666

	DD	imagerel $L$SEH_begin_x25519_fe64_add
	DD	imagerel $L$SEH_end_x25519_fe64_add
	DD	imagerel $L$SEH_info_x25519_fe64_add

	DD	imagerel $L$SEH_begin_x25519_fe64_sub
	DD	imagerel $L$SEH_end_x25519_fe64_sub
	DD	imagerel $L$SEH_info_x25519_fe64_sub

	DD	imagerel $L$SEH_begin_x25519_fe64_tobytes
	DD	imagerel $L$SEH_end_x25519_fe64_tobytes
	DD	imagerel $L$SEH_info_x25519_fe64_tobytes
.pdata	ENDS
.xdata	SEGMENT READONLY ALIGN(8)
ALIGN	8
$L$SEH_info_x25519_fe51_mul::
DB	9,0,0,0
	DD	imagerel full_handler
	DD	imagerel $L$fe51_mul_body,imagerel $L$fe51_mul_epilogue
	DD	88,0
$L$SEH_info_x25519_fe51_sqr::
DB	9,0,0,0
	DD	imagerel full_handler
	DD	imagerel $L$fe51_sqr_body,imagerel $L$fe51_sqr_epilogue
	DD	88,0
$L$SEH_info_x25519_fe51_mul121666::
DB	9,0,0,0
	DD	imagerel full_handler
	DD	imagerel $L$fe51_mul121666_body,imagerel $L$fe51_mul121666_epilogue
	DD	88,0
$L$SEH_info_x25519_fe64_mul::
DB	9,0,0,0
	DD	imagerel full_handler
	DD	imagerel $L$fe64_mul_body,imagerel $L$fe64_mul_epilogue
	DD	72,0
$L$SEH_info_x25519_fe64_sqr::
DB	9,0,0,0
	DD	imagerel full_handler
	DD	imagerel $L$fe64_sqr_body,imagerel $L$fe64_sqr_epilogue
	DD	72,0
$L$SEH_info_x25519_fe64_mul121666::
DB	9,0,0,0
	DD	imagerel short_handler
	DD	imagerel $L$fe64_mul121666_body,imagerel $L$fe64_mul121666_epilogue
$L$SEH_info_x25519_fe64_add::
DB	9,0,0,0
	DD	imagerel short_handler
	DD	imagerel $L$fe64_add_body,imagerel $L$fe64_add_epilogue
$L$SEH_info_x25519_fe64_sub::
DB	9,0,0,0
	DD	imagerel short_handler
	DD	imagerel $L$fe64_sub_body,imagerel $L$fe64_sub_epilogue
$L$SEH_info_x25519_fe64_tobytes::
DB	9,0,0,0
	DD	imagerel short_handler
	DD	imagerel $L$fe64_to_body,imagerel $L$fe64_to_epilogue

.xdata	ENDS
END
