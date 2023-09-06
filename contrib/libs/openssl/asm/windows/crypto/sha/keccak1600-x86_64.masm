OPTION	DOTNAME
.text$	SEGMENT ALIGN(256) 'CODE'


ALIGN	32
__KeccakF1600	PROC PRIVATE

	mov	rax,QWORD PTR[60+rdi]
	mov	rbx,QWORD PTR[68+rdi]
	mov	rcx,QWORD PTR[76+rdi]
	mov	rdx,QWORD PTR[84+rdi]
	mov	rbp,QWORD PTR[92+rdi]
	jmp	$L$oop

ALIGN	32
$L$oop::
	mov	r8,QWORD PTR[((-100))+rdi]
	mov	r9,QWORD PTR[((-52))+rdi]
	mov	r10,QWORD PTR[((-4))+rdi]
	mov	r11,QWORD PTR[44+rdi]

	xor	rcx,QWORD PTR[((-84))+rdi]
	xor	rdx,QWORD PTR[((-76))+rdi]
	xor	rax,r8
	xor	rbx,QWORD PTR[((-92))+rdi]
	xor	rcx,QWORD PTR[((-44))+rdi]
	xor	rax,QWORD PTR[((-60))+rdi]
	mov	r12,rbp
	xor	rbp,QWORD PTR[((-68))+rdi]

	xor	rcx,r10
	xor	rax,QWORD PTR[((-20))+rdi]
	xor	rdx,QWORD PTR[((-36))+rdi]
	xor	rbx,r9
	xor	rbp,QWORD PTR[((-28))+rdi]

	xor	rcx,QWORD PTR[36+rdi]
	xor	rax,QWORD PTR[20+rdi]
	xor	rdx,QWORD PTR[4+rdi]
	xor	rbx,QWORD PTR[((-12))+rdi]
	xor	rbp,QWORD PTR[12+rdi]

	mov	r13,rcx
	rol	rcx,1
	xor	rcx,rax
	xor	rdx,r11

	rol	rax,1
	xor	rax,rdx
	xor	rbx,QWORD PTR[28+rdi]

	rol	rdx,1
	xor	rdx,rbx
	xor	rbp,QWORD PTR[52+rdi]

	rol	rbx,1
	xor	rbx,rbp

	rol	rbp,1
	xor	rbp,r13
	xor	r9,rcx
	xor	r10,rdx
	rol	r9,44
	xor	r11,rbp
	xor	r12,rax
	rol	r10,43
	xor	r8,rbx
	mov	r13,r9
	rol	r11,21
	or	r9,r10
	xor	r9,r8
	rol	r12,14

	xor	r9,QWORD PTR[r15]
	lea	r15,QWORD PTR[8+r15]

	mov	r14,r12
	and	r12,r11
	mov	QWORD PTR[((-100))+rsi],r9
	xor	r12,r10
	not	r10
	mov	QWORD PTR[((-84))+rsi],r12

	or	r10,r11
	mov	r12,QWORD PTR[76+rdi]
	xor	r10,r13
	mov	QWORD PTR[((-92))+rsi],r10

	and	r13,r8
	mov	r9,QWORD PTR[((-28))+rdi]
	xor	r13,r14
	mov	r10,QWORD PTR[((-20))+rdi]
	mov	QWORD PTR[((-68))+rsi],r13

	or	r14,r8
	mov	r8,QWORD PTR[((-76))+rdi]
	xor	r14,r11
	mov	r11,QWORD PTR[28+rdi]
	mov	QWORD PTR[((-76))+rsi],r14


	xor	r8,rbp
	xor	r12,rdx
	rol	r8,28
	xor	r11,rcx
	xor	r9,rax
	rol	r12,61
	rol	r11,45
	xor	r10,rbx
	rol	r9,20
	mov	r13,r8
	or	r8,r12
	rol	r10,3

	xor	r8,r11
	mov	QWORD PTR[((-36))+rsi],r8

	mov	r14,r9
	and	r9,r13
	mov	r8,QWORD PTR[((-92))+rdi]
	xor	r9,r12
	not	r12
	mov	QWORD PTR[((-28))+rsi],r9

	or	r12,r11
	mov	r9,QWORD PTR[((-44))+rdi]
	xor	r12,r10
	mov	QWORD PTR[((-44))+rsi],r12

	and	r11,r10
	mov	r12,QWORD PTR[60+rdi]
	xor	r11,r14
	mov	QWORD PTR[((-52))+rsi],r11

	or	r14,r10
	mov	r10,QWORD PTR[4+rdi]
	xor	r14,r13
	mov	r11,QWORD PTR[52+rdi]
	mov	QWORD PTR[((-60))+rsi],r14


	xor	r10,rbp
	xor	r11,rax
	rol	r10,25
	xor	r9,rdx
	rol	r11,8
	xor	r12,rbx
	rol	r9,6
	xor	r8,rcx
	rol	r12,18
	mov	r13,r10
	and	r10,r11
	rol	r8,1

	not	r11
	xor	r10,r9
	mov	QWORD PTR[((-12))+rsi],r10

	mov	r14,r12
	and	r12,r11
	mov	r10,QWORD PTR[((-12))+rdi]
	xor	r12,r13
	mov	QWORD PTR[((-4))+rsi],r12

	or	r13,r9
	mov	r12,QWORD PTR[84+rdi]
	xor	r13,r8
	mov	QWORD PTR[((-20))+rsi],r13

	and	r9,r8
	xor	r9,r14
	mov	QWORD PTR[12+rsi],r9

	or	r14,r8
	mov	r9,QWORD PTR[((-60))+rdi]
	xor	r14,r11
	mov	r11,QWORD PTR[36+rdi]
	mov	QWORD PTR[4+rsi],r14


	mov	r8,QWORD PTR[((-68))+rdi]

	xor	r10,rcx
	xor	r11,rdx
	rol	r10,10
	xor	r9,rbx
	rol	r11,15
	xor	r12,rbp
	rol	r9,36
	xor	r8,rax
	rol	r12,56
	mov	r13,r10
	or	r10,r11
	rol	r8,27

	not	r11
	xor	r10,r9
	mov	QWORD PTR[28+rsi],r10

	mov	r14,r12
	or	r12,r11
	xor	r12,r13
	mov	QWORD PTR[36+rsi],r12

	and	r13,r9
	xor	r13,r8
	mov	QWORD PTR[20+rsi],r13

	or	r9,r8
	xor	r9,r14
	mov	QWORD PTR[52+rsi],r9

	and	r8,r14
	xor	r8,r11
	mov	QWORD PTR[44+rsi],r8


	xor	rdx,QWORD PTR[((-84))+rdi]
	xor	rbp,QWORD PTR[((-36))+rdi]
	rol	rdx,62
	xor	rcx,QWORD PTR[68+rdi]
	rol	rbp,55
	xor	rax,QWORD PTR[12+rdi]
	rol	rcx,2
	xor	rbx,QWORD PTR[20+rdi]
	xchg	rdi,rsi
	rol	rax,39
	rol	rbx,41
	mov	r13,rdx
	and	rdx,rbp
	not	rbp
	xor	rdx,rcx
	mov	QWORD PTR[92+rdi],rdx

	mov	r14,rax
	and	rax,rbp
	xor	rax,r13
	mov	QWORD PTR[60+rdi],rax

	or	r13,rcx
	xor	r13,rbx
	mov	QWORD PTR[84+rdi],r13

	and	rcx,rbx
	xor	rcx,r14
	mov	QWORD PTR[76+rdi],rcx

	or	rbx,r14
	xor	rbx,rbp
	mov	QWORD PTR[68+rdi],rbx

	mov	rbp,rdx
	mov	rdx,r13

	test	r15,255
	jnz	$L$oop

	lea	r15,QWORD PTR[((-192))+r15]
	DB	0F3h,0C3h		;repret

__KeccakF1600	ENDP


ALIGN	32
KeccakF1600	PROC PRIVATE

	push	rbx

	push	rbp

	push	r12

	push	r13

	push	r14

	push	r15


	lea	rdi,QWORD PTR[100+rdi]
	sub	rsp,200


	not	QWORD PTR[((-92))+rdi]
	not	QWORD PTR[((-84))+rdi]
	not	QWORD PTR[((-36))+rdi]
	not	QWORD PTR[((-4))+rdi]
	not	QWORD PTR[36+rdi]
	not	QWORD PTR[60+rdi]

	lea	r15,QWORD PTR[iotas]
	lea	rsi,QWORD PTR[100+rsp]

	call	__KeccakF1600

	not	QWORD PTR[((-92))+rdi]
	not	QWORD PTR[((-84))+rdi]
	not	QWORD PTR[((-36))+rdi]
	not	QWORD PTR[((-4))+rdi]
	not	QWORD PTR[36+rdi]
	not	QWORD PTR[60+rdi]
	lea	rdi,QWORD PTR[((-100))+rdi]

	add	rsp,200


	pop	r15

	pop	r14

	pop	r13

	pop	r12

	pop	rbp

	pop	rbx

	DB	0F3h,0C3h		;repret

KeccakF1600	ENDP
PUBLIC	SHA3_absorb

ALIGN	32
SHA3_absorb	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_SHA3_absorb::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8
	mov	rcx,r9



	push	rbx

	push	rbp

	push	r12

	push	r13

	push	r14

	push	r15


	lea	rdi,QWORD PTR[100+rdi]
	sub	rsp,232


	mov	r9,rsi
	lea	rsi,QWORD PTR[100+rsp]

	not	QWORD PTR[((-92))+rdi]
	not	QWORD PTR[((-84))+rdi]
	not	QWORD PTR[((-36))+rdi]
	not	QWORD PTR[((-4))+rdi]
	not	QWORD PTR[36+rdi]
	not	QWORD PTR[60+rdi]
	lea	r15,QWORD PTR[iotas]

	mov	QWORD PTR[((216-100))+rsi],rcx

$L$oop_absorb::
	cmp	rdx,rcx
	jc	$L$done_absorb

	shr	rcx,3
	lea	r8,QWORD PTR[((-100))+rdi]

$L$block_absorb::
	mov	rax,QWORD PTR[r9]
	lea	r9,QWORD PTR[8+r9]
	xor	rax,QWORD PTR[r8]
	lea	r8,QWORD PTR[8+r8]
	sub	rdx,8
	mov	QWORD PTR[((-8))+r8],rax
	sub	rcx,1
	jnz	$L$block_absorb

	mov	QWORD PTR[((200-100))+rsi],r9
	mov	QWORD PTR[((208-100))+rsi],rdx
	call	__KeccakF1600
	mov	r9,QWORD PTR[((200-100))+rsi]
	mov	rdx,QWORD PTR[((208-100))+rsi]
	mov	rcx,QWORD PTR[((216-100))+rsi]
	jmp	$L$oop_absorb

ALIGN	32
$L$done_absorb::
	mov	rax,rdx

	not	QWORD PTR[((-92))+rdi]
	not	QWORD PTR[((-84))+rdi]
	not	QWORD PTR[((-36))+rdi]
	not	QWORD PTR[((-4))+rdi]
	not	QWORD PTR[36+rdi]
	not	QWORD PTR[60+rdi]

	add	rsp,232


	pop	r15

	pop	r14

	pop	r13

	pop	r12

	pop	rbp

	pop	rbx

	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_SHA3_absorb::
SHA3_absorb	ENDP
PUBLIC	SHA3_squeeze

ALIGN	32
SHA3_squeeze	PROC PUBLIC
	mov	QWORD PTR[8+rsp],rdi	;WIN64 prologue
	mov	QWORD PTR[16+rsp],rsi
	mov	rax,rsp
$L$SEH_begin_SHA3_squeeze::
	mov	rdi,rcx
	mov	rsi,rdx
	mov	rdx,r8
	mov	rcx,r9



	push	r12

	push	r13

	push	r14


	shr	rcx,3
	mov	r8,rdi
	mov	r12,rsi
	mov	r13,rdx
	mov	r14,rcx
	jmp	$L$oop_squeeze

ALIGN	32
$L$oop_squeeze::
	cmp	r13,8
	jb	$L$tail_squeeze

	mov	rax,QWORD PTR[r8]
	lea	r8,QWORD PTR[8+r8]
	mov	QWORD PTR[r12],rax
	lea	r12,QWORD PTR[8+r12]
	sub	r13,8
	jz	$L$done_squeeze

	sub	rcx,1
	jnz	$L$oop_squeeze

	call	KeccakF1600
	mov	r8,rdi
	mov	rcx,r14
	jmp	$L$oop_squeeze

$L$tail_squeeze::
	mov	rsi,r8
	mov	rdi,r12
	mov	rcx,r13
DB	0f3h,0a4h

$L$done_squeeze::
	pop	r14

	pop	r13

	pop	r12

	mov	rdi,QWORD PTR[8+rsp]	;WIN64 epilogue
	mov	rsi,QWORD PTR[16+rsp]
	DB	0F3h,0C3h		;repret

$L$SEH_end_SHA3_squeeze::
SHA3_squeeze	ENDP
ALIGN	256
	DQ	0,0,0,0,0,0,0,0

iotas::
	DQ	00000000000000001h
	DQ	00000000000008082h
	DQ	0800000000000808ah
	DQ	08000000080008000h
	DQ	0000000000000808bh
	DQ	00000000080000001h
	DQ	08000000080008081h
	DQ	08000000000008009h
	DQ	0000000000000008ah
	DQ	00000000000000088h
	DQ	00000000080008009h
	DQ	0000000008000000ah
	DQ	0000000008000808bh
	DQ	0800000000000008bh
	DQ	08000000000008089h
	DQ	08000000000008003h
	DQ	08000000000008002h
	DQ	08000000000000080h
	DQ	0000000000000800ah
	DQ	0800000008000000ah
	DQ	08000000080008081h
	DQ	08000000000008080h
	DQ	00000000080000001h
	DQ	08000000080008008h

DB	75,101,99,99,97,107,45,49,54,48,48,32,97,98,115,111
DB	114,98,32,97,110,100,32,115,113,117,101,101,122,101,32,102
DB	111,114,32,120,56,54,95,54,52,44,32,67,82,89,80,84
DB	79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64
DB	111,112,101,110,115,115,108,46,111,114,103,62,0

.text$	ENDS
END
