.text	



.globl	_poly1305_init
.private_extern	_poly1305_init
.globl	_poly1305_blocks
.private_extern	_poly1305_blocks
.globl	_poly1305_emit
.private_extern	_poly1305_emit


.p2align	5
_poly1305_init:
	xorq	%rax,%rax
	movq	%rax,0(%rdi)
	movq	%rax,8(%rdi)
	movq	%rax,16(%rdi)

	cmpq	$0,%rsi
	je	L$no_key

	leaq	_poly1305_blocks(%rip),%r10
	leaq	_poly1305_emit(%rip),%r11
	movq	$0x0ffffffc0fffffff,%rax
	movq	$0x0ffffffc0ffffffc,%rcx
	andq	0(%rsi),%rax
	andq	8(%rsi),%rcx
	movq	%rax,24(%rdi)
	movq	%rcx,32(%rdi)
	movq	%r10,0(%rdx)
	movq	%r11,8(%rdx)
	movl	$1,%eax
L$no_key:
	.byte	0xf3,0xc3



.p2align	5
_poly1305_blocks:

L$blocks:
	shrq	$4,%rdx
	jz	L$no_data

	pushq	%rbx

	pushq	%rbp

	pushq	%r12

	pushq	%r13

	pushq	%r14

	pushq	%r15

L$blocks_body:

	movq	%rdx,%r15

	movq	24(%rdi),%r11
	movq	32(%rdi),%r13

	movq	0(%rdi),%r14
	movq	8(%rdi),%rbx
	movq	16(%rdi),%rbp

	movq	%r13,%r12
	shrq	$2,%r13
	movq	%r12,%rax
	addq	%r12,%r13
	jmp	L$oop

.p2align	5
L$oop:
	addq	0(%rsi),%r14
	adcq	8(%rsi),%rbx
	leaq	16(%rsi),%rsi
	adcq	%rcx,%rbp
	mulq	%r14
	movq	%rax,%r9
	movq	%r11,%rax
	movq	%rdx,%r10

	mulq	%r14
	movq	%rax,%r14
	movq	%r11,%rax
	movq	%rdx,%r8

	mulq	%rbx
	addq	%rax,%r9
	movq	%r13,%rax
	adcq	%rdx,%r10

	mulq	%rbx
	movq	%rbp,%rbx
	addq	%rax,%r14
	adcq	%rdx,%r8

	imulq	%r13,%rbx
	addq	%rbx,%r9
	movq	%r8,%rbx
	adcq	$0,%r10

	imulq	%r11,%rbp
	addq	%r9,%rbx
	movq	$-4,%rax
	adcq	%rbp,%r10

	andq	%r10,%rax
	movq	%r10,%rbp
	shrq	$2,%r10
	andq	$3,%rbp
	addq	%r10,%rax
	addq	%rax,%r14
	adcq	$0,%rbx
	adcq	$0,%rbp
	movq	%r12,%rax
	decq	%r15
	jnz	L$oop

	movq	%r14,0(%rdi)
	movq	%rbx,8(%rdi)
	movq	%rbp,16(%rdi)

	movq	0(%rsp),%r15

	movq	8(%rsp),%r14

	movq	16(%rsp),%r13

	movq	24(%rsp),%r12

	movq	32(%rsp),%rbp

	movq	40(%rsp),%rbx

	leaq	48(%rsp),%rsp

L$no_data:
L$blocks_epilogue:
	.byte	0xf3,0xc3




.p2align	5
_poly1305_emit:
L$emit:
	movq	0(%rdi),%r8
	movq	8(%rdi),%r9
	movq	16(%rdi),%r10

	movq	%r8,%rax
	addq	$5,%r8
	movq	%r9,%rcx
	adcq	$0,%r9
	adcq	$0,%r10
	shrq	$2,%r10
	cmovnzq	%r8,%rax
	cmovnzq	%r9,%rcx

	addq	0(%rdx),%rax
	adcq	8(%rdx),%rcx
	movq	%rax,0(%rsi)
	movq	%rcx,8(%rsi)

	.byte	0xf3,0xc3

.byte	80,111,108,121,49,51,48,53,32,102,111,114,32,120,56,54,95,54,52,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0
.p2align	4
.globl	_xor128_encrypt_n_pad

.p2align	4
_xor128_encrypt_n_pad:
	subq	%rdx,%rsi
	subq	%rdx,%rdi
	movq	%rcx,%r10
	shrq	$4,%rcx
	jz	L$tail_enc
	nop
L$oop_enc_xmm:
	movdqu	(%rsi,%rdx,1),%xmm0
	pxor	(%rdx),%xmm0
	movdqu	%xmm0,(%rdi,%rdx,1)
	movdqa	%xmm0,(%rdx)
	leaq	16(%rdx),%rdx
	decq	%rcx
	jnz	L$oop_enc_xmm

	andq	$15,%r10
	jz	L$done_enc

L$tail_enc:
	movq	$16,%rcx
	subq	%r10,%rcx
	xorl	%eax,%eax
L$oop_enc_byte:
	movb	(%rsi,%rdx,1),%al
	xorb	(%rdx),%al
	movb	%al,(%rdi,%rdx,1)
	movb	%al,(%rdx)
	leaq	1(%rdx),%rdx
	decq	%r10
	jnz	L$oop_enc_byte

	xorl	%eax,%eax
L$oop_enc_pad:
	movb	%al,(%rdx)
	leaq	1(%rdx),%rdx
	decq	%rcx
	jnz	L$oop_enc_pad

L$done_enc:
	movq	%rdx,%rax
	.byte	0xf3,0xc3


.globl	_xor128_decrypt_n_pad

.p2align	4
_xor128_decrypt_n_pad:
	subq	%rdx,%rsi
	subq	%rdx,%rdi
	movq	%rcx,%r10
	shrq	$4,%rcx
	jz	L$tail_dec
	nop
L$oop_dec_xmm:
	movdqu	(%rsi,%rdx,1),%xmm0
	movdqa	(%rdx),%xmm1
	pxor	%xmm0,%xmm1
	movdqu	%xmm1,(%rdi,%rdx,1)
	movdqa	%xmm0,(%rdx)
	leaq	16(%rdx),%rdx
	decq	%rcx
	jnz	L$oop_dec_xmm

	pxor	%xmm1,%xmm1
	andq	$15,%r10
	jz	L$done_dec

L$tail_dec:
	movq	$16,%rcx
	subq	%r10,%rcx
	xorl	%eax,%eax
	xorq	%r11,%r11
L$oop_dec_byte:
	movb	(%rsi,%rdx,1),%r11b
	movb	(%rdx),%al
	xorb	%r11b,%al
	movb	%al,(%rdi,%rdx,1)
	movb	%r11b,(%rdx)
	leaq	1(%rdx),%rdx
	decq	%r10
	jnz	L$oop_dec_byte

	xorl	%eax,%eax
L$oop_dec_pad:
	movb	%al,(%rdx)
	leaq	1(%rdx),%rdx
	decq	%rcx
	jnz	L$oop_dec_pad

L$done_dec:
	movq	%rdx,%rax
	.byte	0xf3,0xc3

