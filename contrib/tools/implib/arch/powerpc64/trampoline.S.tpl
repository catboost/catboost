/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl $sym
  .p2align 4
  .type $sym, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden $sym
#endif

  .section ".opd", "aw"
  .align 3
$sym:
  .quad .L.$sym, .TOC.@tocbase, 0

  .previous

.L.$sym:
  .cfi_startproc

1:
  // Load address
  addis 11, 2, .LC0@toc@ha
  ld 11, .LC0@toc@l(11)
  ld 11, $offset(11)

  cmpdi 11, 0
  beq 3f

2: // "Fast" path

  // TODO:
  // We need to somehow get rid of additional stack frame
  // because it breaks arguments which are passed on stack.
  // Currently we work around this by duplicating parameters
  // in new stack frame but this supports only 16 parameters
  // and is very inefficient.

  mflr 0
  std 0, 16(1)
  stdu 1, -176(1)
  .cfi_def_cfa_offset 176
  .cfi_offset lr, 16

  std 2, 40(1)

  ld 0, 176+112(1)
  std 0, 112(1)
  ld 0, 176+120(1)
  std 0, 120(1)
  ld 0, 176+128(1)
  std 0, 128(1)
  ld 0, 176+136(1)
  std 0, 136(1)
  ld 0, 176+144(1)
  std 0, 144(1)
  ld 0, 176+152(1)
  std 0, 152(1)
  ld 0, 176+160(1)
  std 0, 160(1)
  ld 0, 176+168(1)
  std 0, 168(1)

  ld 2, 8(11)
  ld 11, 0(11)

  mtctr 11
  bctrl

  ld 2, 40(1)
  addi 1, 1, 176
  .cfi_def_cfa_offset 0
  ld 0, 16(1)
  mtlr 0
  .cfi_restore lr
  blr

3: // Slow path

  mflr 0
  std 0, 16(1)

  li 0, $number
  std 0, -8(1)

  stdu 1, -128(1)
  .cfi_def_cfa_offset 128
  .cfi_offset lr, 16

  bl _${lib_suffix}_save_regs_and_resolve
  nop

  addi 1, 1, 128
  .cfi_def_cfa_offset 0

  ld 0, 16(1)
  mtlr 0
  .cfi_restore lr

  b 1b

  .long 0
  .quad 0

  .cfi_endproc
