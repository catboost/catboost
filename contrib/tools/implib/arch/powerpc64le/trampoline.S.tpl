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

  // Force callers to save r2
  .localentry $sym, 1

$sym:
  .cfi_startproc

  // Get function address
  mflr 11
  bcl 20, 31, $$+4
0:
  mflr 12
  mtlr 11
  addi 12, 12, ($sym - 0b)

  // Get TOC address
  addis 2, 12, .TOC. - $sym@ha
  addi 2, 2, .TOC. - $sym@l

1:
  // Load address
  addis 12, 2, .LC0@toc@ha
  ld 12, .LC0@toc@l(12)
  ld 12, $offset(12)

  cmpdi 12, 0
  beq 3f

2: // "Fast" path

  mtctr 12
  bctr

3: // Slow path

  mflr 0
  std 0, 16(1)

  li 0, $number
  std 0, -8(1)

  stdu 1, -48(1)
  .cfi_def_cfa_offset 48
  .cfi_offset lr, 16

  bl _${lib_suffix}_save_regs_and_resolve
  nop

  addi 1, 1, 48
  .cfi_def_cfa_offset 0

  ld 0, 16(1)
  mtlr 0
  .cfi_restore lr

  b 1b

  .long 0
  .quad 0

  .cfi_endproc
