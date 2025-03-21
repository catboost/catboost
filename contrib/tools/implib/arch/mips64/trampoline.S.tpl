/*
 * Copyright 2022-2023 Yury Gribov
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
$sym:
  .cfi_startproc

  .set noreorder
  .cpload $$25
  .set nomacro
  .set noat

1:
  // Load address
#if $offset < 32768
  lui $$AT, %hi(%neg(%gp_rel($sym)))
  daddu $$AT, $$AT, $$25
  daddiu $$AT, $$AT, %lo(%neg(%gp_rel($sym)))
  ld $$AT, %got_disp(_${lib_suffix}_tramp_table)($$AT)
  ld $$AT, $offset($$AT)
#else
  PUSH_REG($$2)
  lui $$AT, %hi(%neg(%gp_rel($sym)))
  daddu $$AT, $$AT, $$25
  daddiu $$AT, $$AT, %lo(%neg(%gp_rel($sym)))
  ld $$AT, %got_disp(_${lib_suffix}_tramp_table)($$AT)
  .set macro
  .set at=$$2
  ld $$AT, $offset($$AT)
  .set nomacro
  .set noat
  POP_REG($$2)
#endif

  beqz $$AT, 3f
  nop

2:
  // Fast path
  j $$AT
  move $$25, $$AT

3:
  // Slow path

  PUSH_REG($$25)
  PUSH_REG($$ra)
  PUSH_REG($$gp)

  lui $$gp, %hi(%neg(%gp_rel($sym)))
  daddu $$gp, $$gp, $$25
  daddiu $$gp, $$gp, %lo(%neg(%gp_rel($sym)))

  ld $$25, %call16(_${lib_suffix}_save_regs_and_resolve)($$gp)
  .reloc  4f, R_MIPS_JALR, _${lib_suffix}_save_regs_and_resolve
4: jalr $$25
  li $$AT, $number

  POP_REG($$gp)
  POP_REG($$ra)
  POP_REG($$25)

  j 1b
  nop

  .set macro
  .set reorder

  .cfi_endproc
