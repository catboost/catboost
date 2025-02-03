/*
 * Copyright 2022-2024 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .section .note.GNU-stack,"",@progbits

  .data

  .globl _${lib_suffix}_tramp_table
  .hidden _${lib_suffix}_tramp_table
  .align 8
_${lib_suffix}_tramp_table:
  .zero $table_size

  .text

  .globl _${lib_suffix}_tramp_resolve
  .hidden _${lib_suffix}_tramp_resolve

  .globl _${lib_suffix}_save_regs_and_resolve
  .hidden _${lib_suffix}_save_regs_and_resolve
  .type _${lib_suffix}_save_regs_and_resolve, %function
_${lib_suffix}_save_regs_and_resolve:
  .cfi_startproc

  .set noreorder
  .cpload $$25
  .set nomacro
  .set noat

  // Slow path which calls dlsym, taken only on first call.
  // Registers are saved acc. to "Procedure Call Standard for the MIPS Architecture".
  // For DWARF directives, read https://www.imperialviolet.org/2017/01/18/cfi.html.

  // TODO: push two regs at once here and in trampoline to avoid temporarily unaligned stack

#define PUSH_REG(reg) daddiu $$sp, $$sp, -8; .cfi_adjust_cfa_offset 8; sd reg, 0($$sp); .cfi_rel_offset reg, 0
#define POP_REG(reg) ld reg, 0($$sp); .cfi_restore reg; daddiu $$sp, $$sp, 8; .cfi_adjust_cfa_offset -8

// dwarf_num = 32 + reg_num
#define PUSH_FREG(reg, dwarf_num) daddiu $$sp, $$sp, -8; .cfi_adjust_cfa_offset 8; sdc1 reg, 0($$sp); .cfi_rel_offset dwarf_num, 0
#define POP_FREG(reg, dwarf_num) ldc1 reg, 0($$sp); .cfi_restore dwarf_num; daddiu $$sp, $$sp, 8; .cfi_adjust_cfa_offset -8

  PUSH_REG($$ra)
  PUSH_REG($$gp)
  PUSH_REG($$a0)
  PUSH_REG($$a1)
  PUSH_REG($$a2)
  PUSH_REG($$a3)
  PUSH_REG($$a4)
  PUSH_REG($$a5)
  PUSH_REG($$a6)
  PUSH_REG($$a7)

  PUSH_FREG($$f12, 44)
  PUSH_FREG($$f13, 45)
  PUSH_FREG($$f14, 46)
  PUSH_FREG($$f15, 47)
  PUSH_FREG($$f16, 48)
  PUSH_FREG($$f17, 49)
  PUSH_FREG($$f18, 50)
  PUSH_FREG($$f19, 51)

  // Vector arguments are passed on stack so we don't save vector regs

  lui $$gp, %hi(%neg(%gp_rel(_${lib_suffix}_save_regs_and_resolve)))
  daddu $$gp, $$gp, $$25
  daddiu $$gp, $$gp, %lo(%neg(%gp_rel(_${lib_suffix}_save_regs_and_resolve)))

  move $$a0, $$AT

  ld $$25, %call16(_${lib_suffix}_tramp_resolve)($$gp)
  .reloc  1f, R_MIPS_JALR, _${lib_suffix}_tramp_resolve
1: jalr $$25
  nop

  POP_FREG($$f19, 51)
  POP_FREG($$f18, 50)
  POP_FREG($$f17, 49)
  POP_FREG($$f16, 48)
  POP_FREG($$f15, 47)
  POP_FREG($$f14, 46)
  POP_FREG($$f13, 45)
  POP_FREG($$f12, 44)

  POP_REG($$a7)
  POP_REG($$a6)
  POP_REG($$a5)
  POP_REG($$a4)
  POP_REG($$a3)
  POP_REG($$a2)
  POP_REG($$a1)
  POP_REG($$a0)
  POP_REG($$gp)
  POP_REG($$ra)

  jr $$ra
  nop

  .set macro
  .set reorder

  .cfi_endproc
