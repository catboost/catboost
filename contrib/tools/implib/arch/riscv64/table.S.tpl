/*
 * Copyright 2024 Yury Gribov
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

  // Slow path which calls dlsym, taken only on first call.
  // Registers are saved according to "Procedure Call Standard for the Arm® 64-bit Architecture".
  // For DWARF directives, read https://www.imperialviolet.org/2017/01/18/cfi.html.

  // TODO: cfi_offset/cfi_restore

  addi sp, sp, -144
 .cfi_adjust_cfa_offset 144

  sd ra, 0(sp)
  .cfi_rel_offset ra, 0

  sd a0, 8(sp)
  .cfi_rel_offset a0, 8
  sd a1, 16(sp)
  .cfi_rel_offset a1, 16
  sd a2, 24(sp)
  .cfi_rel_offset a2, 24
  sd a3, 32(sp)
  .cfi_rel_offset a3, 32
  sd a4, 40(sp)
  .cfi_rel_offset a4, 40
  sd a5, 48(sp)
  .cfi_rel_offset a5, 48
  sd a6, 56(sp)
  .cfi_rel_offset a6, 56
  sd a7, 64(sp)
  .cfi_rel_offset a7, 64

  fsd fa0, 72(sp)
  .cfi_rel_offset fa0, 72
  fsd fa1, 80(sp)
  .cfi_rel_offset fa1, 80
  fsd fa2, 88(sp)
  .cfi_rel_offset fa2, 88
  fsd fa3, 96(sp)
  .cfi_rel_offset fa3, 96
  fsd fa4, 104(sp)
  .cfi_rel_offset fa4, 104
  fsd fa5, 112(sp)
  .cfi_rel_offset fa5, 112
  fsd fa6, 120(sp)
  .cfi_rel_offset fa6, 120
  fsd fa7, 128(sp)
  .cfi_rel_offset fa7, 128

  ld a0, 144(sp)

  // TODO: vector arguments

  // Stack is aligned at 16 bytes

  call _${lib_suffix}_tramp_resolve

  fld fa7, 128(sp)
  .cfi_restore fa7
  fld fa6, 120(sp)
  .cfi_restore fa6
  fld fa5, 112(sp)
  .cfi_restore fa5
  fld fa4, 104(sp)
  .cfi_restore fa4
  fld fa3, 96(sp)
  .cfi_restore fa3
  fld fa2, 88(sp)
  .cfi_restore fa2
  fld fa1, 80(sp)
  .cfi_restore fa1
  fld fa0, 72(sp)
  .cfi_restore fa0

  ld a7, 64(sp)
  .cfi_restore a7
  ld a6, 56(sp)
  .cfi_restore a6
  ld a5, 48(sp)
  .cfi_restore a5
  ld a4, 40(sp)
  .cfi_restore a4
  ld a3, 32(sp)
  .cfi_restore a3
  ld a2, 24(sp)
  .cfi_restore a2
  ld a1, 16(sp)
  .cfi_restore a1
  ld a0, 8(sp)
  .cfi_restore a0

  ld ra, 0(sp)
  .cfi_restore ra

  addi sp, sp, 144
  .cfi_def_cfa_offset 0

  jr ra

  .cfi_endproc
