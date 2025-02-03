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
$sym:
  .cfi_startproc

1:
  // Load address

  lla t0, _${lib_suffix}_tramp_table+$offset
  ld t0, 0(t0)

  beq t0, zero, 2f

  // Fast path
  jr t0

2:
  // Slow path

  addi sp, sp, -16
  .cfi_adjust_cfa_offset 16

  sd ra, 8(sp)
  .cfi_rel_offset ra, 8

  li t0, $number
  sd t0, 0(sp)

  // TODO: cfi_offset/cfi_restore

  call _${lib_suffix}_save_regs_and_resolve

  ld ra, 8(sp)
  .cfi_restore ra

  addi sp, sp, 16
  .cfi_def_cfa_offset 0

  j 1b
  .cfi_endproc
