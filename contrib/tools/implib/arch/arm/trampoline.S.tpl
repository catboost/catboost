/*
 * Copyright 2018-2023 Yury Gribov
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
  // TODO: can we do this faster on newer ARMs?
  ldr ip, 3f
2:
  add ip, pc, ip
  ldr ip, [ip]

  cmp ip, #0

  // Fast path
  bxne ip

  // Slow path
  ldr ip, =$number
  push {ip}
  .cfi_adjust_cfa_offset 4
  PUSH_REG(lr)
  bl _${lib_suffix}_save_regs_and_resolve
  POP_REG(lr)
  add sp, #4
  .cfi_adjust_cfa_offset -4
  b 1b

  // Force constant pool for ldr above
  .ltorg

  .cfi_endproc

3:
  .word _${lib_suffix}_tramp_table - (2b + 8) + $offset

