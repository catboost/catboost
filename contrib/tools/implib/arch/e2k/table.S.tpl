/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .data

  .globl _${lib_suffix}_tramp_table
  .hidden _${lib_suffix}_tramp_table
  .ignore strict_delay
  .p2align 3
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

  setwd wsz = 0x1, nfx = 1

  addd 0x0, %g0, %r0

  disp  %ctpr1, _${lib_suffix}_tramp_resolve
  call %ctpr1, wbs = 0

  return %ctpr3
  ct %ctpr3

  .cfi_endproc

