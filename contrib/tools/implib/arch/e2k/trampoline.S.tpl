/*
 * Copyright 2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .globl $sym
  .p2align 3
  .ignore strict_delay
  .type $sym, %function
#ifndef IMPLIB_EXPORT_SHIMS
  .hidden $sym
#endif
$sym:
  .cfi_startproc

  setwd wsz = 0x8, nfx = 0x1

1:
  // Read table address
  {
    rrd %ip, %g0
    addd 0x0, [ _f64 _GLOBAL_OFFSET_TABLE_ ], %g1
  }
  addd %g0, %g1, %g0
  addd %g0, [ _f64 _${lib_suffix}_tramp_table@GOTOFF ], %g0

  // Read current function address
  ldd [%g0 + $offset], %g0

  // NULL?
  {
    cmpesb %g0, 0x0, %pred0
    movtd %g0, %ctpr2
  }

  // Jump to fast path
  ct %ctpr2 ? ~%pred0

  // Or fall through to slow path

2:
  // Initialize parameter
  addd 0x0, _f16s $number, %g0

  // Call resolver
  disp  %ctpr1, _${lib_suffix}_save_regs_and_resolve
  call %ctpr1, wbs = 0x8

  // Return to fast path
  ibranch 1b

  .cfi_endproc

