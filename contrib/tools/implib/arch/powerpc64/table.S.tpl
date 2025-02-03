/*
 * Copyright 2024 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

  .machine power7

  .section .note.GNU-stack,"",@progbits

  .data

  .globl _${lib_suffix}_tramp_table
  .hidden _${lib_suffix}_tramp_table
  .align 8
_${lib_suffix}_tramp_table:
  .zero $table_size

  .section ".toc","aw"
  .align 3
.LC0:
  .quad _${lib_suffix}_tramp_table

  .section ".text"

  .globl _${lib_suffix}_tramp_resolve
  .hidden _${lib_suffix}_tramp_resolve

  .globl _${lib_suffix}_save_regs_and_resolve
  .hidden _${lib_suffix}_save_regs_and_resolve
  .type _${lib_suffix}_save_regs_and_resolve, %function

  .section ".opd", "aw"
  .align 3
_${lib_suffix}_save_regs_and_resolve:
  .quad .L._${lib_suffix}_save_regs_and_resolve, .TOC.@tocbase, 0

  .previous

.L._${lib_suffix}_save_regs_and_resolve:
  .cfi_startproc

  // Slow path which calls dlsym, taken only on first call.
  // Registers are saved acc. to PPC64 ELF ABI
  // For DWARF directives, read https://www.imperialviolet.org/2017/01/18/cfi.html.

  mflr 0
  std 0, 16(1)

  ld 0, 128-8(1)

  std 3, -8(1)
  std 4, -16(1)
  std 5, -24(1)
  std 6, -32(1)
  std 7, -40(1)
  std 8, -48(1)
  std 9, -56(1)
  std 10, -64(1)

  stfd 1, -72(1)
  stfd 2, -80(1)
  stfd 3, -88(1)
  stfd 4, -96(1)
  stfd 5, -104(1)
  stfd 6, -112(1)
  stfd 7, -120(1)
  stfd 8, -128(1)
  stfd 9, -136(1)
  stfd 10, -144(1)
  stfd 11, -152(1)
  stfd 12, -160(1)
  stfd 13, -168(1)

  // TODO: also save Altivec registers

  stdu 1, -256(1)

  .cfi_def_cfa_offset 256
  .cfi_offset r3, -8
  .cfi_offset r4, -16
  .cfi_offset r5, -24
  .cfi_offset r6, -32
  .cfi_offset r7, -40
  .cfi_offset r8, -48
  .cfi_offset r9, -56
  .cfi_offset r10, -64
  .cfi_offset f1, -72
  .cfi_offset f2, -80
  .cfi_offset f3, -88
  .cfi_offset f4, -96
  .cfi_offset f5, -104
  .cfi_offset f6, -112
  .cfi_offset f7, -120
  .cfi_offset f8, -128
  .cfi_offset f9, -136
  .cfi_offset f10, -144
  .cfi_offset f11, -152
  .cfi_offset f12, -160
  .cfi_offset f13, -168

  mr 3, 0

  bl _${lib_suffix}_tramp_resolve
  nop

  addi 1, 1, 256
  .cfi_def_cfa_offset 0

  ld 3, -8(1)
  ld 4, -16(1)
  ld 5, -24(1)
  ld 6, -32(1)
  ld 7, -40(1)
  ld 8, -48(1)
  ld 9, -56(1)
  ld 10, -64(1)

  lfd 1, -72(1)
  lfd 2, -80(1)
  lfd 3, -88(1)
  lfd 4, -96(1)
  lfd 5, -104(1)
  lfd 6, -112(1)
  lfd 7, -120(1)
  lfd 8, -128(1)
  lfd 9, -136(1)
  lfd 10, -144(1)
  lfd 11, -152(1)
  lfd 12, -160(1)
  lfd 13, -168(1)

  ld 0, 16(1)
  mtlr 0

  .cfi_restore r3
  .cfi_restore r4
  .cfi_restore r5
  .cfi_restore r6
  .cfi_restore r7
  .cfi_restore r8
  .cfi_restore r9
  .cfi_restore r10
  .cfi_restore f1
  .cfi_restore f2
  .cfi_restore f3
  .cfi_restore f4
  .cfi_restore f5
  .cfi_restore f6
  .cfi_restore f7
  .cfi_restore f8
  .cfi_restore f9
  .cfi_restore f10
  .cfi_restore f11
  .cfi_restore f12
  .cfi_restore f13

  blr

  .long 0
  .quad 0

  .cfi_endproc
