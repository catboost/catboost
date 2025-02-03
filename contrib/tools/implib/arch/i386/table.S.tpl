/*
 * Copyright 2019-2023 Yury Gribov
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
  .align 4
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

#define PUSH_REG(reg) pushl %reg ; .cfi_adjust_cfa_offset 4; .cfi_rel_offset reg, 0
#define POP_REG(reg) popl %reg ; .cfi_adjust_cfa_offset -4; .cfi_restore reg

  // Slow path which calls dlsym, taken only on first call.
  // All registers are stored to handle arbitrary calling conventions
  // (except XMM/x87 regs in hope they are not used in resolving code).
  // For Dwarf directives, read https://www.imperialviolet.org/2017/01/18/cfi.html.

  .cfi_def_cfa_offset 4  // Return address

  PUSH_REG(eax)
  PUSH_REG(ebx)
  PUSH_REG(ecx)
  PUSH_REG(edx)  // 16

  PUSH_REG(ebp)
  PUSH_REG(edi)
  PUSH_REG(esi)
  pushfl; .cfi_adjust_cfa_offset 4  // 16

  subl $$8, %esp
  .cfi_adjust_cfa_offset 8
  PUSH_REG(eax)

  // TODO: save vector regs

  call _${lib_suffix}_tramp_resolve@PLT  // Stack will be aligned at 16 in call

  addl $$12, %esp
  .cfi_adjust_cfa_offset -12

  popfl; .cfi_adjust_cfa_offset -4
  POP_REG(esi)
  POP_REG(edi)
  POP_REG(ebp)

  POP_REG(edx)
  POP_REG(ecx)
  POP_REG(ebx)
  POP_REG(eax)

  ret

  .cfi_endproc

  .section .text.__implib.x86.get_pc_thunk.ax,"axG",@progbits,__implib.x86.get_pc_thunk.ax,comdat
  .globl __implib.x86.get_pc_thunk.ax
  .hidden __implib.x86.get_pc_thunk.ax
  .type __implib.x86.get_pc_thunk.ax, %function
__implib.x86.get_pc_thunk.ax:
  .cfi_startproc
  movl (%esp), %eax
  ret
  .cfi_endproc

