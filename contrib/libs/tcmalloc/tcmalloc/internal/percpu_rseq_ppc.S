/*
 * Copyright 2019 The TCMalloc Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Rseq critical section functions and restart handlers.
//
// They must also avoid writing the nonvolatile and reserved general purpose
// registers defined by the Power Architecture 64-Bit ELF V2 ABI
//
//  *  r1-r2
//  *  r13
//  *  r14-r31
//
// Finally, note that the restart handler reserves the right to clobber
// condition registers. This means that critical section functions must not
// explicitly or implicitly read condition registers outside of their
// [start, limit) critical regions.

#ifndef __ppc__
#error "percpu_rseq_ppc.S should only be included for PPC builds"
#endif

#include "tcmalloc/internal/percpu.h"

// Use the ELFv2 ABI.
.abiversion 2
.section google_malloc, "ax"

////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////

/*
 * Provide a directive to specify the size of symbol "label", relative to the
 * current location and its start.
 */
#define ENCODE_SIZE(label) .size label, . - label;

// Place the CPU number into the bottom 12 bits of dst. The upper 52 bits are
// unspecified.
//
// See GetCurrentCpu() for notes on the implementation.
#define GET_CPU_UNMASKED(dst) \
    mfspr dst, 259

// Given an unmasked CPU number, put the interesting parts into dst.
#define MASK_CPU(dst, src) \
    clrldi dst, src, 52

// Like GET_CPU_UNMASKED, but guarantees that the upper bits are cleared. May
// be slower than the unmasked version.
#define GET_CPU(dst) \
    GET_CPU_UNMASKED(dst); \
    MASK_CPU(dst, dst)

// This is part of the upstream rseq ABI.  The 4 bytes prior to the abort IP
// must match TCMALLOC_PERCPU_RSEQ_SIGNATURE (as configured by our rseq
// syscall's signature parameter).  This signature is used to annotate valid
// abort IPs (since rseq_cs could live in a user-writable segment).
#define SIGN_ABORT()           \
  .long TCMALLOC_PERCPU_RSEQ_SIGNATURE;

// DEFINE_UPSTREAM_CS triggers the generation of rseq_cs table (the triple of
// start, commit, abort IPs) and a trampoline function.
//
// Upstream API Exposition:
//
//   START_RSEQ() // vvvvv emits a bunch of things
//     global entry point:
//       TOC setup
//     METHOD_critical_abort:
//     local entry point:
//       store rseq_cs to __rseq_abi.rseq_cs, starting restartable sequence
//     METHOD_start:             // Emitted as part of START_RSEQ()
//   // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//
//     GET_CPU...()            // Reads current CPU
//     ...
//     single store            // Commits sequence
//   METHOD_critical_limit:
//     ...return...
//
// START_RSEQ does several things:
// * We need to set up the TOC pointer for global entry points.
// * When restarting, we return to the local entry point, since the TOC pointer
//   is left intact from the restart.  METHOD_critical_abort and local entry
//   point are therefore the same address.
// * It stores to the TLS to register that we're in a restartable sequence with
//   the kernel.
//
// This process is assisted by the DEFINE_UPSTREAM_CS macro, which encodes a
// (rodata) constant table, whose address is used to start the critical
// section, and the abort trampoline.
//
// The trampoline is used because:
// 1.  Restarts are expected to be rare, so the extra jump when restarting is
//     expected to be infrequent.
// 2.  The upstream restartable sequence implementation expects the trailing 4
//     bytes of the abort PC to be "signed" (to prevent manipulation of the PC
//     to an arbitrary choice).  For us, this is
//     TCMALLOC_PERCPU_RSEQ_SIGNATURE.  This value is passed to the kernel
//     during configuration of the rseq syscall.  This would either need to be
//     encoded as a nop* at the start of every restartable sequence, increasing
//     instruction cache pressure, or placed directly before the entry point.
//
//     * The upstream rseq protocol appears to be converging on using a trap
//     instruction (twui), so we cannot allow it to appear anywhere in our
//     actual executed path.
//
// Upon restart, the (upstream) kernel API clears the per-thread restartable
// sequence state. We return to METHOD_abort (rather than METHOD_start), as we
// need to reinitialize this value.

// This macro defines a relocation associated with the provided label to keep
// section GC from discarding it independently of label.
#if !defined(__clang_major__) || __clang_major__ >= 9
#define PINSECTION(label) .reloc 0, R_PPC64_NONE, label
#else
#define PINSECTION(label)
#endif

// TODO(b/141629158):  __rseq_cs only needs to be writeable to allow for
// relocations, but could be read-only for non-PIE builds.
#define DEFINE_UPSTREAM_CS(label)                                 \
  .pushsection __rseq_cs, "aw";                                   \
  .balign 32;                                                     \
  .protected __rseq_cs_##label;                                   \
  .type __rseq_cs_##label,@object;                                \
  .size __rseq_cs_##label,32;                                     \
  __rseq_cs_##label:                                              \
  .long TCMALLOC_PERCPU_RSEQ_VERSION, TCMALLOC_PERCPU_RSEQ_FLAGS; \
  .quad .L##label##_critical_start;                               \
  .quad .L##label##_critical_limit - .L##label##_critical_start;  \
  .quad label##_trampoline;                                       \
  PINSECTION(.L##label##array);                                   \
  .popsection;                                                    \
  .pushsection __rseq_cs_ptr_array, "aw";                         \
  .L##label##array:                                               \
  .quad __rseq_cs_##label;                                        \
  .popsection;                                                    \
  .pushsection rseq_trampoline, "ax";                             \
  SIGN_ABORT();                                                   \
  .globl label##_trampoline;                                      \
  .type  label##_trampoline, @function;                           \
label##_trampoline:                                               \
  .cfi_startproc;                                                 \
  b .L##label##_critical_abort;                                   \
  .cfi_endproc;                                                   \
  .size label##_trampoline, . - label##_trampoline;               \
  .popsection

// With PIE:  We have initial-exec TLS, even in the presence of position
// independent code.
#if !defined(__PIC__) || defined(__PIE__)

#define START_RSEQ(label)                                        \
  .L##label##_gep0:                                              \
  addis %r2, %r12, .TOC.-.L##label##_gep0@ha;                    \
  addi %r2, %r2, .TOC.-.L##label##_gep0@l;                       \
  .L##label##_critical_abort:                                    \
  .L##label##_lep0:                                              \
  .localentry label,.-label;                                     \
  addis %r9, %r2, __rseq_cs_##label@toc@ha;                      \
  addi %r9, %r9, __rseq_cs_##label@toc@l;                        \
  addis %r10, %r13, __rseq_abi@tprel@ha;                         \
  addi %r10, %r10, __rseq_abi@tprel@l;                           \
  std %r9, 8(%r10);                                              \
  .L##label##_critical_start:

#else  /* !defined(__PIC__) || defined(__PIE__) */

// Handle non-initial exec TLS.  When performance matters, we should be using
// initial-exec TLS.
//
// We need to caller-save r3-r8, as they are our arguments to the actual
// restartable sequence code.

#define START_RSEQ(label)                                        \
  .L##label##_gep0:                                              \
  addis %r2, %r12, .TOC.-.L##label##_gep0@ha;                    \
  addi %r2, %r2, .TOC.-.L##label##_gep0@l;                       \
  .L##label##_critical_abort:                                    \
  .L##label##_lep0:                                              \
  .localentry label,.-label;                                     \
  mflr 0;                                                        \
  std  %r0,  0x10(1);                                            \
  std  %r3, -0x10(1);                                            \
  std  %r4, -0x18(1);                                            \
  std  %r5, -0x20(1);                                            \
  std  %r6, -0x28(1);                                            \
  std  %r7, -0x30(1);                                            \
  std  %r8, -0x38(1);                                            \
  stdu %r1, -0x200(1);                                           \
  bl tcmalloc_tls_fetch_pic;                                     \
  nop;                                                           \
  mr   %r10, %r3;                                                \
  addi %r1, %r1, 0x200;                                          \
  ld   %r8, -0x38(1);                                            \
  ld   %r7, -0x30(1);                                            \
  ld   %r6, -0x28(1);                                            \
  ld   %r5, -0x20(1);                                            \
  ld   %r4, -0x18(1);                                            \
  ld   %r3, -0x10(1);                                            \
  ld   %r0,  0x10(1);                                            \
  mtlr 0;                                                        \
  addis %r9, %r2, __rseq_cs_##label@toc@ha;                      \
  addi %r9, %r9, __rseq_cs_##label@toc@l;                        \
  std %r9, 8(%r10);                                              \
  .L##label##_critical_start:

#endif

////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_PerCpuCmpxchg64
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_PerCpuCmpxchg64
.type  TcmallocSlab_Internal_PerCpuCmpxchg64, @function
TcmallocSlab_Internal_PerCpuCmpxchg64:
.LTcmallocSlab_Internal_PerCpuCmpxchg64_entry:
  .cfi_startproc
  // Register use:
  //
  //  *  r3: (Argument: int64) target_cpu
  //  *  r4: (Argument: intptr_t*) p
  //  *  r5: (Argument: intptr_t) old_val
  //  *  r6: (Argument: intptr_t) new_val
  //  *  r7: The current CPU number.
  //  *  r8: The current value of *p.
  //

  START_RSEQ(TcmallocSlab_Internal_PerCpuCmpxchg64)

  // Are we running on the target CPU?
  GET_CPU(%r7)
  cmpd %r7, %r3
  bne .LCAS_wrong_cpu

  // Load the current value of *p.
  ld %r8, 0(%r4)

  // Is the value up to date?
  cmpd %r8, %r5
  bne .LCAS_wrong_value

  // Store the new value, committing the operation.
  std %r6, 0(%r4)
.LTcmallocSlab_Internal_PerCpuCmpxchg64_critical_limit:

  // Return the target CPU, which is already in r3.
  blr

.LCAS_wrong_cpu:
  // Return the current CPU.
  mr %r3, %r7
  blr

.LCAS_wrong_value:
  // Return -1.
  li %r3, -1
  blr

.LTcmallocSlab_Internal_PerCpuCmpxchg64_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_PerCpuCmpxchg64);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_PerCpuCmpxchg64);


////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_Push
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_Push
.type  TcmallocSlab_Internal_Push, @function
TcmallocSlab_Internal_Push:
.LTcmallocSlab_Internal_Push_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) p
  //  *  r6: (Argument: size_t) shift
  //  *  r7: (Argument: uintptr_t) f
  // Return value: current CPU
  // Available r8 r9 r10 r11 r12
  // Note that r12 may be overwritten in rseq_restart_address_internal so
  // cannot be relied upon across restartable sequence boundaries.

  START_RSEQ(TcmallocSlab_Internal_Push)

  GET_CPU(%r8)              // r8  = current CPU, includes MASK operation
  sld %r9, %r8, %r6         // r9  = r8 << shift (r6)
  add %r9, %r3, %r9         // r9  = start of this CPU region
  rldicr %r10, %r4, 3, 60   // r10 = header offset for class size cl (r4)
  add %r10, %r9, %r10       // r10 = slab header addr (class offset + CPU base)
  lhz %r12, 0(%r10)         // r12 = current index
  lhz %r11, 6(%r10)         // r11 = length
  cmpld %cr7, %r11, %r12    // compare current index with length
  ble %cr7, .LTcmallocSlab_Internal_Push_no_capacity
  rldicr %r11, %r12, 3, 60  // r11 = offset of current index
  addi %r12, %r12, 1        // current index += 1
  stdx %r5, %r9, %r11       // store pointer p (r5) into current offset
  sth %r12, 0(%r10)         // update current index

.LTcmallocSlab_Internal_Push_critical_limit:
  mr %r3, %r8               // Return current CPU in r3
  blr

.LTcmallocSlab_Internal_Push_no_capacity:
  mr %r3, %r8               // Place current CPU in r3
  // r7 already contains target function
  b .LPushOverflowTrampoline

.LTcmallocSlab_Internal_Push_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_Push);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_Push);

////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_Push_FixedShift
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_Push_FixedShift
.type  TcmallocSlab_Internal_Push_FixedShift, @function
TcmallocSlab_Internal_Push_FixedShift:
.LTcmallocSlab_Internal_Push_FixedShift_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) p
  //  *  r6: (Argument: uintptr_t) f

  START_RSEQ(TcmallocSlab_Internal_Push_FixedShift)

  GET_CPU_UNMASKED(%r7)   // r7 = unmasked CPU
                          // Mask upper 52 bits of %r7 and shift left in single
                          // operation. Removes the need to have a separate
                          // MASK operation on the critical path.
  clrlsldi %r8, %r7, 52, TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
  add %r8, %r3, %r8       // r8 = start of this CPU region
  rldicr %r9, %r4, 3, 60  // r9 = start of header
  add %r9, %r8, %r9       // r9 = slab header addr
  lhz %r10, 0(%r9)        // r10 = current index
  lhz %r11, 6(%r9)        // r11 = end index
  cmpld %cr7, %r11, %r10  // Check for space
  ble %cr7, .LTcmallocSlab_Internal_Push_FixedShift_no_capacity
  rldicr %r11, %r10, 3, 60  // r11 = offset of current index
  addi %r10, %r10, 1        // current index ++
  stdx %r5, %r8, %r11       // store the item (from r5)
  sth %r10, 0(%r9)          // store current index

.LTcmallocSlab_Internal_Push_FixedShift_critical_limit:
  MASK_CPU(%r3, %r7)     // Return and mask CPU into %r3
  blr

.LTcmallocSlab_Internal_Push_FixedShift_no_capacity:
  MASK_CPU(%r3, %r7)     // Move and mask CPU into %r3
  mr %r7, %r6            // Move target function into r7
  b .LPushOverflowTrampoline

.LTcmallocSlab_Internal_Push_FixedShift_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_Push_FixedShift);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_Push_FixedShift);


////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_Pop
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_Pop
.type  TcmallocSlab_Internal_Pop, @function
TcmallocSlab_Internal_Pop:
.LTcmallocSlab_Internal_Pop_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) f
  //  *  r6: (Argument: size_t) shift
  // Available r7 r8 r9 r10 r11
  // r12 can be used as a temporary within rseq

  START_RSEQ(TcmallocSlab_Internal_Pop)

  GET_CPU(%r7)             // r7 = CPU, includes mask operation
  sld %r12, %r7, %r6       // r12 = CPU shifted by shift (r6)
  add %r12, %r3, %r12      // r12 = start of this CPU region
  rldicr %r8, %r4, 3, 60   // r8 = offset to class size
  add %r8, %r12, %r8       // r8 = slab header addr for class size
  lhz %r9, 0(%r8)          // r9 = current index
  lhz %r10, 4(%r8)         // r10 = begin
  cmpld %cr7, %r10, %r9    // Check that we have items to pop
  bge %cr7, .LTcmallocSlab_Internal_Pop_no_item
  subi %r9, %r9, 1         // r9 = current index --
  rldicr %r10, %r9, 3, 60  // r10 = offset to current item
  ldx %r11, %r12, %r10     // load the item from base + index
  sth %r9, 0(%r8)          // store current index

.LTcmallocSlab_Internal_Pop_critical_limit:
  // Move the item into r3, now that it's safe to do so.
  mr %r3, %r11
  blr

.LTcmallocSlab_Internal_Pop_no_item:
  mr %r3, %r7  // Place CPU into r3
  b .LPopUnderflowTrampoline

.LTcmallocSlab_Internal_Pop_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_Pop);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_Pop);

////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_Pop_FixedShift
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_Pop_FixedShift
.type  TcmallocSlab_Internal_Pop_FixedShift, @function
TcmallocSlab_Internal_Pop_FixedShift:
.LTcmallocSlab_Internal_Pop_FixedShift_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) f

  START_RSEQ(TcmallocSlab_Internal_Pop_FixedShift)

  GET_CPU_UNMASKED(%r6)  // r6 = current CPU
                         // Following instruction combines mask and shift
  clrlsldi %r7, %r6, 52, TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
			 // r7 = header offset
  add %r7, %r3, %r7       // r7 = start of this CPU region
  rldicr %r8, %r4, 3, 60  // r8 = offset of size class
  add %r8, %r7, %r8       // r8 = slab header addr
  lhz %r9, 0(%r8)         // r9 = current index
  lhz %r10, 4(%r8)        // r10 = begin index
  cmpld %cr7, %r10, %r9   // Check that there are elements available
  bge %cr7, .LTcmallocSlab_Internal_Pop_FixedShift_no_item
  subi %r9, %r9, 1         // current index --
  rldicr %r10, %r9, 3, 60  // r10 = offset of current index
  ldx %r11, %r7, %r10      // r11 = load the item
  sth %r9, 0(%r8)          // update current index

.LTcmallocSlab_Internal_Pop_FixedShift_critical_limit:
  // Move the item into r3, now that it's safe to do so.
  mr %r3, %r11
  blr

.LTcmallocSlab_Internal_Pop_FixedShift_no_item:
  MASK_CPU(%r3, %r6)          // Extract CPU from unmasked value in %r6
  b .LPopUnderflowTrampoline

.LTcmallocSlab_Internal_Pop_FixedShift_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_Pop_FixedShift);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_Pop_FixedShift);

////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_PushBatch_FixedShift
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_PushBatch_FixedShift
.type  TcmallocSlab_Internal_PushBatch_FixedShift, @function
TcmallocSlab_Internal_PushBatch_FixedShift:
.LTcmallocSlab_Internal_PushBatch_FixedShift_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) batch
  //  *  r6: (Argument: uintptr_t) len

  START_RSEQ(TcmallocSlab_Internal_PushBatch_FixedShift)

  GET_CPU_UNMASKED(%r7)
  clrlsldi %r8, %r7, 52, TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
  add %r8, %r3, %r8    // r8 - start of this CPU region
  sldi %r9, %r4, 3
  add %r9, %r8, %r9    // r9 - slab header addr
  lhz %r10, 0(%r9)     // r10 - current
  lhz %r11, 6(%r9)     // r11 - end
  sldi %r7, %r6, 3   // r7 - len * 8
  cmpld %cr7, %r11, %r10  // current < end?
  ble %cr7, .LTcmallocSlab_Internal_PushBatch_FixedShift_critical_limit
  sub %r11, %r11, %r10  // r11 - available capacity
  // r11 = min(r11, r6)
  cmpld %cr7, %r6, %r11
  bge %cr7, .LTcmallocSlab_Internal_PushBatch_FixedShift_min
  mr %r11, %r6
.LTcmallocSlab_Internal_PushBatch_FixedShift_min:
  add %r11, %r10, %r11
  sldi %r11, %r11, 3
  sldi %r10, %r10, 3

  // At this point:
  // r5 - batch, r7 - offset in the batch
  // r8 - cpu region, r10 - offset into the cpu region, r11 - limit of offset
.LTcmallocSlab_Internal_PushBatch_FixedShift_loop:
  subi %r7, %r7, 8
  ldx %r12, %r5, %r7  // load the item
  stdx %r12, %r8, %r10  // store the item
  addi %r10, %r10, 8
  cmpld %cr7, %r10, %r11
  bne %cr7, .LTcmallocSlab_Internal_PushBatch_FixedShift_loop
  rotrdi %r10, %r10, 3
  sth %r10, 0(%r9)  // update current

.LTcmallocSlab_Internal_PushBatch_FixedShift_critical_limit:
  // return r6 - r7 / 8
  rotrdi %r7, %r7, 3
  sub %r3, %r6, %r7
  blr

.LTcmallocSlab_Internal_PushBatch_FixedShift_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_PushBatch_FixedShift);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_PushBatch_FixedShift);

////////////////////////////////////////////////////////////////////////
// TcmallocSlab_Internal_PopBatch_FixedShift
////////////////////////////////////////////////////////////////////////

.globl TcmallocSlab_Internal_PopBatch_FixedShift
.type  TcmallocSlab_Internal_PopBatch_FixedShift, @function
TcmallocSlab_Internal_PopBatch_FixedShift:
.LTcmallocSlab_Internal_PopBatch_FixedShift_entry:
  .cfi_startproc
  // Arguments use:
  //  *  r3: (Argument: Slabs*) cpu_0_slab_ptr
  //  *  r4: (Argument: uintptr_t) cl
  //  *  r5: (Argument: uintptr_t) batch
  //  *  r6: (Argument: uintptr_t) len

  START_RSEQ(TcmallocSlab_Internal_PopBatch_FixedShift)

  GET_CPU_UNMASKED(%r7)
  clrlsldi %r7, %r7, 52, TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
  add %r7, %r3, %r7    // r7 - start of this CPU region
  sldi %r8, %r4, 3
  add %r8, %r7, %r8    // r8 - slab header addr
  lhz %r9, 0(%r8)      // r9 - current
  lhz %r10, 4(%r8)     // r10 - begin
  li %r11, 0           // current position in batch
  cmpld %cr7, %r10, %r9
  bge %cr7, .LTcmallocSlab_Internal_PopBatch_FixedShift_critical_limit
  sub %r10, %r9, %r10  // r10 - available items
  // r10 = min(r10, r6)
  cmpld %cr7, %r6, %r10
  bge %cr7, .LTcmallocSlab_Internal_PopBatch_FixedShift_min
  mr %r10, %r6
.LTcmallocSlab_Internal_PopBatch_FixedShift_min:
  sub %r10, %r9, %r10
  sldi %r10, %r10, 3
  sldi %r9, %r9, 3

  // At this point:
  // r5 - batch, r11 - offset in the batch
  // r7 - cpu region, r9 - offset into the cpu region, r10 - limit of offset
.LTcmallocSlab_Internal_PopBatch_FixedShift_loop:
  subi %r9, %r9, 8
  ldx %r12, %r7, %r9  // load the item
  stdx %r12, %r5, %r11  // store the item
  addi %r11, %r11, 8
  cmpld %cr7, %r9, %r10
  bne %cr7, .LTcmallocSlab_Internal_PopBatch_FixedShift_loop
  rotrdi %r9, %r9, 3
  sth %r9, 0(%r8)  // update current

.LTcmallocSlab_Internal_PopBatch_FixedShift_critical_limit:
  rotrdi %r3, %r11, 3
  blr

.LTcmallocSlab_Internal_PopBatch_FixedShift_function_limit:
  .cfi_endproc
ENCODE_SIZE(TcmallocSlab_Internal_PopBatch_FixedShift);
DEFINE_UPSTREAM_CS(TcmallocSlab_Internal_PopBatch_FixedShift);

  // Input: r7 points to the function to tail call. r3...r6 are args for it.
.LPushOverflowTrampoline:
  mtctr %r7
  mr %r12, %r7  // Callee expects r12 to point to its first instruction.
  bctr

  // Input: r5 points to the function to tail call. r3...r4 are args for it.
.LPopUnderflowTrampoline:
  mtctr %r5
  mr %r12, %r5  // Callee expects r12 to point to its first instruction.
  bctr

.section .note.GNU-stack,"",%progbits

