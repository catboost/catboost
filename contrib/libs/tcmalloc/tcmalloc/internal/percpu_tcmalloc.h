// Copyright 2019 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TCMALLOC_INTERNAL_PERCPU_TCMALLOC_H_
#define TCMALLOC_INTERNAL_PERCPU_TCMALLOC_H_

#include <bits/wordsize.h>

#include <atomic>
#include <cstring>

#include "absl/base/dynamic_annotations.h"
#include "absl/base/internal/sysinfo.h"
#include "tcmalloc/internal/mincore.h"
#include "tcmalloc/internal/percpu.h"

#if defined(TCMALLOC_PERCPU_USE_RSEQ)
#if !defined(__clang__)
#define TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO 1
#elif __clang_major__ >= 9 && !__has_feature(speculative_load_hardening)
// asm goto requires the use of Clang 9 or newer:
// https://releases.llvm.org/9.0.0/tools/clang/docs/ReleaseNotes.html#c-language-changes-in-clang
//
// SLH (Speculative Load Hardening) builds do not support asm goto.  We can
// detect these compilation modes since
// https://github.com/llvm/llvm-project/commit/379e68a763097bed55556c6dc7453e4b732e3d68.
#define TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO 1
#if __clang_major__ >= 11
#define TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT 1
#endif

#else
#define TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO 0
#endif
#else
#define TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO 0
#endif

namespace tcmalloc {

struct PerCPUMetadataState {
  size_t virtual_size;
  size_t resident_size;
};

namespace subtle {
namespace percpu {

// Tcmalloc slab for per-cpu caching mode.
// Conceptually it is equivalent to an array of NumClasses PerCpuSlab's,
// and in fallback implementation it is implemented that way. But optimized
// implementation uses more compact layout and provides faster operations.
//
// Methods of this type must only be used in threads where it is known that the
// percpu primitives are available and percpu::IsFast() has previously returned
// 'true'.
//
// The template parameter Shift indicates the number of bits to shift the
// the CPU id in order to get the location of the per-cpu slab. If this
// parameter matches TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT as set in
// percpu_intenal.h then the assembly language versions of push/pop batch
// can be used; otherwise batch operations are emulated.
template <size_t Shift, size_t NumClasses>
class TcmallocSlab {
 public:
  TcmallocSlab() {}

  // Init must be called before any other methods.
  // <alloc> is memory allocation callback (e.g. malloc).
  // <capacity> callback returns max capacity for size class <cl>.
  // <lazy> indicates that per-CPU slabs should be populated on demand
  //
  // Initial capacity is 0 for all slabs.
  void Init(void*(alloc)(size_t size), size_t (*capacity)(size_t cl),
            bool lazy);

  // Only may be called if Init(..., lazy = true) was used.
  void InitCPU(int cpu, size_t (*capacity)(size_t cl));

  // For tests.
  void Destroy(void(free)(void*));

  // Number of elements in cpu/cl slab.
  size_t Length(int cpu, size_t cl) const;

  // Number of elements (currently) allowed in cpu/cl slab.
  size_t Capacity(int cpu, size_t cl) const;

  // If running on cpu, increment the cpu/cl slab's capacity to no greater than
  // min(capacity+len, max_cap) and return the increment applied. Otherwise
  // return 0. Note: max_cap must be the same as returned by capacity callback
  // passed to Init.
  size_t Grow(int cpu, size_t cl, size_t len, size_t max_cap);

  // If running on cpu, decrement the cpu/cl slab's capacity to no less than
  // max(capacity-len, 0) and return the actual decrement applied. Otherwise
  // return 0.
  size_t Shrink(int cpu, size_t cl, size_t len);

  // Add an item (which must be non-zero) to the current CPU's slab. Returns
  // true if add succeeds. Otherwise invokes <f> and returns false (assuming
  // that <f> returns negative value).
  bool Push(size_t cl, void* item, OverflowHandler f);

  // Remove an item (LIFO) from the current CPU's slab. If the slab is empty,
  // invokes <f> and returns its result.
  void* Pop(size_t cl, UnderflowHandler f);

  // Add up to <len> items to the current cpu slab from the array located at
  // <batch>. Returns the number of items that were added (possibly 0). All
  // items not added will be returned at the start of <batch>. Items are only
  // not added if there is no space on the current cpu.
  // REQUIRES: len > 0.
  size_t PushBatch(size_t cl, void** batch, size_t len);

  // Pop up to <len> items from the current cpu slab and return them in <batch>.
  // Returns the number of items actually removed.
  // REQUIRES: len > 0.
  size_t PopBatch(size_t cl, void** batch, size_t len);

  // Remove all items (of all classes) from <cpu>'s slab; reset capacity for all
  // classes to zero.  Then, for each sizeclass, invoke
  // DrainHandler(drain_ctx, cl, <items from slab>, <previous slab capacity>);
  //
  // It is invalid to concurrently execute Drain() for the same CPU; calling
  // Push/Pop/Grow/Shrink concurrently (even on the same CPU) is safe.
  typedef void (*DrainHandler)(void* drain_ctx, size_t cl, void** batch,
                               size_t n, size_t cap);
  void Drain(int cpu, void* drain_ctx, DrainHandler f);

  PerCPUMetadataState MetadataMemoryUsage() const;

  // We use a single continuous region of memory for all slabs on all CPUs.
  // This region is split into NumCPUs regions of size kPerCpuMem (256k).
  // First NumClasses words of each CPU region are occupied by slab
  // headers (Header struct). The remaining memory contain slab arrays.
  struct Slabs {
    std::atomic<int64_t> header[NumClasses];
    void* mem[((1ul << Shift) - sizeof(header)) / sizeof(void*)];
  };
  static_assert(sizeof(Slabs) == (1ul << Shift), "Slabs has unexpected size");

 private:
  // Slab header (packed, atomically updated 64-bit).
  struct Header {
    // All values are word offsets from per-CPU region start.
    // The array is [begin, end).
    uint16_t current;
    // Copy of end. Updated by Shrink/Grow, but is not overwritten by Drain.
    uint16_t end_copy;
    // Lock updates only begin and end with a 32-bit write.
    uint16_t begin;
    uint16_t end;

    // Lock is used by Drain to stop concurrent mutations of the Header.
    // Lock sets begin to 0xffff and end to 0, which makes Push and Pop fail
    // regardless of current value.
    bool IsLocked() const;
    void Lock();
  };

  // We cast Header to std::atomic<int64_t>.
  static_assert(sizeof(Header) == sizeof(std::atomic<int64_t>),
                "bad Header size");

  Slabs* slabs_;

  Slabs* CpuMemoryStart(int cpu) const;
  std::atomic<int64_t>* GetHeader(int cpu, size_t cl) const;
  static Header LoadHeader(std::atomic<int64_t>* hdrp);
  static void StoreHeader(std::atomic<int64_t>* hdrp, Header hdr);
  static int CompareAndSwapHeader(int cpu, std::atomic<int64_t>* hdrp,
                                  Header old, Header hdr);
};

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::Length(int cpu,
                                                      size_t cl) const {
  Header hdr = LoadHeader(GetHeader(cpu, cl));
  return hdr.IsLocked() ? 0 : hdr.current - hdr.begin;
}

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::Capacity(int cpu,
                                                        size_t cl) const {
  Header hdr = LoadHeader(GetHeader(cpu, cl));
  return hdr.IsLocked() ? 0 : hdr.end - hdr.begin;
}

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::Grow(int cpu, size_t cl,
                                                    size_t len,
                                                    size_t max_cap) {
  std::atomic<int64_t>* hdrp = GetHeader(cpu, cl);
  for (;;) {
    Header old = LoadHeader(hdrp);
    if (old.IsLocked() || old.end - old.begin == max_cap) {
      return 0;
    }
    uint16_t n = std::min<uint16_t>(len, max_cap - (old.end - old.begin));
    Header hdr = old;
    hdr.end += n;
    hdr.end_copy += n;
    const int ret = CompareAndSwapHeader(cpu, hdrp, old, hdr);
    if (ret == cpu) {
      return n;
    } else if (ret >= 0) {
      return 0;
    }
  }
}

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::Shrink(int cpu, size_t cl,
                                                      size_t len) {
  std::atomic<int64_t>* hdrp = GetHeader(cpu, cl);
  for (;;) {
    Header old = LoadHeader(hdrp);
    if (old.IsLocked() || old.current == old.end) {
      return 0;
    }
    uint16_t n = std::min<uint16_t>(len, old.end - old.current);
    Header hdr = old;
    hdr.end -= n;
    hdr.end_copy -= n;
    const int ret = CompareAndSwapHeader(cpu, hdrp, old, hdr);
    if (ret == cpu) {
      return n;
    } else if (ret >= 0) {
      return 0;
    }
  }
}

#define TCMALLOC_PERCPU_XSTRINGIFY(s) #s
#define TCMALLOC_PERCPU_STRINGIFY(s) PERCPU_XSTRINGIFY(s)

#if defined(__x86_64__)
template <size_t Shift, size_t NumClasses>
static inline ABSL_ATTRIBUTE_ALWAYS_INLINE int TcmallocSlab_Push(
    typename TcmallocSlab<Shift, NumClasses>::Slabs* slabs, size_t cl,
    void* item, OverflowHandler f) {
#if TCMALLOC_PERCPU_USE_RSEQ
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO
  asm goto(
#else
  bool overflow;
  asm volatile(
#endif
      // TODO(b/141629158):  __rseq_cs only needs to be writeable to allow for
      // relocations, but could be read-only for non-PIE builds.
      ".pushsection __rseq_cs, \"aw?\"\n"
      ".balign 32\n"
      ".local __rseq_cs_TcmallocSlab_Push_%=\n"
      ".type __rseq_cs_TcmallocSlab_Push_%=,@object\n"
      ".size __rseq_cs_TcmallocSlab_Push_%=,32\n"
      "__rseq_cs_TcmallocSlab_Push_%=:\n"
      ".long 0x0\n"
      ".long 0x0\n"
      ".quad 4f\n"
      ".quad 5f - 4f\n"
      ".quad 2f\n"
      ".popsection\n"
#if !defined(__clang_major__) || __clang_major__ >= 9
      ".reloc 0, R_X86_64_NONE, 1f\n"
#endif
      ".pushsection __rseq_cs_ptr_array, \"aw?\"\n"
      "1:\n"
      ".balign 8;"
      ".quad __rseq_cs_TcmallocSlab_Push_%=\n"
      // Force this section to be retained.  It is for debugging, but is
      // otherwise not referenced.
      ".popsection\n"
      ".pushsection .text.unlikely, \"ax?\"\n"
      ".byte 0x0f, 0x1f, 0x05\n"
      ".long %c[rseq_sig]\n"
      ".local TcmallocSlab_Push_trampoline_%=\n"
      ".type TcmallocSlab_Push_trampoline_%=,@function\n"
      "TcmallocSlab_Push_trampoline_%=:\n"
      "2:\n"
      "jmp 3f\n"
      ".popsection\n"
      // Prepare
      //
      // TODO(b/151503411):  Pending widespread availability of LLVM's asm
      // goto with output contraints
      // (https://github.com/llvm/llvm-project/commit/23c2a5ce33f0), we can
      // return the register allocations to the compiler rather than using
      // explicit clobbers.  Prior to this, blocks which use asm goto cannot
      // also specify outputs.
      //
      // r10: Scratch
      // r11: Current
      "3:\n"
      "lea __rseq_cs_TcmallocSlab_Push_%=(%%rip), %%r10\n"
      "mov %%r10, %c[rseq_cs_offset](%[rseq_abi])\n"
      // Start
      "4:\n"
      // scratch = __rseq_abi.cpu_id;
      "movzwl (%[rseq_abi], %[rseq_cpu_offset]), %%r10d\n"
      // scratch = slabs + scratch
      "shl %[shift], %%r10\n"
      "add %[slabs], %%r10\n"
      // r11 = slabs->current;
      "movzwq (%%r10, %[cl], 8), %%r11\n"
      // if (ABSL_PREDICT_FALSE(r11 >= slabs->end)) { goto overflow; }
      "cmp 6(%%r10, %[cl], 8), %%r11w\n"
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO
      "jae %l[overflow_label]\n"
#else
      "jae 5f\n"
  // Important! code below this must not affect any flags (i.e.: ccae)
  // If so, the above code needs to explicitly set a ccae return value.
#endif
      "mov %[item], (%%r10, %%r11, 8)\n"
      "lea 1(%%r11), %%r11\n"
      "mov %%r11w, (%%r10, %[cl], 8)\n"
      // Commit
      "5:\n"
      :
#if !TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO
      [overflow] "=@ccae"(overflow)
#endif
      : [rseq_abi] "r"(&__rseq_abi),
        [rseq_cs_offset] "n"(offsetof(kernel_rseq, rseq_cs)),
        [rseq_cpu_offset] "r"(tcmalloc_virtual_cpu_id_offset),
        [rseq_sig] "in"(TCMALLOC_PERCPU_RSEQ_SIGNATURE), [shift] "in"(Shift),
        [slabs] "r"(slabs), [cl] "r"(cl), [item] "r"(item)
      : "cc", "memory", "r10", "r11"
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO
      : overflow_label
#endif
  );
#if !TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO
  if (ABSL_PREDICT_FALSE(overflow)) {
    goto overflow_label;
  }
#endif
  return 0;
overflow_label:
  // As of 3/2020, LLVM's asm goto (even with output constraints) only provides
  // values for the fallthrough path.  The values on the taken branches are
  // undefined.
  int cpu = VirtualRseqCpuId();
  return f(cpu, cl, item);
#else  // !TCMALLOC_PERCPU_USE_RSEQ
    __builtin_unreachable();
#endif  // !TCMALLOC_PERCPU_USE_RSEQ
}
#endif  // defined(__x86_64__)

template <size_t Shift, size_t NumClasses>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool TcmallocSlab<Shift, NumClasses>::Push(
    size_t cl, void* item, OverflowHandler f) {
  ASSERT(item != nullptr);
#if defined(__x86_64__)
  return TcmallocSlab_Push<Shift, NumClasses>(slabs_, cl, item, f) >= 0;
#else
  if (Shift == TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT) {
    return TcmallocSlab_Push_FixedShift(slabs_, cl, item, f) >= 0;
  } else {
    return TcmallocSlab_Push(slabs_, cl, item, Shift, f) >= 0;
  }
#endif
}

#if defined(__x86_64__)
template <size_t Shift, size_t NumClasses>
static inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* TcmallocSlab_Pop(
    typename TcmallocSlab<Shift, NumClasses>::Slabs* slabs, size_t cl,
    UnderflowHandler f) {
#if TCMALLOC_PERCPU_USE_RSEQ
  void* result;
  void* scratch;
  uintptr_t current;
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
  asm goto
#else
  bool underflow;
  asm
#endif
      (
          // TODO(b/141629158):  __rseq_cs only needs to be writeable to allow
          // for relocations, but could be read-only for non-PIE builds.
          ".pushsection __rseq_cs, \"aw?\"\n"
          ".balign 32\n"
          ".local __rseq_cs_TcmallocSlab_Pop_%=\n"
          ".type __rseq_cs_TcmallocSlab_Pop_%=,@object\n"
          ".size __rseq_cs_TcmallocSlab_Pop_%=,32\n"
          "__rseq_cs_TcmallocSlab_Pop_%=:\n"
          ".long 0x0\n"
          ".long 0x0\n"
          ".quad 4f\n"
          ".quad 5f - 4f\n"
          ".quad 2f\n"
          ".popsection\n"
#if !defined(__clang_major__) || __clang_major__ >= 9
          ".reloc 0, R_X86_64_NONE, 1f\n"
#endif
          ".pushsection __rseq_cs_ptr_array, \"aw?\"\n"
          "1:\n"
          ".balign 8;"
          ".quad __rseq_cs_TcmallocSlab_Pop_%=\n"
          // Force this section to be retained.  It is for debugging, but is
          // otherwise not referenced.
          ".popsection\n"
          ".pushsection .text.unlikely, \"ax?\"\n"
          ".byte 0x0f, 0x1f, 0x05\n"
          ".long %c[rseq_sig]\n"
          ".local TcmallocSlab_Pop_trampoline_%=\n"
          ".type TcmallocSlab_Pop_trampoline_%=,@function\n"
          "TcmallocSlab_Pop_trampoline_%=:\n"
          "2:\n"
          "jmp 3f\n"
          ".popsection\n"
          // Prepare
          "3:\n"
          "lea __rseq_cs_TcmallocSlab_Pop_%=(%%rip), %[scratch];\n"
          "mov %[scratch], %c[rseq_cs_offset](%[rseq_abi])\n"
          // Start
          "4:\n"
          // scratch = __rseq_abi.cpu_id;
          "movzwl (%[rseq_abi], %[rseq_cpu_offset]), %k[scratch]\n"
          // scratch = slabs + scratch
          "shl %[shift], %[scratch]\n"
          "add %[slabs], %[scratch]\n"
          // current = scratch->header[cl].current;
          "movzwq (%[scratch], %[cl], 8), %[current]\n"
          // if (ABSL_PREDICT_FALSE(scratch->header[cl].begin > current))
          "cmp 4(%[scratch], %[cl], 8), %w[current]\n"
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
          "jbe %l[underflow_path]\n"
#else
          "jbe 5f\n"
  // Important! code below this must not affect any flags (i.e.: ccbe)
  // If so, the above code needs to explicitly set a ccbe return value.
#endif
          "mov -16(%[scratch], %[current], 8), %[result]\n"
          // A note about prefetcht0 in Pop:  While this prefetch may appear
          // costly, trace analysis shows the target is frequently used
          // (b/70294962). Stalling on a TLB miss at the prefetch site (which
          // has no deps) and prefetching the line async is better than stalling
          // at the use (which may have deps) to fill the TLB and the cache
          // miss.
          "prefetcht0 (%[result])\n"
          "movq -8(%[scratch], %[current], 8), %[result]\n"
          "lea -1(%[current]), %[current]\n"
          "mov %w[current], (%[scratch], %[cl], 8)\n"
          // Commit
          "5:\n"
          : [result] "=&r"(result),
#if !TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
            [underflow] "=@ccbe"(underflow),
#endif
            [scratch] "=&r"(scratch), [current] "=&r"(current)
          : [rseq_abi] "r"(&__rseq_abi),
            [rseq_cs_offset] "n"(offsetof(kernel_rseq, rseq_cs)),
            [rseq_cpu_offset] "r"(tcmalloc_virtual_cpu_id_offset),
            [rseq_sig] "n"(TCMALLOC_PERCPU_RSEQ_SIGNATURE), [shift] "n"(Shift),
            [slabs] "r"(slabs), [cl] "r"(cl)
          : "cc", "memory"
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
          : underflow_path
#endif
      );
#if !TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
  if (ABSL_PREDICT_FALSE(underflow)) {
    goto underflow_path;
  }
#endif

  return result;
underflow_path:
#if TCMALLOC_PERCPU_USE_RSEQ_ASM_GOTO_OUTPUT
  // As of 3/2020, LLVM's asm goto (even with output constraints) only provides
  // values for the fallthrough path.  The values on the taken branches are
  // undefined.
  int cpu = VirtualRseqCpuId();
#else
  // With asm goto--without output constraints--the value of scratch is
  // well-defined by the compiler and our implementation.  As an optimization on
  // this case, we can avoid looking up cpu_id again, by undoing the
  // transformation of cpu_id to the value of scratch.
  int cpu = reinterpret_cast<typename TcmallocSlab<Shift, NumClasses>::Slabs*>(
                scratch) -
            slabs;
#endif
  return f(cpu, cl);
#else  // !TCMALLOC_PERCPU_USE_RSEQ
    __builtin_unreachable();
#endif  // !TCMALLOC_PERCPU_USE_RSEQ
}
#endif  // defined(__x86_64__)

#undef TCMALLOC_PERCPU_STRINGIFY
#undef TCMALLOC_PERCPU_XSTRINGIFY

template <size_t Shift, size_t NumClasses>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* TcmallocSlab<Shift, NumClasses>::Pop(
    size_t cl, UnderflowHandler f) {
#if defined(__x86_64__)
  return TcmallocSlab_Pop<Shift, NumClasses>(slabs_, cl, f);
#else
  if (Shift == TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT) {
    return TcmallocSlab_Pop_FixedShift(slabs_, cl, f);
  } else {
    return TcmallocSlab_Pop(slabs_, cl, f, Shift);
  }
#endif
}

static inline void* NoopUnderflow(int cpu, size_t cl) { return nullptr; }

static inline int NoopOverflow(int cpu, size_t cl, void* item) { return -1; }

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::PushBatch(size_t cl,
                                                         void** batch,
                                                         size_t len) {
  ASSERT(len != 0);
  if (Shift == TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT) {
#if TCMALLOC_PERCPU_USE_RSEQ
    // TODO(b/159923407): TcmallocSlab_PushBatch_FixedShift needs to be
    // refactored to take a 5th parameter (tcmalloc_virtual_cpu_id_offset) to
    // avoid needing to dispatch on two separate versions of the same function
    // with only minor differences between them.
    switch (tcmalloc_virtual_cpu_id_offset) {
      case offsetof(kernel_rseq, cpu_id):
        return TcmallocSlab_PushBatch_FixedShift(slabs_, cl, batch, len);
#ifdef __x86_64__
      case offsetof(kernel_rseq, vcpu_id):
        return TcmallocSlab_PushBatch_FixedShift_VCPU(slabs_, cl, batch, len);
#endif  // __x86_64__
      default:
        __builtin_unreachable();
    }
#else  // !TCMALLOC_PERCPU_USE_RSEQ
    __builtin_unreachable();
#endif  // !TCMALLOC_PERCPU_USE_RSEQ
  } else {
    size_t n = 0;
    // Push items until either all done or a push fails
    while (n < len && Push(cl, batch[len - 1 - n], NoopOverflow)) {
      n++;
    }
    return n;
  }
}

template <size_t Shift, size_t NumClasses>
inline size_t TcmallocSlab<Shift, NumClasses>::PopBatch(size_t cl, void** batch,
                                                        size_t len) {
  ASSERT(len != 0);
  size_t n = 0;
  if (Shift == TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT) {
#if TCMALLOC_PERCPU_USE_RSEQ
    // TODO(b/159923407): TcmallocSlab_PopBatch_FixedShift needs to be
    // refactored to take a 5th parameter (tcmalloc_virtual_cpu_id_offset) to
    // avoid needing to dispatch on two separate versions of the same function
    // with only minor differences between them.
    switch (tcmalloc_virtual_cpu_id_offset) {
      case offsetof(kernel_rseq, cpu_id):
        n = TcmallocSlab_PopBatch_FixedShift(slabs_, cl, batch, len);
        break;
#ifdef __x86_64__
      case offsetof(kernel_rseq, vcpu_id):
        n = TcmallocSlab_PopBatch_FixedShift_VCPU(slabs_, cl, batch, len);
        break;
#endif  // __x86_64__
      default:
        __builtin_unreachable();
    }

    // PopBatch is implemented in assembly, msan does not know that the returned
    // batch is initialized.
    ANNOTATE_MEMORY_IS_INITIALIZED(batch, n * sizeof(batch[0]));
#else  // !TCMALLOC_PERCPU_USE_RSEQ
    __builtin_unreachable();
#endif  // !TCMALLOC_PERCPU_USE_RSEQ
  } else {
    // Pop items until either all done or a pop fails
    while (n < len && (batch[n] = Pop(cl, NoopUnderflow))) {
      n++;
    }
  }
  return n;
}

template <size_t Shift, size_t NumClasses>
inline typename TcmallocSlab<Shift, NumClasses>::Slabs*
TcmallocSlab<Shift, NumClasses>::CpuMemoryStart(int cpu) const {
  return &slabs_[cpu];
}

template <size_t Shift, size_t NumClasses>
inline std::atomic<int64_t>* TcmallocSlab<Shift, NumClasses>::GetHeader(
    int cpu, size_t cl) const {
  return &CpuMemoryStart(cpu)->header[cl];
}

template <size_t Shift, size_t NumClasses>
inline typename TcmallocSlab<Shift, NumClasses>::Header
TcmallocSlab<Shift, NumClasses>::LoadHeader(std::atomic<int64_t>* hdrp) {
  uint64_t raw = hdrp->load(std::memory_order_relaxed);
  Header hdr;
  memcpy(&hdr, &raw, sizeof(hdr));
  return hdr;
}

template <size_t Shift, size_t NumClasses>
inline void TcmallocSlab<Shift, NumClasses>::StoreHeader(
    std::atomic<int64_t>* hdrp, Header hdr) {
  uint64_t raw;
  memcpy(&raw, &hdr, sizeof(raw));
  hdrp->store(raw, std::memory_order_relaxed);
}

template <size_t Shift, size_t NumClasses>
inline int TcmallocSlab<Shift, NumClasses>::CompareAndSwapHeader(
    int cpu, std::atomic<int64_t>* hdrp, Header old, Header hdr) {
#if __WORDSIZE == 64
  uint64_t old_raw, new_raw;
  memcpy(&old_raw, &old, sizeof(old_raw));
  memcpy(&new_raw, &hdr, sizeof(new_raw));
  return CompareAndSwapUnsafe(cpu, hdrp, static_cast<intptr_t>(old_raw),
                              static_cast<intptr_t>(new_raw));
#else
  Crash(kCrash, __FILE__, __LINE__, "This architecture is not supported.");
#endif
}

template <size_t Shift, size_t NumClasses>
inline bool TcmallocSlab<Shift, NumClasses>::Header::IsLocked() const {
  return begin == 0xffffu;
}

template <size_t Shift, size_t NumClasses>
inline void TcmallocSlab<Shift, NumClasses>::Header::Lock() {
  // Write 0xffff to begin and 0 to end. This blocks new Push'es and Pop's.
  // Note: we write only 4 bytes. The first 4 bytes are left intact.
  // See Drain method for details. tl;dr: C++ does not allow us to legally
  // express this without undefined behavior.
  std::atomic<int32_t>* p = reinterpret_cast<std::atomic<int32_t>*>(&begin);
  Header hdr;
  hdr.begin = 0xffffu;
  hdr.end = 0;
  int32_t raw;
  memcpy(&raw, &hdr.begin, sizeof(raw));
  p->store(raw, std::memory_order_relaxed);
}

template <size_t Shift, size_t NumClasses>
void TcmallocSlab<Shift, NumClasses>::Init(void*(alloc)(size_t size),
                                           size_t (*capacity)(size_t cl),
                                           bool lazy) {
  size_t mem_size = absl::base_internal::NumCPUs() * (1ul << Shift);
  void* backing = alloc(mem_size);
  // MSan does not see writes in assembly.
  ANNOTATE_MEMORY_IS_INITIALIZED(backing, mem_size);
  if (!lazy) {
    memset(backing, 0, mem_size);
  }
  slabs_ = static_cast<Slabs*>(backing);
  size_t bytes_used = 0;
  for (int cpu = 0; cpu < absl::base_internal::NumCPUs(); ++cpu) {
    bytes_used += sizeof(std::atomic<int64_t>) * NumClasses;
    void** elems = slabs_[cpu].mem;

    for (size_t cl = 0; cl < NumClasses; ++cl) {
      size_t cap = capacity(cl);
      CHECK_CONDITION(static_cast<uint16_t>(cap) == cap);

      if (cap == 0) {
        continue;
      }

      if (cap) {
        if (!lazy) {
          // In Pop() we prefetch the item a subsequent Pop() would return; this
          // is slow if it's not a valid pointer. To avoid this problem when
          // popping the last item, keep one fake item before the actual ones
          // (that points, safely, to itself.)
          *elems = elems;
          elems++;
        }

        // One extra element for prefetch
        bytes_used += (cap + 1) * sizeof(void*);
      }

      if (!lazy) {
        // TODO(ckennelly): Consolidate this initialization logic with that in
        // InitCPU.
        size_t offset = elems - reinterpret_cast<void**>(CpuMemoryStart(cpu));
        CHECK_CONDITION(static_cast<uint16_t>(offset) == offset);

        Header hdr;
        hdr.current = offset;
        hdr.begin = offset;
        hdr.end = offset;
        hdr.end_copy = offset;

        StoreHeader(GetHeader(cpu, cl), hdr);
      }

      elems += cap;
      CHECK_CONDITION(reinterpret_cast<char*>(elems) -
                          reinterpret_cast<char*>(CpuMemoryStart(cpu)) <=
                      (1 << Shift));
    }
  }
  // Check for less than 90% usage of the reserved memory
  if (bytes_used * 10 < 9 * mem_size) {
    Log(kLog, __FILE__, __LINE__, "Bytes used per cpu of available", bytes_used,
        mem_size);
  }
}

template <size_t Shift, size_t NumClasses>
void TcmallocSlab<Shift, NumClasses>::InitCPU(int cpu,
                                              size_t (*capacity)(size_t cl)) {
  // TODO(ckennelly): Consolidate this logic with Drain.
  // Phase 1: verify no header is locked
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    Header hdr = LoadHeader(GetHeader(cpu, cl));
    CHECK_CONDITION(!hdr.IsLocked());
  }

  // Phase 2: Stop concurrent mutations.  Locking ensures that there exists no
  // value of current such that begin < current.
  for (bool done = false; !done;) {
    for (size_t cl = 0; cl < NumClasses; ++cl) {
      // Note: this reinterpret_cast and write in Lock lead to undefined
      // behavior, because the actual object type is std::atomic<int64_t>. But
      // C++ does not allow to legally express what we need here: atomic writes
      // of different sizes.
      reinterpret_cast<Header*>(GetHeader(cpu, cl))->Lock();
    }
    FenceCpu(cpu);
    done = true;
    for (size_t cl = 0; cl < NumClasses; ++cl) {
      Header hdr = LoadHeader(GetHeader(cpu, cl));
      if (!hdr.IsLocked()) {
        // Header was overwritten by Grow/Shrink. Retry.
        done = false;
        break;
      }
    }
  }

  // Phase 3: Initialize prefetch target and compute the offsets for the
  // boundaries of each size class' cache.
  void** elems = slabs_[cpu].mem;
  uint16_t begin[NumClasses];
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    size_t cap = capacity(cl);
    CHECK_CONDITION(static_cast<uint16_t>(cap) == cap);

    if (cap) {
      // In Pop() we prefetch the item a subsequent Pop() would return; this is
      // slow if it's not a valid pointer. To avoid this problem when popping
      // the last item, keep one fake item before the actual ones (that points,
      // safely, to itself.)
      *elems = elems;
      elems++;
    }

    size_t offset = elems - reinterpret_cast<void**>(CpuMemoryStart(cpu));
    CHECK_CONDITION(static_cast<uint16_t>(offset) == offset);
    begin[cl] = offset;

    elems += cap;
    CHECK_CONDITION(reinterpret_cast<char*>(elems) -
                        reinterpret_cast<char*>(CpuMemoryStart(cpu)) <=
                    (1 << Shift));
  }

  // Phase 4: Store current.  No restartable sequence will proceed
  // (successfully) as !(begin < current) for all size classes.
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    std::atomic<int64_t>* hdrp = GetHeader(cpu, cl);
    Header hdr = LoadHeader(hdrp);
    hdr.current = begin[cl];
    StoreHeader(hdrp, hdr);
  }
  FenceCpu(cpu);

  // Phase 5: Allow access to this cache.
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    Header hdr;
    hdr.current = begin[cl];
    hdr.begin = begin[cl];
    hdr.end = begin[cl];
    hdr.end_copy = begin[cl];
    StoreHeader(GetHeader(cpu, cl), hdr);
  }
}

template <size_t Shift, size_t NumClasses>
void TcmallocSlab<Shift, NumClasses>::Destroy(void(free)(void*)) {
  free(slabs_);
  slabs_ = nullptr;
}

template <size_t Shift, size_t NumClasses>
void TcmallocSlab<Shift, NumClasses>::Drain(int cpu, void* ctx,
                                            DrainHandler f) {
  CHECK_CONDITION(cpu >= 0);
  CHECK_CONDITION(cpu < absl::base_internal::NumCPUs());

  // Push/Pop/Grow/Shrink can be executed concurrently with Drain.
  // That's not an expected case, but it must be handled for correctness.
  // Push/Pop/Grow/Shrink can only be executed on <cpu> and use rseq primitives.
  // Push only updates current. Pop only updates current and end_copy
  // (it mutates only current but uses 4 byte write for performance).
  // Grow/Shrink mutate end and end_copy using 64-bit stores.

  // We attempt to stop all concurrent operations by writing 0xffff to begin
  // and 0 to end. However, Grow/Shrink can overwrite our write, so we do this
  // in a loop until we know that the header is in quiescent state.

  // Phase 1: collect all begin's (these are not mutated by anybody else).
  uint16_t begin[NumClasses];
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    Header hdr = LoadHeader(GetHeader(cpu, cl));
    CHECK_CONDITION(!hdr.IsLocked());
    begin[cl] = hdr.begin;
  }

  // Phase 2: stop concurrent mutations.
  for (bool done = false; !done;) {
    for (size_t cl = 0; cl < NumClasses; ++cl) {
      // Note: this reinterpret_cast and write in Lock lead to undefined
      // behavior, because the actual object type is std::atomic<int64_t>. But
      // C++ does not allow to legally express what we need here: atomic writes
      // of different sizes.
      reinterpret_cast<Header*>(GetHeader(cpu, cl))->Lock();
    }
    FenceCpu(cpu);
    done = true;
    for (size_t cl = 0; cl < NumClasses; ++cl) {
      Header hdr = LoadHeader(GetHeader(cpu, cl));
      if (!hdr.IsLocked()) {
        // Header was overwritten by Grow/Shrink. Retry.
        done = false;
        break;
      }
    }
  }

  // Phase 3: execute callbacks.
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    Header hdr = LoadHeader(GetHeader(cpu, cl));
    // We overwrote begin and end, instead we use our local copy of begin
    // and end_copy.
    size_t n = hdr.current - begin[cl];
    size_t cap = hdr.end_copy - begin[cl];
    void** batch = reinterpret_cast<void**>(GetHeader(cpu, 0) + begin[cl]);
    f(ctx, cl, batch, n, cap);
  }

  // Phase 4: reset current to beginning of the region.
  // We can't write all 4 fields at once with a single write, because Pop does
  // several non-atomic loads of the fields. Consider that a concurrent Pop
  // loads old current (still pointing somewhere in the middle of the region);
  // then we update all fields with a single write; then Pop loads the updated
  // begin which allows it to proceed; then it decrements current below begin.
  //
  // So we instead first just update current--our locked begin/end guarantee
  // no Push/Pop will make progress.  Once we Fence below, we know no Push/Pop
  // is using the old current, and can safely update begin/end to be an empty
  // slab.
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    std::atomic<int64_t>* hdrp = GetHeader(cpu, cl);
    Header hdr = LoadHeader(hdrp);
    hdr.current = begin[cl];
    StoreHeader(hdrp, hdr);
  }

  // Phase 5: fence and reset the remaining fields to beginning of the region.
  // This allows concurrent mutations again.
  FenceCpu(cpu);
  for (size_t cl = 0; cl < NumClasses; ++cl) {
    std::atomic<int64_t>* hdrp = GetHeader(cpu, cl);
    Header hdr;
    hdr.current = begin[cl];
    hdr.begin = begin[cl];
    hdr.end = begin[cl];
    hdr.end_copy = begin[cl];
    StoreHeader(hdrp, hdr);
  }
}

template <size_t Shift, size_t NumClasses>
PerCPUMetadataState TcmallocSlab<Shift, NumClasses>::MetadataMemoryUsage()
    const {
  PerCPUMetadataState result;
  result.virtual_size = absl::base_internal::NumCPUs() * sizeof(*slabs_);
  result.resident_size = MInCore::residence(slabs_, result.virtual_size);
  return result;
}

}  // namespace percpu
}  // namespace subtle
}  // namespace tcmalloc

#endif  // TCMALLOC_INTERNAL_PERCPU_TCMALLOC_H_
