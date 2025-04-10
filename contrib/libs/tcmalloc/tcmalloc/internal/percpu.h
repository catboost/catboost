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

#ifndef TCMALLOC_INTERNAL_PERCPU_H_
#define TCMALLOC_INTERNAL_PERCPU_H_

#define TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT 18

// TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM defines whether or not we have an
// implementation for the target OS and architecture.
#if defined(__linux__) && \
    (defined(__x86_64__) || defined(__PPC64__) || defined(__aarch64__))
#define TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM 1
#else
#define TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM 0
#endif

#define TCMALLOC_PERCPU_RSEQ_VERSION 0x0
#define TCMALLOC_PERCPU_RSEQ_FLAGS 0x0
#if defined(__x86_64__)
#define TCMALLOC_PERCPU_RSEQ_SIGNATURE 0x53053053
#elif defined(__ppc__)
#define TCMALLOC_PERCPU_RSEQ_SIGNATURE 0x0FE5000B
#elif defined(__aarch64__)
#define TCMALLOC_PERCPU_RSEQ_SIGNATURE 0xd428bc00
#else
// Rather than error, allow us to build, but with an invalid signature.
#define TCMALLOC_PERCPU_RSEQ_SIGNATURE 0x0
#endif

// The constants above this line must be macros since they are shared with the
// RSEQ assembly sources.
#ifndef __ASSEMBLER__

#ifdef __linux__
#include <sched.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "absl/base/dynamic_annotations.h"
#include "absl/base/internal/per_thread_tls.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "tcmalloc/internal/atomic_danger.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/linux_syscall_support.h"
#include "tcmalloc/internal/logging.h"

// TCMALLOC_PERCPU_USE_RSEQ defines whether TCMalloc support for RSEQ on the
// target architecture exists. We currently only provide RSEQ for 64-bit x86 and
// PPC binaries.
#if !defined(TCMALLOC_PERCPU_USE_RSEQ)
#if (ABSL_PER_THREAD_TLS == 1) && (TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM == 1)
#define TCMALLOC_PERCPU_USE_RSEQ 1
#else
#define TCMALLOC_PERCPU_USE_RSEQ 0
#endif
#endif  // !defined(TCMALLOC_PERCPU_USE_RSEQ)

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace subtle {
namespace percpu {

inline constexpr int kRseqUnregister = 1;

// Internal state used for tracking initialization of RseqCpuId()
inline constexpr int kCpuIdUnsupported = -2;
inline constexpr int kCpuIdUninitialized = -1;
inline constexpr int kCpuIdInitialized = 0;

#if TCMALLOC_PERCPU_USE_RSEQ
extern "C" ABSL_PER_THREAD_TLS_KEYWORD volatile kernel_rseq __rseq_abi;

static inline int RseqCpuId() { return __rseq_abi.cpu_id; }

static inline int VirtualRseqCpuId(const size_t virtual_cpu_id_offset) {
#ifdef __x86_64__
  ASSERT(virtual_cpu_id_offset == offsetof(kernel_rseq, cpu_id) ||
         virtual_cpu_id_offset == offsetof(kernel_rseq, vcpu_id));
  return *reinterpret_cast<short *>(reinterpret_cast<uintptr_t>(&__rseq_abi) +
                                    virtual_cpu_id_offset);
#else
  ASSERT(virtual_cpu_id_offset == offsetof(kernel_rseq, cpu_id));
  return RseqCpuId();
#endif
}
#else  // !TCMALLOC_PERCPU_USE_RSEQ
static inline int RseqCpuId() { return kCpuIdUnsupported; }

static inline int VirtualRseqCpuId(const size_t virtual_cpu_id_offset) {
  return kCpuIdUnsupported;
}
#endif

typedef int (*OverflowHandler)(int cpu, size_t cl, void *item);
typedef void *(*UnderflowHandler)(int cpu, size_t cl);

// Functions below are implemented in the architecture-specific percpu_rseq_*.S
// files.
extern "C" {
int TcmallocSlab_Internal_PerCpuCmpxchg64(int target_cpu, intptr_t *p,
                                          intptr_t old_val, intptr_t new_val);

#ifndef __x86_64__
int TcmallocSlab_Internal_Push(void *ptr, size_t cl, void *item, size_t shift,
                               OverflowHandler f);
int TcmallocSlab_Internal_Push_FixedShift(void *ptr, size_t cl, void *item,
                                          OverflowHandler f);
void *TcmallocSlab_Internal_Pop(void *ptr, size_t cl, UnderflowHandler f,
                                size_t shift);
void *TcmallocSlab_Internal_Pop_FixedShift(void *ptr, size_t cl,
                                           UnderflowHandler f);
#endif  // __x86_64__

// Push a batch for a slab which the Shift equal to
// TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
size_t TcmallocSlab_Internal_PushBatch_FixedShift(void *ptr, size_t cl,
                                                  void **batch, size_t len);

// Pop a batch for a slab which the Shift equal to
// TCMALLOC_PERCPU_TCMALLOC_FIXED_SLAB_SHIFT
size_t TcmallocSlab_Internal_PopBatch_FixedShift(void *ptr, size_t cl,
                                                 void **batch, size_t len);

#ifdef __x86_64__
int TcmallocSlab_Internal_PerCpuCmpxchg64_VCPU(int target_cpu, intptr_t *p,
                                               intptr_t old_val,
                                               intptr_t new_val);
size_t TcmallocSlab_Internal_PushBatch_FixedShift_VCPU(void *ptr, size_t cl,
                                                       void **batch,
                                                       size_t len);
size_t TcmallocSlab_Internal_PopBatch_FixedShift_VCPU(void *ptr, size_t cl,
                                                      void **batch, size_t len);
#endif
}

// NOTE:  We skirt the usual naming convention slightly above using "_" to
// increase the visibility of functions embedded into the root-namespace (by
// virtue of C linkage) in the supported case.

// Return whether we are using flat virtual CPUs.
bool UsingFlatVirtualCpus();

inline int GetCurrentCpuUnsafe() {
// On PowerPC, Linux maintains the current CPU in the bottom 12 bits of special
// purpose register SPRG3, which is readable from user mode. References:
//
//   https://github.com/torvalds/linux/blob/164c09978cebebd8b5fc198e9243777dbaecdfa0/arch/powerpc/kernel/vdso.c#L727
//   https://github.com/torvalds/linux/blob/dfb945473ae8528fd885607b6fa843c676745e0c/arch/powerpc/include/asm/reg.h#L966
//   https://github.com/torvalds/linux/blob/dfb945473ae8528fd885607b6fa843c676745e0c/arch/powerpc/include/asm/reg.h#L593
//   https://lists.ozlabs.org/pipermail/linuxppc-dev/2012-July/099011.html
//
// This is intended for VDSO syscalls, but is much faster if we simply inline it
// here, presumably due to the function call and null-check overheads of the
// VDSO version. As of 2014-07 the CPU time costs are something like 1.2 ns for
// the inline version vs 12 ns for VDSO.
#if defined(__PPC64__) && defined(__linux__)
  uint64_t spr;

  // Mark the asm as volatile, so that it is not hoisted out of loops.
  asm volatile("mfspr %0, 0x103;" : "=r"(spr));

  return spr & 0xfff;
#else
  // Elsewhere, use the rseq mechanism.
  return RseqCpuId();
#endif
}

inline int GetCurrentCpu() {
  // We can't use the unsafe version unless we have the appropriate version of
  // the rseq extension. This also allows us a convenient escape hatch if the
  // kernel changes the way it uses special-purpose registers for CPU IDs.
  int cpu = GetCurrentCpuUnsafe();

  // We open-code the check for fast-cpu availability since we do not want to
  // force initialization in the first-call case.  This so done so that we can
  // use this in places where it may not always be safe to initialize and so
  // that it may serve in the future as a proxy for callers such as
  // CPULogicalId() without introducing an implicit dependence on the fast-path
  // extensions. Initialization is also simply unneeded on some platforms.
  if (ABSL_PREDICT_TRUE(cpu >= kCpuIdInitialized)) {
    return cpu;
  }

#ifdef TCMALLOC_HAVE_SCHED_GETCPU
  cpu = sched_getcpu();
  ASSERT(cpu >= 0);
#endif  // TCMALLOC_HAVE_SCHED_GETCPU

  return cpu;
}

inline int GetCurrentVirtualCpuUnsafe(const size_t virtual_cpu_id_offset) {
  return VirtualRseqCpuId(virtual_cpu_id_offset);
}

inline int GetCurrentVirtualCpu(const size_t virtual_cpu_id_offset) {
  // We can't use the unsafe version unless we have the appropriate version of
  // the rseq extension. This also allows us a convenient escape hatch if the
  // kernel changes the way it uses special-purpose registers for CPU IDs.
  int cpu = VirtualRseqCpuId(virtual_cpu_id_offset);

  // We open-code the check for fast-cpu availability since we do not want to
  // force initialization in the first-call case.  This so done so that we can
  // use this in places where it may not always be safe to initialize and so
  // that it may serve in the future as a proxy for callers such as
  // CPULogicalId() without introducing an implicit dependence on the fast-path
  // extensions. Initialization is also simply unneeded on some platforms.
  if (ABSL_PREDICT_TRUE(cpu >= kCpuIdInitialized)) {
    return cpu;
  }

#ifdef TCMALLOC_HAVE_SCHED_GETCPU
  cpu = sched_getcpu();
  ASSERT(cpu >= 0);
#endif  // TCMALLOC_HAVE_SCHED_GETCPU

  return cpu;
}

bool InitFastPerCpu();

inline bool IsFast() {
  if (!TCMALLOC_PERCPU_USE_RSEQ) {
    return false;
  }

  int cpu = RseqCpuId();

  if (ABSL_PREDICT_TRUE(cpu >= kCpuIdInitialized)) {
    return true;
  } else if (ABSL_PREDICT_FALSE(cpu == kCpuIdUnsupported)) {
    return false;
  } else {
    // Sets 'cpu' for next time, and calls EnsureSlowModeInitialized if
    // necessary.
    return InitFastPerCpu();
  }
}

// As IsFast(), but if this thread isn't already initialized, will not
// attempt to do so.
inline bool IsFastNoInit() {
  if (!TCMALLOC_PERCPU_USE_RSEQ) {
    return false;
  }
  int cpu = RseqCpuId();
  return ABSL_PREDICT_TRUE(cpu >= kCpuIdInitialized);
}

// A barrier that prevents compiler reordering.
inline void CompilerBarrier() {
#if defined(__GNUC__)
  __asm__ __volatile__("" : : : "memory");
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// Internal tsan annotations, do not use externally.
// Required as tsan does not natively understand RSEQ.
#ifdef THREAD_SANITIZER
extern "C" {
void __tsan_acquire(void *addr);
void __tsan_release(void *addr);
}
#endif

// TSAN relies on seeing (and rewriting) memory accesses.  It can't
// get at the memory acccesses we make from RSEQ assembler sequences,
// which means it doesn't know about the semantics our sequences
// enforce.  So if we're under TSAN, add barrier annotations.
inline void TSANAcquire(void *p) {
#ifdef THREAD_SANITIZER
  __tsan_acquire(p);
#endif
}

inline void TSANRelease(void *p) {
#ifdef THREAD_SANITIZER
  __tsan_release(p);
#endif
}

inline void TSANMemoryBarrierOn(void *p) {
  TSANAcquire(p);
  TSANRelease(p);
}

// These methods may *only* be called if IsFast() has been called by the current
// thread (and it returned true).
inline int CompareAndSwapUnsafe(int target_cpu, std::atomic<intptr_t> *p,
                                intptr_t old_val, intptr_t new_val,
                                const size_t virtual_cpu_id_offset) {
  TSANMemoryBarrierOn(p);
#if TCMALLOC_PERCPU_USE_RSEQ
  switch (virtual_cpu_id_offset) {
    case offsetof(kernel_rseq, cpu_id):
      return TcmallocSlab_Internal_PerCpuCmpxchg64(
          target_cpu, tcmalloc_internal::atomic_danger::CastToIntegral(p),
          old_val, new_val);
#ifdef __x86_64__
    case offsetof(kernel_rseq, vcpu_id):
      return TcmallocSlab_Internal_PerCpuCmpxchg64_VCPU(
          target_cpu, tcmalloc_internal::atomic_danger::CastToIntegral(p),
          old_val, new_val);
#endif  // __x86_64__
    default:
      __builtin_unreachable();
  }
#else  // !TCMALLOC_PERCPU_USE_RSEQ
  __builtin_unreachable();
#endif  // !TCMALLOC_PERCPU_USE_RSEQ
}

void FenceCpu(int cpu, const size_t virtual_cpu_id_offset);

}  // namespace percpu
}  // namespace subtle
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // !__ASSEMBLER__
#endif  // TCMALLOC_INTERNAL_PERCPU_H_
