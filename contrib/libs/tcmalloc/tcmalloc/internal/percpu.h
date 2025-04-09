#pragma clang system_header
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

// sizeof(Sampler)
#define TCMALLOC_SAMPLER_SIZE 32
// alignof(Sampler)
#define TCMALLOC_SAMPLER_ALIGN 8
// Sampler::HotDataOffset()
#define TCMALLOC_SAMPLER_HOT_OFFSET 24

// Offset from __rseq_abi to the cached slabs address.
#define TCMALLOC_RSEQ_SLABS_OFFSET -4

// The bit denotes that tcmalloc_rseq.slabs contains valid slabs offset.
#define TCMALLOC_CACHED_SLABS_BIT 63
#define TCMALLOC_CACHED_SLABS_MASK (1ul << TCMALLOC_CACHED_SLABS_BIT)

// TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM defines whether or not we have an
// implementation for the target OS and architecture.
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
#define TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM 1
#else
#define TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM 0
#endif

#define TCMALLOC_PERCPU_RSEQ_VERSION 0x0
#define TCMALLOC_PERCPU_RSEQ_FLAGS 0x0
#if defined(__x86_64__)
#define TCMALLOC_PERCPU_RSEQ_SIGNATURE 0x53053053
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

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/linux_syscall_support.h"
#include "tcmalloc/internal/logging.h"

// TCMALLOC_INTERNAL_PERCPU_USE_RSEQ defines whether TCMalloc support for RSEQ
// on the target architecture exists. We currently only provide RSEQ for 64-bit
// x86, Arm binaries.
#if !defined(TCMALLOC_INTERNAL_PERCPU_USE_RSEQ)
#if TCMALLOC_PERCPU_RSEQ_SUPPORTED_PLATFORM == 1
#define TCMALLOC_INTERNAL_PERCPU_USE_RSEQ 1
#else
#define TCMALLOC_INTERNAL_PERCPU_USE_RSEQ 0
#endif
#endif  // !defined(TCMALLOC_INTERNAL_PERCPU_USE_RSEQ)

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {
namespace subtle {
namespace percpu {

inline constexpr int kRseqUnregister = 1;

// Internal state used for tracking initialization of GetRealCpuUnsafe()
inline constexpr int kCpuIdUnsupported = -2;
inline constexpr int kCpuIdUninitialized = -1;
inline constexpr int kCpuIdInitialized = 0;

#if TCMALLOC_INTERNAL_PERCPU_USE_RSEQ
// We provide a per-thread value (defined in percpu_rseq_asm.S) which both
// tracks thread-local initialization state and (with RSEQ) provides an atomic
// in-memory reference for this thread's execution CPU. This value is only
// valid when the thread is currently executing.
// Possible values:
//   Unavailable/uninitialized:
//     { kCpuIdUnsupported, kCpuIdUninitialized }
//   Initialized, available:
//     [0, NumCpus())    (Always updated at context-switch)
//
// CPU slabs region address caching.
// Calculation of the address of the current CPU slabs region is needed for
// allocation/deallocation fast paths, but is quite expensive. Due to variable
// shift and experimental support for "virtual CPUs", the calculation involves
// several additional loads and dependent calculations. Pseudo-code for the
// address calculation is as follows:
//
//   cpu_offset = TcmallocSlab.virtual_cpu_id_offset_;
//   cpu = *(&__rseq_abi + virtual_cpu_id_offset_);
//   slabs_and_shift = TcmallocSlab.slabs_and_shift_;
//   shift = slabs_and_shift & kShiftMask;
//   shifted_cpu = cpu << shift;
//   slabs = slabs_and_shift & kSlabsMask;
//   slabs += shifted_cpu;
//
// To remove this calculation from fast paths, we cache the slabs address
// for the current CPU in thread local storage. However, when a thread is
// rescheduled to another CPU, we somehow need to understand that the cached
// address is not valid anymore. To achieve this, we overlap the top 4 bytes
// of the cached address with __rseq_abi.cpu_id_start. When a thread is
// rescheduled the kernel overwrites cpu_id_start with the current CPU number,
// which gives us the signal that the cached address is not valid anymore.
// To distinguish the high part of the cached address from the CPU number,
// we set the top bit in the cached address, real CPU numbers (<2^31) do not
// have this bit set.
//
// With these arrangements, slabs address calculation on allocation/deallocation
// fast paths reduces to load and check of the cached address:
//
//   slabs = __rseq_abi[-4];
//   if ((slabs & (1 << 63)) == 0) goto slowpath;
//   slabs &= ~(1 << 63);
//
// Note: here we assume little-endian byte order (which is the case for our
// supported architectures). On a little-endian arch, reading 8 bytes starting
// at __rseq_abi-4 gives __rseq_abi[-4...3]. So the tag bit (1<<63) is
// therefore from __rseq_abi[3]. That's also the most significant byte of
// __rseq_abi.cpu_id_start, hence real CPU numbers can't have this bit set
// (assuming <2^31 CPUs).
//
// The slow path does full slabs address calculation and caches it.
//
// Note: this makes __rseq_abi.cpu_id_start unusable for its original purpose.
//
// Since we need to export the __rseq_abi variable (as part of rseq ABI),
// we arrange overlapping of __rseq_abi and the preceding cached slabs
// address in percpu_rseq_asm.S (C++ is not capable of expressing that).
// __rseq_abi must be aligned to 32 bytes as per ABI. We want the cached slabs
// address to be contained within a single cache line (64 bytes), rather than
// split 2 cache lines. To achieve that we locate __rseq_abi in the second
// part of a cache line.
// For performance reasons we also collocate tcmalloc_sampler with __rseq_abi
// in the same cache line.
// InitPerCpu contains checks that the resulting data layout is as expected.

// Top 4 bytes of this variable overlap with __rseq_abi.cpu_id_start.
extern "C" ABSL_CONST_INIT thread_local volatile uintptr_t tcmalloc_slabs
    ABSL_ATTRIBUTE_INITIAL_EXEC;
extern "C" ABSL_CONST_INIT thread_local volatile kernel_rseq __rseq_abi
    ABSL_ATTRIBUTE_INITIAL_EXEC;
extern "C" ABSL_CONST_INIT thread_local volatile int tcmalloc_cached_vcpu
    ABSL_ATTRIBUTE_INITIAL_EXEC;

// Provide weak definitions here to enable more efficient codegen.
// If compiler sees only extern declaration when generating accesses,
// then even with initial-exec model and -fno-PIE compiler has to assume
// that the definition may come from a dynamic library and has to use
// GOT access. When compiler sees even a weak definition, it knows the
// declaration will be in the current module and can generate direct accesses.
ABSL_CONST_INIT thread_local volatile uintptr_t tcmalloc_slabs
    ABSL_ATTRIBUTE_WEAK = {};
ABSL_CONST_INIT thread_local volatile kernel_rseq __rseq_abi
    ABSL_ATTRIBUTE_WEAK = {
        0,      static_cast<unsigned>(kCpuIdUninitialized),   0, 0,
        {0, 0}, {{kCpuIdUninitialized, kCpuIdUninitialized}},
};
ABSL_CONST_INIT thread_local volatile int tcmalloc_cached_vcpu
    ABSL_ATTRIBUTE_WEAK = kCpuIdUninitialized;

inline int GetRealCpuUnsafe() { return __rseq_abi.cpu_id; }
#else  // !TCMALLOC_INTERNAL_PERCPU_USE_RSEQ
inline int GetRealCpuUnsafe() { return kCpuIdUnsupported; }
#endif

// Functions below are implemented in the architecture-specific percpu_rseq_*.S
// files.
extern "C" {
size_t TcmallocSlab_Internal_PushBatch(size_t size_class, void** batch,
                                       size_t len);
size_t TcmallocSlab_Internal_PopBatch(size_t size_class, void** batch,
                                      size_t len,
                                      std::atomic<uint16_t>* begin_ptr);
}  // extern "C"

// NOTE:  We skirt the usual naming convention slightly above using "_" to
// increase the visibility of functions embedded into the root-namespace (by
// virtue of C linkage) in the supported case.

enum class RseqVcpuMode { kNone };
inline RseqVcpuMode GetRseqVcpuMode() { return RseqVcpuMode::kNone; }

// Return whether we are using any kind of virtual CPUs.
inline bool UsingVirtualCpus() {
  return GetRseqVcpuMode() != RseqVcpuMode::kNone;
}

// Return whether we are using flat virtual CPUs (provided by kernel RSEQ).
bool UsingRseqVirtualCpus();

inline int GetRealCpu() {
  // The "unsafe" variant strongly depends on RSEQ.
  int cpu = GetRealCpuUnsafe();

  // We open-code the check for fast-cpu availability since we do not want to
  // force initialization in the first-call case.  This is done so that we can
  // use this in places where it may not always be safe to initialize.
  // Initialization is also simply unneeded on some platforms.
  if (ABSL_PREDICT_TRUE(cpu >= kCpuIdInitialized)) {
    return cpu;
  }

#ifdef TCMALLOC_HAVE_SCHED_GETCPU
  cpu = sched_getcpu();
  TC_ASSERT_GE(cpu, 0);
#endif  // TCMALLOC_HAVE_SCHED_GETCPU

  return cpu;
}

// Static accessors functions for any kind of vCPU IDs, which transparently
// choose the right vCPU source based on the initialized mode. Wrapping it into
// a class helps to restrict access and avoid accidental misuse.
class VirtualCpu {
 public:
  // Returns the last vCPU ID since the last synchronization point. This may be
  // used where the vCPU ID is not used to derive RSEQ-validated state. Returns
  // kCpuIdUninitialized if this thread has never synchronized with a vCPU ID,
  // or kCpuIdUnsupported if RSEQ is not used.
  //
  // This is safe, because without a RSEQ critical section to detect thread
  // preemption, a thread may be preempted at any point and the virtual (or
  // real) CPU may change.
  static int get() {
#if TCMALLOC_INTERNAL_PERCPU_USE_RSEQ
    return tcmalloc_cached_vcpu;
#else   // TCMALLOC_INTERNAL_PERCPU_USE_RSEQ
    return kCpuIdUnsupported;
#endif  // TCMALLOC_INTERNAL_PERCPU_USE_RSEQ
  }

  // Returns the last vCPU ID since the last synchronization point.
  // REQUIRES: Synchronize() has been called by this thread
  static int GetAfterSynchronize() {
    const int ret = get();
    TC_ASSERT_GE(ret, kCpuIdInitialized);
    return ret;
  }

  // Returns the current vCPU ID. Use to synchronize RSEQ-validated state that
  // depends on the (per-CPU mutually exclusive) vCPU ID with the current vCPU
  // ID after a thread preemption was detected. This function may be expensive,
  // so it should only be called on slow paths.
  static int Synchronize();

 private:
  // The return value of Synchronize() may be overridden by tests if they define
  // VirtualCpu::TestSynchronize().
  ABSL_ATTRIBUTE_WEAK static int TestSynchronize();
};

bool InitFastPerCpu();

inline bool IsFast() {
  if (!TCMALLOC_INTERNAL_PERCPU_USE_RSEQ) {
    return false;
  }

  int cpu = GetRealCpuUnsafe();

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
  if (!TCMALLOC_INTERNAL_PERCPU_USE_RSEQ) {
    return false;
  }
  int cpu = GetRealCpuUnsafe();
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
#ifdef ABSL_HAVE_THREAD_SANITIZER
extern "C" {
void __tsan_acquire(void* addr);
void __tsan_release(void* addr);
}
#endif

// TSAN relies on seeing (and rewriting) memory accesses.  It can't
// get at the memory accesses we make from RSEQ assembler sequences,
// which means it doesn't know about the semantics our sequences
// enforce.  So if we're under TSAN, add barrier annotations.
inline void TSANAcquire(void* p) {
#ifdef ABSL_HAVE_THREAD_SANITIZER
  __tsan_acquire(p);
#endif
}

inline void TSANAcquireBatch(void** batch, int n) {
#ifdef ABSL_HAVE_THREAD_SANITIZER
  for (int i = 0; i < n; i++) {
    __tsan_acquire(batch[i]);
  }
#endif
}

inline void TSANRelease(void* p) {
#ifdef ABSL_HAVE_THREAD_SANITIZER
  __tsan_release(p);
#endif
}

inline void TSANReleaseBatch(void** batch, int n) {
#ifdef ABSL_HAVE_THREAD_SANITIZER
  for (int i = 0; i < n; i++) {
    __tsan_release(batch[i]);
  }
#endif
}

void FenceCpu(int vcpu);
void FenceAllCpus();

}  // namespace percpu
}  // namespace subtle
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // !__ASSEMBLER__
#endif  // TCMALLOC_INTERNAL_PERCPU_H_
