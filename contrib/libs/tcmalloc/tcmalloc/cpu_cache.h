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

#ifndef TCMALLOC_CPU_CACHE_H_
#define TCMALLOC_CPU_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/optimization.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/internal/percpu_tcmalloc.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/thread_cache.h"
#include "tcmalloc/tracking.h"

namespace tcmalloc {

class CPUCache {
 public:
  enum class ActivationMode {
    FastPathOn,
    FastPathOffTestOnly,
  };

  // tcmalloc explicitly initializes its global state (to be safe for
  // use in global constructors) so our constructor must be trivial;
  // do all initialization here instead.
  void Activate(ActivationMode mode);

  // Allocate an object of the given size class. When allocation fails
  // (from this cache and after running Refill), OOOHandler(size) is
  // called and its return value is returned from
  // Allocate. OOOHandler is used to parameterize out-of-memory
  // handling (raising exception, returning nullptr, calling
  // new_handler or anything else). "Passing" OOOHandler in this way
  // allows Allocate to be used in tail-call position in fast-path,
  // making Allocate use jump (tail-call) to slow path code.
  template <void* OOMHandler(size_t)>
  void* Allocate(size_t cl);

  // Free an object of the given class.
  void Deallocate(void* ptr, size_t cl);

  // Give the number of bytes in <cpu>'s cache
  uint64_t UsedBytes(int cpu) const;

  // Whether <cpu>'s cache has ever been populated with objects
  bool HasPopulated(int cpu) const;

  PerCPUMetadataState MetadataMemoryUsage() const;

  // Give the number of bytes used in all cpu caches.
  uint64_t TotalUsedBytes() const;

  // Give the number of objects of a given class in all cpu caches.
  uint64_t TotalObjectsOfClass(size_t cl) const;

  // Give the number of bytes unallocated to any sizeclass in <cpu>'s cache.
  uint64_t Unallocated(int cpu) const;

  // Give the per-cpu limit of cache size.
  uint64_t CacheLimit() const;

  // Empty out the cache on <cpu>; move all objects to the central
  // cache.  (If other threads run concurrently on that cpu, we can't
  // guarantee it will be fully empty on return, but if the cpu is
  // unused, this will eliminate stranded memory.)  Returns the number
  // of bytes we sent back.  This function is thread safe.
  uint64_t Reclaim(int cpu);

  // Determine number of bits we should use for allocating per-cpu cache
  // The amount of per-cpu cache is 2 ^ kPerCpuShift
#if defined(TCMALLOC_SMALL_BUT_SLOW)
  static const size_t kPerCpuShift = 12;
#else
  static constexpr size_t kPerCpuShift = 18;
#endif

  // Report statistics
  void Print(TCMalloc_Printer* out) const;
  void PrintInPbtxt(PbtxtRegion* region) const;

 private:
  // Per-size-class freelist resizing info.
  class PerClassResizeInfo {
   public:
    void Init();
    // Updates info on overflow/underflow.
    // <overflow> says if it's overflow or underflow.
    // <grow> is caller approximation of whether we want to grow capacity.
    // <successive> will contain number of successive overflows/underflows.
    // Returns if capacity needs to be grown aggressively (i.e. by batch size).
    bool Update(bool overflow, bool grow, uint32_t* successive);
    uint32_t Tick();

   private:
    std::atomic<int32_t> state_;
    // state_ layout:
    struct State {
      // last overflow/underflow?
      uint32_t overflow : 1;
      // number of times Steal checked this class since the last grow
      uint32_t quiescent_ticks : 15;
      // number of successive overflows/underflows
      uint32_t successive : 16;
    };
    static_assert(sizeof(State) == sizeof(std::atomic<int32_t>),
                  "size mismatch");
  };

  subtle::percpu::TcmallocSlab<kPerCpuShift, kNumClasses> freelist_;

  struct ResizeInfoUnpadded {
    // cache space on this CPU we're not using.  Modify atomically;
    // we don't want to lose space.
    std::atomic<size_t> available;
    // this is just a hint
    std::atomic<size_t> last_steal;
    // Track whether we have initialized this CPU.
    absl::once_flag initialized;
    // Track whether we have ever populated this CPU.
    std::atomic<bool> populated;
    // For cross-cpu operations.
    absl::base_internal::SpinLock lock;
    PerClassResizeInfo per_class[kNumClasses];
  };
  struct ResizeInfo : ResizeInfoUnpadded {
    char pad[ABSL_CACHELINE_SIZE -
             sizeof(ResizeInfoUnpadded) % ABSL_CACHELINE_SIZE];
  };
  // Tracking data for each CPU's cache resizing efforts.
  ResizeInfo* resize_;
  // Track whether we are lazily initializing slabs.  We cannot use the latest
  // value in Parameters, as it can change after initialization.
  bool lazy_slabs_;

  // Return a set of objects to be returned to the Transfer Cache.
  static constexpr int kMaxToReturn = 16;
  struct ObjectsToReturn {
    // The number of slots available for storing objects.
    int count = kMaxToReturn;
    // The size class of the returned object. kNumClasses is the
    // largest value that needs to be stored in cl.
    static_assert(kNumClasses <= std::numeric_limits<unsigned char>::max());
    unsigned char cl[kMaxToReturn];
    void* obj[kMaxToReturn];
  };

  void* Refill(int cpu, size_t cl);

  // This is called after finding a full freelist when attempting to push <ptr>
  // on the freelist for sizeclass <cl>.  The last arg should indicate which
  // CPU's list was full.  Returns 1.
  int Overflow(void* ptr, size_t cl, int cpu);

  // Called on <cl> freelist overflow/underflow on <cpu> to balance cache
  // capacity between size classes. Returns number of objects to return/request
  // from transfer cache. <to_return> will contain objects that need to be
  // freed.
  size_t UpdateCapacity(int cpu, size_t cl, size_t batch_length, bool overflow,
                        ObjectsToReturn* to_return);

  // Tries to obtain up to <desired_increase> bytes of freelist space on <cpu>
  // for <cl> from other <cls>. <to_return> will contain objects that need to be
  // freed.
  void Grow(int cpu, size_t cl, size_t desired_increase,
            ObjectsToReturn* to_return);

  // Tries to steal <bytes> for <cl> on <cpu> from other size classes on that
  // CPU. Returns acquired bytes. <to_return> will contain objects that need to
  // be freed.
  size_t Steal(int cpu, size_t cl, size_t bytes, ObjectsToReturn* to_return);

  static void* NoopUnderflow(int cpu, size_t cl) { return nullptr; }
  static int NoopOverflow(int cpu, size_t cl, void* item) { return -1; }
};

template <void* OOMHandler(size_t)>
inline void* ABSL_ATTRIBUTE_ALWAYS_INLINE CPUCache::Allocate(size_t cl) {
  ASSERT(cl > 0);

  tracking::Report(kMallocHit, cl, 1);
  struct Helper {
    static void* ABSL_ATTRIBUTE_NOINLINE Underflow(int cpu, size_t cl) {
      // we've optimistically reported hit in Allocate, lets undo it and
      // report miss instead.
      tracking::Report(kMallocHit, cl, -1);
      tracking::Report(kMallocMiss, cl, 1);
      void* ret = Static::cpu_cache().Refill(cpu, cl);
      if (ABSL_PREDICT_FALSE(ret == nullptr)) {
        size_t size = Static::sizemap().class_to_size(cl);
        return OOMHandler(size);
      }
      return ret;
    }
  };
  return freelist_.Pop(cl, &Helper::Underflow);
}

inline void ABSL_ATTRIBUTE_ALWAYS_INLINE CPUCache::Deallocate(void* ptr,
                                                              size_t cl) {
  ASSERT(cl > 0);
  tracking::Report(kFreeHit, cl, 1);  // Be optimistic; correct later if needed.

  struct Helper {
    static int ABSL_ATTRIBUTE_NOINLINE Overflow(int cpu, size_t cl, void* ptr) {
      // When we reach here we've already optimistically bumped FreeHits.
      // Fix that.
      tracking::Report(kFreeHit, cl, -1);
      tracking::Report(kFreeMiss, cl, 1);
      return Static::cpu_cache().Overflow(ptr, cl, cpu);
    }
  };
  freelist_.Push(cl, ptr, Helper::Overflow);
}

inline bool UsePerCpuCache() {
  // We expect a fast path of per-CPU caches being active and the thread being
  // registered with rseq.
  if (ABSL_PREDICT_FALSE(!Static::CPUCacheActive())) {
    return false;
  }

  if (ABSL_PREDICT_TRUE(tcmalloc::subtle::percpu::IsFastNoInit())) {
    return true;
  }

  // When rseq is not registered, use this transition edge to shutdown the
  // thread cache for this thread.
  //
  // We call IsFast() on every non-fastpath'd malloc or free since IsFast() has
  // the side-effect of initializing the per-thread state needed for "unsafe"
  // per-cpu operations in case this is the first time a new thread is calling
  // into tcmalloc.
  //
  // If the per-CPU cache for a thread is not initialized, we push ourselves
  // onto the slow path (if !defined(TCMALLOC_DEPRECATED_PERTHREAD)) until this
  // occurs.  See fast_alloc's use of TryRecordAllocationFast.
  if (ABSL_PREDICT_TRUE(tcmalloc::subtle::percpu::IsFast())) {
    ThreadCache::BecomeIdle();
    return true;
  }

  return false;
}

}  // namespace tcmalloc
#endif  // TCMALLOC_CPU_CACHE_H_
