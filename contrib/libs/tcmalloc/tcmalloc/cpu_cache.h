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

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class CPUCache {
 public:
  constexpr CPUCache() = default;

  enum class ActivationMode {
    FastPathOn,
    FastPathOffTestOnly,
  };

  // tcmalloc explicitly initializes its global state (to be safe for
  // use in global constructors) so our constructor must be trivial;
  // do all initialization here instead.
  void Activate(ActivationMode mode);

  // Allocate an object of the given size class. When allocation fails
  // (from this cache and after running Refill), OOMHandler(size) is
  // called and its return value is returned from
  // Allocate. OOMHandler is used to parameterize out-of-memory
  // handling (raising exception, returning nullptr, calling
  // new_handler or anything else). "Passing" OOMHandler in this way
  // allows Allocate to be used in tail-call position in fast-path,
  // making Allocate use jump (tail-call) to slow path code.
  template <void* OOMHandler(size_t)>
  void* Allocate(size_t cl);

  // Free an object of the given class.
  void Deallocate(void* ptr, size_t cl);

  // Give the number of bytes in <cpu>'s cache
  uint64_t UsedBytes(int cpu) const;

  // Give the allocated number of bytes in <cpu>'s cache
  uint64_t Allocated(int cpu) const;

  // Whether <cpu>'s cache has ever been populated with objects
  bool HasPopulated(int cpu) const;

  PerCPUMetadataState MetadataMemoryUsage() const;

  // Give the number of bytes used in all cpu caches.
  uint64_t TotalUsedBytes() const;

  // Give the number of objects of a given class in all cpu caches.
  uint64_t TotalObjectsOfClass(size_t cl) const;

  // Give the number of bytes unallocated to any sizeclass in <cpu>'s cache.
  uint64_t Unallocated(int cpu) const;

  // Gives the total capacity of <cpu>'s cache in bytes.
  //
  // The total capacity of <cpu>'s cache should be equal to the sum of allocated
  // and unallocated bytes for that cache.
  uint64_t Capacity(int cpu) const;

  // Give the per-cpu limit of cache size.
  uint64_t CacheLimit() const;

  // Shuffles per-cpu caches using the number of underflows and overflows that
  // occurred in the prior interval. It selects the top per-cpu caches
  // with highest misses as candidates, iterates through the other per-cpu
  // caches to steal capacity from them and adds the stolen bytes to the
  // available capacity of the per-cpu caches. May be called from any processor.
  //
  // TODO(vgogte): There are quite a few knobs that we can play around with in
  // ShuffleCpuCaches.
  void ShuffleCpuCaches();

  // Sets the lower limit on the capacity that can be stolen from the cpu cache.
  static constexpr double kCacheCapacityThreshold = 0.20;

  // Tries to steal <bytes> for the destination <cpu>. It iterates through the
  // the set of populated cpu caches and steals the bytes from them. A cpu is
  // considered a good candidate to steal from if:
  // (1) the cache is populated
  // (2) the numbers of underflows and overflows are both less than 0.8x those
  // of the destination per-cpu cache
  // (3) source cpu is not the same as the destination cpu
  // (4) capacity of the source cpu/cl is non-zero
  //
  // For a given source cpu, we iterate through the size classes to steal from
  // them. Currently, we use a similar clock-like algorithm from Steal() to
  // identify the cl to steal from.
  void StealFromOtherCache(int cpu, int max_populated_cpu, size_t bytes);

  // Tries to reclaim inactive per-CPU caches. It iterates through the set of
  // populated cpu caches and reclaims the caches that:
  // (1) had same number of used bytes since the last interval,
  // (2) had no change in the number of misses since the last interval.
  void TryReclaimingCaches();

  // Empty out the cache on <cpu>; move all objects to the central
  // cache.  (If other threads run concurrently on that cpu, we can't
  // guarantee it will be fully empty on return, but if the cpu is
  // unused, this will eliminate stranded memory.)  Returns the number
  // of bytes we sent back.  This function is thread safe.
  uint64_t Reclaim(int cpu);

  // Reports number of times the <cpu> has been reclaimed.
  uint64_t GetNumReclaims(int cpu) const;

  // Determine number of bits we should use for allocating per-cpu cache
  // The amount of per-cpu cache is 2 ^ kPerCpuShift
#if defined(TCMALLOC_SMALL_BUT_SLOW)
  static const size_t kPerCpuShift = 12;
#else
  static constexpr size_t kPerCpuShift = 18;
#endif

  struct CpuCacheMissStats {
    size_t underflows;
    size_t overflows;
  };

  // Reports total cache underflows and overflows for <cpu>.
  CpuCacheMissStats GetTotalCacheMissStats(int cpu) const;

  // Reports the cache underflows and overflows for <cpu> that were recorded at
  // the end of the previous interval. It also records current underflows and
  // overflows in the reclaim underflow and overflow stats.
  CpuCacheMissStats GetReclaimCacheMissStats(int cpu) const;

  // Reports cache underflows and overflows for <cpu> this interval.
  CpuCacheMissStats GetIntervalCacheMissStats(int cpu) const;

  // Report statistics
  void Print(Printer* out) const;
  void PrintInPbtxt(PbtxtRegion* region) const;

  void AcquireInternalLocks();
  void ReleaseInternalLocks();

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

  subtle::percpu::TcmallocSlab<kNumClasses> freelist_;

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
    // tracks number of underflows on allocate.
    std::atomic<size_t> total_underflows;
    // tracks number of overflows on deallocate.
    std::atomic<size_t> total_overflows;
    // tracks number of underflows recorded as of the end of the last shuffle
    // interval.
    std::atomic<size_t> shuffle_underflows;
    // tracks number of overflows recorded as of the end of the last shuffle
    // interval.
    std::atomic<size_t> shuffle_overflows;
    // total cache space available on this CPU. This tracks the total
    // allocated and unallocated bytes on this CPU cache.
    std::atomic<size_t> capacity;
    // Number of underflows as of the end of the last resize interval.
    std::atomic<size_t> reclaim_underflows;
    // Number of overflows as of the end of the last resize interval.
    std::atomic<size_t> reclaim_overflows;
    // Used bytes in the cache as of the end of the last resize interval.
    std::atomic<uint64_t> reclaim_used_bytes;
    // Tracks number of times this CPU has been reclaimed.
    std::atomic<size_t> num_reclaims;
  };
  struct ResizeInfo : ResizeInfoUnpadded {
    char pad[ABSL_CACHELINE_SIZE -
             sizeof(ResizeInfoUnpadded) % ABSL_CACHELINE_SIZE];
  };
  // Tracking data for each CPU's cache resizing efforts.
  ResizeInfo* resize_ = nullptr;

  // Track whether we are lazily initializing slabs.  We cannot use the latest
  // value in Parameters, as it can change after initialization.
  bool lazy_slabs_ = false;
  // The maximum capacity of each size class within the slab.
  uint16_t max_capacity_[kNumClasses] = {0};

  // Provides a hint to StealFromOtherCache() so that we can steal from the
  // caches in a round-robin fashion.
  std::atomic<int> last_cpu_cache_steal_ = 0;

  // Return a set of objects to be returned to the Transfer Cache.
  static constexpr int kMaxToReturn = 16;
  struct ObjectsToReturn {
    // The number of slots available for storing objects.
    int count = kMaxToReturn;
    // The size class of the returned object. kNumClasses is the
    // largest value that needs to be stored in cl.
    CompactSizeClass cl[kMaxToReturn];
    void* obj[kMaxToReturn];
  };

  static size_t MaxCapacityHelper(size_t cl) {
    CPUCache& cpu_cache = Static::cpu_cache();
    // Heuristic that the CPUCache has been activated.
    ASSERT(cpu_cache.resize_ != nullptr);
    return cpu_cache.max_capacity_[cl];
  }

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

  // Records a cache underflow or overflow on <cpu>, increments underflow or
  // overflow by 1.
  // <is_malloc> determines whether the associated count corresponds to an
  // underflow or overflow.
  void RecordCacheMissStat(const int cpu, const bool is_malloc);

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
      void* ret = nullptr;
      if (Static::sharded_transfer_cache().should_use(cl)) {
        ret = Static::sharded_transfer_cache().Pop(cl);
      } else {
        tracking::Report(kMallocMiss, cl, 1);
        CPUCache& cache = Static::cpu_cache();
        cache.RecordCacheMissStat(cpu, true);
        ret = cache.Refill(cpu, cl);
      }
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
      if (Static::sharded_transfer_cache().should_use(cl)) {
        Static::sharded_transfer_cache().Push(cl, ptr);
        return 1;
      }
      tracking::Report(kFreeMiss, cl, 1);
      CPUCache& cache = Static::cpu_cache();
      cache.RecordCacheMissStat(cpu, false);
      return cache.Overflow(ptr, cl, cpu);
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

  if (ABSL_PREDICT_TRUE(subtle::percpu::IsFastNoInit())) {
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
  if (ABSL_PREDICT_TRUE(subtle::percpu::IsFast())) {
    ThreadCache::BecomeIdle();
    return true;
  }

  return false;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
#endif  // TCMALLOC_CPU_CACHE_H_
