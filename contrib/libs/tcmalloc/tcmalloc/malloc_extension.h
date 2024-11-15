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

// This file documents extensions supported by TCMalloc. These extensions
// provide hooks for both surfacing telemetric data about TCMalloc's usage and
// tuning the internal implementation of TCMalloc. The internal implementation
// functions use weak linkage, allowing an application to link against the
// extensions without always linking against TCMalloc.

#ifndef TCMALLOC_MALLOC_EXTENSION_H_
#define TCMALLOC_MALLOC_EXTENSION_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/base/policy_checks.h"
#include "absl/base/port.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"

namespace tcmalloc {
namespace tcmalloc_internal {
class AllocationProfilingTokenAccessor;
class AllocationProfilingTokenBase;
class ProfileAccessor;
class ProfileBase;
}  // namespace tcmalloc_internal

enum class ProfileType {
  // Approximation of current heap usage
  kHeap,

  // Fragmentation report
  kFragmentation,

  // Sample of objects that were live at a recent peak of total heap usage. The
  // specifics of when exactly this profile is collected are subject to change.
  kPeakHeap,

  // Sample of objects allocated from the start of allocation profiling until
  // the profile was terminated with Stop().
  kAllocations,

  // Only present to prevent switch statements without a default clause so that
  // we can extend this enumeration without breaking code.
  kDoNotUse,
};

class Profile final {
 public:
  Profile() = default;
  Profile(Profile&&) = default;
  Profile(const Profile&) = delete;

  ~Profile();

  Profile& operator=(Profile&&) = default;
  Profile& operator=(const Profile&) = delete;

  struct Sample {
    static constexpr int kMaxStackDepth = 64;

    int64_t sum;
    int64_t count;  // Total added with this <stack,requested_size,...>

    size_t requested_size;
    size_t requested_alignment;
    size_t allocated_size;

    int depth;
    void* stack[kMaxStackDepth];

    void* user_data;
  };

  void Iterate(absl::FunctionRef<void(const Sample&)> f) const;

  int64_t Period() const;
  ProfileType Type() const;

 private:
  explicit Profile(std::unique_ptr<const tcmalloc_internal::ProfileBase>);

  std::unique_ptr<const tcmalloc_internal::ProfileBase> impl_;
  friend class tcmalloc_internal::ProfileAccessor;
};

class AddressRegion {
 public:
  AddressRegion() {}
  virtual ~AddressRegion();

  // Allocates at least size bytes of memory from this region, aligned with
  // alignment.  Returns a pair containing a pointer to the start the allocated
  // memory and the actual size allocated.  Returns {nullptr, 0} on failure.
  //
  // Alloc must return memory located within the address range given in the call
  // to AddressRegionFactory::Create that created this AddressRegion.
  virtual std::pair<void*, size_t> Alloc(size_t size, size_t alignment) = 0;
};

// Interface to a pluggable address region allocator.
class AddressRegionFactory {
 public:
  enum class UsageHint {
    kNormal,                // Normal usage.
    kInfrequentAllocation,  // TCMalloc allocates from these regions less
                            // frequently than normal regions.
    kInfrequent ABSL_DEPRECATED("Use kInfrequentAllocation") =
        kInfrequentAllocation,
  };

  AddressRegionFactory() {}
  virtual ~AddressRegionFactory();

  // Returns an AddressRegion with the specified start address and size.  hint
  // indicates how the caller intends to use the returned region (helpful for
  // deciding which regions to remap with hugepages, which regions should have
  // pages prefaulted, etc.).  The returned AddressRegion must never be deleted.
  //
  // The caller must have reserved size bytes of address space starting at
  // start_addr with mmap(PROT_NONE) prior to calling this function (so it is
  // safe for Create() to mmap(MAP_FIXED) over the specified address range).
  // start_addr and size are always page-aligned.
  virtual AddressRegion* Create(void* start_addr, size_t size,
                                UsageHint hint) = 0;

  // Gets a human-readable description of the current state of the allocator.
  //
  // The state is stored in the provided buffer.  The number of bytes used (or
  // would have been required, had the buffer been of sufficient size) is
  // returned.
  virtual size_t GetStats(absl::Span<char> buffer);

  // Gets a description of the current state of the allocator in pbtxt format.
  //
  // The state is stored in the provided buffer.  The number of bytes used (or
  // would have been required, had the buffer been of sufficient size) is
  // returned.
  virtual size_t GetStatsInPbtxt(absl::Span<char> buffer);

  // Returns the total number of bytes allocated by MallocInternal().
  static size_t InternalBytesAllocated();

 protected:
  // Dynamically allocates memory for use by AddressRegionFactory.  Particularly
  // useful for creating AddressRegions inside Create().
  //
  // This memory is never freed, so allocate sparingly.
  static void* MallocInternal(size_t size);
};

class MallocExtension final {
 public:
  // Gets a human readable description of the current state of the malloc data
  // structures.
  //
  // See https://github.com/google/tcmalloc/tree/master/docs/stats.md for how to interpret these
  // statistics.
  static std::string GetStats();

  // -------------------------------------------------------------------
  // Control operations for getting malloc implementation specific parameters.
  // Some currently useful properties:
  //
  // generic
  // -------
  // "generic.current_allocated_bytes"
  //      Number of bytes currently allocated by application
  //
  // "generic.heap_size"
  //      Number of bytes in the heap ==
  //            current_allocated_bytes +
  //            fragmentation +
  //            freed (but not released to OS) memory regions
  //
  // tcmalloc
  // --------
  // "tcmalloc.max_total_thread_cache_bytes"
  //      Upper limit on total number of bytes stored across all
  //      per-thread caches.  Default: 16MB.
  //
  // "tcmalloc.current_total_thread_cache_bytes"
  //      Number of bytes used across all thread caches.
  //
  // "tcmalloc.pageheap_free_bytes"
  //      Number of bytes in free, mapped pages in page heap.  These
  //      bytes can be used to fulfill allocation requests.  They
  //      always count towards virtual memory usage, and unless the
  //      underlying memory is swapped out by the OS, they also count
  //      towards physical memory usage.
  //
  // "tcmalloc.pageheap_unmapped_bytes"
  //      Number of bytes in free, unmapped pages in page heap.
  //      These are bytes that have been released back to the OS,
  //      possibly by one of the MallocExtension "Release" calls.
  //      They can be used to fulfill allocation requests, but
  //      typically incur a page fault.  They always count towards
  //      virtual memory usage, and depending on the OS, typically
  //      do not count towards physical memory usage.
  //
  //  "tcmalloc.per_cpu_caches_active"
  //      Whether tcmalloc is using per-CPU caches (1 or 0 respectively).
  // -------------------------------------------------------------------

  // Gets the named property's value or a nullopt if the property is not valid.
  static absl::optional<size_t> GetNumericProperty(absl::string_view property);

  // Marks the current thread as "idle".  This function may optionally be called
  // by threads as a hint to the malloc implementation that any thread-specific
  // resources should be released.  Note: this may be an expensive function, so
  // it should not be called too often.
  //
  // Also, if the code that calls this function will go to sleep for a while, it
  // should take care to not allocate anything between the call to this function
  // and the beginning of the sleep.
  static void MarkThreadIdle();

  // Marks the current thread as "busy".  This function should be called after
  // MarkThreadIdle() if the thread will now do more work.  If this method is
  // not called, performance may suffer.
  static void MarkThreadBusy();

  // Attempts to free any resources associated with cpu <cpu> (in the sense of
  // only being usable from that CPU.)  Returns the number of bytes previously
  // assigned to "cpu" that were freed.  Safe to call from any processor, not
  // just <cpu>.
  static size_t ReleaseCpuMemory(int cpu);

  // Gets the region factory used by the malloc extension instance. Returns null
  // for malloc implementations that do not support pluggable region factories.
  static AddressRegionFactory* GetRegionFactory();

  // Sets the region factory to the specified.
  //
  // Users could register their own region factories by doing:
  //   factory = new MyOwnRegionFactory();
  //   MallocExtension::SetRegionFactory(factory);
  //
  // It's up to users whether to fall back (recommended) to the default region
  // factory (use GetRegionFactory() above) or not. The caller is responsible to
  // any necessary locking.
  static void SetRegionFactory(AddressRegionFactory* a);

  // Tries to release at least num_bytes of free memory back to the OS for
  // reuse.
  //
  // Depending on the state of the malloc implementation, more than num_bytes of
  // memory may be released to the OS.
  //
  // This request may not be completely honored if:
  // * The underlying malloc implementation does not support releasing memory to
  //   the OS.
  // * There are not at least num_bytes of free memory cached, or free memory is
  //   fragmented in ways that keep it from being returned to the OS.
  //
  // Returning memory to the OS can hurt performance in two ways:
  // * Parts of huge pages may be free and returning them to the OS requires
  //   breaking up the huge page they are located on.  This can slow accesses to
  //   still-allocated memory due to increased TLB pressure for the working set.
  // * If the memory is ultimately needed again, pages will need to be faulted
  //   back in.
  static void ReleaseMemoryToSystem(size_t num_bytes);

  struct MemoryLimit {
    // Make a best effort attempt to prevent more than limit bytes of memory
    // from being allocated by the system. In particular, if satisfying a given
    // malloc call would require passing this limit, release as much memory to
    // the OS as needed to stay under it if possible.
    //
    // If hard is set, crash if returning memory is unable to get below the
    // limit.
    //
    // Note:  limit=SIZE_T_MAX implies no limit.
    size_t limit = std::numeric_limits<size_t>::max();
    bool hard = false;

    // Explicitly declare the ctor to put it in the google_malloc section.
    MemoryLimit() = default;
  };

  static MemoryLimit GetMemoryLimit();
  static void SetMemoryLimit(const MemoryLimit& limit);

  // Gets the sampling rate.  Returns a value < 0 if unknown.
  static int64_t GetProfileSamplingRate();
  // Sets the sampling rate for heap profiles.  TCMalloc samples approximately
  // every rate bytes allocated.
  static void SetProfileSamplingRate(int64_t rate);

  // Gets the guarded sampling rate.  Returns a value < 0 if unknown.
  static int64_t GetGuardedSamplingRate();
  // Sets the guarded sampling rate for sampled allocations.  TCMalloc samples
  // approximately every rate bytes allocated, subject to implementation
  // limitations in GWP-ASan.
  //
  // Guarded samples provide probablistic protections against buffer underflow,
  // overflow, and use-after-free when GWP-ASan is active (via calling
  // ActivateGuardedSampling).
  static void SetGuardedSamplingRate(int64_t rate);

  // Switches TCMalloc to guard sampled allocations for underflow, overflow, and
  // use-after-free according to the guarded sample parameter value.
  static void ActivateGuardedSampling();

  // Gets whether TCMalloc is using per-CPU caches.
  static bool PerCpuCachesActive();

  // Extension for unified agent.
  //
  // Should be removed in the future https://st.yandex-team.ru/UNIFIEDAGENT-321
  static void DeactivatePerCpuCaches();

  // Gets the current maximum cache size per CPU cache.
  static int32_t GetMaxPerCpuCacheSize();
  // Sets the maximum cache size per CPU cache.  This is a per-core limit.
  static void SetMaxPerCpuCacheSize(int32_t value);

  // Gets the current maximum thread cache.
  static int64_t GetMaxTotalThreadCacheBytes();
  // Sets the maximum thread cache size.  This is a whole-process limit.
  static void SetMaxTotalThreadCacheBytes(int64_t value);

  // Gets the delayed subrelease interval (0 if delayed subrelease is disabled)
  static absl::Duration GetSkipSubreleaseInterval();
  // Sets the delayed subrelease interval (0 to disable delayed subrelease)
  static void SetSkipSubreleaseInterval(absl::Duration value);

  // Returns the estimated number of bytes that will be allocated for a request
  // of "size" bytes.  This is an estimate: an allocation of "size" bytes may
  // reserve more bytes, but will never reserve fewer.
  static size_t GetEstimatedAllocatedSize(size_t size);

  // Returns the actual number N of bytes reserved by tcmalloc for the pointer
  // p.  This number may be equal to or greater than the number of bytes
  // requested when p was allocated.
  //
  // This function is just useful for statistics collection.  The client must
  // *not* read or write from the extra bytes that are indicated by this call.
  //
  // Example, suppose the client gets memory by calling
  //    p = malloc(10)
  // and GetAllocatedSize(p) returns 16.  The client must only use the first 10
  // bytes p[0..9], and not attempt to read or write p[10..15].
  //
  // p must have been allocated by TCMalloc and must not be an interior pointer
  // -- that is, must be exactly the pointer returned to by malloc() et al., not
  // some offset from that -- and should not have been freed yet.  p may be
  // null.
  static absl::optional<size_t> GetAllocatedSize(const void* p);

  // Returns
  // * kOwned if TCMalloc allocated the memory pointed to by p, or
  // * kNotOwned if allocated elsewhere or p is null.
  //
  // REQUIRES: p must be a value returned from a previous call to malloc(),
  // calloc(), realloc(), memalign(), posix_memalign(), valloc(), pvalloc(),
  // new, or new[], and must refer to memory that is currently allocated (so,
  // for instance, you should not pass in a pointer after having called free()
  // on it).
  enum class Ownership { kUnknown = 0, kOwned, kNotOwned };
  static Ownership GetOwnership(const void* p);

  // Type used by GetProperties.  See comment on GetProperties.
  struct Property {
    size_t value;
  };

  // Returns detailed statistics about the state of TCMalloc.  The map is keyed
  // by the name of the statistic.
  //
  // Common across malloc implementations:
  //  generic.bytes_in_use_by_app  -- Bytes currently in use by application
  //  generic.physical_memory_used -- Overall (including malloc internals)
  //  generic.virtual_memory_used  -- Overall (including malloc internals)
  //
  // Tcmalloc specific properties
  //  tcmalloc.cpu_free            -- Bytes in per-cpu free-lists
  //  tcmalloc.thread_cache_free   -- Bytes in per-thread free-lists
  //  tcmalloc.transfer_cache      -- Bytes in cross-thread transfer caches
  //  tcmalloc.central_cache_free  -- Bytes in central cache
  //  tcmalloc.page_heap_free      -- Bytes in page heap
  //  tcmalloc.page_heap_unmapped  -- Bytes in page heap (no backing phys. mem)
  //  tcmalloc.metadata_bytes      -- Used by internal data structures
  //  tcmalloc.thread_cache_count  -- Number of thread caches in use
  //  tcmalloc.experiment.NAME     -- Experiment NAME is running if 1
  static std::map<std::string, Property> GetProperties();

  static Profile SnapshotCurrent(tcmalloc::ProfileType type);

  // AllocationProfilingToken tracks an active profiling session started with
  // StartAllocationProfiling.  Profiling continues until Stop() is called.
  class AllocationProfilingToken {
   public:
    AllocationProfilingToken() = default;
    AllocationProfilingToken(AllocationProfilingToken&&) = default;
    AllocationProfilingToken(const AllocationProfilingToken&) = delete;
    ~AllocationProfilingToken();

    AllocationProfilingToken& operator=(AllocationProfilingToken&&) = default;
    AllocationProfilingToken& operator=(const AllocationProfilingToken&) =
        delete;

    // Finish the recording started by the corresponding call to
    // StartAllocationProfile, and return samples of calls to each function.  If
    // it is called more than once, subsequent calls will return an empty
    // profile.
    Profile Stop() &&;

   private:
    explicit AllocationProfilingToken(
        std::unique_ptr<tcmalloc_internal::AllocationProfilingTokenBase>);

    std::unique_ptr<tcmalloc_internal::AllocationProfilingTokenBase> impl_;
    friend class tcmalloc_internal::AllocationProfilingTokenAccessor;
  };

  // Start recording a sample of allocation and deallocation calls.  Returns
  // null if the implementation does not support profiling.
  static AllocationProfilingToken StartAllocationProfiling();

  // Runs housekeeping actions for the allocator off of the main allocation path
  // of new/delete.  As of 2020, this includes:
  // * Inspecting the current CPU mask and releasing memory from inaccessible
  //   CPUs.
  // * Releasing GetBackgroundReleaseRate() bytes per second from the page
  //   heap, if that many bytes are free, via ReleaseMemoryToSystem().
  //
  // When linked against TCMalloc, this method does not return.
  static void ProcessBackgroundActions();

  // Return true if ProcessBackgroundActions should be called on this platform.
  // Not all platforms need/support background actions. As of 2021 this
  // includes Apple and Emscripten.
  static bool NeedsProcessBackgroundActions();

  // Specifies a rate in bytes per second.
  //
  // The enum is used to provide strong-typing for the value.
  enum class BytesPerSecond : size_t {};

  // Gets the current release rate (in bytes per second) from the page heap.
  // Zero inhibits the release path.
  static BytesPerSecond GetBackgroundReleaseRate();
  // Specifies the release rate from the page heap.  ProcessBackgroundActions
  // must be called for this to be operative.
  static void SetBackgroundReleaseRate(BytesPerSecond rate);

  // Enables fork support.
  // Allocator will continue to function correctly in the child, after calling fork().
  static void EnableForkSupport();

  using SoftMemoryLimitCallback = void();
  static void SetSoftMemoryLimitHandler(SoftMemoryLimitCallback* handler);
  static SoftMemoryLimitCallback* GetSoftMemoryLimitHandler();

  using CreateSampleUserDataCallback = void*();
  using CopySampleUserDataCallback = void*(void*);
  using DestroySampleUserDataCallback = void(void*);

  // Sets callbacks for lifetime control of custom user data attached to allocation samples
  static void SetSampleUserDataCallbacks(
    CreateSampleUserDataCallback create,
    CopySampleUserDataCallback copy,
    DestroySampleUserDataCallback destroy);
};

}  // namespace tcmalloc

// The nallocx function allocates no memory, but it performs the same size
// computation as the malloc function, and returns the real size of the
// allocation that would result from the equivalent malloc function call.
// Default weak implementation returns size unchanged, but tcmalloc overrides it
// and returns rounded up size. See the following link for details:
// http://www.unix.com/man-page/freebsd/3/nallocx/
extern "C" size_t nallocx(size_t size, int flags) noexcept;

// The sdallocx function deallocates memory allocated by malloc or memalign.  It
// takes a size parameter to pass the original allocation size.
//
// The default weak implementation calls free(), but TCMalloc overrides it and
// uses the size to improve deallocation performance.
extern "C" void sdallocx(void* ptr, size_t size, int flags) noexcept;

namespace tcmalloc {

// Pointer / capacity information as returned by
// tcmalloc_size_returning_operator_new(). See
// tcmalloc_size_returning_operator_new() for more information.
struct sized_ptr_t {
  void* p;
  size_t n;
};

}  // namespace tcmalloc

// Allocates memory of at least the requested size.
//
// Returns a `sized_ptr_t` struct holding the allocated pointer, and the
// capacity of the allocated memory, which may be larger than the requested
// size.
//
// The returned pointer follows the alignment requirements of the standard new
// operator. This function will terminate on failure, except for the APIs
// accepting the std::nothrow parameter which will return {nullptr, 0} on
// failure.
//
// The returned pointer must be freed calling the matching ::operator delete.
//
// If a sized operator delete operator is invoked, then the 'size' parameter
// passed to delete must be greater or equal to the original requested size, and
// less than or equal to the capacity of the allocated memory as returned by the
// `tcmalloc_size_returning_operator_new` method.
//
// If neither the original size or capacity is known, then the non-sized
// operator delete can be invoked, however, this should be avoided, as this is
// substantially less efficient.
//
// The default weak implementation allocates the memory using the corresponding
// (matching) ::operator new(size_t, ...).
//
// This is a prototype API for the extension to C++ "size feedback in operator
// new" proposal:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0901r5.html
extern "C" {
tcmalloc::sized_ptr_t tcmalloc_size_returning_operator_new(size_t size);
tcmalloc::sized_ptr_t tcmalloc_size_returning_operator_new_nothrow(
    size_t size) noexcept;

// Aligned size returning new is only supported for libc++ because of issues
// with libstdcxx.so linkage. See http://b/110969867 for background.
#if defined(__cpp_aligned_new)

// Identical to `tcmalloc_size_returning_operator_new` except that the returned
// memory is aligned according to the `alignment` argument.
tcmalloc::sized_ptr_t tcmalloc_size_returning_operator_new_aligned(
    size_t size, std::align_val_t alignment);
tcmalloc::sized_ptr_t tcmalloc_size_returning_operator_new_aligned_nothrow(
    size_t size, std::align_val_t alignment) noexcept;

#endif  // __cpp_aligned_new

}  // extern "C"

#ifndef MALLOCX_LG_ALIGN
#define MALLOCX_LG_ALIGN(la) (la)
#endif

namespace tcmalloc {
namespace tcmalloc_internal {

// AllocationProfilingTokenBase tracks an on-going profiling session of sampled
// allocations.  The session ends when Stop() is called.
//
// This decouples the implementation details (of TCMalloc) from the interface,
// allowing non-TCMalloc allocators (such as libc and sanitizers) to be provided
// while allowing the library to compile and link.
class AllocationProfilingTokenBase {
 public:
  // Explicitly declare the ctor to put it in the google_malloc section.
  AllocationProfilingTokenBase() = default;

  virtual ~AllocationProfilingTokenBase() = default;

  // Finish recording started during construction of this object.
  //
  // After the first call, Stop() will return an empty profile.
  virtual Profile Stop() && = 0;
};

// ProfileBase contains a profile of allocations.
//
// This decouples the implementation details (of TCMalloc) from the interface,
// allowing non-TCMalloc allocators (such as libc and sanitizers) to be provided
// while allowing the library to compile and link.
class ProfileBase {
 public:
  virtual ~ProfileBase() = default;

  // For each sample in the profile, Iterate invokes the callback f on the
  // sample.
  virtual void Iterate(
      absl::FunctionRef<void(const Profile::Sample&)> f) const = 0;

  // The approximate interval between recorded samples of the event of interest.
  // A period of 1 means every sample was recorded.
  virtual int64_t Period() const = 0;

  // The type of profile (live objects, allocated, etc.).
  virtual ProfileType Type() const = 0;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc

#endif  // TCMALLOC_MALLOC_EXTENSION_H_
