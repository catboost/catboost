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
//
// Common definitions for tcmalloc code.

#ifndef TCMALLOC_COMMON_H_
#define TCMALLOC_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/size_class_info.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

//-------------------------------------------------------------------
// Configuration
//-------------------------------------------------------------------

// There are four different models for tcmalloc which are created by defining a
// set of constant variables differently:
//
// DEFAULT:
//   The default configuration strives for good performance while trying to
//   minimize fragmentation.  It uses a smaller page size to reduce
//   fragmentation, but allocates per-thread and per-cpu capacities similar to
//   TCMALLOC_LARGE_PAGES / TCMALLOC_256K_PAGES.
//
// TCMALLOC_LARGE_PAGES:
//   Larger page sizes increase the bookkeeping granularity used by TCMalloc for
//   its allocations.  This can reduce PageMap size and traffic to the
//   innermost cache (the page heap), but can increase memory footprints.  As
//   TCMalloc will not reuse a page for a different allocation size until the
//   entire page is deallocated, this can be a source of increased memory
//   fragmentation.
//
//   Historically, larger page sizes improved lookup performance for the
//   pointer-to-size lookup in the PageMap that was part of the critical path.
//   With most deallocations leveraging C++14's sized delete feature
//   (https://isocpp.org/files/papers/n3778.html), this optimization is less
//   significant.
//
// TCMALLOC_256K_PAGES
//   This configuration uses an even larger page size (256KB) as the unit of
//   accounting granularity.
//
// TCMALLOC_SMALL_BUT_SLOW:
//   Used for situations where minimizing the memory footprint is the most
//   desirable attribute, even at the cost of performance.
//
// The constants that vary between models are:
//
//   kPageShift - Shift amount used to compute the page size.
//   kNumBaseClasses - Number of size classes serviced by bucket allocators
//   kMaxSize - Maximum size serviced by bucket allocators (thread/cpu/central)
//   kMinThreadCacheSize - The minimum size in bytes of each ThreadCache.
//   kMaxThreadCacheSize - The maximum size in bytes of each ThreadCache.
//   kDefaultOverallThreadCacheSize - The maximum combined size in bytes of all
//     ThreadCaches for an executable.
//   kStealAmount - The number of bytes one ThreadCache will steal from another
//     when the first ThreadCache is forced to Scavenge(), delaying the next
//     call to Scavenge for this thread.

// Older configurations had their own customized macros.  Convert them into
// a page-shift parameter that is checked below.

#ifndef TCMALLOC_PAGE_SHIFT
#ifdef TCMALLOC_SMALL_BUT_SLOW
#define TCMALLOC_PAGE_SHIFT 12
#define TCMALLOC_USE_PAGEMAP3
#elif defined(TCMALLOC_256K_PAGES)
#define TCMALLOC_PAGE_SHIFT 18
#elif defined(TCMALLOC_LARGE_PAGES)
#define TCMALLOC_PAGE_SHIFT 15
#else
#define TCMALLOC_PAGE_SHIFT 13
#endif
#else
#error "TCMALLOC_PAGE_SHIFT is an internal macro!"
#endif

#if TCMALLOC_PAGE_SHIFT == 12
inline constexpr size_t kPageShift = 12;
inline constexpr size_t kNumBaseClasses = 46;
inline constexpr bool kHasExpandedClasses = false;
inline constexpr size_t kMaxSize = 8 << 10;
inline constexpr size_t kMinThreadCacheSize = 4 * 1024;
inline constexpr size_t kMaxThreadCacheSize = 64 * 1024;
inline constexpr size_t kMaxCpuCacheSize = 20 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize = kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = kMinThreadCacheSize;
inline constexpr size_t kDefaultProfileSamplingRate = 1 << 19;
inline constexpr size_t kMinPages = 2;
#elif TCMALLOC_PAGE_SHIFT == 15
inline constexpr size_t kPageShift = 15;
inline constexpr size_t kNumBaseClasses = 78;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingRate = 1 << 21;
inline constexpr size_t kMinPages = 8;
#elif TCMALLOC_PAGE_SHIFT == 18
inline constexpr size_t kPageShift = 18;
inline constexpr size_t kNumBaseClasses = 89;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingRate = 1 << 21;
inline constexpr size_t kMinPages = 8;
#elif TCMALLOC_PAGE_SHIFT == 13
inline constexpr size_t kPageShift = 13;
inline constexpr size_t kNumBaseClasses = 86;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingRate = 1 << 21;
inline constexpr size_t kMinPages = 8;
#else
#error "Unsupported TCMALLOC_PAGE_SHIFT value!"
#endif

// Sanitizers constrain the memory layout which causes problems with the
// enlarged tags required to represent NUMA partitions. Disable NUMA awareness
// to avoid failing to mmap memory.
#if defined(TCMALLOC_NUMA_AWARE) && !defined(MEMORY_SANITIZER) && \
    !defined(THREAD_SANITIZER)
inline constexpr size_t kNumaPartitions = 2;
#else
inline constexpr size_t kNumaPartitions = 1;
#endif

// We have copies of kNumBaseClasses size classes for each NUMA node, followed
// by any expanded classes.
inline constexpr size_t kExpandedClassesStart =
    kNumBaseClasses * kNumaPartitions;
inline constexpr size_t kNumClasses =
    kExpandedClassesStart + (kHasExpandedClasses ? kNumBaseClasses : 0);

// Size classes are often stored as uint32_t values, but there are some
// situations where we need to store a size class with as compact a
// representation as possible (e.g. in PageMap). Here we determine the integer
// type to use in these situations - i.e. the smallest integer type large
// enough to store values in the range [0,kNumClasses).
constexpr size_t kMaxClass = kNumClasses - 1;
using CompactSizeClass =
    std::conditional_t<kMaxClass <= std::numeric_limits<uint8_t>::max(),
                       uint8_t, uint16_t>;

// ~64K classes ought to be enough for anybody, but let's be sure.
static_assert(kMaxClass <= std::numeric_limits<CompactSizeClass>::max());

// Minimum/maximum number of batches in TransferCache per size class.
// Actual numbers depends on a number of factors, see TransferCache::Init
// for details.
inline constexpr size_t kMinObjectsToMove = 2;
inline constexpr size_t kMaxObjectsToMove = 128;

inline constexpr size_t kPageSize = 1 << kPageShift;
// Verify that the page size used is at least 8x smaller than the maximum
// element size in the thread cache.  This guarantees at most 12.5% internal
// fragmentation (1/8). When page size is 256k (kPageShift == 18), the benefit
// of increasing kMaxSize to be multiple of kPageSize is unclear. Object size
// profile data indicates that the number of simultaneously live objects (of
// size >= 256k) tends to be very small. Keeping those objects as 'large'
// objects won't cause too much memory waste, while heap memory reuse can be
// improved. Increasing kMaxSize to be too large has another bad side effect --
// the thread cache pressure is increased, which will in turn increase traffic
// between central cache and thread cache, leading to performance degradation.
static_assert((kMaxSize / kPageSize) >= kMinPages || kPageShift >= 18,
              "Ratio of kMaxSize / kPageSize is too small");

inline constexpr size_t kAlignment = 8;
// log2 (kAlignment)
inline constexpr size_t kAlignmentShift = absl::bit_width(kAlignment - 1u);

// The number of times that a deallocation can cause a freelist to
// go over its max_length() before shrinking max_length().
inline constexpr int kMaxOverages = 3;

// Maximum length we allow a per-thread free-list to have before we
// move objects from it into the corresponding central free-list.  We
// want this big to avoid locking the central free-list too often.  It
// should not hurt to make this list somewhat big because the
// scavenging code will shrink it down when its contents are not in use.
inline constexpr int kMaxDynamicFreeListLength = 8192;

enum class MemoryTag : uint8_t {
  // Sampled, infrequently allocated
  kSampled = 0x0,
  // Not sampled, NUMA partition 0
  kNormalP0 = 0x1,
  // Not sampled, NUMA partition 1
  kNormalP1 = (kNumaPartitions > 1) ? 0x2 : 0xff,
  // Not sampled
  kNormal = kNormalP0,
};

inline constexpr uintptr_t kTagShift = std::min(kAddressBits - 4, 42);
inline constexpr uintptr_t kTagMask = uintptr_t{0x3} << kTagShift;

// Returns true if ptr is tagged.
ABSL_DEPRECATED("Replace with specific tests")
inline bool IsTaggedMemory(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & kTagMask) == 0;
}

inline bool IsSampledMemory(const void* ptr) {
  constexpr uintptr_t kSampledNormalMask = kNumaPartitions > 1 ? 0x3 : 0x1;

  static_assert(static_cast<uintptr_t>(MemoryTag::kNormalP0) &
                kSampledNormalMask);
  static_assert(static_cast<uintptr_t>(MemoryTag::kNormalP1) &
                kSampledNormalMask);

  const uintptr_t tag =
      (reinterpret_cast<uintptr_t>(ptr) & kTagMask) >> kTagShift;
  return (tag & kSampledNormalMask) ==
         static_cast<uintptr_t>(MemoryTag::kSampled);
}

inline bool IsNormalMemory(const void* ptr) { return !IsSampledMemory(ptr); }

inline MemoryTag GetMemoryTag(const void* ptr) {
  return static_cast<MemoryTag>((reinterpret_cast<uintptr_t>(ptr) & kTagMask) >>
                                kTagShift);
}

absl::string_view MemoryTagToLabel(MemoryTag tag);

inline constexpr bool IsExpandedSizeClass(unsigned cl) {
  return kHasExpandedClasses && (cl >= kExpandedClassesStart);
}

#if !defined(TCMALLOC_SMALL_BUT_SLOW) && __SIZEOF_POINTER__ != 4
// Always allocate at least a huge page
inline constexpr size_t kMinSystemAlloc = kHugePageSize;
inline constexpr size_t kMinMmapAlloc = 1 << 30;  // mmap() in 1GiB ranges.
#else
// Allocate in units of 2MiB. This is the size of a huge page for x86, but
// not for Power.
inline constexpr size_t kMinSystemAlloc = 2 << 20;
// mmap() in units of 32MiB. This is a multiple of huge page size for
// both x86 (2MiB) and Power (16MiB)
inline constexpr size_t kMinMmapAlloc = 32 << 20;
#endif

static_assert(kMinMmapAlloc % kMinSystemAlloc == 0,
              "Minimum mmap allocation size is not a multiple of"
              " minimum system allocation size");

inline MemoryTag NumaNormalTag(size_t numa_partition) {
  switch (numa_partition) {
    case 0:
      return MemoryTag::kNormalP0;
    case 1:
      return MemoryTag::kNormalP1;
    default:
      ASSUME(false);
      __builtin_unreachable();
  }
}

inline size_t NumaPartitionFromPointer(void* ptr) {
  if constexpr (kNumaPartitions == 1) {
    return 0;
  }

  switch (GetMemoryTag(ptr)) {
    case MemoryTag::kNormalP1:
      return 1;
    default:
      return 0;
  }
}

// Size-class information + mapping
class SizeMap {
 public:
  // All size classes <= 512 in all configs always have 1 page spans.
  static constexpr size_t kMultiPageSize = 512;
  // Min alignment for all size classes > kMultiPageSize in all configs.
  static constexpr size_t kMultiPageAlignment = 64;
  // log2 (kMultiPageAlignment)
  static constexpr size_t kMultiPageAlignmentShift =
      absl::bit_width(kMultiPageAlignment - 1u);

 private:
  //-------------------------------------------------------------------
  // Mapping from size to size_class and vice versa
  //-------------------------------------------------------------------

  // Sizes <= 1024 have an alignment >= 8.  So for such sizes we have an
  // array indexed by ceil(size/8).  Sizes > 1024 have an alignment >= 128.
  // So for these larger sizes we have an array indexed by ceil(size/128).
  //
  // We flatten both logical arrays into one physical array and use
  // arithmetic to compute an appropriate index.  The constants used by
  // ClassIndex() were selected to make the flattening work.
  //
  // Examples:
  //   Size       Expression                      Index
  //   -------------------------------------------------------
  //   0          (0 + 7) / 8                     0
  //   1          (1 + 7) / 8                     1
  //   ...
  //   1024       (1024 + 7) / 8                  128
  //   1025       (1025 + 127 + (120<<7)) / 128   129
  //   ...
  //   32768      (32768 + 127 + (120<<7)) / 128  376
  static constexpr int kMaxSmallSize = 1024;
  static constexpr size_t kClassArraySize =
      ((kMaxSize + 127 + (120 << 7)) >> 7) + 1;

  // Batch size is the number of objects to move at once.
  typedef unsigned char BatchSize;

  // class_array_ is accessed on every malloc, so is very hot.  We make it the
  // first member so that it inherits the overall alignment of a SizeMap
  // instance.  In particular, if we create a SizeMap instance that's cache-line
  // aligned, this member is also aligned to the width of a cache line.
  CompactSizeClass
      class_array_[kClassArraySize * (kHasExpandedClasses ? 2 : 1)] = {0};

  // Number of objects to move between a per-thread list and a central
  // list in one shot.  We want this to be not too small so we can
  // amortize the lock overhead for accessing the central list.  Making
  // it too big may temporarily cause unnecessary memory wastage in the
  // per-thread free list until the scavenger cleans up the list.
  BatchSize num_objects_to_move_[kNumClasses] = {0};

  // If size is no more than kMaxSize, compute index of the
  // class_array[] entry for it, putting the class index in output
  // parameter idx and returning true. Otherwise return false.
  static inline bool ABSL_ATTRIBUTE_ALWAYS_INLINE
  ClassIndexMaybe(size_t s, uint32_t* idx) {
    if (ABSL_PREDICT_TRUE(s <= kMaxSmallSize)) {
      *idx = (static_cast<uint32_t>(s) + 7) >> 3;
      return true;
    } else if (s <= kMaxSize) {
      *idx = (static_cast<uint32_t>(s) + 127 + (120 << 7)) >> 7;
      return true;
    }
    return false;
  }

  static inline size_t ClassIndex(size_t s) {
    uint32_t ret;
    CHECK_CONDITION(ClassIndexMaybe(s, &ret));
    return ret;
  }

  // Mapping from size class to number of pages to allocate at a time
  unsigned char class_to_pages_[kNumClasses] = {0};

  // Mapping from size class to max size storable in that class
  uint32_t class_to_size_[kNumClasses] = {0};

  // If environment variable defined, use it to override sizes classes.
  // Returns true if all classes defined correctly.
  bool MaybeRunTimeSizeClasses();

 protected:
  // Set the give size classes to be used by TCMalloc.
  void SetSizeClasses(int num_classes, const SizeClassInfo* parsed);

  // Check that the size classes meet all requirements.
  bool ValidSizeClasses(int num_classes, const SizeClassInfo* parsed);

  // Definition of size class that is set in size_classes.cc
  static const SizeClassInfo kSizeClasses[];
  static const int kSizeClassesCount;

  static const SizeClassInfo kExperimentalPow2Below64SizeClasses[];
  static const int kExperimentalPow2Below64SizeClassesCount;
  // kExperimentalPowBelow64SizeClassesCount
  static const SizeClassInfo kExperimentalPow2SizeClasses[];
  static const int kExperimentalPow2SizeClassesCount;

  // Definition of size class that is set in size_classes.cc
  static const SizeClassInfo kLegacySizeClasses[];
  static const int kLegacySizeClassesCount;

 public:
  // constexpr constructor to guarantee zero-initialization at compile-time.  We
  // rely on Init() to populate things.
  constexpr SizeMap() = default;

  // Initialize the mapping arrays
  void Init();

  // Returns the size class for size `size` respecting the alignment
  // requirements of `policy`.
  //
  // Returns true on success. Returns false if either:
  // - the size exceeds the maximum size class size.
  // - the align size is greater or equal to the default page size
  // - no matching properly aligned size class is available
  //
  // Requires that policy.align() returns a non-zero power of 2.
  //
  // When policy.align() = 1 the default alignment of the size table will be
  // used. If policy.align() is constexpr 1 (e.g. when using
  // DefaultAlignPolicy) then alignment-related code will optimize away.
  //
  // TODO(b/171978365): Replace the output parameter with returning
  // absl::optional<uint32_t>.
  template <typename Policy>
  inline bool ABSL_ATTRIBUTE_ALWAYS_INLINE GetSizeClass(Policy policy,
                                                        size_t size,
                                                        uint32_t* cl) {
    const size_t align = policy.align();
    ASSERT(absl::has_single_bit(align));

    if (ABSL_PREDICT_FALSE(align >= kPageSize)) {
      // TODO(b/172060547): Consider changing this to align > kPageSize.
      ABSL_ANNOTATE_MEMORY_IS_UNINITIALIZED(cl, sizeof(*cl));
      return false;
    }

    uint32_t idx;
    if (ABSL_PREDICT_FALSE(!ClassIndexMaybe(size, &idx))) {
      ABSL_ANNOTATE_MEMORY_IS_UNINITIALIZED(cl, sizeof(*cl));
      return false;
    }
    *cl = class_array_[idx] + policy.scaled_numa_partition();

    // Predict that size aligned allocs most often directly map to a proper
    // size class, i.e., multiples of 32, 64, etc, matching our class sizes.
    const size_t mask = (align - 1);
    do {
      if (ABSL_PREDICT_TRUE((class_to_size(*cl) & mask) == 0)) {
        return true;
      }
    } while ((++*cl % kNumBaseClasses) != 0);

    ABSL_ANNOTATE_MEMORY_IS_UNINITIALIZED(cl, sizeof(*cl));
    return false;
  }

  // Returns size class for given size, or 0 if this instance has not been
  // initialized yet. REQUIRES: size <= kMaxSize.
  template <typename Policy>
  inline size_t ABSL_ATTRIBUTE_ALWAYS_INLINE SizeClass(Policy policy,
                                                       size_t size) {
    ASSERT(size <= kMaxSize);
    uint32_t ret = 0;
    GetSizeClass(policy, size, &ret);
    return ret;
  }

  // Get the byte-size for a specified class. REQUIRES: cl <= kNumClasses.
  inline size_t ABSL_ATTRIBUTE_ALWAYS_INLINE class_to_size(size_t cl) {
    ASSERT(cl < kNumClasses);
    return class_to_size_[cl];
  }

  // Mapping from size class to number of pages to allocate at a time
  inline size_t class_to_pages(size_t cl) {
    ASSERT(cl < kNumClasses);
    return class_to_pages_[cl];
  }

  // Number of objects to move between a per-thread list and a central
  // list in one shot.  We want this to be not too small so we can
  // amortize the lock overhead for accessing the central list.  Making
  // it too big may temporarily cause unnecessary memory wastage in the
  // per-thread free list until the scavenger cleans up the list.
  inline SizeMap::BatchSize num_objects_to_move(size_t cl) {
    ASSERT(cl < kNumClasses);
    return num_objects_to_move_[cl];
  }
};

// Linker initialized, so this lock can be accessed at any time.
extern absl::base_internal::SpinLock pageheap_lock;

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_COMMON_H_
