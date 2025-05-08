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
//
// Common definitions for tcmalloc code.

#ifndef TCMALLOC_COMMON_H_
#define TCMALLOC_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cerrno>
#include <limits>
#include <new>
#include <type_traits>

#include "absl/base/internal/spinlock.h"
#include "absl/base/optimization.h"
#include "absl/numeric/bits.h"
#include "absl/strings/string_view.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/memory_tag.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/malloc_extension.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

static_assert(sizeof(void*) == 8);

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
//   TCMALLOC_INTERNAL_32K_PAGES / TCMALLOC_INTERNAL_256K_PAGES.
//
// TCMALLOC_INTERNAL_32K_PAGES:
//   Larger page sizes (32KB) increase the bookkeeping granularity used by
//   TCMalloc for its allocations.  This can reduce PageMap size and traffic to
//   the innermost cache (the page heap), but can increase memory footprints. As
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
// TCMALLOC_INTERNAL_256K_PAGES
//   This configuration uses an even larger page size (256KB) as the unit of
//   accounting granularity.
//
// TCMALLOC_INTERNAL_SMALL_BUT_SLOW:
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
//   kDefaultProfileSamplingInterval - Bytes between sampled allocations.

// Older configurations had their own customized macros.  Convert them into
// a page-shift parameter that is checked below.

#ifndef TCMALLOC_PAGE_SHIFT
#ifdef TCMALLOC_INTERNAL_SMALL_BUT_SLOW
#define TCMALLOC_PAGE_SHIFT 12
#define TCMALLOC_USE_PAGEMAP3
#elif defined(TCMALLOC_INTERNAL_256K_PAGES)
#define TCMALLOC_PAGE_SHIFT 18
#elif defined(TCMALLOC_INTERNAL_32K_PAGES)
#define TCMALLOC_PAGE_SHIFT 15
#else
#define TCMALLOC_PAGE_SHIFT 13
#endif
#else
#error "TCMALLOC_PAGE_SHIFT is an internal macro!"
#endif

#if defined(TCMALLOC_INTERNAL_SMALL_BUT_SLOW) + \
        defined(TCMALLOC_INTERNAL_8K_PAGES) +   \
        defined(TCMALLOC_INTERNAL_256K_PAGES) + \
        defined(TCMALLOC_INTERNAL_32K_PAGES) >  \
    1
#error "At most 1 variant configuration must be used."
#endif

#if TCMALLOC_PAGE_SHIFT == 12
inline constexpr size_t kPageShift = 12;
inline constexpr size_t kNumBaseClasses = 46;
inline constexpr bool kHasExpandedClasses = false;
inline constexpr size_t kMaxSize = 8 << 10;
inline constexpr size_t kMinThreadCacheSize = 4 * 1024;
inline constexpr size_t kMaxThreadCacheSize = 64 * 1024;
inline constexpr size_t kMaxCpuCacheSize = 10 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize = kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = kMinThreadCacheSize;
inline constexpr size_t kDefaultProfileSamplingInterval = 1 << 19;
#elif TCMALLOC_PAGE_SHIFT == 15
inline constexpr size_t kPageShift = 15;
inline constexpr size_t kNumBaseClasses = 78;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 1.5 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingInterval = 1 << 21;
#elif TCMALLOC_PAGE_SHIFT == 18
inline constexpr size_t kPageShift = 18;
inline constexpr size_t kNumBaseClasses = 89;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 1.5 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingInterval = 1 << 21;
#elif TCMALLOC_PAGE_SHIFT == 13
inline constexpr size_t kPageShift = 13;
inline constexpr size_t kNumBaseClasses = 86;
inline constexpr bool kHasExpandedClasses = true;
inline constexpr size_t kMaxSize = 256 * 1024;
inline constexpr size_t kMinThreadCacheSize = kMaxSize * 2;
inline constexpr size_t kMaxThreadCacheSize = 4 << 20;
inline constexpr size_t kMaxCpuCacheSize = 1.5 * 1024 * 1024;
inline constexpr size_t kDefaultOverallThreadCacheSize =
    8u * kMaxThreadCacheSize;
inline constexpr size_t kStealAmount = 1 << 16;
inline constexpr size_t kDefaultProfileSamplingInterval = 1 << 21;
#else
#error "Unsupported TCMALLOC_PAGE_SHIFT value!"
#endif

// Disable NUMA awareness under Sanitizers to avoid failing to mmap memory.
#if defined(TCMALLOC_INTERNAL_NUMA_AWARE)
inline constexpr size_t kNumaPartitions = kSanitizerAddressSpace ? 1 : 2;
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

inline constexpr std::align_val_t kAlignment{8};
// log2 (kAlignment)
inline constexpr size_t kAlignmentShift =
    absl::bit_width(static_cast<size_t>(kAlignment) - 1u);

// The number of times that a deallocation can cause a freelist to
// go over its max_length() before shrinking max_length().
inline constexpr int kMaxOverages = 3;

// Maximum length we allow a per-thread free-list to have before we
// move objects from it into the corresponding transfer cache.  We
// want this big to avoid locking the transfer cache too often.  It
// should not hurt to make this list somewhat big because the
// scavenging code will shrink it down when its contents are not in use.
inline constexpr size_t kMaxDynamicFreeListLength = 8192;

inline constexpr bool ColdFeatureActive() { return kHasExpandedClasses; }

inline constexpr bool IsExpandedSizeClass(unsigned size_class) {
  return kHasExpandedClasses && (size_class >= kExpandedClassesStart);
}

#if !defined(TCMALLOC_INTERNAL_SMALL_BUT_SLOW)
inline constexpr size_t kMinMmapAlloc = 1 << 30;  // mmap() in 1GiB ranges.
#else
// mmap() in units of 32MiB. This is a multiple of huge page size for
// both x86 (2MiB) and Power (16MiB)
inline constexpr size_t kMinMmapAlloc = 32 << 20;
#endif

static_assert(
    kMinMmapAlloc % kHugePageSize == 0,
    "Minimum mmap allocation size is not a multiple of the huge page size");

enum class AllocationAccess {
  kHot,
  kCold,
};

inline AllocationAccess AccessFromPointer(void* ptr) {
  if (!kHasExpandedClasses) {
    TC_ASSERT_NE(GetMemoryTag(ptr), MemoryTag::kCold);
    return AllocationAccess::kHot;
  }

  return ABSL_PREDICT_FALSE(GetMemoryTag(ptr) == MemoryTag::kCold)
             ? AllocationAccess::kCold
             : AllocationAccess::kHot;
}

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

// Linker initialized, so this lock can be accessed at any time.
// Note: `CpuCache::ResizeInfo::lock` must be taken before the `pageheap_lock`
// if both are going to be held simultaneously.
extern absl::base_internal::SpinLock pageheap_lock;

class ABSL_SCOPED_LOCKABLE PageHeapSpinLockHolder {
 public:
  PageHeapSpinLockHolder()
      ABSL_EXCLUSIVE_LOCK_FUNCTION(pageheap_lock) = default;
  ~PageHeapSpinLockHolder() ABSL_UNLOCK_FUNCTION() = default;

 private:
  AllocationGuardSpinLockHolder lock_{&pageheap_lock};
};

// Evaluates a/b, avoiding division by zero.
inline double safe_div(double a, double b) {
  if (b == 0) {
    return 0.;
  } else {
    return a / b;
  }
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_COMMON_H_
