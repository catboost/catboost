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
// Routine that uses sbrk/mmap to allocate memory from the system.
// Useful for implementing malloc.

#ifndef TCMALLOC_SYSTEM_ALLOC_H_
#define TCMALLOC_SYSTEM_ALLOC_H_

#include <asm/unistd.h>
#ifdef __linux__
#include <linux/mempolicy.h>
#endif
#include <stddef.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/exponential_biased.h"
#include "tcmalloc/internal/memory_tag.h"
#include "tcmalloc/internal/numa.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/page_size.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/malloc_extension.h"

#ifndef MADV_FREE
#define MADV_FREE 8
#endif

// The <sys/prctl.h> on some systems may not define these macros yet even though
// the kernel may have support for the new PR_SET_VMA syscall, so we explicitly
// define them here.
#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#endif

#ifndef PR_SET_VMA_ANON_NAME
#define PR_SET_VMA_ANON_NAME 0
#endif

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

struct AddressRange {
  void* ptr;
  size_t bytes;
};

template <typename Topology>
class SystemAllocator {
 public:
  constexpr explicit SystemAllocator(
      const Topology& topology ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : topology_(topology) {}

  // REQUIRES: "alignment" is a power of two or "0" to indicate default
  // alignment REQUIRES: "alignment" and "size" <= kTagMask
  //
  // Allocate and return "bytes" of zeroed memory.  The allocator may optionally
  // return more bytes than asked for (i.e. return an entire "huge" page).
  //
  // The returned pointer is a multiple of "alignment" if non-zero. The
  // returned pointer will always be aligned suitably for holding a
  // void*, double, or size_t. In addition, if this platform defines
  // ABSL_CACHELINE_ALIGNED, the return pointer will always be cacheline
  // aligned.
  //
  // The returned pointer is guaranteed to satisfy GetMemoryTag(ptr) == "tag".
  // Returns nullptr when out of memory.
  [[nodiscard]] AddressRange Allocate(size_t bytes, size_t alignment,
                                      MemoryTag tag);

  // Returns the number of times we failed to give pages back to the OS after a
  // call to Release.
  int release_errors() const {
    return release_errors_.load(std::memory_order_relaxed);
  }

  void set_madvise_preference(MadvisePreference v) {
    madvise_.store(v, std::memory_order_relaxed);
  }

  MadvisePreference madvise_preference() const {
    return madvise_.load(std::memory_order_relaxed);
  }

  // This call is a hint to the operating system that the pages
  // contained in the specified range of memory will not be used for a
  // while, and can be released for use by other processes or the OS.
  // Pages which are released in this way may be destroyed (zeroed) by
  // the OS.  The benefit of this function is that it frees memory for
  // use by the system, the cost is that the pages are faulted back into
  // the address space next time they are touched, which can impact
  // performance.  (Only pages fully covered by the memory region will
  // be released, partial pages will not.)
  //
  // Returns true on success.
  [[nodiscard]] bool Release(void* start, size_t length);

  // This call is the inverse of Release: the pages in this range are in use and
  // should be faulted in.  (In principle this is a best-effort hint, but in
  // practice we will unconditionally fault the range.)
  //
  // REQUIRES: [start, start + length) is a range aligned to 4KiB boundaries.
  inline void Back(void* start, size_t length) {
    // TODO(b/134694141): use madvise when we have better support for that;
    // taking faults is not free.

    // TODO(b/134694141): enable this, if we can avoid causing trouble for apps
    // that routinely make large mallocs they never touch (sigh).
  }

  // Returns the current address region factory.
  [[nodiscard]] AddressRegionFactory* GetRegionFactory() const;

  // Sets the current address region factory to factory.
  void SetRegionFactory(AddressRegionFactory* factory);

  // Reserves using mmap() a region of memory of the requested size and
  // alignment, with the bits specified by kTagMask set according to tag.
  //
  // REQUIRES: pagesize <= alignment <= kTagMask
  // REQUIRES: size <= kTagMask
  [[nodiscard]] void* MmapAligned(size_t size, size_t alignment, MemoryTag tag)
      ABSL_LOCKS_EXCLUDED(spinlock_);

  void AcquireInternalLocks() {
    spinlock_.Lock();
  }
  void ReleaseInternalLocks() {
    spinlock_.Unlock();
  }

 private:
  const Topology& topology_;

  static constexpr size_t kNumaPartitions = Topology::kNumPartitions;

  mutable absl::base_internal::SpinLock spinlock_{
      absl::kConstInit, absl::base_internal::SCHEDULE_KERNEL_ONLY};

  uintptr_t rnd_ ABSL_GUARDED_BY(spinlock_) = 0;
  absl::once_flag rnd_flag_;

  uintptr_t next_sampled_addr_ ABSL_GUARDED_BY(spinlock_) = 0;
  uintptr_t next_selsan_addr_ ABSL_GUARDED_BY(spinlock_) = 0;
  std::array<uintptr_t, kNumaPartitions> next_normal_addr_
      ABSL_GUARDED_BY(spinlock_) = {0};
  uintptr_t next_cold_addr_ ABSL_GUARDED_BY(spinlock_) = 0;
  uintptr_t next_metadata_addr_ ABSL_GUARDED_BY(spinlock_) = 0;

  std::atomic<int> release_errors_{0};
  std::atomic<MadvisePreference> madvise_{MadvisePreference::kDontNeed};

  void DiscardMappedRegions() ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_);

  // Checks that there is sufficient space available in the reserved region
  // for the next allocation, if not allocate a new region.
  // Then returns a pointer to the new memory.
  std::pair<void*, size_t> AllocateFromRegion(size_t size, size_t alignment,
                                              MemoryTag tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_);

  std::array<AddressRegion*, kNumaPartitions> normal_region_
      ABSL_GUARDED_BY(spinlock_){{nullptr}};
  AddressRegion* sampled_region_ ABSL_GUARDED_BY(spinlock_){nullptr};
  AddressRegion* selsan_region_ ABSL_GUARDED_BY(spinlock_){nullptr};
  AddressRegion* cold_region_ ABSL_GUARDED_BY(spinlock_){nullptr};
  AddressRegion* metadata_region_ ABSL_GUARDED_BY(spinlock_){nullptr};

  class MmapRegion final : public AddressRegion {
   public:
    MmapRegion(uintptr_t start, size_t size,
               AddressRegionFactory::UsageHint hint)
        : start_(start), free_size_(size), hint_(hint) {}
    std::pair<void*, size_t> Alloc(size_t size, size_t alignment) override;
    ~MmapRegion() override = default;

   private:
    const uintptr_t start_;
    size_t free_size_;
    const AddressRegionFactory::UsageHint hint_;
  };

  class MmapRegionFactory final : public AddressRegionFactory {
   public:
    constexpr MmapRegionFactory() = default;
    ~MmapRegionFactory() override = default;

    AddressRegion* Create(void* start, size_t size, UsageHint hint) override;
    size_t GetStats(absl::Span<char> buffer) override;
    size_t GetStatsInPbtxt(absl::Span<char> buffer) override;

   private:
    std::atomic<size_t> bytes_reserved_{0};
  };

  MmapRegionFactory mmap_factory_ ABSL_GUARDED_BY(spinlock_);
  AddressRegionFactory* region_factory_ ABSL_GUARDED_BY(spinlock_) =
      &mmap_factory_;

  AddressRegionFactory::UsageHint TagToHint(MemoryTag tag) const;
  void BindMemory(void* base, size_t size, size_t partition) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_);
  uintptr_t RandomMmapHint(size_t size, size_t alignment, MemoryTag tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_);
  [[nodiscard]] void* MmapAlignedLocked(size_t size, size_t alignment,
                                        MemoryTag tag)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_);
  [[nodiscard]] bool ReleasePages(void* start, size_t length) const;
};

namespace system_allocator_internal {

// Check that no bit is set at position ADDRESS_BITS or higher.
template <int ADDRESS_BITS>
void CheckAddressBits(uintptr_t ptr) {
  TC_ASSERT_EQ(ptr >> ADDRESS_BITS, 0);
}

// Specialize for the bit width of a pointer to avoid undefined shift.
template <>
ABSL_ATTRIBUTE_UNUSED inline void CheckAddressBits<8 * sizeof(void*)>(
    uintptr_t ptr) {}

static_assert(kAddressBits <= 8 * sizeof(void*),
              "kAddressBits must be smaller than the pointer size");

// Rounds size down to a multiple of alignment.
inline size_t RoundDown(const size_t size, const size_t alignment) {
  // Checks that the alignment has only one bit set.
  TC_ASSERT(absl::has_single_bit(alignment));
  return (size) & ~(alignment - 1);
}

// Rounds size up to a multiple of alignment.
inline size_t RoundUp(const size_t size, const size_t alignment) {
  return RoundDown(size + alignment - 1, alignment);
}

int MapFixedNoReplaceFlagAvailable();

}  // namespace system_allocator_internal

template <typename Topology>
AddressRange SystemAllocator<Topology>::Allocate(size_t bytes, size_t alignment,
                                                 const MemoryTag tag) {
  // If default alignment is set request the minimum alignment provided by
  // the system.
  alignment = std::max(alignment, GetPageSize());

  // Discard requests that overflow
  if (bytes + alignment < bytes) return {nullptr, 0};

  AllocationGuardSpinLockHolder lock_holder(&spinlock_);

  auto [result, actual_bytes] = AllocateFromRegion(bytes, alignment, tag);

  if (result != nullptr) {
    system_allocator_internal::CheckAddressBits<kAddressBits>(
        reinterpret_cast<uintptr_t>(result) + actual_bytes - 1);
    TC_ASSERT_EQ(GetMemoryTag(result), tag);
  }
  return {result, actual_bytes};
}

template <typename Topology>
AddressRegionFactory* SystemAllocator<Topology>::GetRegionFactory() const {
  AllocationGuardSpinLockHolder lock_holder(&spinlock_);
  return region_factory_;
}

template <typename Topology>
void SystemAllocator<Topology>::SetRegionFactory(
    AddressRegionFactory* factory) {
  AllocationGuardSpinLockHolder lock_holder(&spinlock_);
  DiscardMappedRegions();
  region_factory_ = factory;
}

template <typename Topology>
void SystemAllocator<Topology>::DiscardMappedRegions() {
  std::fill(normal_region_.begin(), normal_region_.end(), nullptr);
  sampled_region_ = nullptr;
  selsan_region_ = nullptr;
  cold_region_ = nullptr;
  metadata_region_ = nullptr;
}

template <typename Topology>
std::pair<void*, size_t> SystemAllocator<Topology>::MmapRegion::Alloc(
    size_t request_size, size_t alignment) {
  using system_allocator_internal::RoundUp;

  // Align on kHugePageSize boundaries to reduce external fragmentation for
  // future allocations.
  size_t size = RoundUp(request_size, kHugePageSize);
  if (size < request_size) return {nullptr, 0};
  alignment = std::max(alignment, kHugePageSize);

  // Tries to allocate size bytes from the end of [start_, start_ + free_size_),
  // aligned to alignment.
  uintptr_t end = start_ + free_size_;
  uintptr_t result = end - size;
  if (result > end) return {nullptr, 0};  // Underflow.
  result &= ~(alignment - 1);
  if (result < start_) return {nullptr, 0};  // Out of memory in region.
  size_t actual_size = end - result;

  TC_ASSERT_EQ(result % GetPageSize(), 0);
  void* result_ptr = reinterpret_cast<void*>(result);
  if (mprotect(result_ptr, actual_size, PROT_READ | PROT_WRITE) != 0) {
    TC_LOG("mprotect(%p, %v) failed (%s)", result_ptr, actual_size,
           strerror(errno));
    return {nullptr, 0};
  }
  // For cold regions (kInfrequentAccess) and sampled regions
  // (kInfrequentAllocation), we want as granular of access telemetry as
  // possible; this hint means we can get 4kiB granularity instead of 2MiB.
  if (hint_ == AddressRegionFactory::UsageHint::kInfrequentAccess ||
      hint_ == AddressRegionFactory::UsageHint::kInfrequentAllocation) {
    // This is only advisory, so ignore the error.
    ErrnoRestorer errno_restorer;
    (void)madvise(result_ptr, actual_size, MADV_NOHUGEPAGE);
  }
  free_size_ -= actual_size;
  return {result_ptr, actual_size};
}

template <typename Topology>
AddressRegion* SystemAllocator<Topology>::MmapRegionFactory::Create(
    void* start, size_t size, UsageHint hint) {
  void* region_space = MallocInternal(sizeof(MmapRegion));
  if (!region_space) return nullptr;
  bytes_reserved_.fetch_add(size, std::memory_order_relaxed);
  return new (region_space)
      MmapRegion(reinterpret_cast<uintptr_t>(start), size, hint);
}

template <typename Topology>
size_t SystemAllocator<Topology>::MmapRegionFactory::GetStats(
    absl::Span<char> buffer) {
  Printer printer(buffer.data(), buffer.size());
  size_t allocated = bytes_reserved_.load(std::memory_order_relaxed);
  constexpr double MiB = 1048576.0;
  printer.printf("MmapSysAllocator: %zu bytes (%.1f MiB) reserved\n", allocated,
                 allocated / MiB);

  return printer.SpaceRequired();
}

template <typename Topology>
size_t SystemAllocator<Topology>::MmapRegionFactory::GetStatsInPbtxt(
    absl::Span<char> buffer) {
  Printer printer(buffer.data(), buffer.size());
  size_t allocated = bytes_reserved_.load(std::memory_order_relaxed);
  printer.printf(" mmap_sys_allocator: %lld\n", allocated);

  return printer.SpaceRequired();
}

template <typename Topology>
std::pair<void*, size_t> SystemAllocator<Topology>::AllocateFromRegion(
    size_t request_size, size_t alignment, const MemoryTag tag) {
  using system_allocator_internal::RoundUp;

  constexpr uintptr_t kTagFree = uintptr_t{1} << kTagShift;

  // We do not support size or alignment larger than kTagFree.
  // TODO(b/141325493): Handle these large allocations.
  if (request_size > kTagFree || alignment > kTagFree) return {nullptr, 0};

  // If we are dealing with large sizes, or large alignments we do not
  // want to throw away the existing reserved region, so instead we
  // return a new region specifically targeted for the request.
  if (request_size > kMinMmapAlloc || alignment > kMinMmapAlloc) {
    // Align on kHugePageSize boundaries to reduce external fragmentation for
    // future allocations.
    size_t size = RoundUp(request_size, kHugePageSize);
    if (size < request_size) return {nullptr, 0};
    alignment = std::max(alignment, kHugePageSize);
    void* ptr = MmapAlignedLocked(size, alignment, tag);
    if (!ptr) return {nullptr, 0};

    const auto region_type = TagToHint(tag);
    AddressRegion* region = region_factory_->Create(ptr, size, region_type);
    if (!region) {
      munmap(ptr, size);
      return {nullptr, 0};
    }
    std::pair<void*, size_t> result = region->Alloc(size, alignment);
    if (result.first != nullptr) {
      TC_ASSERT_EQ(result.first, ptr);
      TC_ASSERT_EQ(result.second, size);
    } else {
      TC_ASSERT_EQ(result.second, 0);
    }
    return result;
  }

  AddressRegion*& region =
      *[&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_) GOOGLE_MALLOC_SECTION {
        switch (tag) {
          case MemoryTag::kNormal:
            return &normal_region_[0];
          case MemoryTag::kNormalP1:
            return &normal_region_[1];
          case MemoryTag::kSampled:
            return &sampled_region_;
          case MemoryTag::kSelSan:
            return &selsan_region_;
          case MemoryTag::kCold:
            return &cold_region_;
          case MemoryTag::kMetadata:
            return &metadata_region_;
        }

        ASSUME(false);
        __builtin_unreachable();
      }();
  // For sizes that fit in our reserved range first of all check if we can
  // satisfy the request from what we have available.
  if (region) {
    std::pair<void*, size_t> result = region->Alloc(request_size, alignment);
    if (result.first) return result;
  }

  // Allocation failed so we need to reserve more memory.
  // Reserve new region and try allocation again.
  void* ptr = MmapAlignedLocked(kMinMmapAlloc, kMinMmapAlloc, tag);
  if (!ptr) return {nullptr, 0};

  const auto region_type = TagToHint(tag);
  region = region_factory_->Create(ptr, kMinMmapAlloc, region_type);
  if (!region) {
    munmap(ptr, kMinMmapAlloc);
    return {nullptr, 0};
  }
  return region->Alloc(request_size, alignment);
}

template <typename Topology>
void* SystemAllocator<Topology>::MmapAligned(size_t size, size_t alignment,
                                             const MemoryTag tag) {
  AllocationGuardSpinLockHolder l(&spinlock_);
  return MmapAlignedLocked(size, alignment, tag);
}

template <typename Topology>
void* SystemAllocator<Topology>::MmapAlignedLocked(size_t size,
                                                   size_t alignment,
                                                   const MemoryTag tag) {
  using system_allocator_internal::MapFixedNoReplaceFlagAvailable;

  TC_ASSERT_LE(size, kTagMask);
  TC_ASSERT_LE(alignment, kTagMask);

  std::optional<int> numa_partition;
  uintptr_t& next_addr =
      *[&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(spinlock_) GOOGLE_MALLOC_SECTION {
        switch (tag) {
          case MemoryTag::kSampled:
            return &next_sampled_addr_;
          case MemoryTag::kSelSan:
            return &next_selsan_addr_;
          case MemoryTag::kNormalP0:
            numa_partition = 0;
            return &next_normal_addr_[0];
          case MemoryTag::kNormalP1:
            numa_partition = 1;
            return &next_normal_addr_[1];
          case MemoryTag::kCold:
            return &next_cold_addr_;
          case MemoryTag::kMetadata:
            return &next_metadata_addr_;
        }

        ASSUME(false);
        __builtin_unreachable();
      }();

  bool first = !next_addr;
  if (!next_addr || next_addr & (alignment - 1) ||
      GetMemoryTag(reinterpret_cast<void*>(next_addr)) != tag ||
      GetMemoryTag(reinterpret_cast<void*>(next_addr + size - 1)) != tag) {
    next_addr = RandomMmapHint(size, alignment, tag);
  }
  const int map_fixed_noreplace_flag = MapFixedNoReplaceFlagAvailable();
  void* hint;
  // Avoid clobbering errno, especially if an initial mmap fails but a
  // subsequent one succeeds.  If we fail to allocate memory, MallocOomPolicy
  // will set errno for us.
  ErrnoRestorer errno_restorer;
  for (int i = 0; i < 1000; ++i) {
    hint = reinterpret_cast<void*>(next_addr);
    TC_ASSERT_EQ(GetMemoryTag(hint), tag);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | map_fixed_noreplace_flag;

    void* result = mmap(hint, size, PROT_NONE, flags, -1, 0);
    if (result == hint) {
      if (numa_partition.has_value()) {
        BindMemory(result, size, *numa_partition);
      }
      // Attempt to keep the next mmap contiguous in the common case.
      next_addr += size;
      TC_CHECK(kAddressBits == std::numeric_limits<uintptr_t>::digits ||
               next_addr <= uintptr_t{1} << kAddressBits);

      TC_ASSERT_EQ(reinterpret_cast<uintptr_t>(result) & (alignment - 1), 0);
      // Give the mmaped region a name based on its tag.
#ifdef __linux__
      // Make a best-effort attempt to name the allocated region based on its
      // tag.
      //
      // The call to prctl() may fail if the kernel was not configured with the
      // CONFIG_ANON_VMA_NAME kernel option.  This is OK since the call is
      // primarily a debugging aid.
      char name[256];
      absl::SNPrintF(name, sizeof(name), "tcmalloc_region_%s",
                     MemoryTagToLabel(tag));
      prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, result, size, name);
#endif  // __linux__
      return result;
    }
    if (map_fixed_noreplace_flag) {
      // If MAP_FIXED_NOREPLACE was correctly detected, we should either get
      // result == hint or MAP_FAILED.  Any other value indicates incorrect
      // detection.
      TC_CHECK_EQ(result, MAP_FAILED);
    } else {
      if (result == MAP_FAILED) {
        TC_LOG("mmap(%p, %v) reservation failed (%s)", hint, size,
               strerror(errno));
        return nullptr;
      }
      if (int err = munmap(result, size)) {
        TC_LOG("munmap(%p, %v) failed (%s)", result, size, strerror(errno));
        TC_ASSERT_EQ(err, 0);
      }
    }
    next_addr = RandomMmapHint(size, alignment, tag);
  }

  TC_LOG(
      "MmapAligned() failed - unable to allocate with tag (hint=%p, size=%v, "
      "alignment=%v) - is something limiting address placement?",
      hint, size, alignment);
  if (first) {
    TC_LOG(
        "Note: the allocation may have failed because TCMalloc assumes a "
        "%u-bit virtual address space size; you may need to rebuild TCMalloc "
        "with TCMALLOC_ADDRESS_BITS defined to your system's virtual address "
        "space size",
        kAddressBits);
  }
  return nullptr;
}

template <typename Topology>
bool SystemAllocator<Topology>::Release(void* start, size_t length) {
  bool result = false;

#if defined(MADV_DONTNEED) || defined(MADV_REMOVE)
  ErrnoRestorer errno_restorer;
  const size_t pagemask = GetPageSize() - 1;

  size_t new_start = reinterpret_cast<size_t>(start);
  size_t end = new_start + length;
  size_t new_end = end;

  // Round up the starting address and round down the ending address
  // to be page aligned:
  new_start = (new_start + GetPageSize() - 1) & ~pagemask;
  new_end = new_end & ~pagemask;

  TC_ASSERT_EQ(new_start & pagemask, 0);
  TC_ASSERT_EQ(new_end & pagemask, 0);
  TC_ASSERT_GE(new_start, reinterpret_cast<size_t>(start));
  TC_ASSERT_LE(new_end, end);

  if (new_end > new_start) {
    void* new_ptr = reinterpret_cast<void*>(new_start);
    size_t new_length = new_end - new_start;

    if (!ReleasePages(new_ptr, new_length)) {
      // Try unlocking.
      int ret;
      do {
        ret = munlock(reinterpret_cast<char*>(new_start), new_end - new_start);
      } while (ret == -1 && errno == EAGAIN);

      if (ret != 0 || !ReleasePages(new_ptr, new_length)) {
        // If we fail to munlock *or* fail our second attempt at madvise,
        // increment our failure count.
        release_errors_.fetch_add(1, std::memory_order_relaxed);
      } else {
        result = true;
      }
    } else {
      result = true;
    }
  }
#endif

  return result;
}

// Bind the memory region spanning `size` bytes starting from `base` to NUMA
// nodes assigned to `partition`. Returns zero upon success, or a standard
// error code upon failure.
template <typename Topology>
void SystemAllocator<Topology>::BindMemory(void* const base, const size_t size,
                                           const size_t partition) const {
  // If NUMA awareness is unavailable or disabled, or the user requested that
  // we don't bind memory then do nothing.
  const NumaBindMode bind_mode = topology_.bind_mode();
  if (!topology_.numa_aware() || bind_mode == NumaBindMode::kNone) {
    return;
  }

  const uint64_t nodemask = topology_.GetPartitionNodes(partition);
  int err =
      syscall(__NR_mbind, base, size, MPOL_BIND | MPOL_F_STATIC_NODES,
              &nodemask, sizeof(nodemask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
  if (err == 0) {
    return;
  }

  if (bind_mode == NumaBindMode::kAdvisory) {
    TC_LOG("Warning: Unable to mbind memory (errno=%d, base=%p, nodemask=%v)",
           errno, base, nodemask);
    return;
  }

  TC_ASSERT_EQ(bind_mode, NumaBindMode::kStrict);
  TC_BUG("Unable to mbind memory (errno=%d, base=%p, nodemask=%v)", errno, base,
         nodemask);
}

template <typename Topology>
AddressRegionFactory::UsageHint SystemAllocator<Topology>::TagToHint(
    MemoryTag tag) const {
  using UsageHint = AddressRegionFactory::UsageHint;
  switch (tag) {
    case MemoryTag::kNormal:
      if (topology_.numa_aware()) {
        return UsageHint::kNormalNumaAwareS0;
      }
      return UsageHint::kNormal;
    case MemoryTag::kNormalP1:
      if (topology_.numa_aware()) {
        return UsageHint::kNormalNumaAwareS1;
      }
      return UsageHint::kNormal;
    case MemoryTag::kSelSan:
      if (topology_.numa_aware()) {
        return UsageHint::kNormalNumaAwareS0;
      }
      return UsageHint::kNormal;
    case MemoryTag::kSampled:
      return UsageHint::kInfrequentAllocation;
    case MemoryTag::kCold:
      return UsageHint::kInfrequentAccess;
    case MemoryTag::kMetadata:
      return UsageHint::kMetadata;
  }

  ASSUME(false);
  __builtin_unreachable();
}

template <typename Topology>
uintptr_t SystemAllocator<Topology>::RandomMmapHint(size_t size,
                                                    size_t alignment,
                                                    const MemoryTag tag) {
  // Rely on kernel's mmap randomization to seed our RNG.
  absl::base_internal::LowLevelCallOnce(
      &rnd_flag_, [&]() GOOGLE_MALLOC_SECTION {
        const size_t page_size = GetPageSize();
        void* seed = mmap(nullptr, page_size, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (seed == MAP_FAILED) {
          TC_BUG("Initial mmap() reservation failed (errno=%v, size=%v)", errno,
                 page_size);
        }
        munmap(seed, page_size);
        spinlock_.AssertHeld();
        rnd_ = reinterpret_cast<uintptr_t>(seed);
      });

#if !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER)
  // We don't use the following bits:
  //
  //  *  The top bits that are forbidden for use by the hardware (or are
  //     required to be set to the same value as the next bit, which we also
  //     don't use).
  //
  //  *  Below that, the top highest the hardware allows us to use, since it is
  //     reserved for kernel space addresses.
  constexpr uintptr_t kAddrMask = (uintptr_t{1} << (kAddressBits - 1)) - 1;
#else
  // MSan and TSan use up all of the lower address space, so we allow use of
  // mid-upper address space when they're active.  This only matters for
  // TCMalloc-internal tests, since sanitizers install their own malloc/free.
  constexpr uintptr_t kAddrMask = (uintptr_t{0xF} << (kAddressBits - 5)) - 1;
#endif

  // Ensure alignment >= size so we're guaranteed the full mapping has the same
  // tag.
  alignment = absl::bit_ceil(std::max(alignment, size));

  rnd_ = ExponentialBiased::NextRandom(rnd_);
  uintptr_t addr = rnd_ & kAddrMask & ~(alignment - 1) & ~kTagMask;
  addr |= static_cast<uintptr_t>(tag) << kTagShift;
  TC_ASSERT_EQ(GetMemoryTag(reinterpret_cast<const void*>(addr)), tag);
  return addr;
}

template <typename Topology>
inline bool SystemAllocator<Topology>::ReleasePages(void* start,
                                                    size_t length) const {
  int ret;
  // Note -- ignoring most return codes, because if this fails it
  // doesn't matter...
  // Moreover, MADV_REMOVE *will* fail (with EINVAL) on private memory,
  // but that's harmless.
#ifdef MADV_REMOVE
  // MADV_REMOVE deletes any backing storage for tmpfs or anonymous shared
  // memory.
  do {
    ret = madvise(start, length, MADV_REMOVE);
  } while (ret == -1 && errno == EAGAIN);

  if (ret == 0) {
    return true;
  }
#endif

#ifdef MADV_FREE
  const bool do_madvfree = [&]() {
    switch (madvise_preference()) {
      case MadvisePreference::kFreeAndDontNeed:
      case MadvisePreference::kFreeOnly:
        return true;
      case MadvisePreference::kDontNeed:
      case MadvisePreference::kNever:
        return false;
    }

    ABSL_UNREACHABLE();
  }();

  if (do_madvfree) {
    do {
      ret = madvise(start, length, MADV_FREE);
    } while (ret == -1 && errno == EAGAIN);
  }
#endif
#ifdef MADV_DONTNEED
  const bool do_madvdontneed = [&]() {
    switch (madvise_preference()) {
      case MadvisePreference::kDontNeed:
      case MadvisePreference::kFreeAndDontNeed:
        return true;
      case MadvisePreference::kFreeOnly:
      case MadvisePreference::kNever:
        return false;
    }

    ABSL_UNREACHABLE();
  }();

  // MADV_DONTNEED drops page table info and any anonymous pages.
  if (do_madvdontneed) {
    do {
      ret = madvise(start, length, MADV_DONTNEED);
    } while (ret == -1 && errno == EAGAIN);
  }
#endif
  if (ret == 0) {
    return true;
  }

  return false;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_SYSTEM_ALLOC_H_
