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

#include "tcmalloc/guarded_page_allocator.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/base/optimization.h"
#include "absl/debugging/stacktrace.h"
#include "absl/numeric/bits.h"
#include "tcmalloc/common.h"
#include "tcmalloc/guarded_allocations.h"
#include "tcmalloc/internal/allocation_guard.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/memory_tag.h"
#include "tcmalloc/internal/page_size.h"
#include "tcmalloc/malloc_extension.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/system-alloc.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

void GuardedPageAllocator::Init(size_t max_allocated_pages,
                                size_t total_pages) {
  TC_CHECK_GT(max_allocated_pages, 0);
  TC_CHECK_LE(max_allocated_pages, total_pages);
  TC_CHECK_LE(total_pages, kGpaMaxPages);
  max_allocated_pages_ = max_allocated_pages;
  total_pages_ = total_pages;

  // If the system page size is larger than kPageSize, we need to use the
  // system page size for this allocator since mprotect operates on full pages
  // only.  This case happens on PPC.
  page_size_ = std::max(kPageSize, static_cast<size_t>(GetPageSize()));
  TC_ASSERT_EQ(page_size_ % kPageSize, 0);

  rand_.Reset(static_cast<uint64_t>(absl::base_internal::CycleClock::Now()) +
              reinterpret_cast<uintptr_t>(this));
  MapPages();
}

void GuardedPageAllocator::Destroy() {
  AllocationGuardSpinLockHolder h(&guarded_page_lock_);
  if (initialized_) {
    size_t len = pages_end_addr_ - pages_base_addr_;
    int err = munmap(reinterpret_cast<void*>(pages_base_addr_), len);
    TC_ASSERT_NE(err, -1);
    (void)err;
    initialized_ = false;
  }
}

// Reset is used by tests to ensure that subsequent allocations will be sampled.
void GuardedPageAllocator::Reset() {
  // Reset sampled/guarded counters so that that we don't skip guarded sampling
  // for a prolonged time due to accumulated stats.
  tc_globals.total_sampled_count_.Add(-tc_globals.total_sampled_count_.value());
  successful_allocations_.Add(-successful_allocations_.value());
  // Allow allocations that are not currently covered by an existing allocation.
  // Fully resetting the stack trace filter is a bad idea, because the pool may
  // not be empty: a later deallocation would try to remove a non-existent entry
  // from the filter.
  stacktrace_filter_.DecayAll();
}

void GuardedPageAllocator::AcquireInternalLocks() {
  guarded_page_lock_.Lock();
}

void GuardedPageAllocator::ReleaseInternalLocks() {
  guarded_page_lock_.Unlock();
}

GuardedAllocWithStatus GuardedPageAllocator::TrySample(
    size_t size, size_t alignment, Length num_pages,
    const StackTrace& stack_trace) {
  if (num_pages != Length(1)) {
    skipped_allocations_toolarge_.Add(1);
    return {nullptr, Profile::Sample::GuardedStatus::LargerThanOnePage};
  }

  const int64_t guarded_sampling_interval =
      tcmalloc::tcmalloc_internal::Parameters::guarded_sampling_interval();
  // Guarded sampling is disabled if guarded_sampling_interval is negative.
  if (guarded_sampling_interval < 0) {
    return {nullptr, Profile::Sample::GuardedStatus::Disabled};
  }
  // Never filter if guarded_sampling_interval == 0, or no samples yet.
  const size_t num_guarded = successful_allocations();
  if (guarded_sampling_interval > 0 && num_guarded > 0) {
    // The guarded page allocator should not exceed the desired sampling rate.
    // To do so, we need to filter allocations while this condition holds:
    //
    //  num_guarded * guarded_interval > num_sampled * profile_interval
    //
    // I.e. if the next guarded allocation should occur at total bytes allocated
    // later than the next sampled allocation. Recall that sampled allocations
    // are a superset of guarded sampled allocations, and num_sampled is always
    // incremented _after_ num_guarded.
    //
    // Assuming that the number of total samples (num_sampled) must always be
    // larger or equal to the guarded samples (num_guarded), and allow for a
    // target num_sampled:num_guarded ratio with up to 1 decimal place, the
    // above can be rewritten as:
    //
    //  guarded_interval * 10 >
    //     ((num_sampled * 10) / num_guarded) * profile_interval
    //
    // This avoids possible overflow if num_sampled or num_guarded grows larger,
    // when individually multiplied by the intervals. We can avoid floating
    // point math as well.
    const int64_t profile_sampling_interval =
        tcmalloc::tcmalloc_internal::Parameters::profile_sampling_interval();
    const int64_t num_sampled = tc_globals.total_sampled_count_.value();
    const int64_t ratio = (num_sampled * 10) / num_guarded;
    if (guarded_sampling_interval * 10 > ratio * profile_sampling_interval) {
      return {nullptr, Profile::Sample::GuardedStatus::RateLimited};
    }

    if (stacktrace_filter_.Contains({stack_trace.stack, stack_trace.depth})) {
      // The probability that we skip a currently covered allocation scales
      // proportional to pool utilization, with pool utilization of 50% or more
      // resulting in always filtering currently covered allocations.
      const size_t usage_pct = (allocated_pages() * 100) / max_allocated_pages_;
      if (rand_.Next() % 50 <= usage_pct) {
        // Decay even if the current allocation is filtered, so that we keep
        // sampling even if we only see the same allocations over and over.
        stacktrace_filter_.Decay();
        skipped_allocations_filtered_.Add(1);
        return {nullptr, Profile::Sample::GuardedStatus::Filtered};
      }
    }
  }
  // The num_pages == 1 constraint ensures that size <= kPageSize.  And
  // since alignments above kPageSize cause size_class == 0, we're also
  // guaranteed alignment <= kPageSize
  //
  // In all cases kPageSize <= GPA::page_size_, so Allocate's preconditions
  // are met.
  return Allocate(size, alignment, stack_trace);
}

GuardedAllocWithStatus GuardedPageAllocator::Allocate(
    size_t size, size_t alignment, const StackTrace& stack_trace) {
  if (size == 0) {
    return {nullptr, Profile::Sample::GuardedStatus::TooSmall};
  }
  const ssize_t free_slot = ReserveFreeSlot();
  if (free_slot == -1) {
    // All slots are reserved.
    return {nullptr, Profile::Sample::GuardedStatus::NoAvailableSlots};
  }

  TC_ASSERT_LE(size, page_size_);
  TC_ASSERT_LE(alignment, page_size_);
  TC_ASSERT(alignment == 0 || absl::has_single_bit(alignment));
  void* result = reinterpret_cast<void*>(SlotToAddr(free_slot));
  if (mprotect(result, page_size_, PROT_READ | PROT_WRITE) == -1) {
    TC_ASSERT(false, "mprotect(.., PROT_READ|PROT_WRITE) failed");
    AllocationGuardSpinLockHolder h(&guarded_page_lock_);
    failed_allocations_.LossyAdd(1);
    successful_allocations_.LossyAdd(-1);
    FreeSlot(free_slot);
    return {nullptr, Profile::Sample::GuardedStatus::MProtectFailed};
  }

  // Place some allocations at end of page for better overflow detection.
  MaybeRightAlign(free_slot, size, alignment, &result);

  // Record stack trace.
  SlotMetadata& d = data_[free_slot];
  // Count the number of pages that have been used at least once.
  if (ABSL_PREDICT_FALSE(d.allocation_start == 0)) {
    pages_touched_.Add(1);
  }

  static_assert(sizeof(d.alloc_trace.stack) == sizeof(stack_trace.stack));
  memcpy(d.alloc_trace.stack, stack_trace.stack,
         stack_trace.depth * sizeof(stack_trace.stack[0]));
  d.alloc_trace.depth = stack_trace.depth;
  d.alloc_trace.thread_id = absl::base_internal::GetTID();
  d.dealloc_trace.depth = 0;
  d.requested_size = size;
  d.allocation_start = reinterpret_cast<uintptr_t>(result);
  d.dealloc_count.store(0, std::memory_order_relaxed);
  TC_ASSERT(!d.write_overflow_detected);
  TC_ASSERT(!alignment || d.allocation_start % alignment == 0);

  stacktrace_filter_.Add({stack_trace.stack, stack_trace.depth}, 1);
  return {result, Profile::Sample::GuardedStatus::Guarded};
}

// To trigger SEGV handler.
static ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NORETURN void ForceTouchPage(
    void* ptr) {
  // Spin, in case this thread is waiting for concurrent mprotect() to finish.
  for (;;) {
    *reinterpret_cast<volatile char*>(ptr) = 'X';
  }
}

void GuardedPageAllocator::Deallocate(void* ptr) {
  TC_ASSERT(PointerIsMine(ptr));
  const uintptr_t page_addr = GetPageAddr(reinterpret_cast<uintptr_t>(ptr));
  const size_t slot = AddrToSlot(page_addr);
  SlotMetadata& d = data_[slot];

  // On double-free, do not overwrite the original deallocation metadata, so
  // that the report produced shows the original deallocation stack trace.
  if (d.dealloc_count.fetch_add(1, std::memory_order_relaxed) != 0) {
    ForceTouchPage(ptr);
  }

  // Record stack trace. Unwinding the stack is expensive, and holding the
  // guarded_page_lock_ should be avoided here.
  d.dealloc_trace.depth =
      absl::GetStackTrace(d.dealloc_trace.stack, kMaxStackDepth,
                          /*skip_count=*/2);
  d.dealloc_trace.thread_id = absl::base_internal::GetTID();

  // Remove allocation (based on allocation stack trace) from filter.
  stacktrace_filter_.Add({d.alloc_trace.stack, d.alloc_trace.depth}, -1);

  // Needs to be done before mprotect() because it accesses the object page to
  // check canary bytes.
  if (WriteOverflowOccurred(slot)) {
    d.write_overflow_detected = true;
  }

  // Calling mprotect() should also be done outside the guarded_page_lock_
  // critical section, since mprotect() can have relatively large latency.
  TC_CHECK_EQ(
      0, mprotect(reinterpret_cast<void*>(page_addr), page_size_, PROT_NONE));

  if (d.write_overflow_detected) {
    ForceTouchPage(ptr);
  }

  AllocationGuardSpinLockHolder h(&guarded_page_lock_);
  FreeSlot(slot);
}

size_t GuardedPageAllocator::GetRequestedSize(const void* ptr) const {
  TC_ASSERT(PointerIsMine(ptr));
  size_t slot = AddrToSlot(GetPageAddr(reinterpret_cast<uintptr_t>(ptr)));
  return data_[slot].requested_size;
}

std::pair<off_t, size_t> GuardedPageAllocator::GetAllocationOffsetAndSize(
    const void* ptr) const {
  TC_ASSERT(PointerIsMine(ptr));
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  const size_t slot = GetNearestSlot(addr);
  return {addr - data_[slot].allocation_start, data_[slot].requested_size};
}

GuardedAllocationsErrorType GuardedPageAllocator::GetStackTraces(
    const void* ptr, GuardedAllocationsStackTrace** alloc_trace,
    GuardedAllocationsStackTrace** dealloc_trace) const {
  TC_ASSERT(PointerIsMine(ptr));
  const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  size_t slot = GetNearestSlot(addr);
  *alloc_trace = &data_[slot].alloc_trace;
  *dealloc_trace = &data_[slot].dealloc_trace;
  return GetErrorType(addr, data_[slot]);
}

// We take guarded samples during periodic profiling samples.  Computes the
// mean number of profiled samples made for every guarded sample.
static int GetChainedInterval() {
  auto guarded_interval = Parameters::guarded_sampling_interval();
  auto sample_interval = Parameters::profile_sampling_interval();
  if (guarded_interval < 0 || sample_interval <= 0) {
    return guarded_interval;
  } else {
    return std::ceil(static_cast<double>(guarded_interval) /
                     static_cast<double>(sample_interval));
  }
}

void GuardedPageAllocator::Print(Printer& out) {
  out.printf(
      "\n"
      "------------------------------------------------\n"
      "GWP-ASan Status\n"
      "------------------------------------------------\n"
      "Successful Allocations: %zu\n"
      "Failed Allocations: %zu\n"
      "Skipped Allocations (No Slots): %zu\n"
      "Skipped Allocations (Filtered): %zu\n"
      "Skipped Allocations (Too Large): %zu\n"
      "Currently Allocated: %zu / %zu\n"
      "Allocated High-Watermark: %zu / %zu\n"
      "Object Pages Touched: %zu / %zu\n"
      "Currently Quarantined: %zu\n"
      "PARAMETER tcmalloc_guarded_sample_parameter %d\n",
      // Successful Allocations
      successful_allocations_.value(),
      // Failed Allocations
      failed_allocations_.value(),
      // Skipped Allocations (No Slots)
      skipped_allocations_noslots_.value(),
      // Skipped Allocations (Filtered)
      skipped_allocations_filtered_.value(),
      // Skipped Allocations (Too Large)
      skipped_allocations_toolarge_.value(),
      // Currently Allocated
      allocated_pages(), max_allocated_pages_,
      // Allocated High-Watermark
      high_allocated_pages_.load(std::memory_order_relaxed),
      max_allocated_pages_,
      // Object Pages Touched
      pages_touched_.value(), total_pages_,
      // Currently Quarantined
      total_pages_ - allocated_pages(),
      // PARAMETER
      GetChainedInterval());
}

void GuardedPageAllocator::PrintInPbtxt(PbtxtRegion& gwp_asan) {
  gwp_asan.PrintI64("successful_allocations", successful_allocations_.value());
  gwp_asan.PrintI64("failed_allocations", failed_allocations_.value());
  gwp_asan.PrintI64("skipped_allocations_noslots",
                    skipped_allocations_noslots_.value());
  gwp_asan.PrintI64("skipped_allocations_filtered",
                    skipped_allocations_filtered_.value());
  gwp_asan.PrintI64("skipped_allocations_toolarge",
                    skipped_allocations_toolarge_.value());
  gwp_asan.PrintI64("allocated_pages", allocated_pages());
  gwp_asan.PrintI64("quarantine_pages", total_pages_ - allocated_pages());
  gwp_asan.PrintI64("high_allocated_pages",
                    high_allocated_pages_.load(std::memory_order_relaxed));
  gwp_asan.PrintI64("max_allocated_pages", max_allocated_pages_);
  gwp_asan.PrintI64("pages_touched", pages_touched_.value());
  gwp_asan.PrintI64("total_pages", total_pages_);
  gwp_asan.PrintI64("tcmalloc_guarded_sample_parameter", GetChainedInterval());
}

// Maps 2 * total_pages_ + 1 pages so that there are total_pages_ unique pages
// we can return from Allocate with guard pages before and after them.
void GuardedPageAllocator::MapPages() {
  AllocationGuardSpinLockHolder h(&guarded_page_lock_);
  TC_ASSERT(!first_page_addr_);
  TC_ASSERT_EQ(page_size_ % GetPageSize(), 0);
  size_t len = (2 * total_pages_ + 1) * page_size_;
  auto base_addr =
      reinterpret_cast<uintptr_t>(tc_globals.system_allocator().MmapAligned(
          len, page_size_, MemoryTag::kSampled));
  TC_ASSERT(base_addr);
  if (!base_addr) return;

  // Tell TCMalloc's PageMap about the memory we own.
  const PageId page = PageIdContaining(reinterpret_cast<void*>(base_addr));
  const Length page_len = BytesToLengthFloor(len);
  if (!tc_globals.pagemap().Ensure(Range(page, page_len))) {
    TC_ASSERT(false, "Failed to notify page map of page-guarded memory.");
    return;
  }

  // Allocate memory for slot metadata.
  data_ = reinterpret_cast<SlotMetadata*>(
      tc_globals.arena().Alloc(sizeof(*data_) * total_pages_));
  for (size_t i = 0; i < total_pages_; ++i) {
    new (&data_[i]) SlotMetadata;
  }

  pages_base_addr_ = base_addr;
  pages_end_addr_ = pages_base_addr_ + len;

  // Align first page to page_size_.
  first_page_addr_ = GetPageAddr(pages_base_addr_ + page_size_);

  initialized_ = true;
}

// Selects a random slot in O(1) time.
ssize_t GuardedPageAllocator::ReserveFreeSlot() {
  AllocationGuardSpinLockHolder h(&guarded_page_lock_);
  if (!initialized_ || !allow_allocations_) return -1;
  if (GetNumAvailablePages() == 0) {
    skipped_allocations_noslots_.Add(1);
    return -1;
  }
  successful_allocations_.LossyAdd(1);

  const size_t slot = GetFreeSlot();
  TC_ASSERT(!used_pages_.GetBit(slot));
  used_pages_.SetBit(slot);

  // Both writes to allocated_pages_ happen under the guarded_page_lock_, so
  // we do not have to use an atomic fetch_add(), which is more expensive due to
  // typically imposing a full memory barrier when lowered on e.g. x86. Recent
  // compiler optimizations will also turn the store(load(relaxed) + N, relaxed)
  // into a simple add instruction.
  const size_t nalloced = allocated_pages_.load(std::memory_order_relaxed) + 1;
  allocated_pages_.store(nalloced, std::memory_order_relaxed);
  if (nalloced > high_allocated_pages_.load(std::memory_order_relaxed)) {
    high_allocated_pages_.store(nalloced, std::memory_order_relaxed);
  }
  return slot;
}

size_t GuardedPageAllocator::GetFreeSlot() {
  const size_t idx = rand_.Next() % total_pages_;
  // Find the closest adjacent free slot to the random index.
  ssize_t slot = used_pages_.FindClearBackwards(idx);
  if (slot >= 0) return slot;
  slot = used_pages_.FindClear(idx);
  TC_ASSERT_LT(slot, total_pages_);
  return slot;
}

void GuardedPageAllocator::FreeSlot(size_t slot) {
  TC_ASSERT_LT(slot, total_pages_);
  TC_ASSERT(used_pages_.GetBit(slot));
  used_pages_.ClearBit(slot);
  // Cheaper decrement - see above.
  allocated_pages_.store(allocated_pages_.load(std::memory_order_relaxed) - 1,
                         std::memory_order_relaxed);
}

uintptr_t GuardedPageAllocator::GetPageAddr(uintptr_t addr) const {
  const uintptr_t addr_mask = ~(page_size_ - 1ULL);
  return addr & addr_mask;
}

uintptr_t GuardedPageAllocator::GetNearestValidPage(uintptr_t addr) const {
  if (addr < first_page_addr_) return first_page_addr_;
  const uintptr_t last_page_addr =
      first_page_addr_ + 2 * (total_pages_ - 1) * page_size_;
  if (addr > last_page_addr) return last_page_addr;
  uintptr_t offset = addr - first_page_addr_;

  // If addr is already on a valid page, just return addr.
  if ((offset / page_size_) % 2 == 0) return addr;

  // ptr points to a guard page, so get nearest valid page.
  const size_t kHalfPageSize = page_size_ / 2;
  if ((offset / kHalfPageSize) % 2 == 0) {
    return addr - kHalfPageSize;  // Round down.
  }
  return addr + kHalfPageSize;  // Round up.
}

size_t GuardedPageAllocator::GetNearestSlot(uintptr_t addr) const {
  return AddrToSlot(GetPageAddr(GetNearestValidPage(addr)));
}

bool GuardedPageAllocator::WriteOverflowOccurred(size_t slot) const {
  if (!ShouldRightAlign(slot)) return false;
  uint8_t magic = GetWriteOverflowMagic(slot);
  uintptr_t alloc_end =
      data_[slot].allocation_start + data_[slot].requested_size;
  uintptr_t page_end = SlotToAddr(slot) + page_size_;
  uintptr_t magic_end = std::min(page_end, alloc_end + kMagicSize);
  for (uintptr_t p = alloc_end; p < magic_end; ++p) {
    if (*reinterpret_cast<uint8_t*>(p) != magic) return true;
  }
  return false;
}

GuardedAllocationsErrorType GuardedPageAllocator::GetErrorType(
    uintptr_t addr, const SlotMetadata& d) const {
  if (!d.allocation_start) return GuardedAllocationsErrorType::kUnknown;
  if (d.dealloc_count.load(std::memory_order_relaxed) >= 2)
    return GuardedAllocationsErrorType::kDoubleFree;
  if (d.write_overflow_detected)
    return GuardedAllocationsErrorType::kBufferOverflowOnDealloc;
  if (d.dealloc_trace.depth > 0) {
    return GuardedAllocationsErrorType::kUseAfterFree;
  }
  if (addr < d.allocation_start) {
    return GuardedAllocationsErrorType::kBufferUnderflow;
  }
  if (addr >= d.allocation_start + d.requested_size) {
    return GuardedAllocationsErrorType::kBufferOverflow;
  }
  return GuardedAllocationsErrorType::kUnknown;
}

uintptr_t GuardedPageAllocator::SlotToAddr(size_t slot) const {
  TC_ASSERT_LT(slot, total_pages_);
  return first_page_addr_ + 2 * slot * page_size_;
}

size_t GuardedPageAllocator::AddrToSlot(uintptr_t addr) const {
  uintptr_t offset = addr - first_page_addr_;
  TC_ASSERT_EQ(offset % page_size_, 0);
  TC_ASSERT_EQ((offset / page_size_) % 2, 0);
  int slot = offset / page_size_ / 2;
  TC_ASSERT(slot >= 0 && slot < total_pages_);
  return slot;
}

void GuardedPageAllocator::MaybeRightAlign(size_t slot, size_t size,
                                           size_t alignment, void** ptr) {
  if (!ShouldRightAlign(slot)) return;
  uintptr_t adjusted_ptr =
      reinterpret_cast<uintptr_t>(*ptr) + page_size_ - size;

  // If alignment == 0, the necessary alignment is never larger than the size
  // rounded up to the next power of 2.  We use this fact to minimize alignment
  // padding between the end of small allocations and their guard pages.
  //
  // For allocations larger than the greater of kAlignment and
  // __STDCPP_DEFAULT_NEW_ALIGNMENT__, we're safe aligning to that value.
  size_t default_alignment =
      std::min(absl::bit_ceil(size),
               std::max(static_cast<size_t>(kAlignment),
                        static_cast<size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));

  // Ensure valid alignment.
  alignment = std::max(alignment, default_alignment);
  uintptr_t alignment_padding = adjusted_ptr & (alignment - 1);
  adjusted_ptr -= alignment_padding;

  // Write magic bytes in alignment padding to detect small overflow writes.
  size_t magic_size = std::min(alignment_padding, kMagicSize);
  memset(reinterpret_cast<void*>(adjusted_ptr + size),
         GetWriteOverflowMagic(slot), magic_size);
  *ptr = reinterpret_cast<void*>(adjusted_ptr);
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
