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
// A Span is a contiguous run of pages.

#ifndef TCMALLOC_SPAN_H_
#define TCMALLOC_SPAN_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/types/span.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/prefetch.h"
#include "tcmalloc/internal/range_tracker.h"
#include "tcmalloc/internal/sampled_allocation.h"
#include "tcmalloc/pages.h"
#include "tcmalloc/sizemap.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Denominator for bitmap scaling factor. The idea is that instead of dividing
// by N we multiply by M = kBitmapScalingDenominator / N and round the resulting
// value.
inline constexpr size_t kBitmapScalingDenominator = 1 << 30;

enum AccessDensityPrediction {
  // Predict that the span would be sparsely-accessed.
  kSparse = 0,
  // Predict that the span would be densely-accessed.
  kDense = 1,
  kPredictionCounts
};

struct SpanAllocInfo {
  size_t objects_per_span;
  AccessDensityPrediction density;
};

// Information kept for a span (a contiguous run of pages).
//
// Spans can be in different states. The current state determines set of methods
// that can be called on the span (and the active member in the union below).
// States are:
//  - SMALL_OBJECT: the span holds multiple small objects.
//    The span is owned by CentralFreeList and is generally on
//    CentralFreeList::nonempty_ list (unless has no free objects).
//  - LARGE_OBJECT: the span holds a single large object.
//    The span can be considered to be owner by user until the object is freed.
//  - SAMPLED: the span holds a single sampled object.
//    The span can be considered to be owner by user until the object is freed.
//    sampled_ == 1.
class Span;
typedef TList<Span> SpanList;

class Span final : public SpanList::Elem {
 public:
  explicit Span(Range r)
      : embed_count_(0),
        freelist_(0),
        allocated_(0),
        cache_size_(0),
        nonempty_index_(0),
        is_donated_(0),
        first_page_(r.p.index()),
        reserved_(0),
        is_large_span_(0),
        sampled_(0),
        large_or_sampled_state_{0, nullptr} {
    TC_ASSERT_GT(r.p, PageId{0});
    TC_CHECK_LT(r.p.index(), static_cast<uint64_t>(1) << kMaxPageIdBits);
    set_num_pages(r.n);
  }

  Span(const Span&) = delete;
  Span& operator=(const Span&) = delete;

  // Allocator/deallocator for spans. Note that these functions are defined
  // in static_vars.h, which is weird: see there for why.
  static Span* New(Range r)
#ifndef TCMALLOC_INTERNAL_LEGACY_LOCKING
      ABSL_LOCKS_EXCLUDED(pageheap_lock)
#endif
          ;
  static void Delete(Span* span);

  static void operator delete(void*) = delete;

  // ---------------------------------------------------------------------------
  // Support for sampled allocations.
  // There is one-to-one correspondence between a sampled allocation and a span.
  // ---------------------------------------------------------------------------

  // Mark this span in the "SAMPLED" state. It will store the corresponding
  // sampled allocation and update some global counters on the total size of
  // sampled allocations.
  void Sample(SampledAllocation* sampled_allocation);

  // Unmark this span from its "SAMPLED" state. It will return the sampled
  // allocation previously passed to Span::Sample() or nullptr if this is a
  // non-sampling span. It will also update the global counters on the total
  // size of sampled allocations.
  // REQUIRES: this is a SAMPLED span.
  [[nodiscard]] SampledAllocation* Unsample();

  // Returns the sampled allocation of the span.
  // pageheap_lock is not required, but caller either needs to hold the lock or
  // ensure by some other means that the sampling state can't be changed
  // concurrently.
  // REQUIRES: this is a SAMPLED span.
  const SampledAllocation& sampled_allocation() const;

  // Is it a sampling span?
  // For debug checks. pageheap_lock is not required, but caller needs to ensure
  // that sampling state can't be changed concurrently.
  bool sampled() const;

  bool donated() const { return is_donated_; }
  void set_donated(bool value) { is_donated_ = value; }

  // ---------------------------------------------------------------------------
  // Span memory range.
  // ---------------------------------------------------------------------------

  // Returns first page of the span.
  PageId first_page() const;

  // Returns the last page in the span.
  PageId last_page() const;

  // Sets span first page.
  void set_first_page(PageId p);

  // Returns start address of the span.
  ABSL_ATTRIBUTE_RETURNS_NONNULL void* start_address() const;

  // Returns number of pages in the span.
  Length num_pages() const;

  // Sets number of pages in the span.
  void set_num_pages(Length len);

  // Total memory bytes in the span.
  size_t bytes_in_span() const;

  // Returns internal fragmentation of the span.
  // REQUIRES: this is a SMALL_OBJECT span.
  double Fragmentation(size_t object_size) const;

  // Returns number of objects allocated in the span.
  uint16_t Allocated() const {
    return allocated_.load(std::memory_order_relaxed);
  }

  // Returns index of the non-empty list to which this span belongs to.
  uint8_t nonempty_index() const { return nonempty_index_; }
  // Records an index of the non-empty list associated with this span.
  void set_nonempty_index(uint8_t index) { nonempty_index_ = index; }

  // ---------------------------------------------------------------------------
  // Freelist management.
  // Used for spans in CentralFreelist to manage free objects.
  // These methods REQUIRE a SMALL_OBJECT span.
  // ---------------------------------------------------------------------------

  // Indicate whether the Span is empty. Size is used to determine whether
  // the span is using a compressed linked list of objects, or a bitmap
  // to hold available objects.
  bool FreelistEmpty(size_t size) const;

  // Pushes ptr onto freelist unless the freelist becomes full, in which case
  // just return false.
  //
  // If the freelist becomes full, we do not push the object onto the freelist.
  [[nodiscard]] bool FreelistPush(void* ptr, size_t size, uint32_t reciprocal,
                                  uint32_t max_cache_size);

  // Pops up to N objects from the freelist and returns them in the batch array.
  // Returns number of objects actually popped.
  [[nodiscard]] size_t FreelistPopBatch(absl::Span<void*> batch, size_t size);

  // Initialize freelist to contain all objects in the span.
  // Pops up to N objects from the freelist and returns them in the batch array.
  // Returns number of objects actually popped.
  [[nodiscard]] int BuildFreelist(size_t size, size_t count,
                                  absl::Span<void*> batch,
                                  uint32_t max_cache_size, uint64_t alloc_time);

  // Prefetch cacheline containing most important span information.
  void Prefetch();

  // IsValidSizeClass verifies size class parameters from the Span perspective.
  static bool IsValidSizeClass(size_t size, size_t pages);

  // Returns true if Span does not touch objects and the <size> is suitable
  // for cold size classes.
  static bool IsNonIntrusive(size_t size);

  // For bitmap'd spans conversion from an offset to an index is performed
  // by multiplying by the scaled reciprocal of the object size.
  static uint32_t CalcReciprocal(size_t size);

  static constexpr size_t kCacheSize = 4;
  static constexpr size_t kLargeCacheSize = 8;
  static constexpr size_t kLargeCacheArraySize = 12;
  static constexpr size_t kMaxCacheBits = 4;
  static constexpr size_t kMaxPageIdBits = kAddressBits - kPageShift;
  static constexpr size_t kReservedBits = 26;

  static_assert(kLargeCacheSize <= (1 << kMaxCacheBits) - 1);

  // When central freelist tracks a span, that span is assured to consist of <
  // kLargeSpanLength number of pages. This allows us to record number of pages
  // in that span in fewer (i.e. kMaxNumPageBits) bits.
  static constexpr size_t kMaxNumPageBits = 6;
  static constexpr Length kLargeSpanLength = Length((1 << kMaxNumPageBits) - 1);

  static std::align_val_t CalcAlignOf(uint32_t max_cache_array_size);
  static size_t CalcSizeOf(uint32_t max_cache_array_size);
  uint64_t AllocTime(size_t size, uint32_t max_span_cache_size) const;

  // Returns true if Span will use bitmap for objects of size <size>.
  static bool UseBitmapForSize(size_t size);

 private:
  // Returns if the span is large (i.e. consists of > kLargeSpanLength number of
  // pages) or is sampled.
  bool is_large_or_sampled() const { return is_large_span_ || sampled_; }

  // See the comment on freelist organization in cc file.
  typedef uint16_t ObjIdx;
  static constexpr ObjIdx kListEnd = -1;

  // Use uint16_t or uint8_t for 16 bit and 8 bit fields instead of bitfields.
  // LLVM will generate widen load/store and bit masking operations to access
  // bitfields and this hurts performance. Although compiler flag
  // -ffine-grained-bitfield-accesses can help the performance if bitfields
  // are used here, but the flag could potentially hurt performance in other
  // cases so it is not enabled by default. For more information, please
  // look at b/35680381 and cl/199502226.
  // For available objects stored as a compressed linked list, the index of
  // the first object in recorded in freelist_.
  struct {
    uint16_t embed_count_;
    uint16_t freelist_;
  };
  std::atomic<uint16_t> allocated_;  // Number of non-free objects
  uint8_t cache_size_ : kMaxCacheBits;
  uint8_t nonempty_index_ : 4;  // The nonempty_ list index for this span.
  // Has this span allocation resulted in a donation to the filler in the page
  // heap? This is used by page heap to compute abandoned pages.
  uint8_t is_donated_ : 1;

  static constexpr size_t kBitmapSize = 8 * sizeof(ObjIdx) * kCacheSize;

  uint64_t first_page_ : kMaxPageIdBits;  // Starting page number.

  uint32_t reserved_ : kReservedBits;
  // Determines if the span consists of > kLargeSpanLength number of pages.
  uint8_t is_large_span_ : 1;
  uint8_t sampled_ : 1;  // Sampled object?

  struct LargeOrSampledState {
    uint64_t num_pages;
    // Used only for sampled spans (SAMPLED state).
    SampledAllocation* sampled_allocation;
  };

  struct SmallSpanState {
    uint64_t num_pages : kMaxNumPageBits;
    union {
      // Used only for spans in CentralFreeList (SMALL_OBJECT state).
      // Embed cache of free objects.
      ObjIdx cache[0];

      // Used for spans with in CentralFreeList with fewer than 64 objects.
      // Each bit is set to one when the object is available, and zero
      // when the object is used.
      Bitmap<kBitmapSize> bitmap{};
    };
  };

  union {
    // When a span consists of greater than kLargeSpanLength number of pages,
    // it's the page heap that is allocating an object > kMaxSize. In such a
    // scenario, use larger number of bits to record the number of pages in that
    // span. Additionally, as central freelist is not tracking that span, we do
    // not need to record the state such as bitmap or cache bits. We also do
    // not need to record that state when the span is sampled.
    LargeOrSampledState large_or_sampled_state_;

    // When a span consists of < kLargeSpanLength number of pages, we can record
    // the number of pages in kMaxNumPageBits number of bits. Additionally, it's
    // likely (although not assured) that the central freelist is tracking that
    // span. So, we additionally need to record cache or bitmap for that span.
    SmallSpanState small_span_state_;
  };

  // Convert object pointer <-> freelist index.
  ObjIdx PtrToIdx(void* ptr, size_t size) const;
  ObjIdx* IdxToPtr(ObjIdx idx, size_t size, uintptr_t start) const;

  // Convert object pointer <-> freelist index for bitmap managed objects.
  ObjIdx BitmapPtrToIdx(void* ptr, size_t size, uint32_t reciprocal) const;
  void* BitmapIdxToPtr(ObjIdx idx, size_t size) const;

  // Helper function for converting a pointer to an index.
  static ObjIdx OffsetToIdx(uintptr_t offset, uint32_t reciprocal);

  size_t ListPopBatch(void** __restrict batch, size_t N, size_t size);

  bool ListPush(void* ptr, size_t size, uint32_t max_cache_size);

  // For spans containing 64 or fewer objects, indicate that the object at the
  // index has been returned. Always returns true.
  bool BitmapPush(void* ptr, size_t size, uint32_t reciprocal);

  // A bitmap is used to indicate object availability for spans containing
  // 64 or fewer objects.
  void BuildBitmap(size_t size, size_t count);

  // For spans with 64 or fewer objects populate batch with up to N objects.
  // Returns number of objects actually popped.
  size_t BitmapPopBatch(absl::Span<void*> batch, size_t size);

  // Friend class to enable more indepth testing of bitmap code.
  friend class SpanTestPeer;
};

inline uint64_t Span::AllocTime(size_t size,
                                uint32_t max_span_cache_size) const {
  uint64_t alloc_time = 0;
  if (max_span_cache_size == Span::kLargeCacheSize && !UseBitmapForSize(size)) {
    memcpy(&alloc_time, &small_span_state_.cache[Span::kLargeCacheSize],
           sizeof(alloc_time));
  }
  return alloc_time;
}

inline Span::ObjIdx* Span::IdxToPtr(ObjIdx idx, size_t size,
                                    uintptr_t start) const {
  TC_ASSERT_EQ(small_span_state_.num_pages, 1);
  TC_ASSERT_EQ(start, first_page().start_uintptr());
  TC_ASSERT_NE(idx, kListEnd);
  uintptr_t off = start + (static_cast<uintptr_t>(idx) << kAlignmentShift);
  ObjIdx* ptr = reinterpret_cast<ObjIdx*>(off);
  TC_ASSERT_EQ(PtrToIdx(ptr, size), idx);
  return ptr;
}

inline Span::ObjIdx Span::PtrToIdx(void* ptr, size_t size) const {
  // Object index is an offset from span start divided by kAlignment.
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  // Classes that use freelist must also use 1 page per span,
  // so don't load first_page_ (may be on a different cache line).
  TC_ASSERT_EQ(small_span_state_.num_pages, 1);
  TC_ASSERT_EQ(PageIdContaining(ptr), first_page());
  uintptr_t off = (p & (kPageSize - 1)) >> kAlignmentShift;
  ObjIdx idx = static_cast<ObjIdx>(off);
  TC_ASSERT_NE(idx, kListEnd);
  TC_ASSERT_EQ(idx, off);
  TC_ASSERT_EQ(p, first_page().start_uintptr() +
                      (static_cast<uintptr_t>(idx) << kAlignmentShift));
  return idx;
}

inline bool Span::FreelistPush(void* ptr, size_t size, uint32_t reciprocal,
                               uint32_t max_cache_size) {
  TC_ASSERT(!is_large_or_sampled());
  const auto allocated = allocated_.load(std::memory_order_relaxed);
  TC_ASSERT_GT(allocated, 0);
  if (ABSL_PREDICT_FALSE(allocated == 1)) {
    return false;
  }
  allocated_.store(allocated - 1, std::memory_order_relaxed);
  // Bitmaps are used to record object availability when there are fewer than
  // 64 objects in a span.
  if (ABSL_PREDICT_FALSE(UseBitmapForSize(size))) {
    return BitmapPush(ptr, size, reciprocal);
  }
  return ListPush(ptr, size, max_cache_size);
}

inline bool Span::ListPush(void* ptr, size_t size, uint32_t max_cache_size) {
  ObjIdx idx = PtrToIdx(ptr, size);
  if (cache_size_ < max_cache_size) {
    // Have empty space in the cache, push there.
    small_span_state_.cache[cache_size_] = idx;
    cache_size_++;
  } else if (ABSL_PREDICT_TRUE(freelist_ != kListEnd) &&
             // -1 because the first slot is used by freelist link.
             ABSL_PREDICT_TRUE(embed_count_ != size / sizeof(ObjIdx) - 1)) {
    // Push onto the first object on freelist.
    // Avoid loading first_page_, we we can infer it from the pointer;
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr) & ~(kPageSize - 1);
    ObjIdx* host = IdxToPtr(freelist_, size, start);
    embed_count_++;
    host[embed_count_] = idx;
  } else {
    // Push onto freelist.
    *reinterpret_cast<ObjIdx*>(ptr) = freelist_;
    freelist_ = idx;
    embed_count_ = 0;
  }
  return true;
}

inline Span::ObjIdx Span::OffsetToIdx(uintptr_t offset, uint32_t reciprocal) {
  // Add kBitmapScalingDenominator / 2 to round to nearest integer.
  return static_cast<ObjIdx>(
      (offset * reciprocal + kBitmapScalingDenominator / 2) /
      kBitmapScalingDenominator);
}

inline Span::ObjIdx Span::BitmapPtrToIdx(void* ptr, size_t size,
                                         uint32_t reciprocal) const {
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t off = static_cast<uint32_t>(p - first_page().start_uintptr());
  ObjIdx idx = OffsetToIdx(off, reciprocal);
  TC_ASSERT_EQ(BitmapIdxToPtr(idx, size), ptr);
  return idx;
}

inline bool Span::BitmapPush(void* ptr, size_t size, uint32_t reciprocal) {
  size_t before = small_span_state_.bitmap.CountBits();
  // TODO(djgove) Conversions to offsets can be computed outside of lock.
  ObjIdx idx = BitmapPtrToIdx(ptr, size, reciprocal);
  // Check that the object is not already returned.
  TC_ASSERT_EQ(small_span_state_.bitmap.GetBit(idx), 0);
  // Set the bit indicating where the object was returned.
  small_span_state_.bitmap.SetBit(idx);
  TC_ASSERT_EQ(before + 1, small_span_state_.bitmap.CountBits());
  return true;
}

inline const SampledAllocation& Span::sampled_allocation() const {
  TC_ASSERT(sampled_);
  TC_ASSERT(is_large_or_sampled());
  return *large_or_sampled_state_.sampled_allocation;
}

inline bool Span::sampled() const { return sampled_; }

inline PageId Span::first_page() const { return PageId(first_page_); }

inline PageId Span::last_page() const {
  if (is_large_or_sampled()) {
    return first_page() + Length(large_or_sampled_state_.num_pages) - Length(1);
  }
  return first_page() + Length(small_span_state_.num_pages) - Length(1);
}

inline void Span::set_first_page(PageId p) {
  TC_ASSERT_GT(p, PageId{0});
  TC_CHECK_LT(p.index(), static_cast<uint64_t>(1) << kMaxPageIdBits);
  first_page_ = p.index();
}

inline void* Span::start_address() const {
  TC_ASSERT_GT(first_page(), PageId{0});
  return first_page().start_addr();
}

inline Length Span::num_pages() const {
  if (is_large_or_sampled()) {
    return Length(large_or_sampled_state_.num_pages);
  }
  return Length(small_span_state_.num_pages);
}

inline void Span::set_num_pages(Length len) {
  if (len > kLargeSpanLength || sampled()) {
    large_or_sampled_state_.num_pages = len.raw_num();
    is_large_span_ = len > kLargeSpanLength;
    return;
  }
  TC_ASSERT_LT(len.raw_num(), 1 << kMaxNumPageBits);
  small_span_state_.num_pages = len.raw_num();
  is_large_span_ = 0;
}

inline size_t Span::bytes_in_span() const ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (is_large_or_sampled()) {
    return Length(large_or_sampled_state_.num_pages).in_bytes();
  }
  return Length(small_span_state_.num_pages).in_bytes();
}

inline bool Span::FreelistEmpty(size_t size) const {
  TC_ASSERT(!is_large_or_sampled());
  if (UseBitmapForSize(size)) {
    return small_span_state_.bitmap.IsZero();
  } else {
    return cache_size_ == 0 && freelist_ == kListEnd;
  }
}

inline void Span::Prefetch() {
  // TODO(b/304135905): Will revisit this.
  // The first 16 bytes of a Span are the next and previous pointers
  // for when it is stored in a linked list. Since the sizeof(Span) is
  // 48 bytes, spans fit into 2 cache lines 50% of the time, with either
  // the first 16-bytes or the last 16-bytes in a different cache line.
  // Prefetch the cacheline that contains the most frequestly accessed
  // data by offseting into the middle of the Span.
  static_assert(sizeof(Span) <= 64, "Update span prefetch offset");
  PrefetchT0(&this->allocated_);
}

inline bool Span::IsValidSizeClass(size_t size, size_t pages) {
  if (Length(pages) > kLargeSpanLength) return false;
  if (Span::UseBitmapForSize(size)) {
    size_t objects = Length(pages).in_bytes() / size;
    return objects <= kBitmapSize;
  } else {
    return pages == 1;
  }
}

inline bool Span::UseBitmapForSize(size_t size) {
  // Can fit kBitmapSize objects into a bitmap, so determine what the minimum
  // object size needs to be in order for that to work. This makes the
  // assumption that we don't increase the number of pages at a point where the
  // object count ends up exceeding kBitmapSize.
  static constexpr size_t kBitmapMinObjectSize = kPageSize / kBitmapSize;
  return size >= kBitmapMinObjectSize;
}

// This is equivalent to UseBitmapForSize, but instrusive-ness is the property
// callers care about, while use of bitmap is an implementation detail.
inline bool Span::IsNonIntrusive(size_t size) { return UseBitmapForSize(size); }

inline std::align_val_t Span::CalcAlignOf(uint32_t max_cache_array_size) {
  TC_ASSERT_GE(max_cache_array_size, kCacheSize);
  TC_ASSERT_LE(max_cache_array_size, kLargeCacheArraySize);
  return static_cast<std::align_val_t>(
      max_cache_array_size == kCacheSize ? 8 : ABSL_CACHELINE_SIZE);
}

inline size_t Span::CalcSizeOf(uint32_t max_cache_array_size) {
  TC_ASSERT_GE(max_cache_array_size, kCacheSize);
  TC_ASSERT_LE(max_cache_array_size, kLargeCacheArraySize);
  size_t ret =
      sizeof(Span) + sizeof(uint16_t) * (max_cache_array_size - kCacheSize);
  ret = (ret + alignof(Span) - 1) & ~(alignof(Span) - 1);
  TC_ASSERT((ret == 48 && max_cache_array_size == kCacheSize) ||
            (ret == 64 && max_cache_array_size == kLargeCacheArraySize));
  return ret;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_SPAN_H_
