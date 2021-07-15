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

#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/pages.h"

namespace tcmalloc {

// Information kept for a span (a contiguous run of pages).
//
// Spans can be in different states. The current state determines set of methods
// that can be called on the span (and the active member in the union below).
// States are:
//  - SMALL_OBJECT: the span holds multiple small objects.
//    The span is owned by CentralFreeList and is generally on
//    CentralFreeList::nonempty_ list (unless has no free objects).
//    location_ == IN_USE.
//  - LARGE_OBJECT: the span holds a single large object.
//    The span can be considered to be owner by user until the object is freed.
//    location_ == IN_USE.
//  - SAMPLED: the span holds a single sampled object.
//    The span can be considered to be owner by user until the object is freed.
//    location_ == IN_USE && sampled_ == 1.
//  - ON_NORMAL_FREELIST: the span has no allocated objects, owned by PageHeap
//    and is on normal PageHeap list.
//    location_ == ON_NORMAL_FREELIST.
//  - ON_RETURNED_FREELIST: the span has no allocated objects, owned by PageHeap
//    and is on returned PageHeap list.
//    location_ == ON_RETURNED_FREELIST.
class Span;
typedef TList<Span> SpanList;

class Span : public SpanList::Elem {
 public:
  // Allocator/deallocator for spans. Note that these functions are defined
  // in static_vars.h, which is weird: see there for why.
  static Span* New(PageId p, Length len)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);
  static void Delete(Span* span) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Remove this from the linked list in which it resides.
  // REQUIRES: this span is on some list.
  void RemoveFromList();

  // locations used to track what list a span resides on.
  enum Location {
    IN_USE,                // not on PageHeap lists
    ON_NORMAL_FREELIST,    // on normal PageHeap list
    ON_RETURNED_FREELIST,  // on returned PageHeap list
  };
  Location location() const;
  void set_location(Location loc);

  // ---------------------------------------------------------------------------
  // Support for sampled allocations.
  // There is one-to-one correspondence between a sampled allocation and a span.
  // ---------------------------------------------------------------------------

  // Mark this span as sampling allocation at the stack. Sets state to SAMPLED.
  void Sample(StackTrace* stack) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Unmark this span as sampling an allocation.
  // Returns stack trace previously passed to Sample,
  // or nullptr if this is a non-sampling span.
  // REQUIRES: this is a SAMPLED span.
  StackTrace* Unsample() ABSL_EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock);

  // Returns stack for the sampled allocation.
  // pageheap_lock is not required, but caller either needs to hold the lock or
  // ensure by some other means that the sampling state can't be changed
  // concurrently.
  // REQUIRES: this is a SAMPLED span.
  StackTrace* sampled_stack() const;

  // Is it a sampling span?
  // For debug checks. pageheap_lock is not required, but caller needs to ensure
  // that sampling state can't be changed concurrently.
  bool sampled() const;

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
  void* start_address() const;

  // Returns number of pages in the span.
  Length num_pages() const;

  // Sets number of pages in the span.
  void set_num_pages(Length len);

  // Total memory bytes in the span.
  size_t bytes_in_span() const;

  // ---------------------------------------------------------------------------
  // Age tracking (for free spans in PageHeap).
  // ---------------------------------------------------------------------------

  uint64_t freelist_added_time() const;
  void set_freelist_added_time(uint64_t t);

  // Sets this span freelist added time to average of this and other times
  // weighted by their sizes.
  // REQUIRES: this is a ON_NORMAL_FREELIST or ON_RETURNED_FREELIST span.
  void AverageFreelistAddedTime(const Span* other);

  // Returns internal fragmentation of the span.
  // REQUIRES: this is a SMALL_OBJECT span.
  double Fragmentation() const;

  // ---------------------------------------------------------------------------
  // Freelist management.
  // Used for spans in CentralFreelist to manage free objects.
  // These methods REQUIRE a SMALL_OBJECT span.
  // ---------------------------------------------------------------------------

  // Span freelist is empty?
  bool FreelistEmpty() const;

  // Pushes ptr onto freelist unless the freelist becomes full,
  // in which case just return false.
  bool FreelistPush(void* ptr, size_t size);

  // Pops up to N objects from the freelist and returns them in the batch array.
  // Returns number of objects actually popped.
  size_t FreelistPopBatch(void** batch, size_t N, size_t size);

  // Reset a Span object to track the range [p, p + n).
  void Init(PageId p, Length n);

  // Initialize freelist to contain all objects in the span.
  void BuildFreelist(size_t size, size_t count);

  // Prefetch cacheline containing most important span information.
  void Prefetch();

  static constexpr size_t kCacheSize = 4;

 private:
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
  uint16_t allocated_;  // Number of non-free objects
  uint16_t embed_count_;
  uint16_t freelist_;
  uint8_t cache_size_;
  uint8_t location_ : 2;  // Is the span on a freelist, and if so, which?
  uint8_t sampled_ : 1;   // Sampled object?

  union {
    // Used only for spans in CentralFreeList (SMALL_OBJECT state).
    // Embed cache of free objects.
    ObjIdx cache_[kCacheSize];

    // Used only for sampled spans (SAMPLED state).
    StackTrace* sampled_stack_;

    // Used only for spans in PageHeap
    // (ON_NORMAL_FREELIST or ON_RETURNED_FREELIST state).
    // Time when this span was added to a freelist.  Units: cycles.  When a span
    // is merged into this one, we set this to the average of now and the
    // current freelist_added_time, weighted by the two spans' sizes.
    uint64_t freelist_added_time_;
  };

  PageId first_page_;  // Starting page number.
  Length num_pages_;   // Number of pages in span.

  // Convert object pointer <-> freelist index.
  ObjIdx PtrToIdx(void* ptr, size_t size) const;
  ObjIdx* IdxToPtr(ObjIdx idx, size_t size) const;

  enum Align { SMALL, LARGE };

  template <Align align>
  ObjIdx* IdxToPtrSized(ObjIdx idx, size_t size) const;

  template <Align align>
  size_t FreelistPopBatchSized(void** __restrict batch, size_t N, size_t size);
};

template <Span::Align align>
Span::ObjIdx* Span::IdxToPtrSized(ObjIdx idx, size_t size) const {
  ASSERT(idx != kListEnd);
  ASSERT(align == Align::LARGE || align == Align::SMALL);
  uintptr_t off =
      first_page_.start_uintptr() +
      (static_cast<uintptr_t>(idx)
       << (align == Align::SMALL ? kAlignmentShift
                                 : SizeMap::kMultiPageAlignmentShift));
  ObjIdx* ptr = reinterpret_cast<ObjIdx*>(off);
  ASSERT(PtrToIdx(ptr, size) == idx);
  return ptr;
}

template <Span::Align align>
size_t Span::FreelistPopBatchSized(void** __restrict batch, size_t N,
                                   size_t size) {
  size_t result = 0;

  // Pop from cache.
  auto csize = cache_size_;
  ASSUME(csize <= kCacheSize);
  auto cache_reads = csize < N ? csize : N;
  for (; result < cache_reads; result++) {
    batch[result] = IdxToPtrSized<align>(cache_[csize - result - 1], size);
  }

  // Store this->cache_size_ one time.
  cache_size_ = csize - result;

  while (result < N) {
    if (freelist_ == kListEnd) {
      break;
    }

    ObjIdx* const host = IdxToPtrSized<align>(freelist_, size);
    uint16_t embed_count = embed_count_;
    ObjIdx current = host[embed_count];

    size_t iter = embed_count;
    if (result + embed_count > N) {
      iter = N - result;
    }
    for (size_t i = 0; i < iter; i++) {
      // Pop from the first object on freelist.
      batch[result + i] = IdxToPtrSized<align>(host[embed_count - i], size);
    }
    embed_count -= iter;
    result += iter;

    // Update current for next cycle.
    current = host[embed_count];

    if (result == N) {
      embed_count_ = embed_count;
      break;
    }

    // The first object on the freelist is empty, pop it.
    ASSERT(embed_count == 0);

    batch[result] = host;
    result++;

    freelist_ = current;
    embed_count_ = size / sizeof(ObjIdx) - 1;
  }
  allocated_ += result;
  return result;
}

inline Span::Location Span::location() const {
  return static_cast<Location>(location_);
}

inline void Span::set_location(Location loc) {
  location_ = static_cast<uint64_t>(loc);
}

inline StackTrace* Span::sampled_stack() const {
  ASSERT(sampled_);
  return sampled_stack_;
}

inline bool Span::sampled() const { return sampled_; }

inline PageId Span::first_page() const { return first_page_; }

inline PageId Span::last_page() const {
  return first_page_ + num_pages_ - Length(1);
}

inline void Span::set_first_page(PageId p) { first_page_ = p; }

inline void* Span::start_address() const { return first_page_.start_addr(); }

inline Length Span::num_pages() const { return num_pages_; }

inline void Span::set_num_pages(Length len) { num_pages_ = len; }

inline size_t Span::bytes_in_span() const { return num_pages_.in_bytes(); }

inline void Span::set_freelist_added_time(uint64_t t) {
  freelist_added_time_ = t;
}

inline uint64_t Span::freelist_added_time() const {
  return freelist_added_time_;
}

inline bool Span::FreelistEmpty() const {
  return cache_size_ == 0 && freelist_ == kListEnd;
}

inline void Span::RemoveFromList() { SpanList::Elem::remove(); }

inline void Span::Prefetch() {
  // The first 16 bytes of a Span are the next and previous pointers
  // for when it is stored in a linked list. Since the sizeof(Span) is
  // 48 bytes, spans fit into 2 cache lines 50% of the time, with either
  // the first 16-bytes or the last 16-bytes in a different cache line.
  // Prefetch the cacheline that contains the most frequestly accessed
  // data by offseting into the middle of the Span.
#if defined(__GNUC__)
#if __WORDSIZE == 32
  // The Span fits in one cache line, so simply prefetch the base pointer.
  static_assert(sizeof(Span) == 32, "Update span prefetch offset");
  __builtin_prefetch(this, 0, 3);
#else
  // The Span can occupy two cache lines, so prefetch the cacheline with the
  // most frequently accessed parts of the Span.
  static_assert(sizeof(Span) == 48, "Update span prefetch offset");
  __builtin_prefetch(&this->allocated_, 0, 3);
#endif
#endif
}

inline void Span::Init(PageId p, Length n) {
#ifndef NDEBUG
  // In debug mode we have additional checking of our list ops; these must be
  // initialized.
  new (this) Span();
#endif
  first_page_ = p;
  num_pages_ = n;
  location_ = IN_USE;
  sampled_ = 0;
}

}  // namespace tcmalloc

#endif  // TCMALLOC_SPAN_H_
