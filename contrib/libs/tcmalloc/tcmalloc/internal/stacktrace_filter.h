#pragma clang system_header
// Copyright 2022 The TCMalloc Authors
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

#ifndef TCMALLOC_INTERNAL_STACKTRACE_FILTER_H_
#define TCMALLOC_INTERNAL_STACKTRACE_FILTER_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "absl/hash/hash.h"
#include "absl/types/span.h"
#include "tcmalloc/internal/config.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

// Counting Bloom Filter (CBF) implementation for stack traces. The template
// parameter kSize denotes the size of the Bloom Filter, and kHashNum the number
// of hash functions used. The probability of false positives is calculated as:
//
//   P(num_stacks) = (1 - e^(-kHashNum * (num_stacks / kSize))) ^ kHashNum
//
// Where `num_stacks` are unique stack traces currently present in the filter.
//
// The main benefit of a CBF (vs. other data structure such as a regular Bloom
// Filter, or a Cache with an eviction policy) is that if the sum of all Add()
// operations becomes zero again, the CBF will no longer contain the item. False
// positives can be mitigated by configuring the CBF according to above formula.
//
// Thread-safety: thread-safe.
template <size_t kSize_, size_t kHashNum_>
class StackTraceFilter {
 public:
  static constexpr size_t kSize = kSize_;
  static constexpr size_t kHashNum = kHashNum_;

  static_assert(kSize > 0, "size must be non-zero");
  static_assert(kHashNum > 0, "number of hashes must be non-zero");

  constexpr StackTraceFilter() = default;

  // Returns true if the filter contains the provided stack trace. See above
  // formula to calculate the probability of false positives.
  bool Contains(absl::Span<void* const> stack_trace) const {
    size_t stack_hash = GetFirstHash(stack_trace);

    for (size_t i = 0; i < kHashNum; ++i) {
      if (!counts_[stack_hash % kSize].load(std::memory_order_relaxed))
        return false;
      stack_hash = GetNextHash(stack_hash);
    }

    return true;
  }

  // Add (or remove if `val` < 0) a stack trace from the filter. The sum of
  // values `val` added to the filter since construction or the last Clear()
  // determines if a stack trace is contained or not: for any non-zero sum,
  // Contains() returns true; false if the sum is zero.
  void Add(absl::Span<void* const> stack_trace, int val) {
    Add(GetFirstHash(stack_trace), val);
  }

 protected:
  static size_t GetFirstHash(absl::Span<void* const> s) {
    return absl::HashOf(s);
  }

  static size_t GetNextHash(size_t prev_hash) {
    return absl::HashOf(prev_hash);
  }

  void Add(size_t stack_hash, int val) {
    for (size_t i = 0; i < kHashNum; ++i) {
      counts_[stack_hash % kSize].fetch_add(val, std::memory_order_relaxed);
      stack_hash = GetNextHash(stack_hash);
    }
  }

 private:
  // Use uint to allow for integer wrap-around; false positives are possible if
  // all kHashNum counts wrap to 0 (which is unlikely with kHashNum > 1).
  std::array<std::atomic<uint16_t>, kSize> counts_ = {};

  friend class StackTraceFilterTest;
};

// Similar to StackTraceFilter, except that entries have to be decayed to be
// fully removed from the filter: the sum value of a stack trace is the added
// value + number of non-negative Add() calls. Each call to Add() or Decay()
// decays a stack trace: the stack trace to be decayed is the one added in the
// kDecaySteps'th previous call to Add().
//
// Thread-safety: thread-safe.
template <size_t kSize, size_t kHashNum, size_t kDecaySteps>
class DecayingStackTraceFilter : public StackTraceFilter<kSize, kHashNum> {
  using Base = StackTraceFilter<kSize, kHashNum>;

 public:
  // Add (or remove if `val` < 0) a stack trace from the filter. On addition
  // (non-negative value), a previously added stack trace is decayed.
  void Add(absl::Span<void* const> stack_trace, int val) {
    const size_t stack_hash = this->GetFirstHash(stack_trace);
    if (val >= 0) {
      Decay(stack_hash);
      // Because 0-valued entries denote unused entries, add 1 to be decayed if
      // this is a non-zero hash (very likely).
      Base::Add(stack_hash, val + !!stack_hash);
    } else {
      // Removal.
      Base::Add(stack_hash, val);
    }
  }

  // Decays a previously added stack trace.
  void Decay() { Decay(0); }

  // Force decay all previously added stack traces.
  void DecayAll() {
    for (int i = 0; i < kDecaySteps; ++i) {
      Decay();
    }
  }

 private:
  // Replace the entry in the current ring buffer position with `replace_hash`
  // and decay the previous stack trace. Advances to the next position.
  void Decay(size_t replace_hash) {
    const size_t pos = pos_.fetch_add(1, std::memory_order_relaxed);
    const size_t decay_hash =
        ring(pos).exchange(replace_hash, std::memory_order_relaxed);
    // 0-valued entries denote unused entries.
    if (decay_hash) Base::Add(decay_hash, -1);
  }

  auto& ring(size_t pos) { return ring_[pos % ring_.size()]; }

  std::array<std::atomic<size_t>, kDecaySteps> ring_ = {};
  std::atomic<size_t> pos_ = 0;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_STACKTRACE_FILTER_H_
