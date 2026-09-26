#pragma clang system_header
// Copyright 2024 The TCMalloc Authors
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

#ifndef TCMALLOC_INTERNAL_GWP_ASAN_STATE_H_
#define TCMALLOC_INTERNAL_GWP_ASAN_STATE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>

#include "absl/types/span.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class GwpAsanState {
 public:
  constexpr GwpAsanState() = default;

  enum class Type {
    kNone,
    kMismatchedDelete,
    kDoubleFree,
  };

  Type type() const { return type_; }

  bool triggered() const { return type_ != Type::kNone; }

  std::optional<absl::Span<void* const>> AllocationStack() const {
    TC_ASSERT_NE(type_, Type::kNone);

    if (!allocation_stack_depth_.has_value()) {
      return std::nullopt;
    }

    return absl::MakeSpan(allocation_stack_, *allocation_stack_depth_);
  }

  std::optional<absl::Span<void* const>> DeallocationStack() const {
    TC_ASSERT_NE(type_, Type::kNone);
    if (!deallocation_stack_depth_.has_value()) {
      return std::nullopt;
    }

    return absl::MakeSpan(deallocation_stack_, *deallocation_stack_depth_);
  }

  size_t provided_min() const {
    TC_ASSERT_EQ(type_, Type::kMismatchedDelete);
    return provided_min_;
  }
  size_t provided_max() const {
    TC_ASSERT_EQ(type_, Type::kMismatchedDelete);
    return provided_max_;
  }

  size_t minimum_size() const {
    TC_ASSERT_EQ(type_, Type::kMismatchedDelete);
    return minimum_;
  }

  size_t maximum_size() const {
    TC_ASSERT_EQ(type_, Type::kMismatchedDelete);
    return maximum_;
  }

  void RecordMismatch(
      size_t provided_min, size_t provided_max, size_t minimum, size_t maximum,
      std::optional<absl::Span<void* const>> allocation_stack,
      std::optional<absl::Span<void* const>> deallocation_stack) {
    type_ = Type::kMismatchedDelete;

    provided_min_ = provided_min;
    provided_max_ = provided_max;
    minimum_ = minimum;
    maximum_ = maximum;

    if (allocation_stack.has_value()) {
      size_t allocation_stack_depth =
          std::min<size_t>(kMaxStackDepth, allocation_stack->size());
      memcpy(allocation_stack_, allocation_stack->data(),
             sizeof(void*) * allocation_stack_depth);
      allocation_stack_depth_ = allocation_stack_depth;
    } else {
      allocation_stack_depth_ = std::nullopt;
    }

    if (deallocation_stack.has_value()) {
      size_t deallocation_stack_depth =
          std::min<size_t>(kMaxStackDepth, deallocation_stack->size());
      memcpy(deallocation_stack_, deallocation_stack->data(),
             sizeof(void*) * deallocation_stack_depth);
      deallocation_stack_depth_ = deallocation_stack_depth;
    } else {
      deallocation_stack_depth_ = std::nullopt;
    }
  }

  void RecordDoubleFree(absl::Span<void* const> deallocation_stack) {
    type_ = Type::kDoubleFree;

    size_t deallocation_stack_depth =
        std::min<size_t>(kMaxStackDepth, deallocation_stack.size());
    memcpy(deallocation_stack_, deallocation_stack.data(),
           sizeof(void*) * deallocation_stack_depth);
    deallocation_stack_depth_ = deallocation_stack_depth;
  }

 private:
  Type type_ = Type::kNone;
  size_t provided_min_ = 0, provided_max_ = 0, minimum_ = 0, maximum_ = 0;

  void* allocation_stack_[kMaxStackDepth] = {};
  std::optional<size_t> allocation_stack_depth_ = std::nullopt;
  void* deallocation_stack_[kMaxStackDepth] = {};
  std::optional<size_t> deallocation_stack_depth_ = std::nullopt;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_GWP_ASAN_STATE_H_
