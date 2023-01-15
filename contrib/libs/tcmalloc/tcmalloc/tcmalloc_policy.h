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
// This file defines policies used when allocation memory.
//
// An allocation policy encapsulates three policies:
//
// - Out of memory policy.
//   Dictates how to handle OOM conditions.
//
//   struct OomPolicyTemplate {
//     // Invoked when we failed to allocate memory
//     // Must either terminate, throw, or return nullptr
//     static void* handle_oom(size_t size);
//   };
//
// - Alignment policy
//   Dictates alignment to use for an allocation.
//   Must be trivially copyable.
//
//   struct AlignPolicyTemplate {
//     // Returns the alignment to use for the memory allocation,
//     // or 1 to use small allocation table alignments (8 bytes)
//     // Returned value Must be a non-zero power of 2.
//     size_t align() const;
//   };
//
// - Hook invocation policy
//   dictates invocation of allocation hooks
//
//   struct HooksPolicyTemplate {
//     // Returns true if allocation hooks must be invoked.
//     static bool invoke_hooks();
//   };

#ifndef TCMALLOC_TCMALLOC_POLICY_H_
#define TCMALLOC_TCMALLOC_POLICY_H_

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <cstddef>

#include "tcmalloc/internal/logging.h"

namespace tcmalloc {

// NullOomPolicy: returns nullptr
struct NullOomPolicy {
  static inline constexpr void* handle_oom(size_t size) { return nullptr; }

  static constexpr bool can_return_nullptr() { return true; }
};

// MallocOomPolicy: sets errno to ENOMEM and returns nullptr
struct MallocOomPolicy {
  static inline void* handle_oom(size_t size) {
    errno = ENOMEM;
    return nullptr;
  }

  static constexpr bool can_return_nullptr() { return true; }
};

// CppOomPolicy: terminates the program
struct CppOomPolicy {
  static ABSL_ATTRIBUTE_NOINLINE ABSL_ATTRIBUTE_NORETURN void* handle_oom(
      size_t size) {
    Crash(kCrashWithStats, __FILE__, __LINE__,
          "Unable to allocate (new failed)", size);
    __builtin_unreachable();
  }

  static constexpr bool can_return_nullptr() { return false; }
};

// DefaultAlignPolicy: use default small size table based allocation
struct DefaultAlignPolicy {
  // Important: the value here is explicitly '1' to indicate that the used
  // alignment is the default alignment of the size tables in tcmalloc.
  // The constexpr value of 1 will optimize out the alignment checks and
  // iterations in the GetSizeClass() calls for default aligned allocations.
  static constexpr size_t align() { return 1; }
};

// MallocAlignPolicy: use std::max_align_t allocation
struct MallocAlignPolicy {
  static constexpr size_t align() { return alignof(std::max_align_t); }
};

// AlignAsPolicy: use user provided alignment
class AlignAsPolicy {
 public:
  AlignAsPolicy() = delete;
  explicit constexpr AlignAsPolicy(size_t value) : value_(value) {}
  explicit constexpr AlignAsPolicy(std::align_val_t value)
      : AlignAsPolicy(static_cast<size_t>(value)) {}

  size_t constexpr align() const { return value_; }

 private:
  size_t value_;
};

// InvokeHooksPolicy: invoke memory allocation hooks
struct InvokeHooksPolicy {
  static constexpr bool invoke_hooks() { return true; }
};

// NoHooksPolicy: do not invoke memory allocation hooks
struct NoHooksPolicy {
  static constexpr bool invoke_hooks() { return false; }
};

// TCMallocPolicy defines the compound policy object containing
// the OOM, alignment and hooks policies.
// Is trivially constructible, copyable and destructible.
template <typename OomPolicy = CppOomPolicy,
          typename AlignPolicy = DefaultAlignPolicy,
          typename HooksPolicy = InvokeHooksPolicy>
class TCMallocPolicy {
 public:
  constexpr TCMallocPolicy() = default;
  explicit constexpr TCMallocPolicy(AlignPolicy align) : align_(align) {}

  // OOM policy
  static void* handle_oom(size_t size) { return OomPolicy::handle_oom(size); }

  // Alignment policy
  constexpr size_t align() const { return align_.align(); }

  // Hooks policy
  static constexpr bool invoke_hooks() { return HooksPolicy::invoke_hooks(); }

  // Returns this policy aligned as 'align'
  template <typename align_t>
  constexpr TCMallocPolicy<OomPolicy, AlignAsPolicy, HooksPolicy> AlignAs(
      align_t align) const {
    return TCMallocPolicy<OomPolicy, AlignAsPolicy, HooksPolicy>(
        AlignAsPolicy{align});
  }

  // Returns this policy with a nullptr OOM policy.
  constexpr TCMallocPolicy<NullOomPolicy, AlignPolicy, HooksPolicy> Nothrow()
      const {
    return TCMallocPolicy<NullOomPolicy, AlignPolicy, HooksPolicy>(align_);
  }

  // Returns this policy with NewAllocHook invocations disabled.
  constexpr TCMallocPolicy<OomPolicy, AlignPolicy, NoHooksPolicy>
  WithoutHooks()
      const {
    return TCMallocPolicy<OomPolicy, AlignPolicy, NoHooksPolicy>(align_);
  }

  static constexpr bool can_return_nullptr() {
    return OomPolicy::can_return_nullptr();
  }

 private:
  AlignPolicy align_;
};

using CppPolicy = TCMallocPolicy<CppOomPolicy, DefaultAlignPolicy>;
using MallocPolicy = TCMallocPolicy<MallocOomPolicy, MallocAlignPolicy>;

}  // namespace tcmalloc

#endif  // TCMALLOC_TCMALLOC_POLICY_H_
