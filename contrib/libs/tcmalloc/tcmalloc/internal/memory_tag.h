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

#ifndef TCMALLOC_INTERNAL_MEMORY_TAG_H_
#define TCMALLOC_INTERNAL_MEMORY_TAG_H_

#include <algorithm>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal {

enum class MemoryTag : uint8_t {
  // Sampled, infrequently allocated
  kSampled = 0x0,
  // Normal memory, NUMA partition 0
  kNormalP0 = kSanitizerAddressSpace ? 0x1 : 0x4,
  // Normal memory, NUMA partition 1
  kNormalP1 = kSanitizerAddressSpace ? 0xff : 0x6,
  // Normal memory
  kNormal = kNormalP0,
  // Cold
  kCold = 0x2,
  // Metadata
  kMetadata = 0x3,
  // SelSan sampled spans, kept separately because we need to quickly
  // distinguish them from the rest during delete and they also consume
  // shadow memory. 0xfe is an arbitrary value that shouldn't be used.
  kSelSan = kSelSanPresent ? 0x1 : 0xfe,
};

inline constexpr uintptr_t kTagShift = std::min(kAddressBits - 4, 42);
inline constexpr uintptr_t kTagMask =
    uintptr_t{kSanitizerAddressSpace ? 0x3 : 0x7} << kTagShift;

inline MemoryTag GetMemoryTag(const void* ptr) {
  return static_cast<MemoryTag>((reinterpret_cast<uintptr_t>(ptr) & kTagMask) >>
                                kTagShift);
}

inline bool IsNormalMemory(const void* ptr) {
  // This is slightly faster than checking kNormalP0/P1 separetly.
  static_assert((static_cast<uint8_t>(MemoryTag::kNormalP0) &
                 (static_cast<uint8_t>(MemoryTag::kSampled) |
                  static_cast<uint8_t>(MemoryTag::kCold))) == 0);
  bool res = (static_cast<uintptr_t>(GetMemoryTag(ptr)) &
              static_cast<uintptr_t>(MemoryTag::kNormal)) != 0;
  TC_ASSERT(res == (GetMemoryTag(ptr) == MemoryTag::kNormalP0 ||
                    GetMemoryTag(ptr) == MemoryTag::kNormalP1),
            "ptr=%p res=%d tag=%d", ptr, res,
            static_cast<int>(GetMemoryTag(ptr)));
  return res;
}

inline bool IsSelSanMemory(const void* ptr) {
  // This is a faster way to check for SelSan memory provided we already know
  // it's not a normal memory, and assuming it's not kMetadata (both assumptions
  // are checked by the assert below). A straightforward comparison with kSelSan
  // leads to extraction/check of 2 bits (these use 2 8-byte immediates);
  // this check can be done with a single BT instruction.
  // kSelSanPresent part allows to optimize away branches in non SelSan build.
  bool res =
      kSelSanPresent && (static_cast<uintptr_t>(GetMemoryTag(ptr)) &
                         static_cast<uintptr_t>(MemoryTag::kSelSan)) != 0;
  TC_ASSERT_EQ(res, GetMemoryTag(ptr) == MemoryTag::kSelSan);
  return res;
}

absl::string_view MemoryTagToLabel(MemoryTag tag);

}  // namespace tcmalloc::tcmalloc_internal
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_INTERNAL_MEMORY_TAG_H_
