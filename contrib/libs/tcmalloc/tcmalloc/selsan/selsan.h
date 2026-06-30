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

#ifndef TCMALLOC_SELSAN_SELSAN_H_
#define TCMALLOC_SELSAN_SELSAN_H_

#include <stddef.h>

#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/logging.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc::tcmalloc_internal::selsan {

#ifdef TCMALLOC_INTERNAL_SELSAN

#if defined(__PIE__) || defined(__PIC__)
inline constexpr uintptr_t kPieBuild = true;
#else
inline constexpr uintptr_t kPieBuild = false;
#endif

#if defined(__x86_64__)
// Note: this is not necessary equal to kAddressBits since we need to cover
// everything kernel can mmap, rather than just the heap.
inline constexpr uintptr_t kAddressSpaceBits = 47;
inline constexpr uintptr_t kTagShift = 57;
#ifdef TCMALLOC_INTERNAL_SELSAN_FAKE_MODE
// The fake mode allows to realistically benchmark selsan on x86 w/o LAM.
// We set kTagUnsetMask to always reset the same tag bit we set during
// retagging. As the result the top byte is always 0 (so no need for LAM).
// But the compiler cannot prove the tag increment won't overflow to other bits,
// so the instruction sequence should be equivalent to the real one, and all
// of the shadow accesses and instrumentation are exactly the same as well.
inline constexpr uintptr_t kTagUnsetMask = 2ul << kTagShift;
#else
// This mask is unset after incrementing tag on pointers to restore
// the canonical bit after potential tag overflow into the canonical bit.
inline constexpr uintptr_t kTagUnsetMask = 1ul << 63;
#endif
#elif defined(__aarch64__)
inline constexpr uintptr_t kAddressSpaceBits = 48;
inline constexpr uintptr_t kTagShift = 56;
inline constexpr uintptr_t kTagUnsetMask = 0;
#else
#error "Unsupported platform."
#endif

inline constexpr uintptr_t kShadowShift = 4;
inline constexpr uintptr_t kShadowScale = 1 << kShadowShift;

// In pie builds we use 0 shadow offset since it's the most efficient to encode
// in instructions. In non-pie builds we cannot use 0 since the executable
// is at 0, instead we use 4GB-2MB because (1) <4GB offsets can be encoded
// efficiently on x86, (2) we want the smallest offset from 4GB to give as much
// memory as possible to the executable, and (3) 2MB alignment allows to use
// huge pages for shadow.
#ifdef TCMALLOC_SELSAN_TEST_SHADOW_OVERRIDE
extern uintptr_t kShadowBase;
#else
inline constexpr uintptr_t kShadowBase =
    kPieBuild ? 0 : (1ul << 32) - (2ul << 20);
// Hex representation of the const for source grepping and compiler flags.
static_assert(kPieBuild || kShadowBase == 0xffe00000);
#endif
inline constexpr uintptr_t kShadowOffset = kPieBuild ? 64 << 10 : 0;

inline ABSL_ATTRIBUTE_ALWAYS_INLINE bool IsEnabled() {
  extern bool enabled;
  return enabled;
}

// Says if a given span allocation should be allocation from selsan page heap.
bool ShouldSample();

int SamplingPercent();
void SetSamplingPercent(int v);

inline ABSL_ATTRIBUTE_ALWAYS_INLINE size_t RoundUpObjectSize(size_t size) {
  TC_ASSERT(IsEnabled());
  return (size + kShadowScale - 1) & ~(kShadowScale - 1);
}

#if __has_builtin(__builtin_memset_inline)
template <size_t kBlockSize>
ABSL_ATTRIBUTE_ALWAYS_INLINE void SetTagTail(unsigned char* p, size_t size,
                                             unsigned char tag) {
  __builtin_memset_inline(
      p + (size + kShadowScale - 1 - kBlockSize * kShadowScale) / kShadowScale,
      tag, kBlockSize);
}

template <size_t kBlockSize>
ABSL_ATTRIBUTE_ALWAYS_INLINE void SetTagPair(unsigned char* p, size_t size,
                                             unsigned char tag) {
  __builtin_memset_inline(p, tag, kBlockSize);
  SetTagTail<kBlockSize>(p, size, tag);
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SetTag(uintptr_t ptr, size_t size,
                                                unsigned char tag) {
  TC_ASSERT_NE(size, 0);
  uintptr_t off = (ptr << (64 - kTagShift)) >> (64 - kTagShift + kShadowShift);
  auto* p = reinterpret_cast<unsigned char*>(kShadowBase + off);
  if (size <= 2 * kShadowScale) {
    SetTagPair<1>(p, size, tag);
  } else if (size <= 4 * kShadowScale) {
    SetTagPair<2>(p, size, tag);
  } else if (size <= 8 * kShadowScale) {
    SetTagPair<4>(p, size, tag);
  } else if (size <= 16 * kShadowScale) {
    SetTagPair<8>(p, size, tag);
  } else if (size <= 32 * kShadowScale) {
    SetTagPair<16>(p, size, tag);
  } else if (size <= 64 * kShadowScale) {
    SetTagPair<32>(p, size, tag);
  } else if (size <= 128 * kShadowScale) {
    // This is affected by clang codegen bug, which leads to duplicate register
    // initialization, but it's unclear how to nicely work-around it without
    // resorting to machine-specific intrinsics:
    // https://github.com/llvm/llvm-project/issues/69895
    SetTagPair<64>(p, size, tag);
  } else {
    const size_t kUnroll = 4;
    const size_t kBlockSize = 32;
    size_t i = 0;
    for (; i < size / kShadowScale - (kUnroll * kBlockSize - 1);
         i += kUnroll * kBlockSize) {
      for (size_t j = 0; j < kUnroll; j++) {
        __builtin_memset_inline(p + i + j * kBlockSize, tag, kBlockSize);
      }
      // Work-around clang bug https://github.com/llvm/llvm-project/issues/56876
      // Without this clang emits memset call for __builtin_memset_inline.
      asm("");
    }
    for (; i < size / kShadowScale - (kBlockSize - 1); i += kBlockSize) {
      __builtin_memset_inline(p + i, tag, kBlockSize);
      asm("");
    }
    SetTagTail<kBlockSize>(p, size, tag);
  }
}
#else   // #if __has_builtin(__builtin_memset_inline)
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void SetTag(uintptr_t ptr, size_t size,
                                                unsigned char tag) {
  TC_ASSERT_NE(size, 0);
  uintptr_t off = (ptr << (64 - kTagShift)) >> (64 - kTagShift + kShadowShift);
  auto* p = reinterpret_cast<unsigned char*>(kShadowBase + off);
  for (size_t i = 0; i < RoundUpObjectSize(size) / kShadowScale; i++) {
    p[i] = tag;
  }
}
#endif  // #if __has_builtin(__builtin_memset_inline)

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* RemoveTag(const void* ptr) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) &
                                 ((1ul << kTagShift) - 1));
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* ResetTag(void* ptr, size_t size) {
  TC_ASSERT(IsEnabled());
  TC_ASSERT_EQ(size % kShadowScale, 0);
  SetTag(reinterpret_cast<uintptr_t>(ptr), size, 0);
  return RemoveTag(ptr);
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE void* UpdateTag(void* ptr, size_t size) {
  TC_ASSERT(IsEnabled());
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  p += 1ul << (kTagShift + 1);
  p &= ~kTagUnsetMask;
  SetTag(reinterpret_cast<uintptr_t>(ptr), size, p >> 56);
  return reinterpret_cast<void*>(p);
}

void PrintTextStats(Printer& out);
void PrintPbtxtStats(PbtxtRegion& out);

#else  // #ifdef TCMALLOC_INTERNAL_SELSAN

inline size_t RoundUpObjectSize(size_t size) { return size; }
inline void* ResetTag(void* ptr, size_t size) { return ptr; }
inline void* UpdateTag(void* ptr, size_t size) { return ptr; }
inline void* RemoveTag(const void* ptr) { return const_cast<void*>(ptr); }
inline bool IsEnabled() { return false; }
inline bool ShouldSample() { return false; }
inline int SamplingPercent() { return 0; }
inline void SetSamplingPercent(int v) {}
inline void PrintTextStats(Printer& out) {}
inline void PrintPbtxtStats(PbtxtRegion& out) {}

#endif  // #ifdef TCMALLOC_INTERNAL_SELSAN

}  // namespace tcmalloc::tcmalloc_internal::selsan
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_SELSAN_SELSAN_H_
