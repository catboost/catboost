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

#ifndef TCMALLOC_INTERNAL_CONFIG_H_
#define TCMALLOC_INTERNAL_CONFIG_H_

#include <stddef.h>

// TCMALLOC_HAVE_SCHED_GETCPU is defined when the system implements
// sched_getcpu(3) as by glibc and it's imitators.
#if defined(__linux__) || defined(__ros__)
#define TCMALLOC_HAVE_SCHED_GETCPU 1
#else
#undef TCMALLOC_HAVE_SCHED_GETCPU
#endif

namespace tcmalloc {

#if defined __x86_64__
// All current and planned x86_64 processors only look at the lower 48 bits
// in virtual to physical address translation.  The top 16 are thus unused.
// TODO(b/134686025): Under what operating systems can we increase it safely to
// 17? This lets us use smaller page maps.  On first allocation, a 36-bit page
// map uses only 96 KB instead of the 4.5 MB used by a 52-bit page map.
inline constexpr int kAddressBits =
    (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);
#elif defined __powerpc64__ && defined __linux__
// Linux(4.12 and above) on powerpc64 supports 128TB user virtual address space
// by default, and up to 512TB if user space opts in by specifing hint in mmap.
// See comments in arch/powerpc/include/asm/processor.h
// and arch/powerpc/mm/mmap.c.
inline constexpr int kAddressBits =
    (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 49);
#elif defined __aarch64__ && defined __linux__
// According to Documentation/arm64/memory.txt of kernel 3.16,
// AARCH64 kernel supports 48-bit virtual addresses for both user and kernel.
inline constexpr int kAddressBits =
    (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);
#else
inline constexpr int kAddressBits = 8 * sizeof(void*);
#endif

#if defined(__x86_64__)
// x86 has 2 MiB huge pages
static constexpr size_t kHugePageShift = 21;
#elif defined(__PPC64__)
static constexpr size_t kHugePageShift = 24;
#elif defined __aarch64__ && defined __linux__
static constexpr size_t kHugePageShift = 21;
#else
// ...whatever, guess something big-ish
static constexpr size_t kHugePageShift = 21;
#endif

static constexpr size_t kHugePageSize = static_cast<size_t>(1)
                                        << kHugePageShift;

}  // namespace tcmalloc

#endif  // TCMALLOC_INTERNAL_CONFIG_H_
