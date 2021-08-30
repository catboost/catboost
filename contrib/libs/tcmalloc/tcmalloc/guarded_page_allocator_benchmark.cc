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

#include <unistd.h>

#include "absl/base/internal/spinlock.h"
#include "benchmark/benchmark.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/logging.h"

namespace tcmalloc {
namespace {

static constexpr size_t kMaxGpaPages =
    tcmalloc::GuardedPageAllocator::kGpaMaxPages;

// Size of pages used by GuardedPageAllocator.
static size_t PageSize() {
  static const size_t page_size =
      std::max(kPageSize, static_cast<size_t>(getpagesize()));
  return page_size;
}

void BM_AllocDealloc(benchmark::State& state) {
  static tcmalloc::GuardedPageAllocator* gpa = []() {
    auto gpa = new tcmalloc::GuardedPageAllocator;
    absl::base_internal::SpinLockHolder h(&tcmalloc::pageheap_lock);
    gpa->Init(kMaxGpaPages, kMaxGpaPages);
    gpa->AllowAllocations();
    return gpa;
  }();
  size_t alloc_size = state.range(0);
  for (auto _ : state) {
    char* ptr = reinterpret_cast<char*>(gpa->Allocate(alloc_size, 0));
    CHECK_CONDITION(ptr != nullptr);
    ptr[0] = 'X';               // Page fault first page.
    ptr[alloc_size - 1] = 'X';  // Page fault last page.
    gpa->Deallocate(ptr);
  }
}

BENCHMARK(BM_AllocDealloc)->Range(1, PageSize());
BENCHMARK(BM_AllocDealloc)->Arg(1)->ThreadRange(1, kMaxGpaPages);

}  // namespace
}  // namespace tcmalloc
