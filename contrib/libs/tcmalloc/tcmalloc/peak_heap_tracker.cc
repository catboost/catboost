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

#include "tcmalloc/peak_heap_tracker.h"

#include <stdio.h>

#include "absl/base/internal/spinlock.h"
#include "absl/memory/memory.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/page_heap_allocator.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/sampler.h"
#include "tcmalloc/span.h"
#include "tcmalloc/stack_trace_table.h"
#include "tcmalloc/static_vars.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

bool PeakHeapTracker::IsNewPeak() {
  return peak_sampled_heap_size_.value() == 0 ||
         (static_cast<double>(Static::sampled_objects_size_.value()) /
              peak_sampled_heap_size_.value() >
          Parameters::peak_sampling_heap_growth_fraction());
}

void PeakHeapTracker::MaybeSaveSample() {
  if (Parameters::peak_sampling_heap_growth_fraction() <= 0 || !IsNewPeak()) {
    return;
  }

  absl::base_internal::SpinLockHolder h(&pageheap_lock);

  // double-check in case another allocation was sampled (or a sampled
  // allocation freed) while we were waiting for the lock
  if (!IsNewPeak()) {
    return;
  }
  peak_sampled_heap_size_.LossyAdd(Static::sampled_objects_size_.value() -
                                   peak_sampled_heap_size_.value());

  StackTrace *t = peak_sampled_span_stacks_, *next = nullptr;
  while (t != nullptr) {
    next = reinterpret_cast<StackTrace*>(t->stack[kMaxStackDepth - 1]);
    Static::DestroySampleUserData(t->user_data);
    Static::stacktrace_allocator().Delete(t);
    t = next;
  }

  next = nullptr;
  for (Span* s : Static::sampled_objects_) {
    t = Static::stacktrace_allocator().New();

    StackTrace* sampled_stack = s->sampled_stack();
    *t = *sampled_stack;
    t->user_data = Static::CopySampleUserData(sampled_stack->user_data);
    if (t->depth == kMaxStackDepth) {
      t->depth = kMaxStackDepth - 1;
    }
    t->stack[kMaxStackDepth - 1] = reinterpret_cast<void*>(next);
    next = t;
  }
  peak_sampled_span_stacks_ = t;
}

std::unique_ptr<ProfileBase> PeakHeapTracker::DumpSample() const {
  auto profile = absl::make_unique<StackTraceTable>(
      ProfileType::kPeakHeap, Sampler::GetSamplePeriod(), true, true);

  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  for (StackTrace* t = peak_sampled_span_stacks_; t != nullptr;
       t = reinterpret_cast<StackTrace*>(t->stack[kMaxStackDepth - 1])) {
    profile->AddTrace(1.0, *t);
  }
  return profile;
}

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
