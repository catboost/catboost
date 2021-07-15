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

#include "tcmalloc/transfer_cache.h"

#include <string.h>

#include <algorithm>
#include <atomic>

#include "absl/base/attributes.h"
#include "tcmalloc/common.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/tracking.h"

namespace tcmalloc {
#ifndef TCMALLOC_SMALL_BUT_SLOW

size_t TransferCacheManager::class_to_size(int size_class) {
  return Static::sizemap().class_to_size(size_class);
}
size_t TransferCacheManager::num_objects_to_move(int size_class) {
  return Static::sizemap().num_objects_to_move(size_class);
}
void* TransferCacheManager::Alloc(size_t size) {
  return Static::arena().Alloc(size);
}

int TransferCacheManager::DetermineSizeClassToEvict() {
  int t = next_to_evict_.load(std::memory_order_relaxed);
  if (t >= kNumClasses) t = 1;
  next_to_evict_.store(t + 1, std::memory_order_relaxed);

  // Ask nicely first.
  if (use_lock_free_cache_) {
    if (cache_[t].lock_free.HasSpareCapacity()) return t;
  } else {
    if (cache_[t].legacy.HasSpareCapacity()) return t;
  }

  // But insist on the second try.
  t = next_to_evict_.load(std::memory_order_relaxed);
  if (t >= kNumClasses) t = 1;
  next_to_evict_.store(t + 1, std::memory_order_relaxed);
  return t;
}

#endif
}  // namespace tcmalloc
