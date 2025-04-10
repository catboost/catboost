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

#include <fcntl.h>
#include <string.h>

#include <algorithm>
#include <atomic>

#include "absl/base/attributes.h"
#include "tcmalloc/common.h"
#include "tcmalloc/experiment.h"
#include "tcmalloc/guarded_page_allocator.h"
#include "tcmalloc/internal/cache_topology.h"
#include "tcmalloc/internal/environment.h"
#include "tcmalloc/internal/linked_list.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/optimization.h"
#include "tcmalloc/internal/util.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/tracking.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

absl::string_view TransferCacheImplementationToLabel(
    TransferCacheImplementation type) {
  switch (type) {
    case TransferCacheImplementation::Legacy:
      return "LEGACY";
    case TransferCacheImplementation::None:
      return "NO_TRANSFERCACHE";
    case TransferCacheImplementation::Ring:
      return "RING";
    default:
      ASSUME(false);
  }
}

#ifndef TCMALLOC_SMALL_BUT_SLOW

size_t StaticForwarder::class_to_size(int size_class) {
  return Static::sizemap().class_to_size(size_class);
}
size_t StaticForwarder::num_objects_to_move(int size_class) {
  return Static::sizemap().num_objects_to_move(size_class);
}
void *StaticForwarder::Alloc(size_t size, int alignment) {
  return Static::arena().Alloc(size, alignment);
}

void ShardedTransferCacheManager::Init() {
  if (!IsExperimentActive(
          Experiment::TEST_ONLY_TCMALLOC_SHARDED_TRANSFER_CACHE)) {
    return;
  }
  num_shards_ = BuildCpuToL3CacheMap(l3_cache_index_);
  cache_ = reinterpret_cast<Cache *>(Static::arena().Alloc(
      sizeof(Cache) * kNumClasses * num_shards_, ABSL_CACHELINE_SIZE));
  ASSERT(cache_ != nullptr);
  for (int shard = 0; shard < num_shards_; ++shard) {
    for (int cl = 0; cl < kNumClasses; ++cl) {
      const int index = shard * kNumClasses + cl;
      const int size_per_object = Static::sizemap().class_to_size(cl);
      static constexpr int k12MB = 12 << 20;
      static constexpr int min_size = 4096;
      const int use_this_size_class = size_per_object >= min_size;
      const int capacity = use_this_size_class ? k12MB / size_per_object : 0;
      active_for_class_[cl] = use_this_size_class;
      new (&cache_[index].tc)
          TransferCache(nullptr, capacity > 0 ? cl : 0, {capacity, capacity});
      cache_[index].tc.freelist().Init(cl);
    }
  }
}

size_t ShardedTransferCacheManager::TotalBytes() {
  if (cache_ == nullptr) return 0;
  size_t out = 0;
  for (int shard = 0; shard < num_shards_; ++shard) {
    for (int cl = 0; cl < kNumClasses; ++cl) {
      const int bytes_per_entry = Static::sizemap().class_to_size(cl);
      if (bytes_per_entry <= 0) continue;
      const int index = shard * kNumClasses + cl;
      out += cache_[index].tc.tc_length() * bytes_per_entry;
    }
  }
  return out;
}

void ShardedTransferCacheManager::BackingTransferCache::InsertRange(
    absl::Span<void *> batch) const {
  Static::transfer_cache().InsertRange(size_class_, batch);
}

ABSL_MUST_USE_RESULT int
ShardedTransferCacheManager::BackingTransferCache::RemoveRange(void **batch,
                                                               int n) const {
  return Static::transfer_cache().RemoveRange(size_class_, batch, n);
}

TransferCacheImplementation TransferCacheManager::ChooseImplementation() {
  // Prefer ring, if we're forcing it on.
  if (IsExperimentActive(
          Experiment::TEST_ONLY_TCMALLOC_RING_BUFFER_TRANSFER_CACHE)) {
    return TransferCacheImplementation::Ring;
  }

  // Consider opt-outs
  const char *e = thread_safe_getenv("TCMALLOC_INTERNAL_TRANSFERCACHE_CONTROL");
  if (e) {
    if (e[0] == '0') {
      return TransferCacheImplementation::Legacy;
    }
    if (e[0] == '1') {
      return TransferCacheImplementation::Ring;
    }
    Crash(kCrash, __FILE__, __LINE__, "bad env var", e);
  }

  // Otherwise, default to ring.
  return TransferCacheImplementation::Ring;
}

int TransferCacheManager::DetermineSizeClassToEvict() {
  int t = next_to_evict_.load(std::memory_order_relaxed);
  if (t >= kNumClasses) t = 1;
  next_to_evict_.store(t + 1, std::memory_order_relaxed);

  // Ask nicely first.
  if (implementation_ == TransferCacheImplementation::Ring) {
    if (cache_[t].rbtc.HasSpareCapacity(t)) return t;
  } else {
    if (cache_[t].tc.HasSpareCapacity(t)) return t;
  }

  // But insist on the second try.
  t = next_to_evict_.load(std::memory_order_relaxed);
  if (t >= kNumClasses) t = 1;
  next_to_evict_.store(t + 1, std::memory_order_relaxed);
  return t;
}

#endif

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END
