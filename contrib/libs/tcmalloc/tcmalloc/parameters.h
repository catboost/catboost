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

#ifndef TCMALLOC_PARAMETERS_H_
#define TCMALLOC_PARAMETERS_H_

#include <atomic>
#include <cmath>
#include <string>

#include "absl/base/internal/spinlock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tcmalloc/internal/config.h"
#include "tcmalloc/internal/parameter_accessors.h"
#include "tcmalloc/malloc_extension.h"

GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

class Parameters {
 public:
  static MallocExtension::BytesPerSecond background_release_rate() {
    return background_release_rate_.load(std::memory_order_relaxed);
  }

  static void set_background_release_rate(
      MallocExtension::BytesPerSecond value) {
    TCMalloc_Internal_SetBackgroundReleaseRate(static_cast<size_t>(value));
  }

  static uint64_t heap_size_hard_limit();
  static void set_heap_size_hard_limit(uint64_t value);

  static bool hpaa_subrelease();
  static void set_hpaa_subrelease(bool value);

  static int64_t guarded_sampling_rate() {
    return guarded_sampling_rate_.load(std::memory_order_relaxed);
  }

  static void set_guarded_sampling_rate(int64_t value) {
    TCMalloc_Internal_SetGuardedSamplingRate(value);
  }

  static int32_t max_per_cpu_cache_size() {
    return max_per_cpu_cache_size_.load(std::memory_order_relaxed);
  }

  static void set_max_per_cpu_cache_size(int32_t value) {
    TCMalloc_Internal_SetMaxPerCpuCacheSize(value);
  }

  static int64_t max_total_thread_cache_bytes() {
    return max_total_thread_cache_bytes_.load(std::memory_order_relaxed);
  }

  static void set_max_total_thread_cache_bytes(int64_t value) {
    TCMalloc_Internal_SetMaxTotalThreadCacheBytes(value);
  }

  static double peak_sampling_heap_growth_fraction() {
    return peak_sampling_heap_growth_fraction_.load(std::memory_order_relaxed);
  }

  static void set_peak_sampling_heap_growth_fraction(double value) {
    TCMalloc_Internal_SetPeakSamplingHeapGrowthFraction(value);
  }

  static bool shuffle_per_cpu_caches() {
    return shuffle_per_cpu_caches_enabled_.load(std::memory_order_relaxed);
  }

  static bool reclaim_idle_per_cpu_caches() {
    return reclaim_idle_per_cpu_caches_enabled_.load(std::memory_order_relaxed);
  }

  static bool lazy_per_cpu_caches() {
    return lazy_per_cpu_caches_enabled_.load(std::memory_order_relaxed);
  }

  static void set_lazy_per_cpu_caches(bool value) {
    TCMalloc_Internal_SetLazyPerCpuCachesEnabled(value);
  }

  static bool per_cpu_caches() {
    return per_cpu_caches_enabled_.load(std::memory_order_relaxed);
  }

  static void set_per_cpu_caches(bool value) {
    TCMalloc_Internal_SetPerCpuCachesEnabled(value);
  }

  static int64_t profile_sampling_rate() {
    return profile_sampling_rate_.load(std::memory_order_relaxed);
  }

  static void set_profile_sampling_rate(int64_t value) {
    TCMalloc_Internal_SetProfileSamplingRate(value);
  }

  static void set_filler_skip_subrelease_interval(absl::Duration value) {
    TCMalloc_Internal_SetHugePageFillerSkipSubreleaseInterval(value);
  }

  static absl::Duration filler_skip_subrelease_interval();

 private:
  friend void ::TCMalloc_Internal_SetBackgroundReleaseRate(size_t v);
  friend void ::TCMalloc_Internal_SetGuardedSamplingRate(int64_t v);
  friend void ::TCMalloc_Internal_SetHPAASubrelease(bool v);
  friend void ::TCMalloc_Internal_SetShufflePerCpuCachesEnabled(bool v);
  friend void ::TCMalloc_Internal_SetReclaimIdlePerCpuCachesEnabled(bool v);
  friend void ::TCMalloc_Internal_SetLazyPerCpuCachesEnabled(bool v);
  friend void ::TCMalloc_Internal_SetMaxPerCpuCacheSize(int32_t v);
  friend void ::TCMalloc_Internal_SetMaxTotalThreadCacheBytes(int64_t v);
  friend void ::TCMalloc_Internal_SetPeakSamplingHeapGrowthFraction(double v);
  friend void ::TCMalloc_Internal_SetPerCpuCachesEnabled(bool v);
  friend void ::TCMalloc_Internal_SetProfileSamplingRate(int64_t v);

  friend void ::TCMalloc_Internal_SetHugePageFillerSkipSubreleaseInterval(
      absl::Duration v);

  static std::atomic<MallocExtension::BytesPerSecond> background_release_rate_;
  static std::atomic<int64_t> guarded_sampling_rate_;
  static std::atomic<bool> shuffle_per_cpu_caches_enabled_;
  static std::atomic<bool> reclaim_idle_per_cpu_caches_enabled_;
  static std::atomic<bool> lazy_per_cpu_caches_enabled_;
  static std::atomic<int32_t> max_per_cpu_cache_size_;
  static std::atomic<int64_t> max_total_thread_cache_bytes_;
  static std::atomic<double> peak_sampling_heap_growth_fraction_;
  static std::atomic<bool> per_cpu_caches_enabled_;
  static std::atomic<int64_t> profile_sampling_rate_;
};

}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_PARAMETERS_H_
