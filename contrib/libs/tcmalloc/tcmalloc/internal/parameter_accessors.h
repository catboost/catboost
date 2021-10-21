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

#ifndef TCMALLOC_INTERNAL_PARAMETER_ACCESSORS_H_
#define TCMALLOC_INTERNAL_PARAMETER_ACCESSORS_H_

#include "absl/base/attributes.h"
#include "absl/time/time.h"

extern "C" {

ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetBackgroundReleaseRate(
    size_t value);
ABSL_ATTRIBUTE_WEAK uint64_t TCMalloc_Internal_GetHeapSizeHardLimit();
ABSL_ATTRIBUTE_WEAK bool TCMalloc_Internal_GetHPAASubrelease();
ABSL_ATTRIBUTE_WEAK void
TCMalloc_Internal_GetHugePageFillerSkipSubreleaseInterval(absl::Duration* v);
ABSL_ATTRIBUTE_WEAK bool TCMalloc_Internal_GetShufflePerCpuCachesEnabled();
ABSL_ATTRIBUTE_WEAK bool TCMalloc_Internal_GetReclaimIdlePerCpuCachesEnabled();
ABSL_ATTRIBUTE_WEAK bool TCMalloc_Internal_GetLazyPerCpuCachesEnabled();
ABSL_ATTRIBUTE_WEAK double
TCMalloc_Internal_GetPeakSamplingHeapGrowthFraction();
ABSL_ATTRIBUTE_WEAK bool TCMalloc_Internal_GetPerCpuCachesEnabled();
ABSL_ATTRIBUTE_WEAK size_t TCMalloc_Internal_GetStats(char* buffer,
                                                      size_t buffer_length);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetGuardedSamplingRate(int64_t v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetHeapSizeHardLimit(uint64_t v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetHPAASubrelease(bool v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetShufflePerCpuCachesEnabled(
    bool v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetReclaimIdlePerCpuCachesEnabled(
    bool v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetLazyPerCpuCachesEnabled(bool v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetMaxPerCpuCacheSize(int32_t v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetMaxTotalThreadCacheBytes(
    int64_t v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetPeakSamplingHeapGrowthFraction(
    double v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetPerCpuCachesEnabled(bool v);
ABSL_ATTRIBUTE_WEAK void TCMalloc_Internal_SetProfileSamplingRate(int64_t v);
ABSL_ATTRIBUTE_WEAK void
TCMalloc_Internal_SetHugePageFillerSkipSubreleaseInterval(absl::Duration v);
}

#endif  // TCMALLOC_INTERNAL_PARAMETER_ACCESSORS_H_
