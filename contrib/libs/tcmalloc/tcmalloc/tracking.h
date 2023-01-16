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

#ifndef TCMALLOC_TRACKING_H_
#define TCMALLOC_TRACKING_H_
// Optional support for tracking various stats in tcmalloc.  For each
// sizeclass, we track:
//  * # of mallocs
//     * ...that hit the fast path
//  * # of frees
//     * ...that hit the fast path
//
// both on each CPU and on each thread.
//
// If disabled (TCMALLOC_TRACK_ALLOCS not defined), it has no runtime cost in
// time or space.
//
// If enabled and an implementation provided, we issue calls to record various
// statistics about cache hit rates.

#include <stddef.h>
#include <sys/types.h>

#include <map>
#include <string>

#include "absl/base/internal/per_thread_tls.h"
#include "absl/base/internal/spinlock.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/internal/percpu.h"
#include "tcmalloc/malloc_extension.h"

// Uncomment here or pass --copt=-DTCMALLOC_TRACK_ALLOCS at build time if you
// want tracking.
#ifndef TCMALLOC_TRACK_ALLOCS
// #define TCMALLOC_TRACK_ALLOCS
#endif
GOOGLE_MALLOC_SECTION_BEGIN
namespace tcmalloc {
namespace tcmalloc_internal {

#if 1
#define TCMALLOC_HAVE_TRACKING 0
#endif

// We track various kinds of events on each thread and each cpu.  Each
// event is broken down by sizeclass where it happened.
// To track a new event, add a enum value here, insert calls to
// Tracking::Report() where the event occurs, and add a printable name
// to the event in kTrackingStatNames (in tracking.cc).  Optionally
// print the stat somehow in State::Print.
enum TrackingStat {
  kMallocHit = 0,   // malloc that took the fast path
  kMallocMiss = 1,  // malloc that didn't
  kFreeHit = 2,     // ibid. for free
  kFreeMiss = 3,
  kFreeScavenges = 4,    // # of frees that leads to scavenge
  kFreeTruncations = 5,  // # of frees that leads to list truncation
  kTCInsertHit = 6,  // # of times the returned object list hits transfer cache.
  kTCInsertMiss = 7,  // # of times the object list misses the transfer cache.
  kTCRemoveHit = 8,   // # of times object list fetching hits transfer cache.
  kTCRemoveMiss = 9,  // # of times object list fetching misses transfer cache.
  kTCElementsPlunder = 10,  // # of elements plundered from the transfer cache.
  kNumTrackingStats = 11,
};

namespace tracking {

// Report <count> occurences of <stat> associated with sizeclass <cl>.
void Report(TrackingStat stat, size_t cl, ssize_t count);

// Dump all tracking data to <out>.  We could support various other
// mechanisms for data delivery without too much trouble...
void Print(Printer* out);

// Call on startup during tcmalloc initialization.
void Init();

// Fill <result> with information for each stat type (broken down by
// sizeclass if level == kDetailed.)
void GetProperties(std::map<std::string, MallocExtension::Property>* result);

#if !TCMALLOC_HAVE_TRACKING
// no tracking, these are all no-ops
inline void Report(TrackingStat stat, size_t cl, ssize_t count) {}
inline void Print(Printer* out) {}
inline void Init() {}
inline void GetProperties(
    std::map<std::string, MallocExtension::Property>* result) {}
#endif

}  // namespace tracking
}  // namespace tcmalloc_internal
}  // namespace tcmalloc
GOOGLE_MALLOC_SECTION_END

#endif  // TCMALLOC_TRACKING_H_
