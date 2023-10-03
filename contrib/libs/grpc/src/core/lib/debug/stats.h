//
//
// Copyright 2017 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//

#ifndef GRPC_SRC_CORE_LIB_DEBUG_STATS_H
#define GRPC_SRC_CORE_LIB_DEBUG_STATS_H

#include <grpc/support/port_platform.h>

#include <stdint.h>

#include <util/generic/string.h>
#include <util/string/cast.h>
#include <vector>

#include "y_absl/strings/string_view.h"
#include "y_absl/types/span.h"

#include "src/core/lib/debug/histogram_view.h"
#include "src/core/lib/debug/stats_data.h"
#include "src/core/lib/gprpp/no_destruct.h"

namespace grpc_core {

inline GlobalStatsCollector& global_stats() {
  return *NoDestructSingleton<GlobalStatsCollector>::Get();
}

namespace stats_detail {
TString StatsAsJson(y_absl::Span<const uint64_t> counters,
                        y_absl::Span<const y_absl::string_view> counter_name,
                        y_absl::Span<const HistogramView> histograms,
                        y_absl::Span<const y_absl::string_view> histogram_name);
}

template <typename T>
TString StatsAsJson(T* data) {
  std::vector<HistogramView> histograms;
  for (int i = 0; i < static_cast<int>(T::Histogram::COUNT); i++) {
    histograms.push_back(
        data->histogram(static_cast<typename T::Histogram>(i)));
  }
  return stats_detail::StatsAsJson(
      y_absl::Span<const uint64_t>(data->counters,
                                 static_cast<int>(T::Counter::COUNT)),
      T::counter_name, histograms, T::histogram_name);
}

}  // namespace grpc_core

#endif  // GRPC_SRC_CORE_LIB_DEBUG_STATS_H
