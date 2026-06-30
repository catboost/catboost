// Copyright 2022 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: log/internal/log_sink_set.h
// -----------------------------------------------------------------------------

#ifndef Y_ABSL_LOG_INTERNAL_LOG_SINK_SET_H_
#define Y_ABSL_LOG_INTERNAL_LOG_SINK_SET_H_

#include "y_absl/base/config.h"
#include "y_absl/log/log_entry.h"
#include "y_absl/log/log_sink.h"
#include "y_absl/types/span.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace log_internal {

// Returns true if a globally-registered `LogSink`'s `Send()` is currently
// being invoked on this thread.
bool ThreadIsLoggingToLogSink();

// This function may log to two sets of sinks:
//
// * If `extra_sinks_only` is true, it will dispatch only to `extra_sinks`.
//   `LogMessage::ToSinkAlso` and `LogMessage::ToSinkOnly` are used to attach
//    extra sinks to the entry.
// * Otherwise it will also log to the global sinks set. This set is managed
//   by `y_absl::AddLogSink` and `y_absl::RemoveLogSink`.
void LogToSinks(const y_absl::LogEntry& entry,
                y_absl::Span<y_absl::LogSink*> extra_sinks, bool extra_sinks_only);

// Implementation for operations with log sink set.
void AddLogSink(y_absl::LogSink* sink);
void RemoveLogSink(y_absl::LogSink* sink);
void FlushLogSinks();

}  // namespace log_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_LOG_INTERNAL_LOG_SINK_SET_H_
