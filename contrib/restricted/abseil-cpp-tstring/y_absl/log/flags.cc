//
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

#include "y_absl/log/internal/flags.h"

#include <stddef.h>

#include <algorithm>
#include <cstdlib>
#include <util/generic/string.h>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/log_severity.h"
#include "y_absl/flags/flag.h"
#include "y_absl/flags/marshalling.h"
#include "y_absl/log/globals.h"
#include "y_absl/log/internal/config.h"
#include "y_absl/log/internal/vlog_config.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace log_internal {
namespace {

void SyncLoggingFlags() {
  y_absl::SetFlag(&FLAGS_minloglevel, static_cast<int>(y_absl::MinLogLevel()));
  y_absl::SetFlag(&FLAGS_log_prefix, y_absl::ShouldPrependLogPrefix());
}

bool RegisterSyncLoggingFlags() {
  log_internal::SetLoggingGlobalsListener(&SyncLoggingFlags);
  return true;
}

Y_ABSL_ATTRIBUTE_UNUSED const bool unused = RegisterSyncLoggingFlags();

template <typename T>
T GetFromEnv(const char* varname, T dflt) {
  const char* val = ::getenv(varname);
  if (val != nullptr) {
    TString err;
    Y_ABSL_INTERNAL_CHECK(y_absl::ParseFlag(val, &dflt, &err), err.c_str());
  }
  return dflt;
}

constexpr y_absl::LogSeverityAtLeast StderrThresholdDefault() {
  return y_absl::LogSeverityAtLeast::kError;
}

}  // namespace
}  // namespace log_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

Y_ABSL_FLAG(int, stderrthreshold,
          static_cast<int>(y_absl::log_internal::StderrThresholdDefault()),
          "Log messages at or above this threshold level are copied to stderr.")
    .OnUpdate([] {
      y_absl::log_internal::RawSetStderrThreshold(
          static_cast<y_absl::LogSeverityAtLeast>(
              y_absl::GetFlag(FLAGS_stderrthreshold)));
    });

Y_ABSL_FLAG(int, minloglevel, static_cast<int>(y_absl::LogSeverityAtLeast::kInfo),
          "Messages logged at a lower level than this don't actually "
          "get logged anywhere")
    .OnUpdate([] {
      y_absl::log_internal::RawSetMinLogLevel(
          static_cast<y_absl::LogSeverityAtLeast>(
              y_absl::GetFlag(FLAGS_minloglevel)));
    });

Y_ABSL_FLAG(TString, log_backtrace_at, "",
          "Emit a backtrace when logging at file:linenum.")
    .OnUpdate([] {
      const TString log_backtrace_at =
          y_absl::GetFlag(FLAGS_log_backtrace_at);
      if (log_backtrace_at.empty()) {
        y_absl::ClearLogBacktraceLocation();
        return;
      }

      const size_t last_colon = log_backtrace_at.rfind(':');
      if (last_colon == log_backtrace_at.npos) {
        y_absl::ClearLogBacktraceLocation();
        return;
      }

      const y_absl::string_view file =
          y_absl::string_view(log_backtrace_at).substr(0, last_colon);
      int line;
      if (!y_absl::SimpleAtoi(
              y_absl::string_view(log_backtrace_at).substr(last_colon + 1),
              &line)) {
        y_absl::ClearLogBacktraceLocation();
        return;
      }
      y_absl::SetLogBacktraceLocation(file, line);
    });

Y_ABSL_FLAG(bool, log_prefix, true,
          "Prepend the log prefix to the start of each log line")
    .OnUpdate([] {
      y_absl::log_internal::RawEnableLogPrefix(y_absl::GetFlag(FLAGS_log_prefix));
    });

Y_ABSL_FLAG(int, v, 0,
          "Show all VLOG(m) messages for m <= this. Overridable by --vmodule.")
    .OnUpdate([] {
      y_absl::log_internal::UpdateGlobalVLogLevel(y_absl::GetFlag(FLAGS_v));
    });

Y_ABSL_FLAG(
    TString, vmodule, "",
    "per-module log verbosity level."
    " Argument is a comma-separated list of <module name>=<log level>."
    " <module name> is a glob pattern, matched against the filename base"
    " (that is, name ignoring .cc/.h./-inl.h)."
    " A pattern without slashes matches just the file name portion, otherwise"
    " the whole file path below the workspace root"
    " (still without .cc/.h./-inl.h) is matched."
    " ? and * in the glob pattern match any single or sequence of characters"
    " respectively including slashes."
    " <log level> overrides any value given by --v.")
    .OnUpdate([] {
      y_absl::log_internal::UpdateVModule(y_absl::GetFlag(FLAGS_vmodule));
    });
