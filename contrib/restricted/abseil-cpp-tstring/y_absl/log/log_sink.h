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
// File: log/log_sink.h
// -----------------------------------------------------------------------------
//
// This header declares the interface class `y_absl::LogSink`.

#ifndef Y_ABSL_LOG_LOG_SINK_H_
#define Y_ABSL_LOG_LOG_SINK_H_

#include "y_absl/base/config.h"
#include "y_absl/log/log_entry.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

// y_absl::LogSink
//
// `y_absl::LogSink` is an interface which can be extended to intercept and
// process particular messages (with `LOG.ToSinkOnly()` or
// `LOG.ToSinkAlso()`) or all messages (if registered with
// `y_absl::AddLogSink`).  Implementations must not take any locks that might be
// held by the `LOG` caller.
class LogSink {
 public:
  virtual ~LogSink() = default;

  // LogSink::Send()
  //
  // `Send` is called synchronously during the log statement.  `Send` must be
  // thread-safe.
  //
  // It is safe to use `LOG` within an implementation of `Send`.  `ToSinkOnly`
  // and `ToSinkAlso` are safe in general but can be used to create an infinite
  // loop if you try.
  virtual void Send(const y_absl::LogEntry& entry) = 0;

  // LogSink::Flush()
  //
  // Sinks that buffer messages should override this method to flush the buffer
  // and return.  `Flush` must be thread-safe.
  virtual void Flush() {}

 protected:
  LogSink() = default;
  // Implementations may be copyable and/or movable.
  LogSink(const LogSink&) = default;
  LogSink& operator=(const LogSink&) = default;

 private:
  // https://lld.llvm.org/missingkeyfunction.html#missing-key-function
  virtual void KeyFunction() const final;  // NOLINT(readability/inheritance)
};

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_LOG_LOG_SINK_H_
